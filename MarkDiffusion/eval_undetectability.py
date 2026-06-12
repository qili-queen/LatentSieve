"""Empirical undetectability evaluation for MIRAGE watermarked latents/images.

Tests whether watermarked latents/images are statistically distinguishable from
unwatermarked ones. For a CCS submission claiming computational indistinguishability.

Tests performed:
  1. Latent-space statistical tests (KS, Anderson-Darling, chi-squared sign,
     norm test, moment comparison)
  2. Classifier-based detection (logistic regression + MLP on latents and CLIP
     features)
  3. Image-space quality tests (FID, CLIP score, Inception Score)
  4. Comparison with RSWST (control: shows RSWST IS distinguishable)

References:
  - Christ & Gunn, "Pseudorandom Error-Correcting Codes", ICLR 2025
  - T2SMark (text-to-speech watermarking undetectability evaluation)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats as scipy_stats

from runtime_utils import build_cached_pipe, configure_runtime, resolve_device
from utils.diffusion_config import DiffusionConfig
from utils.media_utils import get_random_latents
from watermark.auto_watermark import AutoWatermark

configure_runtime()

PROMPTS = [
    "A beautiful sunset over the ocean with golden clouds",
    "A cinematic mountain lake at dawn with pine trees and fog",
    "A close-up portrait of an elderly fisherman with weathered skin",
    "A rainy Tokyo street at night with neon reflections and umbrellas",
    "A red vintage car parked in a wide desert under blue sky",
    "A cozy independent bookstore interior with warm lamps and wooden shelves",
    "A snow-covered cabin in a dense pine forest at twilight",
    "A macro photograph of a bee on a bright sunflower",
    "An aerial view of a coastal city at sunrise with soft haze",
    "A futuristic laboratory with glass walls, white robots, and holograms",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_jsonable(value):
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def reseed_watermark(watermark, seed):
    """Re-seed a watermark object so the next generation uses a fresh latent."""
    watermark.config.gen_seed = seed
    watermark.config.init_latents_seed = seed
    latents_rng = torch.Generator(device=watermark.config.device)
    latents_rng.manual_seed(seed)
    latent_kwargs = {
        "height": watermark.config.image_size[0],
        "width": watermark.config.image_size[1],
        "generator": latents_rng,
    }
    if watermark.config.num_frames >= 1:
        latent_kwargs["num_frames"] = watermark.config.num_frames
    watermark.config.init_latents = get_random_latents(
        watermark.config.pipe, **latent_kwargs,
    ).to(watermark.config.device, dtype=watermark.config.dtype)


def get_latent_pair(watermark, seed):
    """Return (z_T, z'_T) -- the unwatermarked and watermarked latent tensors.

    z_T is simply config.init_latents (the random Gaussian draw).
    z'_T is produced by utils.inject_watermark() which applies sign-flip encoding.
    Both are returned BEFORE diffusion generation.
    """
    reseed_watermark(watermark, seed)
    z_T = watermark.config.init_latents.clone().detach().cpu().float()
    z_prime_T = watermark.utils.inject_watermark().clone().detach().cpu().float()
    return z_T, z_prime_T


def get_rswst_latent_pair(watermark, seed):
    """Same as get_latent_pair but for RSWST watermark."""
    reseed_watermark(watermark, seed)
    z_T = watermark.config.init_latents.clone().detach().cpu().float()
    z_prime_T = watermark.utils.inject_watermark().clone().detach().cpu().float()
    return z_T, z_prime_T


# ---------------------------------------------------------------------------
# 1. Latent-space statistical tests
# ---------------------------------------------------------------------------

def per_coordinate_ks_test(latents_flat):
    """Per-coordinate KS test against N(0,1).

    latents_flat: (N, D) tensor -- N samples, D coordinates.
    Returns dict with mean/median/max p-value and fraction of coordinates
    where we reject H0 at alpha=0.05.
    """
    N, D = latents_flat.shape
    p_values = np.zeros(D)
    for j in range(D):
        col = latents_flat[:, j].numpy()
        _, p = scipy_stats.kstest(col, "norm", args=(0, 1))
        p_values[j] = p
    reject_frac = float((p_values < 0.05).mean())
    return {
        "mean_pvalue": float(p_values.mean()),
        "median_pvalue": float(np.median(p_values)),
        "min_pvalue": float(p_values.min()),
        "max_pvalue": float(p_values.max()),
        "reject_fraction_alpha005": reject_frac,
        "num_coordinates": D,
        "num_samples": N,
    }


def anderson_darling_test(latents_flat):
    """Anderson-Darling test on all coordinates pooled vs N(0,1).

    Also runs per-coordinate AD and reports summary.
    """
    pooled = latents_flat.numpy().flatten()
    # Subsample if pooled is very large (AD is O(n log n))
    if len(pooled) > 500_000:
        rng = np.random.default_rng(42)
        pooled = rng.choice(pooled, size=500_000, replace=False)
    result = scipy_stats.anderson(pooled, dist="norm")
    # Per-coordinate
    N, D = latents_flat.shape
    num_test = min(D, 500)  # test a subset of coordinates
    ad_stats = []
    for j in range(num_test):
        col = latents_flat[:, j].numpy()
        r = scipy_stats.anderson(col, dist="norm")
        ad_stats.append(r.statistic)
    return {
        "pooled_statistic": float(result.statistic),
        "pooled_critical_values": {
            f"{sl}%": float(cv) for sl, cv in zip(result.significance_level, result.critical_values)
        },
        "per_coord_ad_mean": float(np.mean(ad_stats)),
        "per_coord_ad_max": float(np.max(ad_stats)),
        "per_coord_num_tested": num_test,
    }


def chi_squared_sign_test(latents_flat):
    """Test whether signs of watermarked coordinates are uniform {-1,+1}.

    Under H0 (indistinguishable), each coordinate is equally likely to be
    positive or negative. We use a chi-squared test on positive counts.
    """
    N, D = latents_flat.shape
    positive_counts = (latents_flat > 0).float().sum(dim=0).numpy()  # shape (D,)
    negative_counts = N - positive_counts
    # Per-coordinate binomial test (two-sided): fraction of coords that reject
    p_values = np.zeros(D)
    for j in range(D):
        # Binomial test: H0 is p=0.5
        try:
            p_values[j] = scipy_stats.binomtest(int(positive_counts[j]), N, 0.5).pvalue
        except AttributeError:
            # Fallback for older scipy
            p_values[j] = scipy_stats.binom_test(int(positive_counts[j]), N, 0.5)
    reject_frac = float((p_values < 0.05).mean())
    # Also: aggregate chi-squared across all coordinates
    expected = N / 2.0
    chi2_per_coord = (positive_counts - expected) ** 2 / expected + (negative_counts - expected) ** 2 / expected
    # Under H0, sum of D independent chi2(1) ~ chi2(D)
    chi2_total = float(chi2_per_coord.sum())
    chi2_pvalue = float(scipy_stats.chi2.sf(chi2_total, df=D))
    return {
        "per_coord_reject_fraction_alpha005": reject_frac,
        "per_coord_mean_pvalue": float(p_values.mean()),
        "aggregate_chi2_statistic": chi2_total,
        "aggregate_chi2_df": D,
        "aggregate_chi2_pvalue": chi2_pvalue,
        "num_coordinates": D,
        "num_samples": N,
    }


def norm_test(latents_flat):
    """Compare ||z'_T||^2 distribution to chi-squared(D).

    Each sample's squared norm should follow chi2(D) if coordinates are iid N(0,1).
    """
    N, D = latents_flat.shape
    sq_norms = (latents_flat ** 2).sum(dim=1).numpy()  # shape (N,)
    # KS test against chi2(D)
    ks_stat, ks_pval = scipy_stats.kstest(sq_norms, "chi2", args=(D,))
    # Moments comparison
    expected_mean = D
    expected_var = 2 * D
    return {
        "squared_norm_mean": float(sq_norms.mean()),
        "squared_norm_std": float(sq_norms.std()),
        "expected_mean_chi2_D": float(expected_mean),
        "expected_std_chi2_D": float(np.sqrt(expected_var)),
        "ks_statistic_vs_chi2": float(ks_stat),
        "ks_pvalue_vs_chi2": float(ks_pval),
        "num_samples": N,
        "dimension": D,
    }


def moment_comparison(wm_flat, unwm_flat):
    """Compare mean, variance, skewness, kurtosis of watermarked vs unwatermarked."""
    results = {}
    for name, data in [("watermarked", wm_flat), ("unwatermarked", unwm_flat)]:
        vals = data.numpy().flatten()
        # Subsample if huge
        if len(vals) > 1_000_000:
            rng = np.random.default_rng(42)
            vals = rng.choice(vals, size=1_000_000, replace=False)
        results[name] = {
            "mean": float(np.mean(vals)),
            "variance": float(np.var(vals)),
            "skewness": float(scipy_stats.skew(vals)),
            "kurtosis": float(scipy_stats.kurtosis(vals)),  # excess kurtosis
        }
    # Two-sample KS test between watermarked and unwatermarked
    wm_sub = wm_flat.numpy().flatten()
    unwm_sub = unwm_flat.numpy().flatten()
    if len(wm_sub) > 500_000:
        rng = np.random.default_rng(99)
        wm_sub = rng.choice(wm_sub, size=500_000, replace=False)
        unwm_sub = rng.choice(unwm_sub, size=500_000, replace=False)
    ks_stat, ks_pval = scipy_stats.ks_2samp(wm_sub, unwm_sub)
    results["two_sample_ks"] = {
        "statistic": float(ks_stat),
        "pvalue": float(ks_pval),
    }
    # N(0,1) reference
    results["reference_N01"] = {
        "mean": 0.0,
        "variance": 1.0,
        "skewness": 0.0,
        "kurtosis": 0.0,
    }
    return results


def run_latent_statistical_tests(wm_latents_list, unwm_latents_list):
    """Run all latent-space statistical tests.

    Args:
        wm_latents_list: list of watermarked latent tensors (each shape [1, C, H, W])
        unwm_latents_list: list of unwatermarked latent tensors
    Returns:
        dict with all test results
    """
    wm_flat = torch.stack([z.flatten() for z in wm_latents_list])    # (N, D)
    unwm_flat = torch.stack([z.flatten() for z in unwm_latents_list])  # (N, D)

    print("  Running per-coordinate KS test (watermarked)...")
    ks_wm = per_coordinate_ks_test(wm_flat)
    print(f"    reject fraction = {ks_wm['reject_fraction_alpha005']:.4f}, mean p = {ks_wm['mean_pvalue']:.4f}")

    print("  Running per-coordinate KS test (unwatermarked)...")
    ks_unwm = per_coordinate_ks_test(unwm_flat)
    print(f"    reject fraction = {ks_unwm['reject_fraction_alpha005']:.4f}, mean p = {ks_unwm['mean_pvalue']:.4f}")

    print("  Running Anderson-Darling test (watermarked)...")
    ad_wm = anderson_darling_test(wm_flat)
    print(f"    pooled AD statistic = {ad_wm['pooled_statistic']:.4f}")

    print("  Running Anderson-Darling test (unwatermarked)...")
    ad_unwm = anderson_darling_test(unwm_flat)
    print(f"    pooled AD statistic = {ad_unwm['pooled_statistic']:.4f}")

    print("  Running chi-squared sign test (watermarked)...")
    sign_wm = chi_squared_sign_test(wm_flat)
    print(f"    reject fraction = {sign_wm['per_coord_reject_fraction_alpha005']:.4f}")

    print("  Running chi-squared sign test (unwatermarked)...")
    sign_unwm = chi_squared_sign_test(unwm_flat)
    print(f"    reject fraction = {sign_unwm['per_coord_reject_fraction_alpha005']:.4f}")

    print("  Running norm test (watermarked)...")
    norm_wm = norm_test(wm_flat)
    print(f"    KS p-value vs chi2(D) = {norm_wm['ks_pvalue_vs_chi2']:.4f}")

    print("  Running norm test (unwatermarked)...")
    norm_unwm = norm_test(unwm_flat)
    print(f"    KS p-value vs chi2(D) = {norm_unwm['ks_pvalue_vs_chi2']:.4f}")

    print("  Running moment comparison...")
    moments = moment_comparison(wm_flat, unwm_flat)

    return {
        "per_coordinate_ks": {"watermarked": ks_wm, "unwatermarked": ks_unwm},
        "anderson_darling": {"watermarked": ad_wm, "unwatermarked": ad_unwm},
        "chi_squared_sign": {"watermarked": sign_wm, "unwatermarked": sign_unwm},
        "norm_test": {"watermarked": norm_wm, "unwatermarked": norm_unwm},
        "moment_comparison": moments,
    }


# ---------------------------------------------------------------------------
# 2. Classifier-based detection
# ---------------------------------------------------------------------------

def train_classifier_test(wm_flat, unwm_flat, test_fraction=0.3):
    """Train logistic regression and MLP to classify watermarked vs unwatermarked.

    If indistinguishable, both classifiers should achieve ~50% accuracy.
    Returns dict with train/test accuracy for each classifier.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    N = wm_flat.shape[0]
    X = np.concatenate([wm_flat.numpy(), unwm_flat.numpy()], axis=0)
    y = np.concatenate([np.ones(N), np.zeros(N)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=42, stratify=y,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}

    # Logistic Regression
    print("    Training logistic regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr.fit(X_train_s, y_train)
    lr_train_acc = accuracy_score(y_train, lr.predict(X_train_s))
    lr_test_acc = accuracy_score(y_test, lr.predict(X_test_s))
    lr_test_proba = lr.predict_proba(X_test_s)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_test_proba)
    results["logistic_regression"] = {
        "train_accuracy": float(lr_train_acc),
        "test_accuracy": float(lr_test_acc),
        "test_auc": float(lr_auc),
    }
    print(f"      train_acc={lr_train_acc:.4f}, test_acc={lr_test_acc:.4f}, AUC={lr_auc:.4f}")

    # MLP (2 hidden layers, 256 units)
    print("    Training MLP classifier...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 256), max_iter=500, random_state=42,
        early_stopping=True, validation_fraction=0.15,
        learning_rate_init=0.001, batch_size=min(64, N),
    )
    mlp.fit(X_train_s, y_train)
    mlp_train_acc = accuracy_score(y_train, mlp.predict(X_train_s))
    mlp_test_acc = accuracy_score(y_test, mlp.predict(X_test_s))
    mlp_test_proba = mlp.predict_proba(X_test_s)[:, 1]
    mlp_auc = roc_auc_score(y_test, mlp_test_proba)
    results["mlp"] = {
        "train_accuracy": float(mlp_train_acc),
        "test_accuracy": float(mlp_test_acc),
        "test_auc": float(mlp_auc),
        "architecture": "MLP(256,256)",
    }
    print(f"      train_acc={mlp_train_acc:.4f}, test_acc={mlp_test_acc:.4f}, AUC={mlp_auc:.4f}")

    return results


def run_classifier_tests_latent(wm_latents_list, unwm_latents_list):
    """Run classifier-based detection on raw latent vectors."""
    wm_flat = torch.stack([z.flatten() for z in wm_latents_list])
    unwm_flat = torch.stack([z.flatten() for z in unwm_latents_list])
    print("  Running classifier-based detection on latent vectors...")
    return train_classifier_test(wm_flat, unwm_flat)


def run_classifier_tests_images(wm_images, unwm_images, device="cpu"):
    """Run classifier-based detection on generated images using CLIP features.

    Falls back to raw pixel features if CLIP is unavailable.
    """
    print("  Extracting features from generated images...")
    clip_available = False
    try:
        from transformers import CLIPModel, CLIPProcessor
        clip_model_name = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(clip_model_name, local_files_only=True)
        clip_model = CLIPModel.from_pretrained(clip_model_name, local_files_only=True).to(device)
        clip_model.eval()
        clip_available = True
        print("    Using CLIP features")
    except Exception as e:
        print(f"    CLIP unavailable ({e}), falling back to raw pixel features")

    features_wm = []
    features_unwm = []

    if clip_available:
        with torch.no_grad():
            for img in wm_images:
                inputs = processor(images=img, return_tensors="pt").to(device)
                feat = clip_model.get_image_features(**inputs)
                features_wm.append(feat.cpu().squeeze(0).numpy())
            for img in unwm_images:
                inputs = processor(images=img, return_tensors="pt").to(device)
                feat = clip_model.get_image_features(**inputs)
                features_unwm.append(feat.cpu().squeeze(0).numpy())
        del clip_model
        torch.cuda.empty_cache() if "cuda" in device else None
        feature_type = "clip"
    else:
        # Raw pixel features (downscaled)
        import torchvision.transforms as T
        transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
        for img in wm_images:
            features_wm.append(transform(img).flatten().numpy())
        for img in unwm_images:
            features_unwm.append(transform(img).flatten().numpy())
        feature_type = "raw_pixel_64x64"

    wm_feat = np.stack(features_wm)
    unwm_feat = np.stack(features_unwm)
    print(f"    Feature shape: {wm_feat.shape}")

    classifier_results = train_classifier_test(
        torch.from_numpy(wm_feat), torch.from_numpy(unwm_feat),
    )
    classifier_results["feature_type"] = feature_type
    return classifier_results


# ---------------------------------------------------------------------------
# 3. Image-space statistical tests
# ---------------------------------------------------------------------------

def compute_image_stats(wm_images, unwm_images, device="cpu"):
    """Compute FID, CLIP score distributions, and Inception Score.

    These are best-effort: requires torchmetrics or clean-fid.
    Falls back gracefully if libraries are unavailable.
    """
    results = {}

    # --- FID ---
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        import torchvision.transforms as T
        print("  Computing FID...")
        fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        transform = T.Compose([T.Resize((299, 299)), T.ToTensor()])
        for img in wm_images:
            t = transform(img).unsqueeze(0).to(device)
            fid_metric.update(t, real=False)
        for img in unwm_images:
            t = transform(img).unsqueeze(0).to(device)
            fid_metric.update(t, real=True)
        fid_val = fid_metric.compute().item()
        results["fid"] = {"value": float(fid_val), "note": "unwatermarked=real, watermarked=fake"}
        print(f"    FID = {fid_val:.4f}")
        del fid_metric
    except Exception as e:
        print(f"  FID computation failed: {e}")
        results["fid"] = {"value": None, "error": str(e)}

    # --- CLIP Score distributions ---
    try:
        from transformers import CLIPModel, CLIPProcessor
        print("  Computing CLIP cosine similarities...")
        clip_model_name = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(clip_model_name, local_files_only=True)
        clip_model = CLIPModel.from_pretrained(clip_model_name, local_files_only=True).to(device)
        clip_model.eval()

        def _clip_features(images):
            feats = []
            with torch.no_grad():
                for img in images:
                    inputs = processor(images=img, return_tensors="pt").to(device)
                    f = clip_model.get_image_features(**inputs)
                    feats.append(f.cpu().squeeze(0))
            return torch.stack(feats)

        wm_feats = _clip_features(wm_images)
        unwm_feats = _clip_features(unwm_images)
        # Pairwise cosine similarity between matching wm and unwm images
        cos_sims = torch.nn.functional.cosine_similarity(wm_feats, unwm_feats, dim=1).numpy()
        # Norms
        wm_norms = wm_feats.norm(dim=1).numpy()
        unwm_norms = unwm_feats.norm(dim=1).numpy()
        results["clip_cosine_similarity"] = {
            "mean": float(cos_sims.mean()),
            "std": float(cos_sims.std()),
            "min": float(cos_sims.min()),
            "max": float(cos_sims.max()),
        }
        results["clip_feature_norms"] = {
            "watermarked_mean": float(wm_norms.mean()),
            "unwatermarked_mean": float(unwm_norms.mean()),
        }
        # Two-sample KS on CLIP feature norms
        ks_stat, ks_pval = scipy_stats.ks_2samp(wm_norms, unwm_norms)
        results["clip_norm_ks_test"] = {
            "statistic": float(ks_stat),
            "pvalue": float(ks_pval),
        }
        print(f"    CLIP cosine sim: mean={cos_sims.mean():.4f}, std={cos_sims.std():.4f}")
        del clip_model
        torch.cuda.empty_cache() if "cuda" in device else None
    except Exception as e:
        print(f"  CLIP score computation failed: {e}")
        results["clip_cosine_similarity"] = {"error": str(e)}

    # --- Inception Score ---
    try:
        from torchmetrics.image.inception import InceptionScore
        import torchvision.transforms as T
        print("  Computing Inception Score...")
        transform = T.Compose([T.Resize((299, 299)), T.ToTensor()])

        is_wm = InceptionScore(normalize=True).to(device)
        for img in wm_images:
            t = transform(img).unsqueeze(0).to(device)
            is_wm.update(t)
        is_wm_mean, is_wm_std = is_wm.compute()

        is_unwm = InceptionScore(normalize=True).to(device)
        for img in unwm_images:
            t = transform(img).unsqueeze(0).to(device)
            is_unwm.update(t)
        is_unwm_mean, is_unwm_std = is_unwm.compute()

        results["inception_score"] = {
            "watermarked": {"mean": float(is_wm_mean), "std": float(is_wm_std)},
            "unwatermarked": {"mean": float(is_unwm_mean), "std": float(is_unwm_std)},
        }
        print(f"    IS(wm)={is_wm_mean:.4f}+-{is_wm_std:.4f}, IS(unwm)={is_unwm_mean:.4f}+-{is_unwm_std:.4f}")
        del is_wm, is_unwm
    except Exception as e:
        print(f"  Inception Score computation failed: {e}")
        results["inception_score"] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# 4. RSWST comparison (control)
# ---------------------------------------------------------------------------

def run_rswst_comparison(pipe, scheduler, device, dtype, seeds, args):
    """Run the same latent-space tests on RSWST to show it IS distinguishable."""
    print("\n=== RSWST Comparison (Control) ===")
    rswst_config = DiffusionConfig(
        scheduler=scheduler, pipe=pipe, device=device,
        image_size=(args.size, args.size),
        num_inference_steps=args.steps, num_inversion_steps=args.steps,
        guidance_scale=7.5, gen_seed=42, inversion_type="ddim", dtype=dtype,
    )
    rswst_wm = AutoWatermark.load(
        "RSWST",
        algorithm_config=args.rswst_config,
        diffusion_config=rswst_config,
    )

    rswst_wm_latents = []
    rswst_unwm_latents = []
    for si, seed in enumerate(seeds):
        print(f"  [{si+1}/{len(seeds)}] seed={seed}")
        z_T, z_prime_T = get_rswst_latent_pair(rswst_wm, seed)
        rswst_unwm_latents.append(z_T)
        rswst_wm_latents.append(z_prime_T)

    print("  Running latent statistical tests on RSWST...")
    rswst_stats = run_latent_statistical_tests(rswst_wm_latents, rswst_unwm_latents)

    print("  Running classifier tests on RSWST latents...")
    if len(rswst_wm_latents) >= 10:
        rswst_classifier = run_classifier_tests_latent(rswst_wm_latents, rswst_unwm_latents)
    else:
        rswst_classifier = {"note": "too few samples for classifier test"}

    return {
        "latent_statistical_tests": rswst_stats,
        "classifier_tests": rswst_classifier,
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_summary_table(mirage_results, rswst_results=None):
    """Build a human-readable summary table comparing MIRAGE vs RSWST."""
    rows = []
    header = ["Test", "Metric", "MIRAGE", "RSWST (control)", "Verdict"]
    rows.append(header)

    def safe_get(d, *keys, default="N/A"):
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return default
        return d

    # KS test reject fraction
    m_ks = safe_get(mirage_results, "latent_statistical_tests", "per_coordinate_ks", "watermarked", "reject_fraction_alpha005")
    r_ks = safe_get(rswst_results, "latent_statistical_tests", "per_coordinate_ks", "watermarked", "reject_fraction_alpha005") if rswst_results else "N/A"
    verdict = "PASS" if isinstance(m_ks, float) and m_ks < 0.10 else "FAIL"
    rows.append(["KS test (N(0,1))", "reject fraction", f"{m_ks:.4f}" if isinstance(m_ks, float) else str(m_ks),
                  f"{r_ks:.4f}" if isinstance(r_ks, float) else str(r_ks), verdict])

    # AD test
    m_ad = safe_get(mirage_results, "latent_statistical_tests", "anderson_darling", "watermarked", "pooled_statistic")
    r_ad = safe_get(rswst_results, "latent_statistical_tests", "anderson_darling", "watermarked", "pooled_statistic") if rswst_results else "N/A"
    rows.append(["Anderson-Darling", "pooled statistic", f"{m_ad:.4f}" if isinstance(m_ad, float) else str(m_ad),
                  f"{r_ad:.4f}" if isinstance(r_ad, float) else str(r_ad), ""])

    # Sign test
    m_sign = safe_get(mirage_results, "latent_statistical_tests", "chi_squared_sign", "watermarked", "aggregate_chi2_pvalue")
    r_sign = safe_get(rswst_results, "latent_statistical_tests", "chi_squared_sign", "watermarked", "aggregate_chi2_pvalue") if rswst_results else "N/A"
    verdict = "PASS" if isinstance(m_sign, float) and m_sign > 0.01 else "FAIL"
    rows.append(["Chi-sq sign test", "aggregate p-value", f"{m_sign:.4e}" if isinstance(m_sign, float) else str(m_sign),
                  f"{r_sign:.4e}" if isinstance(r_sign, float) else str(r_sign), verdict])

    # Norm test
    m_norm = safe_get(mirage_results, "latent_statistical_tests", "norm_test", "watermarked", "ks_pvalue_vs_chi2")
    r_norm = safe_get(rswst_results, "latent_statistical_tests", "norm_test", "watermarked", "ks_pvalue_vs_chi2") if rswst_results else "N/A"
    verdict = "PASS" if isinstance(m_norm, float) and m_norm > 0.01 else "FAIL"
    rows.append(["Norm vs chi2(D)", "KS p-value", f"{m_norm:.4e}" if isinstance(m_norm, float) else str(m_norm),
                  f"{r_norm:.4e}" if isinstance(r_norm, float) else str(r_norm), verdict])

    # Two-sample KS
    m_2ks = safe_get(mirage_results, "latent_statistical_tests", "moment_comparison", "two_sample_ks", "pvalue")
    r_2ks = safe_get(rswst_results, "latent_statistical_tests", "moment_comparison", "two_sample_ks", "pvalue") if rswst_results else "N/A"
    verdict = "PASS" if isinstance(m_2ks, float) and m_2ks > 0.01 else "FAIL"
    rows.append(["2-sample KS", "p-value", f"{m_2ks:.4e}" if isinstance(m_2ks, float) else str(m_2ks),
                  f"{r_2ks:.4e}" if isinstance(r_2ks, float) else str(r_2ks), verdict])

    # Classifier (latent)
    m_cls = safe_get(mirage_results, "classifier_latent", "logistic_regression", "test_accuracy")
    r_cls = safe_get(rswst_results, "classifier_tests", "logistic_regression", "test_accuracy") if rswst_results else "N/A"
    verdict = "PASS" if isinstance(m_cls, float) and abs(m_cls - 0.5) < 0.10 else "FAIL"
    rows.append(["LR classifier", "test accuracy", f"{m_cls:.4f}" if isinstance(m_cls, float) else str(m_cls),
                  f"{r_cls:.4f}" if isinstance(r_cls, float) else str(r_cls), verdict])

    m_mlp = safe_get(mirage_results, "classifier_latent", "mlp", "test_accuracy")
    r_mlp = safe_get(rswst_results, "classifier_tests", "mlp", "test_accuracy") if rswst_results else "N/A"
    verdict = "PASS" if isinstance(m_mlp, float) and abs(m_mlp - 0.5) < 0.10 else "FAIL"
    rows.append(["MLP classifier", "test accuracy", f"{m_mlp:.4f}" if isinstance(m_mlp, float) else str(m_mlp),
                  f"{r_mlp:.4f}" if isinstance(r_mlp, float) else str(r_mlp), verdict])

    # FID
    m_fid = safe_get(mirage_results, "image_stats", "fid", "value")
    rows.append(["FID (wm vs unwm)", "score", f"{m_fid:.4f}" if isinstance(m_fid, float) else str(m_fid),
                  "N/A", ""])

    # Image classifier
    m_img_cls = safe_get(mirage_results, "classifier_image", "logistic_regression", "test_accuracy")
    verdict = "PASS" if isinstance(m_img_cls, float) and abs(m_img_cls - 0.5) < 0.10 else "FAIL"
    rows.append(["Image LR classifier", "test accuracy",
                  f"{m_img_cls:.4f}" if isinstance(m_img_cls, float) else str(m_img_cls), "N/A", verdict])

    return rows


def print_summary_table(rows):
    """Pretty-print the summary table."""
    col_widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    print("\n" + sep)
    for ri, row in enumerate(rows):
        line = "|" + "|".join(f" {str(row[i]).ljust(col_widths[i])} " for i in range(len(row))) + "|"
        print(line)
        if ri == 0:
            print(sep)
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Empirical undetectability evaluation for MIRAGE watermark"
    )
    parser.add_argument("--algorithm", default="MIRAGE")
    parser.add_argument("--model", default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--config", default="config/MIRAGE_presence.json")
    parser.add_argument("--rswst-config", default="config/RSWST_presence.json")
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--num-seeds", type=int, default=100,
                        help="Number of latent samples to generate (default: 100)")
    parser.add_argument("--max-prompts", type=int, default=0,
                        help="Max prompts for image generation (0=all)")
    parser.add_argument("--output", default="undetectability_results.json")
    parser.add_argument("--skip-images", action="store_true",
                        help="Skip image generation (latent-only tests)")
    parser.add_argument("--skip-rswst", action="store_true",
                        help="Skip RSWST comparison")
    args = parser.parse_args()

    device, dtype = resolve_device(args.gpu)

    # Build pipeline
    print("Loading pipeline...")
    pipe, scheduler = build_cached_pipe(args.model, device, dtype)

    # Build MIRAGE watermark
    print("Loading MIRAGE watermark...")
    diffusion_config = DiffusionConfig(
        scheduler=scheduler, pipe=pipe, device=device,
        image_size=(args.size, args.size),
        num_inference_steps=args.steps, num_inversion_steps=args.steps,
        guidance_scale=7.5, gen_seed=42, inversion_type="ddim", dtype=dtype,
    )
    watermark = AutoWatermark.load(
        args.algorithm,
        algorithm_config=args.config,
        diffusion_config=diffusion_config,
    )

    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))

    # -----------------------------------------------------------------------
    # Collect latent pairs
    # -----------------------------------------------------------------------
    print(f"\n=== Collecting {len(seeds)} latent pairs (MIRAGE) ===")
    mirage_wm_latents = []
    mirage_unwm_latents = []
    for si, seed in enumerate(seeds):
        if (si + 1) % 20 == 0 or si == 0:
            print(f"  [{si+1}/{len(seeds)}] seed={seed}")
        z_T, z_prime_T = get_latent_pair(watermark, seed)
        mirage_unwm_latents.append(z_T)
        mirage_wm_latents.append(z_prime_T)

    # -----------------------------------------------------------------------
    # 1. Latent-space statistical tests
    # -----------------------------------------------------------------------
    print("\n=== 1. Latent-Space Statistical Tests (MIRAGE) ===")
    latent_stats = run_latent_statistical_tests(mirage_wm_latents, mirage_unwm_latents)

    # -----------------------------------------------------------------------
    # 2. Classifier-based detection (latent)
    # -----------------------------------------------------------------------
    print("\n=== 2. Classifier-Based Detection (Latent) ===")
    if len(seeds) >= 10:
        classifier_latent = run_classifier_tests_latent(mirage_wm_latents, mirage_unwm_latents)
    else:
        classifier_latent = {"note": "too few samples for classifier test (need >= 10)"}
        print("  Skipped: too few samples")

    # -----------------------------------------------------------------------
    # 3. Image-space tests (optional)
    # -----------------------------------------------------------------------
    classifier_image = {}
    image_stats = {}
    if not args.skip_images:
        prompts = PROMPTS[:]
        if args.max_prompts > 0:
            prompts = prompts[:args.max_prompts]
        # Use a subset of seeds for image generation (expensive)
        img_seeds = seeds[:min(len(seeds), 20)]

        print(f"\n=== 3. Image-Space Tests ({len(prompts)} prompts x {len(img_seeds)} seeds) ===")
        wm_images = []
        unwm_images = []
        for pi, prompt in enumerate(prompts):
            for si, seed in enumerate(img_seeds):
                idx = pi * len(img_seeds) + si + 1
                total = len(prompts) * len(img_seeds)
                print(f"  [{idx}/{total}] Generating images: prompt={pi}, seed={seed}")
                reseed_watermark(watermark, seed)
                wm_img = watermark.generate_watermarked_media(prompt)
                reseed_watermark(watermark, seed)
                unwm_img = watermark.generate_unwatermarked_media(prompt)
                wm_images.append(wm_img)
                unwm_images.append(unwm_img)

        # Image-space statistical tests
        print("\n  Image-space statistical tests...")
        image_stats = compute_image_stats(wm_images, unwm_images, device=device)

        # Classifier on images
        print("\n  Classifier-based detection on images...")
        if len(wm_images) >= 10:
            classifier_image = run_classifier_tests_images(wm_images, unwm_images, device=device)
        else:
            classifier_image = {"note": "too few image samples for classifier test"}
    else:
        print("\n=== 3. Image-Space Tests: SKIPPED ===")

    # -----------------------------------------------------------------------
    # 4. RSWST comparison
    # -----------------------------------------------------------------------
    rswst_results = None
    if not args.skip_rswst:
        rswst_results = run_rswst_comparison(pipe, scheduler, device, dtype, seeds, args)
    else:
        print("\n=== 4. RSWST Comparison: SKIPPED ===")

    # -----------------------------------------------------------------------
    # Assemble results
    # -----------------------------------------------------------------------
    mirage_results = {
        "latent_statistical_tests": latent_stats,
        "classifier_latent": classifier_latent,
        "classifier_image": classifier_image,
        "image_stats": image_stats,
    }

    summary_rows = build_summary_table(mirage_results, rswst_results)
    print_summary_table(summary_rows)

    full_results = {
        "config": {
            "algorithm": args.algorithm,
            "model": args.model,
            "device": device,
            "steps": args.steps,
            "size": args.size,
            "mirage_config": args.config,
            "rswst_config": args.rswst_config,
            "num_seeds": args.num_seeds,
            "seed_start": args.seed_start,
            "skip_images": args.skip_images,
            "skip_rswst": args.skip_rswst,
            "max_prompts": args.max_prompts,
        },
        "mirage": to_jsonable(mirage_results),
        "rswst": to_jsonable(rswst_results) if rswst_results else None,
        "summary_table": [
            {h: row[i] for i, h in enumerate(summary_rows[0])}
            for row in summary_rows[1:]
        ],
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(full_results, indent=2, ensure_ascii=False))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

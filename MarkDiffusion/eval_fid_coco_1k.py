"""FID evaluation using 1000 MS-COCO captions for any watermarking algorithm.

Generates N watermarked and N unwatermarked images using the same COCO captions
and matching seeds, then computes FID, Inception Score, and CLIP cosine
similarity (feature-level).

Supported algorithms (via --algorithm):
  - MIRAGE : AutoWatermark framework (config/MIRAGE_presence.json)
  - PRC    : AutoWatermark framework (config/PRC.json)
  - GS     : AutoWatermark framework (config/GS.json)
  - T2SMark: Custom pipeline with T2SMark encoder

Usage:
  python qi_eval_fid_coco_1k.py --algorithm MIRAGE --config config/MIRAGE_presence.json --gpu 0
  python qi_eval_fid_coco_1k.py --algorithm T2SMark --gpu 0
  python qi_eval_fid_coco_1k.py --algorithm PRC --config config/PRC.json --gpu 1
  python qi_eval_fid_coco_1k.py --algorithm GS --config config/GS.json --gpu 2
"""
import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from qi_runtime_utils import build_cached_pipe, configure_qi_runtime, resolve_qi_device

configure_qi_runtime()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "huanzi05/stable-diffusion-2-1-base"
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
IMAGE_SIZE = 512

# T2SMark hyperparameters
T2S_TAU = 0.674
T2S_KEY_LENGTH = 16
T2S_MSG_LENGTH = 256
T2S_KEY_CHANNEL_IDX = 0

# COCO dataset path
COCO_PARQUET = Path(__file__).resolve().parent / "dataset" / "mscoco" / "mscoco.parquet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_jsonable(value):
    """Recursively convert numpy / torch types to JSON-serializable Python types."""
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


def load_coco_prompts(parquet_path: str | Path, num_prompts: int = 1000,
                      seed: int = 42) -> list[str]:
    """Load COCO captions from a parquet file.

    Reads the TEXT column, deduplicates, then samples ``num_prompts`` captions
    with the given random seed.
    """
    df = pd.read_parquet(parquet_path)
    captions = df["TEXT"].dropna().drop_duplicates().tolist()
    rng = np.random.RandomState(seed)
    if len(captions) < num_prompts:
        raise ValueError(
            f"Only {len(captions)} unique captions in {parquet_path}, "
            f"but {num_prompts} requested."
        )
    indices = rng.choice(len(captions), size=num_prompts, replace=False)
    return [captions[i] for i in indices]


def reseed_watermark(watermark, seed):
    """Re-seed a watermark object so the next generation uses a fresh latent."""
    from utils.media_utils import get_random_latents

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


# ---------------------------------------------------------------------------
# Inception v3 feature extraction
# ---------------------------------------------------------------------------

def pil_to_inception_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL image to Inception-compatible tensor (299x299, normalized)."""
    transform = T.Compose([
        T.Resize((299, 299), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img)


def get_inception_model_with_logits(device: str):
    """Load Inception v3 that returns both pool features and logits."""
    from torchvision.models import inception_v3, Inception_V3_Weights
    import torch.nn as nn

    class InceptionDual(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = inception_v3(
                weights=Inception_V3_Weights.DEFAULT, transform_input=False,
            )
            self.base.eval()

        def forward(self, x):
            x = self.base.Conv2d_1a_3x3(x)
            x = self.base.Conv2d_2a_3x3(x)
            x = self.base.Conv2d_2b_3x3(x)
            x = self.base.maxpool1(x)
            x = self.base.Conv2d_3b_1x1(x)
            x = self.base.Conv2d_4a_3x3(x)
            x = self.base.maxpool2(x)
            x = self.base.Mixed_5b(x)
            x = self.base.Mixed_5c(x)
            x = self.base.Mixed_5d(x)
            x = self.base.Mixed_6a(x)
            x = self.base.Mixed_6b(x)
            x = self.base.Mixed_6c(x)
            x = self.base.Mixed_6d(x)
            x = self.base.Mixed_6e(x)
            x = self.base.Mixed_7a(x)
            x = self.base.Mixed_7b(x)
            x = self.base.Mixed_7c(x)
            x = self.base.avgpool(x)
            pool = torch.flatten(x, 1)   # (N, 2048)
            logits = self.base.fc(pool)   # (N, 1000)
            return pool, logits

    model = InceptionDual()
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def extract_features_and_logits(
    images: list[Image.Image],
    model,
    device: str,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract both 2048-d pool features and 1000-d logits."""
    all_pool = []
    all_logits = []
    for i in range(0, len(images), batch_size):
        batch_pil = images[i:i + batch_size]
        batch_tensor = torch.stack(
            [pil_to_inception_tensor(img) for img in batch_pil],
        )
        batch_tensor = batch_tensor.to(device)
        pool, logits = model(batch_tensor)
        all_pool.append(pool.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_pool, axis=0), np.concatenate(all_logits, axis=0)


# ---------------------------------------------------------------------------
# FID / IS / CLIP cosine similarity
# ---------------------------------------------------------------------------

def compute_fid_from_features(
    feats_real: np.ndarray,
    feats_fake: np.ndarray,
) -> float:
    """Compute FID from two (N, 2048) feature arrays."""
    from scipy import linalg

    mu_real = np.mean(feats_real, axis=0)
    mu_fake = np.mean(feats_fake, axis=0)
    sigma_real = np.cov(feats_real, rowvar=False)
    sigma_fake = np.cov(feats_fake, rowvar=False)

    diff = mu_real - mu_fake
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.imag(covmean), 0, atol=1e-3):
            print(
                f"  WARNING: sqrtm imaginary component "
                f"(max={np.max(np.abs(np.imag(covmean)))})"
            )
        covmean = np.real(covmean)

    fid = float(
        diff @ diff
        + np.trace(sigma_real)
        + np.trace(sigma_fake)
        - 2 * np.trace(covmean)
    )
    return fid


def compute_inception_score(
    features_logits: np.ndarray,
    splits: int = 10,
) -> tuple[float, float]:
    """Compute Inception Score from softmax predictions."""
    from scipy.special import softmax

    preds = softmax(features_logits, axis=1)
    N = preds.shape[0]
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits):(k + 1) * (N // splits)]
        if len(part) == 0:
            continue
        kl_divs = part * (
            np.log(part + 1e-16)
            - np.log(np.mean(part, axis=0, keepdims=True) + 1e-16)
        )
        kl_mean = np.mean(np.sum(kl_divs, axis=1))
        split_scores.append(np.exp(kl_mean))
    return float(np.mean(split_scores)), float(np.std(split_scores))


@torch.no_grad()
def compute_clip_cosine_similarity(
    images_a: list[Image.Image],
    images_b: list[Image.Image],
    device: str,
    batch_size: int = 32,
) -> dict:
    """Compute CLIP feature-level cosine similarity between paired image sets.

    Uses CLIP ViT-B/32 from torchvision / open_clip.  Falls back gracefully
    if the library is not installed.
    """
    try:
        import open_clip
    except ImportError:
        return {"mean": None, "note": "open_clip not installed (pip install open_clip_torch)"}

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device,
    )
    model.eval()

    def _extract_clip(images: list[Image.Image]) -> np.ndarray:
        all_feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            tensors = torch.stack([preprocess(img) for img in batch]).to(device)
            feats = model.encode_image(tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu().numpy())
        return np.concatenate(all_feats, axis=0)

    feats_a = _extract_clip(images_a)
    feats_b = _extract_clip(images_b)

    # Per-pair cosine similarity (features are already L2-normalised)
    cos_sims = np.sum(feats_a * feats_b, axis=1)

    return {
        "mean": float(cos_sims.mean()),
        "std": float(cos_sims.std()),
        "min": float(cos_sims.min()),
        "max": float(cos_sims.max()),
        "median": float(np.median(cos_sims)),
    }


# ---------------------------------------------------------------------------
# T2SMark generation helpers
# ---------------------------------------------------------------------------

def _setup_t2smark():
    """Import T2SMark and add it to sys.path."""
    t2smark_root = Path(__file__).resolve().parent.parent / "T2SMark"
    if str(t2smark_root) not in sys.path:
        sys.path.insert(0, str(t2smark_root))
    from src.t2s import T2SMark
    return T2SMark


def generate_t2smark_image(
    pipe,
    t2s,
    prompt: str,
    seed: int,
    key_length: int,
    msg_length: int,
    key_channel_idx: int,
    num_inference_steps: int,
    guidance_scale: float,
    device: str,
) -> Image.Image:
    """Generate a single T2SMark watermarked image."""
    gen = torch.Generator(device="cpu").manual_seed(seed)

    key = torch.randint(0, 2, (key_length,), generator=gen)
    msg = torch.randint(0, 2, (msg_length,), generator=gen)

    latent_channels = pipe.unet.config.in_channels  # 4
    latent_h = IMAGE_SIZE // pipe.vae_scale_factor   # 64
    latent_w = IMAGE_SIZE // pipe.vae_scale_factor   # 64

    wm_channel_latent = t2s.encode(msg, key)  # shape: [1, H, W]

    latent_gen = torch.Generator(device="cpu").manual_seed(seed)
    full_latent = torch.randn(
        1, latent_channels, latent_h, latent_w, generator=latent_gen,
    )
    full_latent[0, key_channel_idx] = wm_channel_latent

    full_latent = full_latent.to(device=device, dtype=pipe.unet.dtype)

    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            latents=full_latent,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
    return output.images[0]


def generate_unwatermarked_image(
    pipe,
    prompt: str,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    device: str,
) -> Image.Image:
    """Generate a single unwatermarked image with a random N(0,1) latent."""
    latent_channels = pipe.unet.config.in_channels
    latent_h = IMAGE_SIZE // pipe.vae_scale_factor
    latent_w = IMAGE_SIZE // pipe.vae_scale_factor

    latent_gen = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(
        1, latent_channels, latent_h, latent_w, generator=latent_gen,
    )
    latents = latents.to(device=device, dtype=pipe.unet.dtype)

    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
    return output.images[0]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: Path) -> dict | None:
    if checkpoint_path.exists():
        with checkpoint_path.open("r") as f:
            return json.load(f)
    return None


def save_checkpoint(checkpoint_path: Path, data: dict):
    checkpoint_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "FID evaluation using MS-COCO captions: watermarked vs "
            "unwatermarked images for any watermarking algorithm."
        ),
    )
    parser.add_argument(
        "--algorithm",
        required=True,
        choices=["MIRAGE", "T2SMark", "PRC", "GS"],
        help="Watermarking algorithm to evaluate.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Config JSON path for AutoWatermark algorithms "
            "(e.g. config/MIRAGE_presence.json). "
            "Required for MIRAGE, PRC, GS; ignored for T2SMark."
        ),
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of COCO prompts to use (default: 1000)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: qi_fid_coco_1k_{algorithm}.json)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save checkpoint every N samples (default: 100)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for checkpoints and temp images (default: auto tmpdir)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Keep generated images on disk after evaluation",
    )
    parser.add_argument(
        "--inception-batch-size",
        type=int,
        default=32,
        help="Batch size for Inception feature extraction (default: 32)",
    )
    args = parser.parse_args()

    algorithm = args.algorithm
    num_prompts = args.num_prompts

    # Default output path
    if args.output is None:
        args.output = f"qi_fid_coco_1k_{algorithm}.json"

    # Validate config for AutoWatermark algorithms
    if algorithm in ("MIRAGE", "PRC", "GS") and args.config is None:
        default_configs = {
            "MIRAGE": "config/MIRAGE_presence.json",
            "PRC": "config/PRC.json",
            "GS": "config/GS.json",
        }
        args.config = default_configs[algorithm]
        print(f"  No --config given, using default: {args.config}")

    device, dtype = resolve_qi_device(args.gpu)

    # -------------------------------------------------------------------
    # Setup directories
    # -------------------------------------------------------------------
    if args.checkpoint_dir:
        work_dir = Path(args.checkpoint_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup_work_dir = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix=f"qi_fid_coco_1k_{algorithm}_"))
        cleanup_work_dir = not args.save_images

    wm_img_dir = work_dir / "watermarked"
    unwm_img_dir = work_dir / "unwatermarked"
    wm_img_dir.mkdir(exist_ok=True)
    unwm_img_dir.mkdir(exist_ok=True)
    checkpoint_path = work_dir / "checkpoint.json"

    print(f"=== FID COCO-1K Evaluation ({algorithm}) ===")
    print(f"  model={MODEL_NAME}")
    print(f"  num_prompts={num_prompts}, device={device}, steps={NUM_INFERENCE_STEPS}")
    print(f"  guidance_scale={GUIDANCE_SCALE}, image_size={IMAGE_SIZE}")
    print(f"  work_dir={work_dir}")

    # -------------------------------------------------------------------
    # Load COCO captions
    # -------------------------------------------------------------------
    print(f"\nLoading COCO captions from {COCO_PARQUET} ...")
    prompts = load_coco_prompts(COCO_PARQUET, num_prompts=num_prompts, seed=42)
    print(f"  Loaded {len(prompts)} unique COCO captions.")

    # Each prompt gets seed = prompt_index
    sample_specs = [(prompts[i], i) for i in range(num_prompts)]

    # -------------------------------------------------------------------
    # Resume from checkpoint
    # -------------------------------------------------------------------
    start_idx = 0
    if args.resume:
        ckpt = load_checkpoint(checkpoint_path)
        if ckpt is not None:
            start_idx = ckpt.get("completed", 0)
            print(f"  Resuming from checkpoint: {start_idx}/{num_prompts} completed")
        else:
            print("  No checkpoint found, starting from scratch")

    # -------------------------------------------------------------------
    # Load pipeline and watermark
    # -------------------------------------------------------------------
    print("\nLoading pipeline...")
    pipe, scheduler = build_cached_pipe(MODEL_NAME, device, dtype)

    # Algorithm-specific setup
    watermark = None  # for AutoWatermark algorithms
    t2s = None        # for T2SMark

    if algorithm == "T2SMark":
        T2SMarkClass = _setup_t2smark()
        latent_h = IMAGE_SIZE // pipe.vae_scale_factor
        latent_w = IMAGE_SIZE // pipe.vae_scale_factor
        single_channel_shape = [1, latent_h, latent_w]
        t2s = T2SMarkClass(
            m=T2S_MSG_LENGTH, tau=T2S_TAU, latent_shape=single_channel_shape,
        )
        print(f"  T2SMark initialized: m={t2s.m}, n={t2s.n}, k={t2s.k}, r={t2s.r}")
    else:
        # MIRAGE, PRC, GS -- all use AutoWatermark
        from utils.diffusion_config import DiffusionConfig
        from watermark.auto_watermark import AutoWatermark

        diffusion_config = DiffusionConfig(
            scheduler=scheduler,
            pipe=pipe,
            device=device,
            image_size=(IMAGE_SIZE, IMAGE_SIZE),
            num_inference_steps=NUM_INFERENCE_STEPS,
            num_inversion_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            gen_seed=0,
            inversion_type="ddim",
            dtype=dtype,
        )
        watermark = AutoWatermark.load(
            algorithm,
            algorithm_config=args.config,
            diffusion_config=diffusion_config,
        )
        print(f"  AutoWatermark loaded: {algorithm} (config={args.config})")

    # -------------------------------------------------------------------
    # Generate images
    # -------------------------------------------------------------------
    print(f"\n=== Generating {num_prompts} image pairs ===")
    t_start = time.time()

    for idx in range(start_idx, num_prompts):
        prompt, seed = sample_specs[idx]

        if algorithm == "T2SMark":
            # Watermarked
            wm_img = generate_t2smark_image(
                pipe, t2s, prompt, seed,
                key_length=T2S_KEY_LENGTH,
                msg_length=T2S_MSG_LENGTH,
                key_channel_idx=T2S_KEY_CHANNEL_IDX,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                device=device,
            )
            # Unwatermarked
            unwm_img = generate_unwatermarked_image(
                pipe, prompt, seed,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                device=device,
            )
        else:
            # AutoWatermark: MIRAGE, PRC, GS
            reseed_watermark(watermark, seed)
            wm_img = watermark.generate_watermarked_media(prompt)

            reseed_watermark(watermark, seed)
            unwm_img = watermark.generate_unwatermarked_media(prompt)

        # Save images to disk
        wm_img.save(str(wm_img_dir / f"{idx:06d}.png"))
        unwm_img.save(str(unwm_img_dir / f"{idx:06d}.png"))

        completed = idx + 1
        if completed % 10 == 0 or completed == num_prompts:
            elapsed = time.time() - t_start
            rate = completed - start_idx
            if rate > 0:
                per_sample = elapsed / rate
                eta = per_sample * (num_prompts - completed)
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
            else:
                eta_str = "N/A"
            print(
                f"  [{completed}/{num_prompts}] seed={seed} | "
                f"elapsed={elapsed:.0f}s | ETA={eta_str}"
            )

        # Checkpoint
        if completed % args.checkpoint_interval == 0:
            save_checkpoint(checkpoint_path, {
                "completed": completed,
                "num_prompts": num_prompts,
                "algorithm": algorithm,
                "model": MODEL_NAME,
                "config": args.config,
            })
            print(f"    [checkpoint saved at {completed}]")

    gen_elapsed = time.time() - t_start
    print(
        f"\nGeneration complete: {num_prompts} pairs in {gen_elapsed:.1f}s "
        f"({gen_elapsed / max(num_prompts - start_idx, 1):.2f}s/pair)"
    )

    # -------------------------------------------------------------------
    # Load saved images back for metric computation
    # -------------------------------------------------------------------
    def load_images_from_dir(img_dir: Path, count: int) -> list[Image.Image]:
        images = []
        for i in range(count):
            img_path = img_dir / f"{i:06d}.png"
            images.append(Image.open(str(img_path)).convert("RGB"))
        return images

    print("\n=== Loading images for metric computation ===")
    print("  Loading watermarked images...")
    wm_images = load_images_from_dir(wm_img_dir, num_prompts)
    print("  Loading unwatermarked images...")
    unwm_images = load_images_from_dir(unwm_img_dir, num_prompts)

    # -------------------------------------------------------------------
    # Extract Inception features
    # -------------------------------------------------------------------
    print("\n=== Extracting Inception v3 features ===")
    inception_model = get_inception_model_with_logits(device)

    print(f"  Extracting features for {len(wm_images)} watermarked images...")
    feats_wm, logits_wm = extract_features_and_logits(
        wm_images, inception_model, device, batch_size=args.inception_batch_size,
    )

    print(f"  Extracting features for {len(unwm_images)} unwatermarked images...")
    feats_unwm, logits_unwm = extract_features_and_logits(
        unwm_images, inception_model, device, batch_size=args.inception_batch_size,
    )

    del inception_model
    if "cuda" in device:
        torch.cuda.empty_cache()

    # Save features for potential reuse
    np.save(str(work_dir / "features_wm.npy"), feats_wm)
    np.save(str(work_dir / "features_unwm.npy"), feats_unwm)
    np.save(str(work_dir / "logits_wm.npy"), logits_wm)
    np.save(str(work_dir / "logits_unwm.npy"), logits_unwm)
    print(f"  Features saved: wm={feats_wm.shape}, unwm={feats_unwm.shape}")

    # -------------------------------------------------------------------
    # Compute metrics
    # -------------------------------------------------------------------
    results = {
        "config": {
            "algorithm": algorithm,
            "model": MODEL_NAME,
            "device": device,
            "steps": NUM_INFERENCE_STEPS,
            "size": IMAGE_SIZE,
            "guidance_scale": GUIDANCE_SCALE,
            "num_prompts": num_prompts,
            "prompt_source": "MS-COCO (mscoco.parquet, column TEXT)",
            "config_file": args.config,
            "generation_time_seconds": gen_elapsed,
        },
        "metrics": {},
    }

    if algorithm == "T2SMark":
        results["config"].update({
            "tau": T2S_TAU,
            "key_length": T2S_KEY_LENGTH,
            "msg_length": T2S_MSG_LENGTH,
            "key_channel_idx": T2S_KEY_CHANNEL_IDX,
        })

    # --- FID ---
    print("\n=== Computing FID (Inception v3, 2048-d pool features) ===")
    fid_value = compute_fid_from_features(feats_unwm, feats_wm)
    results["metrics"]["fid_inception_v3"] = {
        "value": float(fid_value),
        "note": "unwatermarked=real, watermarked=fake, 2048-d pool features",
    }
    print(f"  FID = {fid_value:.4f}")

    # --- Inception Score ---
    print("\n=== Computing Inception Score ===")
    is_wm_mean, is_wm_std = compute_inception_score(logits_wm)
    is_unwm_mean, is_unwm_std = compute_inception_score(logits_unwm)
    results["metrics"]["inception_score"] = {
        "watermarked": {"mean": float(is_wm_mean), "std": float(is_wm_std)},
        "unwatermarked": {"mean": float(is_unwm_mean), "std": float(is_unwm_std)},
        "difference": float(abs(is_wm_mean - is_unwm_mean)),
    }
    print(f"  IS(wm)  = {is_wm_mean:.4f} +/- {is_wm_std:.4f}")
    print(f"  IS(unwm)= {is_unwm_mean:.4f} +/- {is_unwm_std:.4f}")

    # --- CLIP cosine similarity (feature-level) ---
    print("\n=== Computing CLIP cosine similarity (feature-level) ===")
    clip_cos_result = compute_clip_cosine_similarity(
        wm_images, unwm_images, device, batch_size=args.inception_batch_size,
    )
    results["metrics"]["clip_cosine_similarity"] = clip_cos_result
    if clip_cos_result.get("mean") is not None:
        print(
            f"  CLIP cos sim (paired): mean={clip_cos_result['mean']:.4f}, "
            f"std={clip_cos_result['std']:.4f}"
        )
    else:
        print(f"  CLIP cos sim skipped: {clip_cos_result.get('note', 'unknown')}")

    # --- Inception feature-space diagnostics ---
    print("\n=== Inception feature-space diagnostics ===")
    from numpy.linalg import norm as np_norm

    cos_sims = np.array([
        np.dot(feats_wm[i], feats_unwm[i])
        / (np_norm(feats_wm[i]) * np_norm(feats_unwm[i]) + 1e-8)
        for i in range(num_prompts)
    ])
    results["metrics"]["inception_feature_cosine_similarity"] = {
        "mean": float(cos_sims.mean()),
        "std": float(cos_sims.std()),
        "min": float(cos_sims.min()),
        "max": float(cos_sims.max()),
        "median": float(np.median(cos_sims)),
    }
    print(
        f"  Inception feature cosine sim (paired): "
        f"mean={cos_sims.mean():.4f}, std={cos_sims.std():.4f}"
    )

    l2_dists = np.array([
        np_norm(feats_wm[i] - feats_unwm[i]) for i in range(num_prompts)
    ])
    results["metrics"]["inception_feature_l2_distance"] = {
        "mean": float(l2_dists.mean()),
        "std": float(l2_dists.std()),
        "min": float(l2_dists.min()),
        "max": float(l2_dists.max()),
        "median": float(np.median(l2_dists)),
    }
    print(
        f"  Inception feature L2 dist (paired): "
        f"mean={l2_dists.mean():.4f}, std={l2_dists.std():.4f}"
    )

    # --- PSNR and SSIM (paired, pixel-level) ---
    print("\n=== Computing PSNR and SSIM (paired) ===")
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        psnr_vals, ssim_vals = [], []
        for i in range(num_prompts):
            wm_np = np.array(wm_images[i]).astype(np.float64)
            unwm_np = np.array(unwm_images[i]).astype(np.float64)
            psnr_vals.append(peak_signal_noise_ratio(unwm_np, wm_np, data_range=255))
            ssim_vals.append(structural_similarity(unwm_np, wm_np, channel_axis=2, data_range=255))
        results["metrics"]["psnr"] = {
            "mean": float(np.mean(psnr_vals)), "std": float(np.std(psnr_vals)),
            "min": float(np.min(psnr_vals)), "max": float(np.max(psnr_vals)),
        }
        results["metrics"]["ssim"] = {
            "mean": float(np.mean(ssim_vals)), "std": float(np.std(ssim_vals)),
            "min": float(np.min(ssim_vals)), "max": float(np.max(ssim_vals)),
        }
        print(f"  PSNR: mean={np.mean(psnr_vals):.2f}, std={np.std(psnr_vals):.2f}")
        print(f"  SSIM: mean={np.mean(ssim_vals):.4f}, std={np.std(ssim_vals):.4f}")
    except ImportError:
        results["metrics"]["psnr"] = {"error": "skimage not installed"}
        results["metrics"]["ssim"] = {"error": "skimage not installed"}
        print("  Skipped (skimage not installed)")

    # --- LPIPS (paired, perceptual) ---
    print("\n=== Computing LPIPS (paired) ===")
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
        lpips_transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
        lpips_vals = []
        with torch.no_grad():
            for i in range(0, num_prompts, 32):
                batch_wm = torch.stack([lpips_transform(wm_images[j]) for j in range(i, min(i+32, num_prompts))]).to(device)
                batch_unwm = torch.stack([lpips_transform(unwm_images[j]) for j in range(i, min(i+32, num_prompts))]).to(device)
                # LPIPS expects [-1, 1]
                dists = lpips_fn(batch_wm * 2 - 1, batch_unwm * 2 - 1)
                lpips_vals.extend(dists.squeeze().cpu().tolist() if dists.numel() > 1 else [dists.item()])
        results["metrics"]["lpips"] = {
            "mean": float(np.mean(lpips_vals)), "std": float(np.std(lpips_vals)),
            "min": float(np.min(lpips_vals)), "max": float(np.max(lpips_vals)),
        }
        print(f"  LPIPS: mean={np.mean(lpips_vals):.4f}, std={np.std(lpips_vals):.4f}")
        del lpips_fn
    except ImportError:
        results["metrics"]["lpips"] = {"error": "lpips not installed (pip install lpips)"}
        print("  Skipped (lpips not installed)")

    # --- KID (Kernel Inception Distance, unbiased FID alternative) ---
    print("\n=== Computing KID (Kernel Inception Distance) ===")
    def compute_kid(feats_real, feats_fake, num_subsets=100, subset_size=None):
        n = min(len(feats_real), len(feats_fake))
        if subset_size is None:
            subset_size = min(n, 100)
        kid_values = []
        for _ in range(num_subsets):
            idx_r = np.random.choice(n, subset_size, replace=False)
            idx_f = np.random.choice(n, subset_size, replace=False)
            fr, ff = feats_real[idx_r], feats_fake[idx_f]
            krr = (fr @ fr.T / fr.shape[1] + 1) ** 3
            kff = (ff @ ff.T / ff.shape[1] + 1) ** 3
            krf = (fr @ ff.T / fr.shape[1] + 1) ** 3
            m = subset_size
            kid_val = (krr.sum() - np.trace(krr)) / (m*(m-1)) + \
                      (kff.sum() - np.trace(kff)) / (m*(m-1)) - \
                      2 * krf.mean()
            kid_values.append(kid_val)
        return float(np.mean(kid_values)), float(np.std(kid_values))
    kid_mean, kid_std = compute_kid(feats_unwm, feats_wm)
    results["metrics"]["kid"] = {"mean": kid_mean, "std": kid_std}
    print(f"  KID = {kid_mean:.6f} +/- {kid_std:.6f}")

    # --- CLIP Score (text-image alignment) ---
    print("\n=== Computing CLIP Score (text-image alignment) ===")
    try:
        import open_clip
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=device,
        )
        clip_model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        clip_scores_wm, clip_scores_unwm = [], []
        with torch.no_grad():
            for i in range(0, num_prompts, 32):
                end = min(i + 32, num_prompts)
                batch_wm_t = torch.stack([clip_preprocess(wm_images[j]) for j in range(i, end)]).to(device)
                batch_unwm_t = torch.stack([clip_preprocess(unwm_images[j]) for j in range(i, end)]).to(device)
                batch_text = tokenizer(prompts[i:end]).to(device)
                img_feats_wm = clip_model.encode_image(batch_wm_t)
                img_feats_unwm = clip_model.encode_image(batch_unwm_t)
                text_feats = clip_model.encode_text(batch_text)
                img_feats_wm = img_feats_wm / img_feats_wm.norm(dim=-1, keepdim=True)
                img_feats_unwm = img_feats_unwm / img_feats_unwm.norm(dim=-1, keepdim=True)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                clip_scores_wm.extend((img_feats_wm * text_feats).sum(dim=-1).cpu().tolist())
                clip_scores_unwm.extend((img_feats_unwm * text_feats).sum(dim=-1).cpu().tolist())
        results["metrics"]["clip_score"] = {
            "watermarked": {"mean": float(np.mean(clip_scores_wm)), "std": float(np.std(clip_scores_wm))},
            "unwatermarked": {"mean": float(np.mean(clip_scores_unwm)), "std": float(np.std(clip_scores_unwm))},
            "difference": float(np.mean(clip_scores_wm) - np.mean(clip_scores_unwm)),
        }
        print(f"  CLIP Score (wm):   {np.mean(clip_scores_wm):.4f} +/- {np.std(clip_scores_wm):.4f}")
        print(f"  CLIP Score (unwm): {np.mean(clip_scores_unwm):.4f} +/- {np.std(clip_scores_unwm):.4f}")
        print(f"  Difference:        {np.mean(clip_scores_wm) - np.mean(clip_scores_unwm):.4f}")
        del clip_model
    except ImportError:
        results["metrics"]["clip_score"] = {"error": "open_clip not installed"}
        print("  Skipped (open_clip not installed)")

    # Free image lists
    del wm_images, unwm_images
    if "cuda" in device:
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    # Write outputs
    # -------------------------------------------------------------------
    output_path = Path(args.output)
    output_path.write_text(
        json.dumps(to_jsonable(results), indent=2, ensure_ascii=False),
    )
    print(f"\nJSON results written to: {output_path}")

    # -------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------
    if cleanup_work_dir and not args.save_images:
        print(f"\nCleaning up temp directory: {work_dir}")
        shutil.rmtree(str(work_dir), ignore_errors=True)
    else:
        print(f"\nImages and features preserved in: {work_dir}")

    # -------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Algorithm:          {algorithm}")
    print(f"  Model:              {MODEL_NAME}")
    print(f"  Num prompts:        {num_prompts} (MS-COCO)")
    print(f"  FID (Inception v3): {fid_value:.4f}")
    print(f"  IS (watermarked):   {is_wm_mean:.4f} +/- {is_wm_std:.4f}")
    print(f"  IS (unwatermarked): {is_unwm_mean:.4f} +/- {is_unwm_std:.4f}")
    if clip_cos_result.get("mean") is not None:
        print(f"  CLIP cos sim:       {clip_cos_result['mean']:.4f} +/- {clip_cos_result['std']:.4f}")
    print(f"  Inception cos sim:  {cos_sims.mean():.4f} +/- {cos_sims.std():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

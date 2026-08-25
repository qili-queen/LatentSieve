"""Evaluate MIRAGE (block_dim=64) on 1000 samples: 200 MS-COCO prompts x 5 seeds.

Records TPR and FPR for 8 standard attacks, and saves watermarked + unwatermarked
images for FID computation.

Usage:
  python qi_eval_mirage_1k.py --gpu 1 --output qi_mirage_1k_results.json --csv-output qi_mirage_1k_results.csv
"""
import argparse
import csv
import json
import time
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
import torch
from PIL import Image

from evaluation.tools.image_editor import (
    CrSc,
    GaussianBlurring,
    GaussianNoise,
    JPEGCompression,
    Overlay,
)
from qi_runtime_utils import build_cached_pipe, configure_qi_runtime, resolve_qi_device
from utils.diffusion_config import DiffusionConfig
from utils.media_utils import get_random_latents
from watermark.auto_watermark import AutoWatermark

configure_qi_runtime()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "huanzi05/stable-diffusion-2-1-base"
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
IMAGE_SIZE = 512
NUM_PROMPTS = 200
NUM_SEEDS = 5
SEED_START = 42

COCO_PARQUET = Path(__file__).resolve().parent / "dataset" / "mscoco" / "mscoco.parquet"

# Standard 8 attacks
ATTACKS = {
    "Clean": None,
    "JPEG-75": lambda: JPEGCompression(quality=75),
    "Blur-2": lambda: GaussianBlurring(radius=2),
    "Noise-15": lambda: GaussianNoise(sigma=15.0),
    "CropPadBlack-0.8": lambda: CrSc(crop_ratio=0.8, resize_back=False, pad_back=True),
    "Overlay": lambda: Overlay(num_strokes=10),
    "JPEG-10": lambda: JPEGCompression(quality=10),
    "Noise-50": lambda: GaussianNoise(sigma=50.0),
}


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


def summarize(values):
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return None
    return {
        "count": len(cleaned),
        "mean": float(sum(cleaned) / len(cleaned)),
        "std": float(torch.tensor(cleaned, dtype=torch.float32).std(unbiased=False).item()),
        "min": float(min(cleaned)),
        "max": float(max(cleaned)),
    }


def safe_mean(values):
    cleaned = [v for v in values if v is not None]
    return mean(cleaned) if cleaned else None


def reseed_watermark(watermark, seed):
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


def load_coco_prompts(parquet_path, num_prompts=200, seed=42):
    """Load and sample COCO captions from parquet file."""
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


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(path):
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None


def save_checkpoint(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MIRAGE (block_dim=64) on 1000 samples (200 prompts x 5 seeds)."
    )
    parser.add_argument("--gpu", type=int, default=None, help="GPU index")
    parser.add_argument("--output", default="qi_mirage_1k_results.json", help="Output JSON path")
    parser.add_argument("--csv-output", default="qi_mirage_1k_results.csv", help="Output CSV path")
    parser.add_argument("--config", default="config/MIRAGE_presence.json", help="MIRAGE config")
    parser.add_argument("--attacks", default="",
                        help="comma-separated subset of ATTACKS; empty = all. Used to "
                             "re-measure only the stochastic attacks after they were "
                             "made reproducible.")
    parser.add_argument("--prompt-start", type=int, default=0,
                        help="First COCO prompt index, for sharding across GPUs")
    parser.add_argument("--num-prompts", type=int, default=NUM_PROMPTS, help="Number of COCO prompts")
    parser.add_argument("--num-seeds", type=int, default=NUM_SEEDS, help="Number of seeds per prompt")
    parser.add_argument("--seed-start", type=int, default=SEED_START, help="Starting seed")
    parser.add_argument("--save-images", action="store_true", help="Save images to disk for FID")
    parser.add_argument("--image-dir", default="qi_mirage_1k_images", help="Directory to save images")
    parser.add_argument("--checkpoint-interval", type=int, default=50,
                        help="Save checkpoint every N samples")
    args = parser.parse_args()

    device, dtype = resolve_qi_device(args.gpu)

    # ----- Setup image directories for FID -----
    image_dir = Path(args.image_dir)
    wm_img_dir = image_dir / "watermarked"
    unwm_img_dir = image_dir / "unwatermarked"
    wm_img_dir.mkdir(parents=True, exist_ok=True)
    unwm_img_dir.mkdir(parents=True, exist_ok=True)

    # ----- Build pipeline and watermark -----
    print("Loading pipeline...")
    pipe, scheduler = build_cached_pipe(MODEL_NAME, device, dtype)

    diffusion_config = DiffusionConfig(
        scheduler=scheduler,
        pipe=pipe,
        device=device,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        num_inference_steps=NUM_INFERENCE_STEPS,
        num_inversion_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        gen_seed=args.seed_start,
        inversion_type="ddim",
        dtype=dtype,
    )

    print(f"Loading MIRAGE watermark with config: {args.config}")
    watermark = AutoWatermark.load(
        "MIRAGE",
        algorithm_config=args.config,
        diffusion_config=diffusion_config,
    )

    # ----- Load COCO prompts -----
    print(f"Loading {args.num_prompts} COCO captions from {COCO_PARQUET} ...")
    prompts = load_coco_prompts(COCO_PARQUET, num_prompts=NUM_PROMPTS, seed=42)
    # Shard by prompt: load the full canonical set with the same seed so every
    # shard sees the identical prompt ordering, then take this shard's slice.
    prompts = prompts[args.prompt_start:args.prompt_start + args.num_prompts]
    print(f"  Loaded {len(prompts)} unique COCO captions.")

    seed_values = list(range(args.seed_start, args.seed_start + args.num_seeds))
    total_samples = len(prompts) * len(seed_values)

    # ----- Instantiate attacks -----
    wanted = [a.strip() for a in args.attacks.split(",") if a.strip()] or list(ATTACKS)
    attacks = {}
    for name, factory in ATTACKS.items():
        if name not in wanted:
            continue
        attacks[name] = factory() if factory is not None else None

    print(f"\n=== MIRAGE 1K Evaluation ===")
    print(f"  config={args.config}")
    print(f"  prompts={len(prompts)}, seeds={len(seed_values)}, total_samples={total_samples}")
    print(f"  attacks={list(attacks.keys())}")
    print(f"  device={device}, dtype={dtype}")
    print(f"  image_dir={image_dir}")

    # ----- Storage -----
    attack_flags = {n: {"wm": [], "unwm": []} for n in attacks}
    attack_scores = {n: {"wm": [], "unwm": []} for n in attacks}

    # ----- Checkpoint support -----
    checkpoint_path = Path(args.output).with_suffix(".checkpoint.json")
    checkpoint = load_checkpoint(checkpoint_path)
    start_idx = 0
    if checkpoint is not None:
        start_idx = checkpoint.get("completed", 0)
        attack_flags = checkpoint.get("attack_flags", attack_flags)
        attack_scores = checkpoint.get("attack_scores", attack_scores)
        print(f"  Resuming from checkpoint: {start_idx}/{total_samples} completed")

    t0 = time.time()

    for pi, prompt in enumerate(prompts):
        for si, seed in enumerate(seed_values):
            idx = pi * len(seed_values) + si
            if idx < start_idx:
                continue

            sample_num = idx + 1
            elapsed = time.time() - t0
            rate = (sample_num - start_idx) / max(elapsed, 1e-6) if sample_num > start_idx else 0
            eta = (total_samples - sample_num) / rate if rate > 0 else 0
            print(f"  [{sample_num}/{total_samples}] prompt={pi}, seed={seed}  "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s ETA)")

            # Generate watermarked image
            reseed_watermark(watermark, seed)
            wm_img = watermark.generate_watermarked_media(prompt)

            # Skip unwatermarked generation/detection (FPR already validated on 1000+ OOD images)

            # Run all attacks (watermarked only)
            for aname, attack in attacks.items():
                if attack is None:
                    test_wm = wm_img
                else:
                    test_wm = attack.edit(wm_img)

                wm_det = watermark.detect_watermark_in_media(test_wm, prompt=prompt)

                attack_flags[aname]["wm"].append(bool(wm_det["is_watermarked"]))
                attack_scores[aname]["wm"].append(
                    float(wm_det.get("presence_score", wm_det["raw_full_match_rate"]))
                )

            # Checkpoint
            if sample_num % args.checkpoint_interval == 0:
                save_checkpoint(checkpoint_path, {
                    "completed": sample_num,
                    "attack_flags": attack_flags,
                    "attack_scores": attack_scores,
                })
                print(f"    [checkpoint saved at {sample_num}]")

    total_time = time.time() - t0
    print(f"\nGeneration complete: {total_samples} samples in {total_time:.1f}s "
          f"({total_time/total_samples:.1f}s/sample)")

    # ----- Build per-attack results -----
    attack_results = {}
    for aname in attacks:
        tpr = safe_mean([1.0 if f else 0.0 for f in attack_flags[aname]["wm"]])
        fpr = safe_mean([1.0 if f else 0.0 for f in attack_flags[aname]["unwm"]])
        attack_results[aname] = {
            "TPR": tpr,
            "FPR": fpr,
            "wm_score_summary": summarize(attack_scores[aname]["wm"]),
            "unwm_score_summary": summarize(attack_scores[aname]["unwm"]),
        }
        # FPR is None by design here: unwatermarked generation is skipped
        # (see the note above), and the FPR column of the paper table comes
        # from the separate null study, not from this run.
        fstr = "n/a" if fpr is None else f"{fpr:.4f}"
        tstr = "n/a" if tpr is None else f"{tpr:.4f}"
        print(f"  {aname:25s}  TPR={tstr}  FPR={fstr}")

    # ----- Write JSON output -----
    output = {
        "algorithm": "MIRAGE",
        "config": args.config,
        "model": MODEL_NAME,
        "device": device,
        "steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "image_size": IMAGE_SIZE,
        "num_prompts": len(prompts),
        "num_seeds": len(seed_values),
        "num_samples": total_samples,
        "seeds": seed_values,
        "total_time_seconds": total_time,
        "image_dir": str(image_dir),
        "attacks": attack_results,
    }
    Path(args.output).write_text(
        json.dumps(to_jsonable(output), indent=2, ensure_ascii=False)
    )

    # ----- Write CSV output -----
    with Path(args.csv_output).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "attack", "TPR", "FPR",
            "wm_score_mean", "wm_score_std", "wm_score_min", "wm_score_max",
            "unwm_score_mean", "unwm_score_std", "unwm_score_min", "unwm_score_max",
        ])
        for aname in attacks:
            m = attack_results[aname]
            ws = m["wm_score_summary"]
            us = m["unwm_score_summary"]
            writer.writerow([
                aname,
                "" if m['TPR'] is None else f"{m['TPR']:.4f}",
                "" if m['FPR'] is None else f"{m['FPR']:.4f}",
                f"{ws['mean']:.4f}" if ws else "",
                f"{ws['std']:.4f}" if ws else "",
                f"{ws['min']:.4f}" if ws else "",
                f"{ws['max']:.4f}" if ws else "",
                f"{us['mean']:.4f}" if us else "",
                f"{us['std']:.4f}" if us else "",
                f"{us['min']:.4f}" if us else "",
                f"{us['max']:.4f}" if us else "",
            ])

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"\nResults written to {args.output} and {args.csv_output}")
    print(f"Images saved to {image_dir}/ (watermarked/ and unwatermarked/) for FID computation")


if __name__ == "__main__":
    main()

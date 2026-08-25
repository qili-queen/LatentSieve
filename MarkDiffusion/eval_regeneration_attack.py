"""Evaluate MIRAGE watermark robustness against diffusion-based regeneration attacks.

Uses SDEdit-style img2img regeneration: add noise to a watermarked image at
varying strength levels, then denoise through the Stable Diffusion pipeline.
Tests both white-box (same prompt) and black-box (empty prompt) settings.

Measures watermark detection rate, score statistics, and image quality metrics
(SSIM, PSNR) at each noise level.
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from qi_runtime_utils import build_cached_pipe, configure_qi_runtime, resolve_qi_device
from utils.diffusion_config import DiffusionConfig
from utils.media_utils import get_random_latents
from watermark.auto_watermark import AutoWatermark

configure_qi_runtime()

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

STRENGTHS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Subset of strengths to also test with empty (black-box) prompt
BLACKBOX_STRENGTHS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# ---------------------------------------------------------------------------
# Helpers (shared with other qi_eval scripts)
# ---------------------------------------------------------------------------


def to_jsonable(value):
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
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


def resolve_seed_values(seed_start, num_seeds, seeds):
    if seeds:
        return [int(item.strip()) for item in seeds.split(",") if item.strip()]
    return list(range(seed_start, seed_start + num_seeds))


def compute_image_quality(original: Image.Image, regenerated: Image.Image):
    """Compute SSIM and PSNR between two PIL images."""
    orig_arr = np.array(original).astype(np.float64)
    regen_arr = np.array(regenerated).astype(np.float64)
    # Ensure same size
    if orig_arr.shape != regen_arr.shape:
        regenerated = regenerated.resize(original.size, Image.LANCZOS)
        regen_arr = np.array(regenerated).astype(np.float64)
    ssim_val = ssim(orig_arr, regen_arr, data_range=255.0, channel_axis=2)
    psnr_val = psnr(orig_arr, regen_arr, data_range=255.0)
    return float(ssim_val), float(psnr_val)


def extract_score(det_result):
    """Extract the watermark detection score from a detection result dict."""
    return float(det_result.get("presence_score", det_result["raw_full_match_rate"]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MIRAGE watermark robustness against diffusion-based regeneration attacks."
    )
    parser.add_argument("--algorithm", default="MIRAGE")
    parser.add_argument("--model", default="huanzi05/stable-diffusion-2-1-base")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--config", default="config/MIRAGE_presence.json")
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--prompt-start", type=int, default=0)
    parser.add_argument("--prompt-end", type=int, default=-1)
    parser.add_argument("--max-prompts", type=int, default=10)
    parser.add_argument("--output", default="qi_regeneration_attack.json")
    parser.add_argument("--csv-output", default="qi_regeneration_attack.csv")
    parser.add_argument(
        "--regen-steps", type=int, default=50,
        help="Number of inference steps for the img2img regeneration pipeline.",
    )
    args = parser.parse_args()

    device, dtype = resolve_qi_device(args.gpu)
    pipe, scheduler = build_cached_pipe(args.model, device, dtype)
    diffusion_config = DiffusionConfig(
        scheduler=scheduler, pipe=pipe, device=device,
        image_size=(args.size, args.size),
        num_inference_steps=args.steps, num_inversion_steps=args.steps,
        guidance_scale=7.5, gen_seed=args.seed_start, inversion_type="ddim",
        dtype=dtype,
    )
    watermark = AutoWatermark.load(
        args.algorithm,
        algorithm_config=args.config,
        diffusion_config=diffusion_config,
    )

    # Build img2img pipeline sharing the same model components
    img2img_pipe = StableDiffusionImg2ImgPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        unet=pipe.unet,
        scheduler=pipe.scheduler,
        safety_checker=None,
        feature_extractor=pipe.feature_extractor if hasattr(pipe, "feature_extractor") else None,
        requires_safety_checker=False,
    ).to(device)

    prompt_end = None if args.prompt_end < 0 else args.prompt_end
    prompts = PROMPTS[args.prompt_start:prompt_end]
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]
    seed_values = resolve_seed_values(args.seed_start, args.num_seeds, args.seeds)

    total_samples = len(prompts) * len(seed_values)
    print(f"=== Regeneration Attack Evaluation ({args.algorithm}) ===")
    print(f"  prompts={len(prompts)}, seeds={len(seed_values)}, samples={total_samples}")
    print(f"  strengths (white-box)={STRENGTHS}")
    print(f"  strengths (black-box)={BLACKBOX_STRENGTHS}")

    # Storage keyed by "whitebox_{strength}" or "blackbox_{strength}"
    buckets = {}
    for s in STRENGTHS:
        key = f"whitebox_{s}"
        buckets[key] = {
            "wm_flags": [], "wm_scores": [],
            "ssim": [], "psnr": [],
        }
    for s in BLACKBOX_STRENGTHS:
        key = f"blackbox_{s}"
        buckets[key] = {
            "wm_flags": [], "wm_scores": [],
            "ssim": [], "psnr": [],
        }

    # Clean baseline
    clean_wm_flags, clean_wm_scores = [], []

    for pi, prompt in enumerate(prompts):
        for si, seed in enumerate(seed_values):
            idx = pi * len(seed_values) + si + 1
            print(f"  [{idx}/{total_samples}] prompt={pi}, seed={seed}")

            # Generate watermarked image
            reseed_watermark(watermark, seed)
            wm_img = watermark.generate_watermarked_media(prompt)

            # Clean baseline detection
            wm_det = watermark.detect_watermark_in_media(wm_img, prompt=prompt)
            clean_wm_flags.append(bool(wm_det["is_watermarked"]))
            clean_wm_scores.append(extract_score(wm_det))

            # White-box regeneration: use the SAME prompt
            for strength in STRENGTHS:
                regen_generator = torch.Generator(device=device)
                regen_generator.manual_seed(seed)
                regen_img = img2img_pipe(
                    prompt=prompt,
                    image=wm_img,
                    strength=strength,
                    num_inference_steps=args.regen_steps,
                    guidance_scale=7.5,
                    generator=regen_generator,
                ).images[0]

                det = watermark.detect_watermark_in_media(regen_img, prompt=prompt)
                ssim_val, psnr_val = compute_image_quality(wm_img, regen_img)

                bkey = f"whitebox_{strength}"
                buckets[bkey]["wm_flags"].append(bool(det["is_watermarked"]))
                buckets[bkey]["wm_scores"].append(extract_score(det))
                buckets[bkey]["ssim"].append(ssim_val)
                buckets[bkey]["psnr"].append(psnr_val)

            # Black-box regeneration: use EMPTY prompt
            for strength in BLACKBOX_STRENGTHS:
                regen_generator = torch.Generator(device=device)
                regen_generator.manual_seed(seed)
                regen_img = img2img_pipe(
                    prompt="",
                    image=wm_img,
                    strength=strength,
                    num_inference_steps=args.regen_steps,
                    guidance_scale=7.5,
                    generator=regen_generator,
                ).images[0]

                det = watermark.detect_watermark_in_media(regen_img, prompt=prompt)
                ssim_val, psnr_val = compute_image_quality(wm_img, regen_img)

                bkey = f"blackbox_{strength}"
                buckets[bkey]["wm_flags"].append(bool(det["is_watermarked"]))
                buckets[bkey]["wm_scores"].append(extract_score(det))
                buckets[bkey]["ssim"].append(ssim_val)
                buckets[bkey]["psnr"].append(psnr_val)

    # -----------------------------------------------------------------------
    # Build JSON output
    # -----------------------------------------------------------------------
    output = {
        "algorithm": args.algorithm,
        "model": args.model,
        "config": args.config,
        "steps": args.steps,
        "regen_steps": args.regen_steps,
        "size": args.size,
        "num_prompts": len(prompts),
        "num_seeds": len(seed_values),
        "num_samples": total_samples,
        "prompts": prompts,
        "seeds": seed_values,
        "clean_baseline": {
            "wm_detect_rate": mean([1.0 if f else 0.0 for f in clean_wm_flags]),
            "wm_score_summary": summarize(clean_wm_scores),
        },
        "whitebox": {},
        "blackbox": {},
    }

    for strength in STRENGTHS:
        bkey = f"whitebox_{strength}"
        b = buckets[bkey]
        output["whitebox"][str(strength)] = {
            "wm_detect_rate": mean([1.0 if f else 0.0 for f in b["wm_flags"]]),
            "wm_score_summary": summarize(b["wm_scores"]),
            "ssim_summary": summarize(b["ssim"]),
            "psnr_summary": summarize(b["psnr"]),
        }

    for strength in BLACKBOX_STRENGTHS:
        bkey = f"blackbox_{strength}"
        b = buckets[bkey]
        output["blackbox"][str(strength)] = {
            "wm_detect_rate": mean([1.0 if f else 0.0 for f in b["wm_flags"]]),
            "wm_score_summary": summarize(b["wm_scores"]),
            "ssim_summary": summarize(b["ssim"]),
            "psnr_summary": summarize(b["psnr"]),
        }

    Path(args.output).write_text(
        json.dumps(to_jsonable(output), indent=2, ensure_ascii=False)
    )

    # -----------------------------------------------------------------------
    # Build CSV output
    # -----------------------------------------------------------------------
    with Path(args.csv_output).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "setting", "strength",
            "wm_detect_rate (TPR)", "wm_score_mean", "wm_score_min",
            "ssim_mean", "psnr_mean",
        ])
        # Clean baseline row
        clean_s = output["clean_baseline"]["wm_score_summary"]
        writer.writerow([
            "clean", "0.0",
            f"{output['clean_baseline']['wm_detect_rate']:.4f}",
            f"{clean_s['mean']:.4f}",
            f"{clean_s['min']:.4f}",
            "1.0000",
            "inf",
        ])
        # White-box rows
        for strength in STRENGTHS:
            m = output["whitebox"][str(strength)]
            writer.writerow([
                "whitebox", str(strength),
                f"{m['wm_detect_rate']:.4f}",
                f"{m['wm_score_summary']['mean']:.4f}",
                f"{m['wm_score_summary']['min']:.4f}",
                f"{m['ssim_summary']['mean']:.4f}",
                f"{m['psnr_summary']['mean']:.4f}",
            ])
        # Black-box rows
        for strength in BLACKBOX_STRENGTHS:
            m = output["blackbox"][str(strength)]
            writer.writerow([
                "blackbox", str(strength),
                f"{m['wm_detect_rate']:.4f}",
                f"{m['wm_score_summary']['mean']:.4f}",
                f"{m['wm_score_summary']['min']:.4f}",
                f"{m['ssim_summary']['mean']:.4f}",
                f"{m['psnr_summary']['mean']:.4f}",
            ])

    print(f"\nResults written to {args.output} and {args.csv_output}")


if __name__ == "__main__":
    main()

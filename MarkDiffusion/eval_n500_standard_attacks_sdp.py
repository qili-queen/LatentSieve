"""Stage 2: Standard 8 attacks × 5 schemes × N=500.

Reads cached watermarked images from eval_n500/{scheme}_wm/, applies each of
8 standard image-domain attacks, and runs each scheme's detector. Records TPR
per (scheme, attack) cell.

Schemes: MIRAGE, PRC, GS, TR, T2SMark.
Attacks: JPEG-75, Blur-2, Noise-15, CropPadBlack-0.8, Brightness-1.2, Mask,
Overlay, AdaptiveNoise.

Usage:
  python eval_n500_standard_attacks.py --gpu 0 --schemes MIRAGE,PRC
"""

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

oti = tqdm.__init__
@functools.wraps(oti)
def st(s, *a, **k): k['disable'] = True; oti(s, *a, **k)
tqdm.__init__ = st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.tools.image_editor import (
    AdaptiveNoiseInjection, Brightness, CrSc, GaussianBlurring,
    GaussianNoise, JPEGCompression, Mask, Overlay,
)
from runtime_utils import build_cached_pipe, configure_runtime, resolve_device
from utils.diffusion_config import DiffusionConfig
from utils.media_utils import get_random_latents
from watermark.auto_watermark import AutoWatermark

configure_runtime()

MODEL = "stabilityai/stable-diffusion-2-1-base"
STEPS = 50
GS_SCALE = 7.5
SIZE = 512

from sdp_n500_config import PROMPTS

WM_DIR_BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "eval_n500_sdp"
)

SCHEME_CONFIGS = {
    "MIRAGE": "config/MIRAGE_presence.json",
    "PRC":    "config/PRC.json",
    "GS":     "config/GS.json",
    "TR":     "config/TR.json",
}


def make_attacks():
    return {
        "Clean": None,
        "JPEG-75": JPEGCompression(quality=75),
        "Blur-2": GaussianBlurring(radius=2),
        "Noise-15": GaussianNoise(sigma=15.0),
        "CropPadBlack-0.8": CrSc(crop_ratio=0.8, resize_back=False,
                                  pad_back=True),
        "Brightness-1.2": Brightness(factor=1.2),
        "Mask": Mask(mask_ratio=0.2),
        "Overlay": Overlay(num_strokes=10),
        "AdaptiveNoise": AdaptiveNoiseInjection(intensity=0.5),
    }


def reseed(wm, seed):
    wm.config.gen_seed = seed
    wm.config.init_latents_seed = seed
    r = torch.Generator(device=wm.config.device).manual_seed(seed)
    wm.config.init_latents = get_random_latents(
        wm.config.pipe, height=SIZE, width=SIZE, generator=r
    ).to(wm.config.device, dtype=wm.config.dtype)


def detect_with_score(wm, image, prompt):
    """Detect watermark; return (is_watermarked, score)."""
    r = wm.detect_watermark_in_media(image, prompt=prompt)
    is_wm = bool(r.get("is_watermarked", False))
    # Score: prefer weighted_full_match_rate (MIRAGE), then bit_acc (GS), then
    # 1 - l1_distance (TR), then score (PRC).
    if "weighted_full_match_rate" in r:
        score = float(r["weighted_full_match_rate"])
    elif "bit_acc" in r:
        score = float(r["bit_acc"])
    elif "l1_distance" in r:
        score = float(r["l1_distance"])
    elif "score" in r:
        score = float(r["score"])
    else:
        score = 0.0
    return is_wm, score


def evaluate_markdiff_scheme(scheme, gpu, num_prompts, seeds_per_prompt,
                              output_dir):
    device, dtype = resolve_device(gpu)
    pipe, scheduler = build_cached_pipe(MODEL, device, dtype)
    dc = DiffusionConfig(
        scheduler=scheduler, pipe=pipe, device=device,
        image_size=(SIZE, SIZE), num_inference_steps=STEPS,
        num_inversion_steps=STEPS, guidance_scale=GS_SCALE,
        gen_seed=42, inversion_type="ddim", dtype=dtype,
    )
    config_path = SCHEME_CONFIGS[scheme]
    wm = AutoWatermark.load(scheme, algorithm_config=config_path,
                             diffusion_config=dc)

    wm_dir = os.path.join(WM_DIR_BASE, f"{scheme}_wm")
    if not os.path.isdir(wm_dir):
        raise FileNotFoundError(f"WM image dir not found: {wm_dir}")

    attacks = make_attacks()
    total = num_prompts * seeds_per_prompt
    results = {name: [] for name in attacks}

    out_path = os.path.join(output_dir, f"{scheme}_standard_attacks.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
            results = existing.get("results_per_attack", results)
            print(f"  {scheme}: resumed from {sum(len(v) for v in results.values())} existing records",
                  flush=True)

    print(f"  {scheme}: evaluating {total} images × {len(attacks)} attacks",
          flush=True)
    t_start = time.time()

    for pi in range(num_prompts):
        prompt = PROMPTS[pi]
        for si in range(seeds_per_prompt):
            seed = pi * 100 + si
            idx = pi * seeds_per_prompt + si
            img_path = os.path.join(wm_dir, f"img_{idx:04d}.png")
            if not os.path.exists(img_path):
                print(f"  WARN: missing image {img_path}", flush=True)
                continue

            wm_img = Image.open(img_path).convert("RGB")

            for attack_name, attack in attacks.items():
                # Skip if already evaluated
                if any(r.get("idx") == idx
                       for r in results[attack_name]):
                    continue

                if attack is None:
                    attacked = wm_img
                else:
                    attacked = attack.edit(wm_img.copy(), {})

                reseed(wm, seed)
                is_wm, score = detect_with_score(wm, attacked, prompt)
                results[attack_name].append({
                    "idx": idx, "seed": seed, "prompt_idx": pi,
                    "is_watermarked": is_wm, "score": score,
                })

            if (idx + 1) % 25 == 0 or idx + 1 == total:
                elapsed = time.time() - t_start
                rate = (idx + 1) / max(elapsed, 1)
                eta_s = (total - idx - 1) / max(rate, 0.01)
                print(f"    [{idx+1}/{total}] {scheme} elapsed={elapsed/60:.1f}min ETA={eta_s/60:.1f}min",
                      flush=True)

                # Save partial results
                summary = {}
                for atk, recs in results.items():
                    if recs:
                        summary[atk] = {
                            "n": len(recs),
                            "tpr": float(np.mean(
                                [r["is_watermarked"] for r in recs]
                            )),
                            "score_mean": float(np.mean(
                                [r["score"] for r in recs]
                            )),
                        }
                output = {
                    "scheme": scheme,
                    "num_prompts": num_prompts,
                    "seeds_per_prompt": seeds_per_prompt,
                    "results_per_attack": results,
                    "summary": summary,
                }
                with open(out_path, "w") as f:
                    json.dump(output, f, indent=2, default=str)

    # Final summary
    summary = {}
    for atk, recs in results.items():
        if recs:
            summary[atk] = {
                "n": len(recs),
                "tpr": float(np.mean([r["is_watermarked"] for r in recs])),
                "score_mean": float(np.mean([r["score"] for r in recs])),
            }
    output = {
        "scheme": scheme,
        "num_prompts": num_prompts,
        "seeds_per_prompt": seeds_per_prompt,
        "results_per_attack": results,
        "summary": summary,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n=== {scheme} SUMMARY ({total} images) ===", flush=True)
    print(f"  {'attack':<20} {'TPR':>8} {'score_mean':>12}", flush=True)
    for atk in attacks.keys():
        if atk in summary:
            d = summary[atk]
            print(f"  {atk:<20} {d['tpr']*100:>7.1f}% {d['score_mean']:>12.4f}",
                  flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--schemes", type=str, default="MIRAGE,PRC,GS,TR",
                        help="comma-separated schemes (T2SMark requires special pipeline)")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--seeds-per-prompt", type=int, default=5)
    parser.add_argument("--output-dir", type=str,
                        default="eval_n500_sdp/standard_attacks_results")
    args = parser.parse_args()

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Schemes: {schemes}", flush=True)
    print(f"Total per scheme: {args.num_prompts * args.seeds_per_prompt}",
          flush=True)
    print(f"Attacks: 9 (Clean + 8 attacks)", flush=True)
    print(f"GPU: {args.gpu}", flush=True)
    print(f"Output: {args.output_dir}", flush=True)

    for scheme in schemes:
        if scheme not in SCHEME_CONFIGS:
            print(f"SKIP unknown scheme: {scheme}", flush=True)
            continue
        print(f"\n=== Evaluating {scheme} ===", flush=True)
        evaluate_markdiff_scheme(
            scheme, args.gpu, args.num_prompts, args.seeds_per_prompt,
            args.output_dir,
        )


if __name__ == "__main__":
    main()

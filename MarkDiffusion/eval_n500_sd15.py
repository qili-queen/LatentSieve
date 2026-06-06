"""Stage 8 — SD-1.5 cross-backbone evaluation at N=500.

Generates MIRAGE watermarked images on SD-1.5 (different VAE / U-Net from SD-2.1),
applies 9 standard attacks, and runs MIRAGE detection.

Usage:
  python eval_n500_sd15.py --gpu 2
"""

import argparse
import functools
import json
import os
import sys
import time

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

MODEL = "sd-legacy/stable-diffusion-v1-5"
STEPS = 50
GS_SCALE = 7.5
SIZE = 512

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

WM_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "eval_n500/MIRAGE_sd15_wm"
)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--seeds-per-prompt", type=int, default=50)
    parser.add_argument("--output", type=str,
                        default="eval_n500/sd15_mirage.json")
    args = parser.parse_args()

    os.makedirs(WM_DIR, exist_ok=True)

    device, dtype = resolve_device(args.gpu)
    pipe, scheduler = build_cached_pipe(MODEL, device, dtype)
    dc = DiffusionConfig(
        scheduler=scheduler, pipe=pipe, device=device,
        image_size=(SIZE, SIZE), num_inference_steps=STEPS,
        num_inversion_steps=STEPS, guidance_scale=GS_SCALE,
        gen_seed=42, inversion_type="ddim", dtype=dtype,
    )
    wm = AutoWatermark.load(
        "MIRAGE", algorithm_config="config/MIRAGE_presence.json",
        diffusion_config=dc,
    )

    total = args.num_prompts * args.seeds_per_prompt
    print(f"SD-1.5 MIRAGE eval, N={total}", flush=True)
    print(f"Model: {MODEL}", flush=True)
    print(f"WM dir: {WM_DIR}", flush=True)

    attacks = make_attacks()
    results = {name: [] for name in attacks}
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f).get("results_per_attack", results)
        for a in attacks:
            if a not in results:
                results[a] = []
        print(f"  Resumed; existing counts: "
              f"{[len(results[a]) for a in attacks]}", flush=True)

    t_start = time.time()
    counter = 0

    for pi in range(args.num_prompts):
        prompt = PROMPTS[pi]
        for si in range(args.seeds_per_prompt):
            seed = pi * 100 + si
            idx = pi * args.seeds_per_prompt + si
            counter += 1
            wm_path = os.path.join(WM_DIR, f"img_{idx:04d}.png")

            # Generate watermarked image if not cached
            if os.path.exists(wm_path):
                wm_img = Image.open(wm_path).convert("RGB")
            else:
                reseed(wm, seed)
                wm_img = wm.generate_watermarked_media(prompt)
                wm_img.save(wm_path)

            for atk_name, atk in attacks.items():
                if any(r.get("idx") == idx for r in results[atk_name]):
                    continue
                if atk is None:
                    attacked = wm_img
                else:
                    attacked = atk.edit(wm_img.copy(), {})
                reseed(wm, seed)
                r = wm.detect_watermark_in_media(attacked, prompt=prompt)
                results[atk_name].append({
                    "idx": idx, "seed": seed, "prompt_idx": pi,
                    "is_watermarked": bool(
                        r.get("is_watermarked", False)
                    ),
                    "ba": float(
                        r.get("weighted_full_match_rate", 0.5)
                    ),
                })

            if counter % 25 == 0 or counter == total:
                elapsed = time.time() - t_start
                rate = counter / max(elapsed, 1)
                eta_s = (total - counter) / max(rate, 0.01)
                print(f"  [{counter}/{total}] elapsed={elapsed/60:.1f}min "
                      f"ETA={eta_s/60:.1f}min", flush=True)

                summary = {}
                for atk_name, recs in results.items():
                    if recs:
                        summary[atk_name] = {
                            "n": len(recs),
                            "tpr": float(np.mean(
                                [r["is_watermarked"] for r in recs]
                            )),
                            "ba_mean": float(np.mean(
                                [r["ba"] for r in recs]
                            )),
                        }
                output = {
                    "scheme": "MIRAGE",
                    "model": MODEL,
                    "num_prompts": args.num_prompts,
                    "seeds_per_prompt": args.seeds_per_prompt,
                    "results_per_attack": results,
                    "summary": summary,
                }
                with open(args.output, "w") as f:
                    json.dump(output, f, indent=2, default=str)

    print("\n=== SD-1.5 SUMMARY ===", flush=True)
    print(f"  {'attack':<20} {'TPR':>8} {'BA_mean':>10}", flush=True)
    for atk_name in attacks.keys():
        recs = results.get(atk_name, [])
        if recs:
            tpr = float(np.mean([r["is_watermarked"] for r in recs]))
            ba = float(np.mean([r["ba"] for r in recs]))
            print(f"  {atk_name:<20} {tpr*100:>7.1f}% {ba:>10.4f}",
                  flush=True)


if __name__ == "__main__":
    main()

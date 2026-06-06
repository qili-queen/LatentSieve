"""Stage 8: Attack-strength sweep on MIRAGE × N=500.

Uses cached watermarked images from eval_n500/MIRAGE_wm/.
Sweeps JPEG, Noise, Blur, CropPad strengths and records TPR / BA.

Usage:
  python eval_n500_strength_sweep.py --gpu 1
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
    CrSc, GaussianBlurring, GaussianNoise, JPEGCompression,
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

# Strength sweep: each attack at 5 strengths
ATTACK_SWEEPS = {
    "JPEG": {
        "factory": lambda q: JPEGCompression(quality=int(q)),
        "strengths": [90, 75, 50, 30, 10],
    },
    "Noise": {
        "factory": lambda s: GaussianNoise(sigma=float(s)),
        "strengths": [5, 15, 25, 40, 50],
    },
    "Blur": {
        "factory": lambda r: GaussianBlurring(radius=int(r)),
        "strengths": [2, 3, 4, 5, 6],
    },
    "CropPad": {
        "factory": lambda c: CrSc(
            crop_ratio=float(c), resize_back=False, pad_back=True
        ),
        "strengths": [0.8, 0.7, 0.6, 0.5, 0.4],
    },
}

WM_DIR_BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "eval_n500_sdp"
)


def reseed(wm, seed):
    wm.config.gen_seed = seed
    wm.config.init_latents_seed = seed
    r = torch.Generator(device=wm.config.device).manual_seed(seed)
    wm.config.init_latents = get_random_latents(
        wm.config.pipe, height=SIZE, width=SIZE, generator=r
    ).to(wm.config.device, dtype=wm.config.dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--seeds-per-prompt", type=int, default=5)
    parser.add_argument("--output", type=str,
                        default="eval_n500_sdp/strength_sweep_mirage.json")
    args = parser.parse_args()

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

    wm_dir = os.path.join(WM_DIR_BASE, "MIRAGE_wm")
    total = args.num_prompts * args.seeds_per_prompt
    print(f"MIRAGE strength sweep, N={total}", flush=True)

    # Resume if output exists
    results = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f).get("results", {})
        print(f"  Resumed; existing keys: {list(results.keys())}", flush=True)

    # Initialize results structure
    for atk_type, sweep in ATTACK_SWEEPS.items():
        if atk_type not in results:
            results[atk_type] = {}
        for s in sweep["strengths"]:
            key = str(s)
            if key not in results[atk_type]:
                results[atk_type][key] = []

    t_start = time.time()
    counter = 0
    total_ops = (sum(len(s["strengths"]) for s in ATTACK_SWEEPS.values())
                 + 1) * total  # +1 for clean
    op_counter = 0

    # Clean baseline
    if "Clean" not in results:
        results["Clean"] = {"0": []}

    for pi in range(args.num_prompts):
        prompt = PROMPTS[pi]
        for si in range(args.seeds_per_prompt):
            seed = pi * 100 + si
            idx = pi * args.seeds_per_prompt + si
            counter += 1
            img_path = os.path.join(wm_dir, f"img_{idx:04d}.png")
            if not os.path.exists(img_path):
                continue
            wm_img = Image.open(img_path).convert("RGB")

            # Clean detection
            if not any(r.get("idx") == idx for r in results["Clean"]["0"]):
                reseed(wm, seed)
                r = wm.detect_watermark_in_media(wm_img, prompt=prompt)
                results["Clean"]["0"].append({
                    "idx": idx, "seed": seed, "prompt_idx": pi,
                    "is_watermarked": bool(r.get("is_watermarked", False)),
                    "ba": float(r.get("weighted_full_match_rate", 0.5)),
                })
            op_counter += 1

            # Each attack at each strength
            for atk_type, sweep in ATTACK_SWEEPS.items():
                for strength in sweep["strengths"]:
                    key = str(strength)
                    if any(r.get("idx") == idx
                           for r in results[atk_type][key]):
                        op_counter += 1
                        continue

                    attack = sweep["factory"](strength)
                    attacked = attack.edit(wm_img.copy(), {})
                    reseed(wm, seed)
                    r = wm.detect_watermark_in_media(
                        attacked, prompt=prompt
                    )
                    results[atk_type][key].append({
                        "idx": idx, "seed": seed, "prompt_idx": pi,
                        "is_watermarked": bool(
                            r.get("is_watermarked", False)
                        ),
                        "ba": float(
                            r.get("weighted_full_match_rate", 0.5)
                        ),
                    })
                    op_counter += 1

            if counter % 25 == 0 or counter == total:
                elapsed = time.time() - t_start
                rate = counter / max(elapsed, 1)
                eta_s = (total - counter) / max(rate, 0.01)
                print(f"  [{counter}/{total}] elapsed={elapsed/60:.1f}min "
                      f"ETA={eta_s/60:.1f}min", flush=True)

                # Save progress
                summary = {}
                for atk_type, by_strength in results.items():
                    summary[atk_type] = {}
                    for k, recs in by_strength.items():
                        if recs:
                            summary[atk_type][k] = {
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
                    "num_prompts": args.num_prompts,
                    "seeds_per_prompt": args.seeds_per_prompt,
                    "results": results,
                    "summary": summary,
                }
                with open(args.output, "w") as f:
                    json.dump(output, f, indent=2, default=str)

    print("\n=== SUMMARY ===", flush=True)
    for atk_type, by_strength in results.items():
        if not any(by_strength.values()):
            continue
        print(f"\n{atk_type}:", flush=True)
        for k in sorted(by_strength.keys(), key=lambda x: float(x)):
            recs = by_strength[k]
            if recs:
                tpr = float(np.mean([r["is_watermarked"] for r in recs]))
                ba = float(np.mean([r["ba"] for r in recs]))
                print(f"  strength={k:>5} TPR={tpr*100:>5.1f}% BA={ba:.4f} (n={len(recs)})",
                      flush=True)


if __name__ == "__main__":
    main()

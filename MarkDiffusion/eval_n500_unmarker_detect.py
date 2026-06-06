"""Stage 6 — UnMarker detection at N=200.

Reads attacked images from eval_n500/{scheme}_attacked_s1/ and _s2/.
Runs each scheme's detector and records TPR.

Usage:
  python eval_n500_unmarker_detect.py --gpu 2 --schemes MIRAGE,PRC,GS
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

from runtime_utils import build_cached_pipe, configure_runtime, resolve_device
from utils.diffusion_config import DiffusionConfig
from utils.media_utils import get_random_latents
from watermark.auto_watermark import AutoWatermark

configure_runtime()

MODEL = "stabilityai/stable-diffusion-2-1-base"
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

WM_DIR_BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "eval_n500"
)

SCHEME_CONFIGS = {
    "MIRAGE": "config/MIRAGE_presence.json",
    "PRC":    "config/PRC.json",
    "GS":     "config/GS.json",
    "TR":     "config/TR.json",
}

SEEDS_PER_PROMPT = 50  # matches eval_n500 layout


def reseed(wm, seed):
    wm.config.gen_seed = seed
    wm.config.init_latents_seed = seed
    r = torch.Generator(device=wm.config.device).manual_seed(seed)
    wm.config.init_latents = get_random_latents(
        wm.config.pipe, height=SIZE, width=SIZE, generator=r
    ).to(wm.config.device, dtype=wm.config.dtype)


def detect_with_scheme(wm, image, prompt, scheme):
    r = wm.detect_watermark_in_media(image, prompt=prompt)
    is_wm = bool(r.get("is_watermarked", False))
    if scheme == "MIRAGE":
        ba = float(r.get("weighted_full_match_rate", 0.5))
    elif scheme == "PRC":
        ba = 1.0 if r.get("is_watermarked", False) else 0.0
    elif scheme == "GS":
        ba = float(r.get("bit_acc", 0.5))
    elif scheme == "TR":
        ba = float(r.get("l1_distance", 999))
    else:
        ba = 0.0
    return is_wm, ba


def evaluate_scheme_unmarker(scheme, gpu, max_images=200):
    device, dtype = resolve_device(gpu)
    pipe, scheduler = build_cached_pipe(MODEL, device, dtype)
    dc = DiffusionConfig(
        scheduler=scheduler, pipe=pipe, device=device,
        image_size=(SIZE, SIZE), num_inference_steps=STEPS,
        num_inversion_steps=STEPS, guidance_scale=GS_SCALE,
        gen_seed=42, inversion_type="ddim", dtype=dtype,
    )
    wm = AutoWatermark.load(scheme,
                             algorithm_config=SCHEME_CONFIGS[scheme],
                             diffusion_config=dc)

    out_path = os.path.join(
        WM_DIR_BASE, f"unmarker_{scheme}_detect.json"
    )
    results = {"s1": [], "s2": [], "clean": []}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f).get("results", results)
        print(f"  {scheme}: resumed; counts {[len(v) for v in results.values()]}",
              flush=True)

    t_start = time.time()

    for idx in range(max_images):
        pi = idx // SEEDS_PER_PROMPT
        si = idx % SEEDS_PER_PROMPT
        seed = pi * 100 + si
        prompt = PROMPTS[pi]

        for stage in ["clean", "s1", "s2"]:
            if any(r.get("idx") == idx for r in results[stage]):
                continue
            if stage == "clean":
                img_path = os.path.join(
                    WM_DIR_BASE, f"{scheme}_wm/img_{idx:04d}.png"
                )
            else:
                img_path = os.path.join(
                    WM_DIR_BASE, f"{scheme}_attacked_{stage}/img_{idx:04d}.png"
                )
            if not os.path.exists(img_path):
                continue

            img = Image.open(img_path).convert("RGB")
            reseed(wm, seed)
            is_wm, ba = detect_with_scheme(wm, img, prompt, scheme)
            results[stage].append({
                "idx": idx, "seed": seed, "prompt_idx": pi,
                "is_watermarked": is_wm, "ba": ba,
            })

        if (idx + 1) % 25 == 0 or idx + 1 == max_images:
            elapsed = time.time() - t_start
            print(f"  {scheme} [{idx+1}/{max_images}] elapsed={elapsed/60:.1f}min",
                  flush=True)
            # Save partial
            summary = {}
            for stage, recs in results.items():
                if recs:
                    summary[stage] = {
                        "n": len(recs),
                        "tpr": float(np.mean(
                            [r["is_watermarked"] for r in recs]
                        )),
                        "ba_mean": float(np.mean(
                            [r["ba"] for r in recs]
                        )),
                    }
            output = {"scheme": scheme, "results": results,
                       "summary": summary}
            with open(out_path, "w") as f:
                json.dump(output, f, indent=2, default=str)

    print(f"\n=== {scheme} UnMarker SUMMARY (N={max_images}) ===",
          flush=True)
    for stage, recs in results.items():
        if recs:
            tpr = float(np.mean([r["is_watermarked"] for r in recs]))
            ba = float(np.mean([r["ba"] for r in recs]))
            print(f"  {stage:<6} TPR={tpr*100:>5.1f}%  BA={ba:.4f}  "
                  f"(n={len(recs)})", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--schemes", type=str,
                        default="MIRAGE,PRC,GS,TR")
    parser.add_argument("--max-images", type=int, default=200)
    args = parser.parse_args()

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    for scheme in schemes:
        if scheme not in SCHEME_CONFIGS:
            print(f"SKIP {scheme}: not a markdiff scheme", flush=True)
            continue
        print(f"\n=== Detecting {scheme} ===", flush=True)
        evaluate_scheme_unmarker(scheme, args.gpu, args.max_images)


if __name__ == "__main__":
    main()

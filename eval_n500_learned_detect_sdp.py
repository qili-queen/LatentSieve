"""Stage 7 — Detection on learned-attack images for 4 markdiff schemes.

Reads attacked images from eval_n500/{scheme}_attacked_{attacker}/
Runs each scheme's detector. T2SMark handled separately.

Usage:
  python eval_n500_learned_detect.py --gpu 0 --schemes MIRAGE,PRC,GS,TR
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

from sdp_n500_config import PROMPTS

BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "eval_n500_sdp"
)

SCHEME_CONFIGS = {
    "MIRAGE": "config/MIRAGE_presence.json",
    "PRC":    "config/PRC.json",
    "GS":     "config/GS.json",
    "TR":     "config/TR.json",
}

SEEDS_PER_PROMPT = 5

ATTACKERS = ["cheng2020-anchor_3", "diff_attacker_60", "Ctrlgen"]


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
        ba = 1.0 if is_wm else 0.0
    elif scheme == "GS":
        ba = float(r.get("bit_acc", 0.5))
    elif scheme == "TR":
        ba = float(r.get("l1_distance", 999))
    else:
        ba = 0.0
    return is_wm, ba


def evaluate(scheme, gpu, max_images=500):
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

    out_path = os.path.join(BASE, f"learned_{scheme}_detect.json")
    results = {a: [] for a in ATTACKERS}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f).get("results", results)
        for a in ATTACKERS:
            if a not in results:
                results[a] = []
        print(f"  {scheme}: resumed; counts {[len(results[a]) for a in ATTACKERS]}",
              flush=True)

    t_start = time.time()

    for idx in range(max_images):
        pi = idx // SEEDS_PER_PROMPT
        si = idx % SEEDS_PER_PROMPT
        seed = pi * 100 + si
        prompt = PROMPTS[pi]

        for attacker in ATTACKERS:
            if any(r.get("idx") == idx for r in results[attacker]):
                continue
            img_path = os.path.join(
                BASE, f"{scheme}_attacked_{attacker}/img_{idx:04d}.png"
            )
            if not os.path.exists(img_path):
                continue

            img = Image.open(img_path).convert("RGB")
            reseed(wm, seed)
            is_wm, ba = detect_with_scheme(wm, img, prompt, scheme)
            results[attacker].append({
                "idx": idx, "seed": seed, "prompt_idx": pi,
                "is_watermarked": is_wm, "ba": ba,
            })

        if (idx + 1) % 25 == 0 or idx + 1 == max_images:
            elapsed = time.time() - t_start
            print(f"  {scheme} [{idx+1}/{max_images}] elapsed={elapsed/60:.1f}min",
                  flush=True)
            summary = {}
            for a, recs in results.items():
                if recs:
                    summary[a] = {
                        "n": len(recs),
                        "tpr": float(np.mean(
                            [r["is_watermarked"] for r in recs]
                        )),
                        "ba_mean": float(np.mean(
                            [r["ba"] for r in recs]
                        )),
                    }
            output = {"scheme": scheme, "results": results, "summary": summary}
            with open(out_path, "w") as f:
                json.dump(output, f, indent=2, default=str)

    print(f"\n=== {scheme} Learned-attack SUMMARY ===", flush=True)
    for a, recs in results.items():
        if recs:
            tpr = float(np.mean([r["is_watermarked"] for r in recs]))
            ba = float(np.mean([r["ba"] for r in recs]))
            print(f"  {a:<24} TPR={tpr*100:>5.1f}% BA={ba:.4f} (n={len(recs)})",
                  flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--schemes", type=str, default="MIRAGE,PRC,GS,TR")
    parser.add_argument("--max-images", type=int, default=500)
    args = parser.parse_args()

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    for scheme in schemes:
        if scheme not in SCHEME_CONFIGS:
            print(f"SKIP {scheme}", flush=True)
            continue
        print(f"\n=== Detecting {scheme} ===", flush=True)
        evaluate(scheme, args.gpu, args.max_images)


if __name__ == "__main__":
    main()

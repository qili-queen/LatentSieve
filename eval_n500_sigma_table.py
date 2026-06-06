"""Stage 8 — Per-coord σ_eff table at N=500.

Measures per-coordinate sign-agreement rate p_j stratified by |z_T_orig|
bins, and computes σ_eff = 1 / tan(π (p - 1/2)) for each attack.

For each watermarked image at seed s:
  1. z_T_orig = MIRAGE.inject_watermark() (deterministic from seed s)
  2. Apply attack to image
  3. Invert attacked image -> z_rec
  4. Per coord j: agreement_j = (sign(z_T_orig_j) == sign(z_rec_j))
  5. Stratify by bins of |z_T_orig_j|

Output: per-attack mean p per bin (table 9 style).

Usage:
  python eval_n500_sigma_table.py --gpu 1
"""

import argparse
import functools
import json
import math
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
    GaussianBlurring, GaussianNoise, JPEGCompression,
)
from runtime_utils import build_cached_pipe, configure_runtime, resolve_device
from utils.diffusion_config import DiffusionConfig
from utils.media_utils import (
    get_random_latents, get_media_latents, transform_to_model_format,
)
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

# 6 attacks for stratification (matches paper Table 9)
ATTACKS = {
    "Noise-15": lambda: GaussianNoise(sigma=15.0),
    "Noise-30": lambda: GaussianNoise(sigma=30.0),
    "JPEG-50":  lambda: JPEGCompression(quality=50),
    "JPEG-30":  lambda: JPEGCompression(quality=30),
    "Blur-2":   lambda: GaussianBlurring(radius=2),
    "Blur-4":   lambda: GaussianBlurring(radius=4),
}

# |z| bins (centers from paper Table 9: 0.12, 0.38, 0.62, 0.88, 1.12, 1.88)
# Boundaries: [0, 0.25, 0.5, 0.75, 1.0, 1.5, 4.0]
BIN_EDGES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 4.0]
BIN_CENTERS = [(BIN_EDGES[i] + BIN_EDGES[i+1]) / 2 for i in range(len(BIN_EDGES)-1)]

WM_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "eval_n500/MIRAGE_wm"
)


def reseed(wm, seed):
    wm.config.gen_seed = seed
    wm.config.init_latents_seed = seed
    r = torch.Generator(device=wm.config.device).manual_seed(seed)
    wm.config.init_latents = get_random_latents(
        wm.config.pipe, height=SIZE, width=SIZE, generator=r
    ).to(wm.config.device, dtype=wm.config.dtype)


def invert_image(wm, image, prompt):
    do_cfg = wm.config.guidance_scale > 1.0
    pe, ne = wm.config.pipe.encode_prompt(
        prompt=prompt, device=wm.config.device,
        do_classifier_free_guidance=do_cfg, num_images_per_prompt=1,
    )
    text_emb = torch.cat([ne, pe]) if do_cfg else pe
    img_t = transform_to_model_format(
        image, target_size=wm.config.image_size[0]
    ).unsqueeze(0).to(wm.config.device, dtype=text_emb.dtype)
    img_lat = get_media_latents(
        pipe=wm.config.pipe, media=img_t, sample=False, decoder_inv=False,
    )
    rev = wm.config.inversion.forward_diffusion(
        latents=img_lat, text_embeddings=text_emb,
        guidance_scale=wm.config.guidance_scale,
        num_inference_steps=wm.config.num_inference_steps,
    )[-1]
    return rev.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--seeds-per-prompt", type=int, default=50)
    parser.add_argument("--output", type=str,
                        default="eval_n500/sigma_table_mirage.json")
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

    total = args.num_prompts * args.seeds_per_prompt
    print(f"Sigma table: N={total} × {len(ATTACKS)} attacks", flush=True)
    print(f"|z| bins (centers): {BIN_CENTERS}", flush=True)

    # Storage: per attack, per bin, list of match rates
    bin_match = {a: {i: [] for i in range(len(BIN_CENTERS))} for a in ATTACKS}
    bin_total = {a: {i: [] for i in range(len(BIN_CENTERS))} for a in ATTACKS}

    # Per-sample done flags (for resume)
    done_keys = set()
    if os.path.exists(args.output):
        try:
            with open(args.output) as f:
                existing = json.load(f)
                done_keys = set(existing.get("done_keys", []))
                for k, v in existing.get("bin_match", {}).items():
                    for atk in ATTACKS:
                        bin_match[atk][int(k.split("_")[1])] = []
            print(f"  Resumed from {len(done_keys)} done", flush=True)
        except Exception as e:
            print(f"  Resume failed: {e}", flush=True)

    t_start = time.time()
    counter = 0

    for pi in range(args.num_prompts):
        prompt = PROMPTS[pi]
        for si in range(args.seeds_per_prompt):
            seed = pi * 100 + si
            idx = pi * args.seeds_per_prompt + si
            counter += 1
            key = f"{pi}_{seed}"
            if key in done_keys:
                continue

            img_path = os.path.join(WM_DIR, f"img_{idx:04d}.png")
            if not os.path.exists(img_path):
                continue
            wm_img = Image.open(img_path).convert("RGB")

            # Reconstruct z_T_orig (deterministic from seed)
            reseed(wm, seed)
            with torch.no_grad():
                z_T_orig = wm.utils.inject_watermark().detach().flatten().cpu()

            abs_z = z_T_orig.abs().numpy()
            sign_orig = torch.sign(z_T_orig).numpy().astype(np.int8)

            # Bin assignment: which bin each coord belongs to
            bin_idx = np.digitize(abs_z, BIN_EDGES[1:-1])  # 0..N-1

            for atk_name, atk_factory in ATTACKS.items():
                attack = atk_factory()
                attacked = attack.edit(wm_img.copy(), {})
                reseed(wm, seed)
                with torch.no_grad():
                    z_rec = invert_image(wm, attacked, prompt).flatten().cpu()
                sign_rec = torch.sign(z_rec).numpy().astype(np.int8)
                match = (sign_orig == sign_rec).astype(np.int8)

                # For each bin, count matches
                for b in range(len(BIN_CENTERS)):
                    mask = (bin_idx == b)
                    if mask.sum() > 0:
                        bin_match[atk_name][b].append(
                            int(match[mask].sum())
                        )
                        bin_total[atk_name][b].append(int(mask.sum()))

            done_keys.add(key)

            if counter % 10 == 0 or counter == total:
                elapsed = time.time() - t_start
                rate = counter / max(elapsed, 1)
                eta_s = (total - counter) / max(rate, 0.01)
                print(f"  [{counter}/{total}] elapsed={elapsed/60:.1f}min "
                      f"ETA={eta_s/60:.1f}min", flush=True)

                # Compute current p per bin per attack
                summary = {}
                for atk in ATTACKS:
                    summary[atk] = {}
                    for b, center in enumerate(BIN_CENTERS):
                        total_match = sum(bin_match[atk][b])
                        total_count = sum(bin_total[atk][b])
                        if total_count > 0:
                            p = total_match / total_count
                            # sigma_eff: avoid div by zero at p=0.5
                            if abs(p - 0.5) > 1e-6:
                                sigma_eff = 1.0 / math.tan(
                                    math.pi * (p - 0.5)
                                ) if p > 0.5 else float("inf")
                            else:
                                sigma_eff = float("inf")
                            summary[atk][f"bin_{b}"] = {
                                "center": center, "p": float(p),
                                "sigma_eff": float(sigma_eff)
                                    if sigma_eff != float("inf") else None,
                                "n_samples": total_count,
                            }
                output = {
                    "config": vars(args),
                    "bin_edges": BIN_EDGES,
                    "bin_centers": BIN_CENTERS,
                    "summary": summary,
                    "done_keys": sorted(done_keys),
                }
                with open(args.output, "w") as f:
                    json.dump(output, f, indent=2, default=str)

    print("\n=== Sigma Table (Table 9 style) ===", flush=True)
    print(f"  bin centers: {BIN_CENTERS}", flush=True)
    print(f"  {'attack':<10} | " + " | ".join(
        [f"{c:>6.2f}" for c in BIN_CENTERS]
    ), flush=True)
    print("-" * (10 + 9 * len(BIN_CENTERS)), flush=True)
    for atk in ATTACKS:
        ps = []
        for b in range(len(BIN_CENTERS)):
            total_match = sum(bin_match[atk][b])
            total_count = sum(bin_total[atk][b])
            if total_count > 0:
                p = total_match / total_count
                ps.append(f"{p:>6.3f}")
            else:
                ps.append("   ---")
        print(f"  {atk:<10} | " + " | ".join(ps), flush=True)


if __name__ == "__main__":
    main()

"""Stage 4 — GZS detector ablation at N=500.

Compares 3 detectors on identical attacked images:
  - mirage:  calibrated reliability w_j = r_j (offline calibrated)
  - gzs:     per-image magnitude w_j = mean(|z_rec_block_j|)
  - uniform: w_j = 1

Uses cached eval_n500/MIRAGE_wm/ + applies 8 standard attacks. For each
attacked image, inverts once and computes 3 weighted-BA scores.

Usage:
  python eval_n500_gzs.py --gpu 0
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
    AdaptiveNoiseInjection, Brightness, CrSc, GaussianBlurring,
    GaussianNoise, JPEGCompression, Mask, Overlay,
)
from runtime_utils import build_cached_pipe, configure_runtime, resolve_device
from utils.diffusion_config import DiffusionConfig
from utils.media_utils import (
    get_media_latents, get_random_latents, transform_to_model_format,
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

WM_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "eval_n500/MIRAGE_wm"
)

THRESHOLD = 0.67
ERASURE = 32


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


def detect_with_weights(z_rec, sign_pattern, selected_block_indices, weights,
                        block_dim, threshold=THRESHOLD, erasure=ERASURE):
    """Compute weighted BA detection. weights=None → uniform."""
    z_flat = z_rec.detach().to(dtype=torch.float32).flatten()
    flat_blocks = z_flat.reshape(-1, block_dim)
    selected_blocks = flat_blocks[selected_block_indices]
    block_correlations = (selected_blocks * sign_pattern).sum(dim=1)
    block_matches = (block_correlations > 0).float().cpu().numpy()
    num_blocks = block_matches.size

    if weights is None:
        weights_np = np.ones(num_blocks, dtype=np.float32)
    else:
        weights_np = weights.detach().cpu().numpy().astype(np.float32)

    keep_mask = np.ones(num_blocks, dtype=bool)
    if erasure > 0 and num_blocks > erasure:
        block_confidences = torch.abs(block_correlations).cpu().numpy()
        if weights is None:
            erased_idx = np.argsort(block_confidences)[:erasure]
        else:
            erased_idx = np.argsort(weights_np)[:erasure]
        keep_mask[erased_idx] = False

    kept_matches = block_matches[keep_mask]
    kept_weights = weights_np[keep_mask]
    if kept_weights.sum() > 0:
        ba = float((kept_weights * kept_matches).sum() / kept_weights.sum())
    else:
        ba = float(kept_matches.mean())
    return ba, bool(ba > threshold)


def gzs_weights_for(z_rec, selected_block_indices, block_dim):
    z_flat = z_rec.detach().to(dtype=torch.float32).flatten()
    flat_blocks = z_flat.reshape(-1, block_dim)
    sel = flat_blocks[selected_block_indices]
    return torch.abs(sel).mean(dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--seeds-per-prompt", type=int, default=50)
    parser.add_argument("--output", type=str,
                        default="eval_n500/gzs_comparison.json")
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

    selected_block_indices = wm.utils.selected_block_indices
    block_dim = wm.config.block_dim
    mirage_weights = wm.utils.selected_weights.detach().to(
        device=device, dtype=torch.float32
    )

    total = args.num_prompts * args.seeds_per_prompt
    print(f"GZS comparison N={total}", flush=True)

    attacks = make_attacks()
    detectors = ["uniform", "gzs", "mirage"]
    results = {a: {d: [] for d in detectors} for a in attacks}
    if os.path.exists(args.output):
        try:
            with open(args.output) as f:
                results = json.load(f).get("results", results)
            for a in attacks:
                if a not in results:
                    results[a] = {d: [] for d in detectors}
                else:
                    for d in detectors:
                        if d not in results[a]:
                            results[a][d] = []
            print(f"  Resumed; counts: "
                  f"{[len(results[a]['mirage']) for a in attacks]}",
                  flush=True)
        except Exception as e:
            print(f"  Resume failed: {e}", flush=True)
            results = {a: {d: [] for d in detectors} for a in attacks}

    t_start = time.time()
    counter = 0

    for pi in range(args.num_prompts):
        prompt = PROMPTS[pi]
        for si in range(args.seeds_per_prompt):
            seed = pi * 100 + si
            idx = pi * args.seeds_per_prompt + si
            counter += 1
            img_path = os.path.join(WM_DIR, f"img_{idx:04d}.png")
            if not os.path.exists(img_path):
                continue
            wm_img = Image.open(img_path).convert("RGB")

            for atk_name, atk in attacks.items():
                if any(r.get("idx") == idx
                       for r in results[atk_name]["mirage"]):
                    continue
                if atk is None:
                    attacked = wm_img
                else:
                    attacked = atk.edit(wm_img.copy(), {})

                reseed(wm, seed)
                # Regenerate sign_pattern (per-image nonce)
                if wm.config.use_per_image_nonce:
                    wm.utils.sign_pattern = wm.utils._generate_sign_pattern(
                        image_nonce=seed
                    )
                sign_pattern = wm.utils.sign_pattern.detach().to(
                    device=device, dtype=torch.float32
                )

                z_rec = invert_image(wm, attacked, prompt).to(
                    device=device
                )

                # 3 detectors
                ba_u, det_u = detect_with_weights(
                    z_rec, sign_pattern, selected_block_indices,
                    None, block_dim,
                )
                gzs_w = gzs_weights_for(
                    z_rec, selected_block_indices, block_dim
                )
                ba_g, det_g = detect_with_weights(
                    z_rec, sign_pattern, selected_block_indices,
                    gzs_w, block_dim,
                )
                ba_m, det_m = detect_with_weights(
                    z_rec, sign_pattern, selected_block_indices,
                    mirage_weights, block_dim,
                )

                rec = {"idx": idx, "seed": seed, "prompt_idx": pi}
                results[atk_name]["uniform"].append(
                    {**rec, "ba": ba_u, "is_watermarked": det_u}
                )
                results[atk_name]["gzs"].append(
                    {**rec, "ba": ba_g, "is_watermarked": det_g}
                )
                results[atk_name]["mirage"].append(
                    {**rec, "ba": ba_m, "is_watermarked": det_m}
                )

            if counter % 25 == 0 or counter == total:
                elapsed = time.time() - t_start
                rate = counter / max(elapsed, 1)
                eta_s = (total - counter) / max(rate, 0.01)
                print(f"  [{counter}/{total}] elapsed={elapsed/60:.1f}min "
                      f"ETA={eta_s/60:.1f}min", flush=True)

                summary = {}
                for a in attacks:
                    summary[a] = {}
                    for d in detectors:
                        recs = results[a][d]
                        if recs:
                            summary[a][d] = {
                                "n": len(recs),
                                "tpr": float(np.mean(
                                    [r["is_watermarked"] for r in recs]
                                )),
                                "ba_mean": float(np.mean(
                                    [r["ba"] for r in recs]
                                )),
                            }
                output = {
                    "num_prompts": args.num_prompts,
                    "seeds_per_prompt": args.seeds_per_prompt,
                    "results": results,
                    "summary": summary,
                }
                with open(args.output, "w") as f:
                    json.dump(output, f, indent=2, default=str)

    print("\n=== GZS COMPARISON SUMMARY ===", flush=True)
    print(f"  {'attack':<20} | {'uniform_TPR':>12} | {'gzs_TPR':>10} | {'mirage_TPR':>12}",
          flush=True)
    for a in attacks:
        u = float(np.mean(
            [r["is_watermarked"] for r in results[a]["uniform"]]
        )) if results[a]["uniform"] else 0
        g = float(np.mean(
            [r["is_watermarked"] for r in results[a]["gzs"]]
        )) if results[a]["gzs"] else 0
        m = float(np.mean(
            [r["is_watermarked"] for r in results[a]["mirage"]]
        )) if results[a]["mirage"] else 0
        print(f"  {a:<20} | {u*100:>11.1f}% | {g*100:>9.1f}% | {m*100:>11.1f}%",
              flush=True)


if __name__ == "__main__":
    main()

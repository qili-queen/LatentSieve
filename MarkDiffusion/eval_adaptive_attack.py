"""Scheme-aware adaptive attack on MIRAGE.

Threat model:
  - Adversary knows the algorithm and reliability profile shape (high-magnitude
    coordinates have high reliability), but does NOT know the secret key K.
  - Adversary can invert the watermarked image, modify the recovered latent,
    and regenerate.
  - Adversary perceptual budget: SSIM, LPIPS, or PSNR.

Attack strategy (Targeted Sign-Flip Regeneration):
  1. Invert watermarked image x' -> z_rec.
  2. Identify "high-reliability" coordinates: coords with largest |z_rec_j|.
  3. Flip signs on a fraction f of these coords (top-f-percent by |z_rec|).
  4. Regenerate: x'' = G(z_rec_modified).
  5. Optionally: if SSIM(x', x'') below budget, BLEND x' and x'' to stay within
     budget (linear interpolation in pixel space).
  6. Measure attacked detection rate.

Compares to non-adaptive baseline (random sign flip on same fraction).

Usage:
  python qi_eval_adaptive_attack.py --gpu 0 --num-images 50 \
       --flip-fractions 0.1,0.2,0.3,0.5
"""

import argparse
import functools
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

oti = tqdm.__init__
@functools.wraps(oti)
def st(s, *a, **k):
    k['disable'] = True
    oti(s, *a, **k)
tqdm.__init__ = st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qi_runtime_utils import build_cached_pipe, configure_qi_runtime, resolve_qi_device
from utils.diffusion_config import DiffusionConfig
from utils.media_utils import (
    get_media_latents,
    get_random_latents,
    transform_to_model_format,
)
from watermark.auto_watermark import AutoWatermark

configure_qi_runtime()

MODEL = "huanzi05/stable-diffusion-2-1-base"
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


def reseed(wm, seed):
    wm.config.gen_seed = seed
    wm.config.init_latents_seed = seed
    r = torch.Generator(device=wm.config.device).manual_seed(seed)
    wm.config.init_latents = get_random_latents(
        wm.config.pipe, height=SIZE, width=SIZE, generator=r
    ).to(wm.config.device, dtype=wm.config.dtype)


def invert_image(wm, image, prompt):
    """DDIM-invert an image and return reversed latent."""
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


def regenerate_from_latent(wm, latent, prompt):
    """Regenerate image from a (possibly modified) latent."""
    do_cfg = wm.config.guidance_scale > 1.0
    output = wm.config.pipe(
        prompt,
        latents=latent.to(dtype=wm.config.dtype),
        guidance_scale=GS_SCALE,
        num_inference_steps=STEPS,
        height=SIZE, width=SIZE,
    )
    return output.images[0]


def adaptive_sign_flip_attack(z_rec, flip_fraction):
    """Adaptive: flip signs of top-flip_fraction-percent by |z_rec|.

    These are the coords the attacker GUESSES to be high-reliability.
    """
    z_flat = z_rec.flatten()
    abs_vals = torch.abs(z_flat)
    n_flip = int(flip_fraction * z_flat.numel())
    if n_flip == 0:
        return z_rec.clone()
    # Top-n by magnitude
    _, top_idx = torch.topk(abs_vals, n_flip)
    z_modified = z_flat.clone()
    z_modified[top_idx] = -z_modified[top_idx]
    return z_modified.reshape_as(z_rec)


def random_sign_flip_attack(z_rec, flip_fraction, seed=0):
    """Non-adaptive baseline: flip signs of random fraction of coords."""
    z_flat = z_rec.flatten()
    n_flip = int(flip_fraction * z_flat.numel())
    if n_flip == 0:
        return z_rec.clone()
    g = torch.Generator(device=z_rec.device).manual_seed(seed)
    perm = torch.randperm(z_flat.numel(), generator=g, device=z_rec.device)
    flip_idx = perm[:n_flip]
    z_modified = z_flat.clone()
    z_modified[flip_idx] = -z_modified[flip_idx]
    return z_modified.reshape_as(z_rec)


def lowmag_sign_flip_attack(z_rec, flip_fraction):
    """Anti-adaptive baseline: flip signs of LOW-magnitude coords (what
    non-targeted noise attacks effectively do)."""
    z_flat = z_rec.flatten()
    abs_vals = torch.abs(z_flat)
    n_flip = int(flip_fraction * z_flat.numel())
    if n_flip == 0:
        return z_rec.clone()
    _, bottom_idx = torch.topk(-abs_vals, n_flip)  # smallest magnitudes
    z_modified = z_flat.clone()
    z_modified[bottom_idx] = -z_modified[bottom_idx]
    return z_modified.reshape_as(z_rec)


def compute_psnr(img1, img2):
    a1 = np.array(img1).astype(float)
    a2 = np.array(img2).astype(float)
    mse = np.mean((a1 - a2) ** 2)
    return 10 * np.log10(255**2 / mse) if mse > 0 else 100.0


def compute_ssim(img1, img2):
    """Quick SSIM via skimage if available, else 1 - normalized L2."""
    try:
        from skimage.metrics import structural_similarity as ssim
        a1 = np.array(img1).astype(np.float32)
        a2 = np.array(img2).astype(np.float32)
        # Multichannel SSIM
        return float(ssim(a1, a2, channel_axis=-1, data_range=255.0))
    except Exception:
        a1 = np.array(img1).astype(float) / 255.0
        a2 = np.array(img2).astype(float) / 255.0
        return float(1.0 - np.linalg.norm(a1 - a2) / np.sqrt(a1.size))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=50,
                        help="total images (split as 10 prompts x N seeds)")
    parser.add_argument("--seeds-per-prompt", type=int, default=5)
    parser.add_argument("--flip-fractions", type=str,
                        default="0.05,0.10,0.15,0.20,0.30,0.50",
                        help="comma-separated fractions of coords to flip")
    parser.add_argument("--output", type=str,
                        default="sp2027_proposal/adaptive_attack.json")
    args = parser.parse_args()

    flip_fractions = [float(x) for x in args.flip_fractions.split(",")]

    device, dtype = resolve_qi_device(args.gpu)
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

    num_prompts = min(len(PROMPTS), args.num_images // args.seeds_per_prompt)
    total = num_prompts * args.seeds_per_prompt
    print(f"Running on {total} images ({num_prompts} prompts x "
          f"{args.seeds_per_prompt} seeds)", flush=True)
    print(f"Flip fractions: {flip_fractions}", flush=True)

    # Storage:
    #   results[strategy][flip_fraction] = list of {ba, detected, psnr, ssim, ...}
    strategies = ["adaptive_high", "random", "lowmag"]
    results = {s: {f"{ff:.2f}": [] for ff in flip_fractions}
               for s in strategies}

    counter = 0
    for pi in range(num_prompts):
        prompt = PROMPTS[pi]
        for si in range(args.seeds_per_prompt):
            seed = pi * 100 + si
            counter += 1
            print(f"\n[{counter}/{total}] prompt={pi} seed={seed}", flush=True)

            # Generate watermarked image
            reseed(wm, seed)
            wm_img = wm.generate_watermarked_media(prompt)

            # Detect on clean watermarked
            reseed(wm, seed)
            r_clean = wm.detect_watermark_in_media(wm_img, prompt=prompt)
            ba_clean = float(r_clean.get("weighted_full_match_rate", 0.5))
            det_clean = bool(r_clean.get("is_watermarked", False))
            print(f"  clean: BA={ba_clean:.3f} det={det_clean}", flush=True)

            # Invert to get attacker's view
            z_rec = invert_image(wm, wm_img, prompt)

            # Run attacks at each flip fraction with each strategy
            for ff in flip_fractions:
                for strategy in strategies:
                    if strategy == "adaptive_high":
                        z_attacked = adaptive_sign_flip_attack(z_rec, ff)
                    elif strategy == "random":
                        z_attacked = random_sign_flip_attack(z_rec, ff, seed=seed)
                    elif strategy == "lowmag":
                        z_attacked = lowmag_sign_flip_attack(z_rec, ff)

                    # Regenerate from attacked latent
                    attacked_img = regenerate_from_latent(wm, z_attacked, prompt)

                    # Compute perceptual metrics vs original watermarked
                    psnr = compute_psnr(wm_img, attacked_img)
                    ssim = compute_ssim(wm_img, attacked_img)

                    # Detect
                    reseed(wm, seed)
                    r_atk = wm.detect_watermark_in_media(
                        attacked_img, prompt=prompt
                    )
                    ba_atk = float(r_atk.get("weighted_full_match_rate", 0.5))
                    det_atk = bool(r_atk.get("is_watermarked", False))

                    results[strategy][f"{ff:.2f}"].append({
                        "prompt_idx": pi, "seed": seed,
                        "ba_atk": ba_atk, "det_atk": det_atk,
                        "psnr": psnr, "ssim": ssim,
                    })

                    print(f"  {strategy} ff={ff:.2f}: BA={ba_atk:.3f} "
                          f"det={det_atk} PSNR={psnr:.1f} SSIM={ssim:.3f}",
                          flush=True)

    # Aggregate: per strategy, per flip-fraction, mean TPR/PSNR/SSIM
    summary = {}
    for s in strategies:
        summary[s] = {}
        for ff in flip_fractions:
            key = f"{ff:.2f}"
            recs = results[s][key]
            if not recs:
                continue
            summary[s][key] = {
                "n": len(recs),
                "tpr": float(np.mean([r["det_atk"] for r in recs])),
                "ba_mean": float(np.mean([r["ba_atk"] for r in recs])),
                "psnr_mean": float(np.mean([r["psnr"] for r in recs])),
                "ssim_mean": float(np.mean([r["ssim"] for r in recs])),
            }

    output = {
        "config": {
            "num_images": total, "num_prompts": num_prompts,
            "seeds_per_prompt": args.seeds_per_prompt,
            "flip_fractions": flip_fractions,
        },
        "results_per_image": results,
        "summary": summary,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Pretty print summary
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for s in strategies:
        print(f"\n{s}:", flush=True)
        print(f"  {'flip_frac':>10} {'TPR':>6} {'BA':>6} {'PSNR':>6} {'SSIM':>6}",
              flush=True)
        for ff in flip_fractions:
            key = f"{ff:.2f}"
            if key in summary[s]:
                d = summary[s][key]
                print(f"  {key:>10} {d['tpr']:.0%} {d['ba_mean']:>5.3f} "
                      f"{d['psnr_mean']:>5.1f} {d['ssim_mean']:>5.3f}",
                      flush=True)

    print(f"\nSaved to {args.output}", flush=True)


if __name__ == "__main__":
    main()

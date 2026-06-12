"""Erasure-aware white-box PGD adaptive attack on MIRAGE with LPIPS measurement.

Threat model (strongest):
  - Attacker knows MIRAGE algorithm AND secret key K (worst-case key compromise)
  - Attacker knows the reliability profile and the erasure rule (top-32 of 256)
  - Attacker runs latent-space PGD with erasure-aware soft-BA loss
  - Constraint: L2 latent budget (swept). Post-hoc LPIPS measurement.

The loss is erasure-aware: a sigmoid-thresholded keep mask over the per-block
confidence steers the attacker toward the high-confidence blocks that survive
MIRAGE's top-32-of-256 erasure, rather than the blocks the detector discards.

Usage:
  python eval_n200_adaptive_whitebox_lpips.py --gpu 0 \
      --num-images 200 --latent-budgets 5.0,20.0,50.0,100.0,200.0
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

import lpips as lpips_module

from runtime_utils import build_cached_pipe, configure_runtime, resolve_device
from utils.diffusion_config import DiffusionConfig
from utils.media_utils import (
    get_media_latents,
    get_random_latents,
    transform_to_model_format,
)
from watermark.auto_watermark import AutoWatermark

configure_runtime()

MODEL = "stabilityai/stable-diffusion-2-1-base"
STEPS = 50
GS_SCALE = 7.5
SIZE = 512

from sdp_n500_config import PROMPTS


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


def regenerate_from_latent(wm, latent, prompt):
    output = wm.config.pipe(
        prompt,
        latents=latent.to(dtype=wm.config.dtype),
        guidance_scale=GS_SCALE,
        num_inference_steps=STEPS,
        height=SIZE, width=SIZE,
    )
    return output.images[0]


def erasure_aware_loss(z_perturbed, sign_pattern, selected_block_indices,
                        weights, block_dim, n_erase=32, temp=0.05):
    """Erasure-aware soft-BA: attacker knows top-32 of 256 will be erased.

    Confidence per block = |normalized correlation|.
    Soft keep mask = sigmoid((conf - threshold) / temp), threshold = 12.5%-quantile.
    Loss = weighted average over keep_mask * weights.

    Attacker minimizing this is incentivized to target HIGH-confidence blocks
    (those will survive erasure and contribute to detection).
    """
    z_flat = z_perturbed.flatten()
    flat_blocks = z_flat.reshape(-1, block_dim)
    selected = flat_blocks[selected_block_indices]  # (B, d)

    block_corr_raw = (selected * sign_pattern).sum(dim=1)  # (B,)
    block_norm = torch.linalg.norm(selected, dim=1).clamp(min=1e-6)
    normalized_corr = block_corr_raw / (block_norm * (block_dim ** 0.5))  # (B,) in [-1, +1]

    confidence = normalized_corr.abs()  # (B,)

    # Detached threshold: 12.5% quantile of confidence
    with torch.no_grad():
        n_blocks = confidence.numel()
        threshold = torch.quantile(confidence, n_erase / n_blocks)

    keep_mask = torch.sigmoid((confidence - threshold) / temp)  # ~1 for top-224, ~0 for bottom-32

    weight_eff = keep_mask * weights
    weight_sum = weight_eff.sum().clamp(min=1e-8)
    score = (weight_eff * normalized_corr).sum() / weight_sum
    return score


def pgd_attack_erasure_aware(z_rec_init, sign_pattern, selected_block_indices,
                              weights, block_dim, latent_budget,
                              n_steps=100, lr=0.05, verbose=False):
    """Erasure-aware latent-space PGD."""
    delta = torch.zeros_like(z_rec_init, requires_grad=True)
    losses = []

    for step in range(n_steps):
        z_perturbed = z_rec_init + delta
        loss = erasure_aware_loss(z_perturbed, sign_pattern,
                                    selected_block_indices, weights, block_dim)
        if torch.isnan(loss):
            break
        losses.append(float(loss.item()))

        grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]
        delta = (delta.detach() - lr * grad.sign()).detach()

        # L2 projection
        delta_norm = delta.norm()
        if delta_norm > latent_budget:
            delta = delta * (latent_budget / delta_norm)
        delta.requires_grad_()

        if verbose and step % 20 == 0:
            print(f"    PGD step {step}: loss={loss.item():.4f} "
                  f"||delta||={delta.norm().item():.3f}", flush=True)

    return (z_rec_init + delta).detach(), losses


def compute_psnr(img1, img2):
    a1 = np.array(img1).astype(float)
    a2 = np.array(img2).astype(float)
    mse = np.mean((a1 - a2) ** 2)
    return 10 * np.log10(255**2 / mse) if mse > 0 else 100.0


def compute_ssim(img1, img2):
    try:
        from skimage.metrics import structural_similarity as ssim
        a1 = np.array(img1).astype(np.float32)
        a2 = np.array(img2).astype(np.float32)
        return float(ssim(a1, a2, channel_axis=-1, data_range=255.0))
    except Exception:
        return 0.0


def img_to_lpips_tensor(img, device):
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def compute_lpips(lpips_net, img1, img2, device):
    t1 = img_to_lpips_tensor(img1, device)
    t2 = img_to_lpips_tensor(img2, device)
    with torch.no_grad():
        return float(lpips_net(t1, t2).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--seeds-per-prompt", type=int, default=1)
    parser.add_argument("--latent-budgets", type=str,
                        default="5.0,20.0,50.0,100.0,200.0")
    parser.add_argument("--pgd-steps", type=int, default=100)
    parser.add_argument("--pgd-lr", type=float, default=0.05)
    parser.add_argument("--start-image", type=int, default=0,
                        help="Skip first N images (for parallel sharding)")
    parser.add_argument("--output", type=str,
                        default="results/adaptive_whitebox_lpips.json")
    args = parser.parse_args()

    latent_budgets = [float(x) for x in args.latent_budgets.split(",")]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

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
    lpips_net = lpips_module.LPIPS(net='alex').to(device).eval()

    selected_block_indices = wm.utils.selected_block_indices
    block_dim = wm.config.block_dim

    num_prompts = min(len(PROMPTS), args.num_images // args.seeds_per_prompt)
    total = num_prompts * args.seeds_per_prompt
    print(f"PGD ERASURE-AWARE attack on MIRAGE: {total} images "
          f"(starting at offset {args.start_image})", flush=True)
    print(f"Latent budgets: {latent_budgets}", flush=True)
    print(f"PGD steps: {args.pgd_steps}, lr: {args.pgd_lr}", flush=True)

    results = {f"{lb:.1f}": [] for lb in latent_budgets}

    counter = 0
    for pi in range(num_prompts):
        prompt = PROMPTS[pi]
        for si in range(args.seeds_per_prompt):
            seed = pi * 100 + si
            counter += 1
            if counter <= args.start_image:
                continue
            print(f"\n[{counter}/{total}] prompt={pi} seed={seed}", flush=True)

            # Generate watermarked image
            reseed(wm, seed)
            wm_img = wm.generate_watermarked_media(prompt)

            # Refresh sign pattern with per-image nonce
            if wm.config.use_per_image_nonce:
                wm.utils.sign_pattern = wm.utils._generate_sign_pattern(
                    image_nonce=wm.config.init_latents_seed
                )
            sign_pattern = wm.utils.sign_pattern.detach().to(
                device=device, dtype=torch.float32
            )
            weights = wm.utils.selected_weights.detach().to(
                device=device, dtype=torch.float32
            )

            # Detect on clean watermarked
            reseed(wm, seed)
            r_clean = wm.detect_watermark_in_media(wm_img, prompt=prompt)
            ba_clean = float(r_clean.get("weighted_full_match_rate", 0.5))
            det_clean = bool(r_clean.get("is_watermarked", False))
            print(f"  clean: BA={ba_clean:.3f} det={det_clean}", flush=True)

            # Invert once for PGD initialization
            z_rec_init = invert_image(wm, wm_img, prompt).to(dtype=torch.float32)

            for lb in latent_budgets:
                key = f"{lb:.1f}"
                z_attacked, losses = pgd_attack_erasure_aware(
                    z_rec_init.clone(), sign_pattern,
                    selected_block_indices, weights, block_dim,
                    latent_budget=lb,
                    n_steps=args.pgd_steps, lr=args.pgd_lr,
                )

                attacked_img = regenerate_from_latent(wm, z_attacked, prompt)

                psnr = compute_psnr(wm_img, attacked_img)
                ssim = compute_ssim(wm_img, attacked_img)
                lpips_val = compute_lpips(lpips_net, wm_img, attacked_img, device)

                reseed(wm, seed)
                r = wm.detect_watermark_in_media(attacked_img, prompt=prompt)
                ba_post = float(r.get("weighted_full_match_rate", 0.5))
                det_post = bool(r.get("is_watermarked", False))

                results[key].append({
                    "prompt_idx": pi, "seed": seed,
                    "latent_budget": lb,
                    "ba_clean": ba_clean,
                    "ba_attacked": ba_post,
                    "det_attacked": det_post,
                    "psnr": psnr, "ssim": ssim, "lpips": lpips_val,
                    "loss_init": losses[0] if losses else None,
                    "loss_final": losses[-1] if losses else None,
                })
                print(f"  LB={lb:.1f}: BA={ba_post:.3f} det={det_post} "
                      f"PSNR={psnr:.1f} LPIPS={lpips_val:.3f}", flush=True)

            # Incremental save every 10 images
            if counter % 10 == 0:
                with open(args.output, "w") as f:
                    json.dump({
                        "config": vars(args),
                        "results_per_image": results,
                        "progress": counter,
                    }, f, indent=2, default=str)

    # Final summary
    summary = {}
    for lb in latent_budgets:
        key = f"{lb:.1f}"
        recs = results[key]
        if not recs:
            continue
        summary[key] = {
            "n": len(recs),
            "tpr": float(np.mean([r["det_attacked"] for r in recs])),
            "ba_mean": float(np.mean([r["ba_attacked"] for r in recs])),
            "ba_std": float(np.std([r["ba_attacked"] for r in recs])),
            "psnr_mean": float(np.mean([r["psnr"] for r in recs])),
            "ssim_mean": float(np.mean([r["ssim"] for r in recs])),
            "lpips_mean": float(np.mean([r["lpips"] for r in recs])),
            "lpips_std": float(np.std([r["lpips"] for r in recs])),
            "loss_init_mean": float(np.mean([r["loss_init"] for r in recs
                                              if r["loss_init"] is not None])),
            "loss_final_mean": float(np.mean([r["loss_final"] for r in recs
                                                if r["loss_final"] is not None])),
        }

    with open(args.output, "w") as f:
        json.dump({
            "config": vars(args),
            "results_per_image": results,
            "summary": summary,
        }, f, indent=2, default=str)

    print("\n" + "=" * 70, flush=True)
    print("ERASURE-AWARE WHITE-BOX ATTACK SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"  {'budget':>7} {'TPR':>6} {'BA':>6} {'PSNR':>6} "
          f"{'SSIM':>6} {'LPIPS':>7}", flush=True)
    for lb in latent_budgets:
        key = f"{lb:.1f}"
        if key in summary:
            d = summary[key]
            print(f"  {key:>7} {d['tpr']:>5.1%} {d['ba_mean']:>5.3f} "
                  f"{d['psnr_mean']:>5.1f} {d['ssim_mean']:>5.3f} "
                  f"{d['lpips_mean']:>6.3f}", flush=True)
    print(f"\nSaved to {args.output}", flush=True)


if __name__ == "__main__":
    main()

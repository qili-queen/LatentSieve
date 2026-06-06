"""GZS detector ablation under SHIFT on SDP slice, N=200.

Applies SHIFT trajectory deflection (lambda=0.6, 0.7) to MIRAGE-watermarked
images, then computes 4 detector variants on each recovered latent:
  (a) uniform_no_erase: w_j=1, erasure=0
  (b) uniform_erase:    w_j=1, erasure=32
  (c) gzs_erase:        w_j=mean(|z|) per block, erasure=32
  (d) mirage_erase:     w_j=calibrated reliability, erasure=32

Matches the structure of tab:gzs-comparison in the paper but on SDP N=200.

Usage:
  CUDA_VISIBLE_DEVICES=2 python eval_n500_gzs_shift_sdp.py --gpu 0
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
from diffusers import DDIMScheduler
from tqdm import tqdm

oti = tqdm.__init__
@functools.wraps(oti)
def st(s, *a, **k): k['disable'] = True; oti(s, *a, **k)
tqdm.__init__ = st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
THRESHOLD = 0.67
ERASURE = 32
LAMBDAS = [0.6, 0.7]

from sdp_n500_config import PROMPTS

WM_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "eval_n500_sdp", "MIRAGE_wm",
)


def reseed(wm, seed):
    wm.config.gen_seed = seed
    wm.config.init_latents_seed = seed
    r = torch.Generator(device=wm.config.device).manual_seed(seed)
    wm.config.init_latents = get_random_latents(
        wm.config.pipe, height=SIZE, width=SIZE, generator=r
    ).to(wm.config.device, dtype=wm.config.dtype)


def vae_encode(pipe, image, device, dtype):
    t = transform_to_model_format(image, target_size=SIZE).unsqueeze(0).to(
        device, dtype=dtype
    )
    return get_media_latents(pipe=pipe, media=t, sample=False,
                              decoder_inv=False)


def vae_decode(pipe, latents):
    image = pipe.vae.decode(
        latents / pipe.vae.config.scaling_factor, return_dict=False
    )[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    image_np = (image[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype(
        np.uint8
    )
    return Image.fromarray(image_np)


@torch.no_grad()
def shift_attack(pipe, image, text_emb, lam, device, dtype):
    z0 = vae_encode(pipe, image, device, dtype)
    sched = DDIMScheduler.from_pretrained(
        MODEL, subfolder="scheduler", local_files_only=True
    )
    t_l = int(lam * 1000)
    at = sched.alphas_cumprod[min(t_l, 999)]
    zt = torch.sqrt(at) * z0 + torch.sqrt(1 - at) * torch.randn_like(z0)

    pipe.scheduler = DDIMScheduler.from_pretrained(
        MODEL, subfolder="scheduler", local_files_only=True
    )
    pipe.scheduler.set_timesteps(STEPS)
    ts = pipe.scheduler.timesteps
    si = next(i for i, t in enumerate(ts) if t.item() <= t_l)
    la = zt
    for t in ts[si:]:
        li = torch.cat([la] * 2)
        li = pipe.scheduler.scale_model_input(li, t)
        np_ = pipe.unet(li, t, encoder_hidden_states=text_emb).sample
        nu, nt = np_.chunk(2)
        ng = nu + GS_SCALE * (nt - nu)
        apt = pipe.scheduler.alphas_cumprod[t]
        pt = t - 1000 // STEPS
        if pt >= 0:
            ap = pipe.scheduler.alphas_cumprod[max(pt, 0)]
        else:
            ap = torch.tensor(1.0, device=device)
        x0p = (la - torch.sqrt(1 - apt) * ng) / torch.sqrt(apt)
        sig = 0.5 * torch.sqrt(
            (1 - ap) / (1 - apt) * (1 - apt / ap)
        ).clamp(min=0)
        pd = torch.sqrt(torch.clamp(1 - ap - sig**2, min=0)) * ng
        la = torch.sqrt(ap) * x0p + pd + sig * torch.randn_like(la)
    return vae_decode(pipe, la)


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
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--output", type=str,
                        default="eval_n500_sdp/gzs_shift_comparison.json")
    parser.add_argument("--lambdas", type=str, default="0.6,0.7")
    args = parser.parse_args()

    lambdas = [float(x) for x in args.lambdas.split(",")]

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

    print(f"GZS×SHIFT ablation N={args.max_images} per lambda", flush=True)

    detectors = ["uniform_no_erase", "uniform_erase", "gzs_erase",
                 "mirage_erase"]
    results = {str(l): {d: [] for d in detectors} for l in lambdas}
    if os.path.exists(args.output):
        try:
            with open(args.output) as f:
                results = json.load(f).get("results", results)
            for l in lambdas:
                results.setdefault(str(l), {d: [] for d in detectors})
                for d in detectors:
                    results[str(l)].setdefault(d, [])
            print(f"  Resumed; counts: "
                  f"{[(l, len(results[str(l)]['mirage_erase'])) for l in lambdas]}",
                  flush=True)
        except Exception as e:
            print(f"  Resume failed: {e}", flush=True)
            results = {str(l): {d: [] for d in detectors} for l in lambdas}

    SEEDS_PER_PROMPT = 5

    t_start = time.time()
    for idx in range(args.max_images):
        pi = idx // SEEDS_PER_PROMPT
        si = idx % SEEDS_PER_PROMPT
        seed = pi * 100 + si
        prompt = PROMPTS[pi]
        img_path = os.path.join(WM_DIR, f"img_{idx:04d}.png")
        if not os.path.exists(img_path):
            continue
        wm_img = Image.open(img_path).convert("RGB")

        # Regenerate sign_pattern (per-image nonce) once per image
        reseed(wm, seed)
        if wm.config.use_per_image_nonce:
            wm.utils.sign_pattern = wm.utils._generate_sign_pattern(
                image_nonce=seed
            )
        sign_pattern = wm.utils.sign_pattern.detach().to(
            device=device, dtype=torch.float32
        )

        # Encode text once per image
        do_cfg = GS_SCALE > 1.0
        pe, ne = pipe.encode_prompt(
            prompt=prompt, device=device,
            do_classifier_free_guidance=do_cfg, num_images_per_prompt=1,
        )
        text_emb = torch.cat([ne, pe]) if do_cfg else pe

        for lam in lambdas:
            lam_key = str(lam)
            if any(r.get("idx") == idx for r in results[lam_key]["mirage_erase"]):
                continue
            attacked = shift_attack(pipe, wm_img, text_emb, lam, device, dtype)
            z_rec = invert_image(wm, attacked, prompt).to(device=device)

            # 4 detectors
            ba_u_ne, det_u_ne = detect_with_weights(
                z_rec, sign_pattern, selected_block_indices,
                None, block_dim, erasure=0,
            )
            ba_u_e, det_u_e = detect_with_weights(
                z_rec, sign_pattern, selected_block_indices,
                None, block_dim, erasure=ERASURE,
            )
            gzs_w = gzs_weights_for(
                z_rec, selected_block_indices, block_dim
            )
            ba_g_e, det_g_e = detect_with_weights(
                z_rec, sign_pattern, selected_block_indices,
                gzs_w, block_dim, erasure=ERASURE,
            )
            ba_m_e, det_m_e = detect_with_weights(
                z_rec, sign_pattern, selected_block_indices,
                mirage_weights, block_dim, erasure=ERASURE,
            )

            rec = {"idx": idx, "seed": seed, "prompt_idx": pi}
            results[lam_key]["uniform_no_erase"].append(
                {**rec, "ba": ba_u_ne, "is_watermarked": det_u_ne}
            )
            results[lam_key]["uniform_erase"].append(
                {**rec, "ba": ba_u_e, "is_watermarked": det_u_e}
            )
            results[lam_key]["gzs_erase"].append(
                {**rec, "ba": ba_g_e, "is_watermarked": det_g_e}
            )
            results[lam_key]["mirage_erase"].append(
                {**rec, "ba": ba_m_e, "is_watermarked": det_m_e}
            )

        if (idx + 1) % 25 == 0 or idx + 1 == args.max_images:
            elapsed = time.time() - t_start
            rate = (idx + 1) / max(elapsed, 1)
            eta_s = (args.max_images - idx - 1) / max(rate, 0.01)
            print(f"  [{idx+1}/{args.max_images}] elapsed={elapsed/60:.1f}min "
                  f"ETA={eta_s/60:.1f}min", flush=True)

            summary = {}
            for l in lambdas:
                lk = str(l)
                summary[lk] = {}
                for d in detectors:
                    recs = results[lk][d]
                    if recs:
                        bas = np.array([r["ba"] for r in recs])
                        summary[lk][d] = {
                            "n": len(recs),
                            "tpr": float(np.mean(
                                [r["is_watermarked"] for r in recs]
                            )),
                            "ba_mean": float(bas.mean()),
                            "ba_std": float(bas.std()),
                            "ba_min": float(bas.min()),
                        }
            output = {"results": results, "summary": summary}
            with open(args.output, "w") as f:
                json.dump(output, f, indent=2, default=str)

    print("\n=== GZS×SHIFT SDP SUMMARY ===", flush=True)
    for l in lambdas:
        lk = str(l)
        print(f"\nSHIFT lambda={l}:", flush=True)
        for d in detectors:
            recs = results[lk][d]
            if recs:
                bas = np.array([r["ba"] for r in recs])
                tpr = float(np.mean(
                    [r["is_watermarked"] for r in recs]
                ))
                print(f"  {d:<20}  BA mean={bas.mean():.4f} std={bas.std():.4f} "
                      f"min={bas.min():.4f}  TPR@0.67={tpr*100:.1f}%",
                      flush=True)


if __name__ == "__main__":
    main()

"""Per-coordinate sign agreement under different attack families.

For each (attack, strength), measure the fraction of watermark-carrying
coordinates whose sign survives the round trip: attack -> DDIM inversion.
This p is the quantity in Observation 3 (closed-form robustness bound).

Run in markdiff env.
Usage: python eval_sign_agreement_attacks.py --gpu 0
"""
import argparse
import functools
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

oti = tqdm.__init__
@functools.wraps(oti)
def st(s, *a, **k): k['disable']=True; oti(s, *a, **k)
tqdm.__init__ = st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runtime_utils import build_cached_pipe, configure_runtime, resolve_device
from utils.diffusion_config import DiffusionConfig
from utils.media_utils import get_media_latents, get_random_latents, transform_to_model_format
from watermark.auto_watermark import AutoWatermark
from evaluation.tools.image_editor import GaussianBlurring, GaussianNoise, JPEGCompression

configure_runtime()

MODEL = "stabilityai/stable-diffusion-2-1-base"
STEPS = 50; GS_SCALE = 7.5; SIZE = 512

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

ATTACKS = {
    "Noise-5": GaussianNoise(sigma=5.0),
    "Noise-15": GaussianNoise(sigma=15.0),
    "Noise-30": GaussianNoise(sigma=30.0),
    "Noise-50": GaussianNoise(sigma=50.0),
    "JPEG-90": JPEGCompression(quality=90),
    "JPEG-75": JPEGCompression(quality=75),
    "JPEG-50": JPEGCompression(quality=50),
    "JPEG-30": JPEGCompression(quality=30),
    "JPEG-10": JPEGCompression(quality=10),
    "Blur-2": GaussianBlurring(radius=2),
    "Blur-4": GaussianBlurring(radius=4),
    "Blur-6": GaussianBlurring(radius=6),
}


def reseed_wm(wm, seed):
    wm.config.gen_seed = seed
    wm.config.init_latents_seed = seed
    r = torch.Generator(device=wm.config.device).manual_seed(seed)
    wm.config.init_latents = get_random_latents(
        wm.config.pipe, height=SIZE, width=SIZE, generator=r
    ).to(wm.config.device, dtype=wm.config.dtype)


def invert_image(wm, image, prompt):
    """DDIM-invert an image using MIRAGE's pipeline. Returns reversed_latents."""
    do_cfg = wm.config.guidance_scale > 1.0
    prompt_embeds, neg_embeds = wm.config.pipe.encode_prompt(
        prompt=prompt,
        device=wm.config.device,
        do_classifier_free_guidance=do_cfg,
        num_images_per_prompt=1,
    )
    text_emb = torch.cat([neg_embeds, prompt_embeds]) if do_cfg else prompt_embeds

    img_tensor = transform_to_model_format(
        image, target_size=wm.config.image_size[0]
    ).unsqueeze(0).to(wm.config.device, dtype=text_emb.dtype)
    img_latents = get_media_latents(
        pipe=wm.config.pipe, media=img_tensor, sample=False, decoder_inv=False,
    )
    reversed_latents = wm.config.inversion.forward_diffusion(
        latents=img_latents,
        text_embeddings=text_emb,
        guidance_scale=wm.config.guidance_scale,
        num_inference_steps=wm.config.num_inference_steps,
    )[-1]
    return reversed_latents.detach()


def compute_sign_agreement(wm, reversed_latents):
    """Compute per-coord sign agreement on selected watermark-carrying coords."""
    sign_pattern = wm.utils.sign_pattern  # shape (num_blocks, block_dim)
    selected_block_indices = wm.utils.selected_block_indices  # shape (num_blocks,)

    flat = reversed_latents.detach().to(dtype=torch.float32).flatten()
    flat_blocks = flat.reshape(-1, wm.config.block_dim)
    rec_blocks = flat_blocks[selected_block_indices]  # (num_blocks, block_dim)

    rec_signs = torch.sign(rec_blocks)
    # Treat exact zeros as matching (rare; avoids biasing p down)
    rec_signs[rec_signs == 0] = sign_pattern[rec_signs == 0]

    agree = (rec_signs == sign_pattern.to(rec_signs.device)).float()
    return float(agree.mean().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--output", default="sign_agreement_attacks.json")
    args = parser.parse_args()

    device, dtype = resolve_device(args.gpu)
    pipe, scheduler = build_cached_pipe(MODEL, device, dtype)
    dc = DiffusionConfig(
        scheduler=scheduler, pipe=pipe, device=device,
        image_size=(SIZE, SIZE), num_inference_steps=STEPS,
        num_inversion_steps=STEPS, guidance_scale=GS_SCALE,
        gen_seed=42, inversion_type="ddim", dtype=dtype,
    )
    wm = AutoWatermark.load("MIRAGE", algorithm_config="config/MIRAGE_presence.json", diffusion_config=dc)

    num_prompts = len(PROMPTS)
    num_samples = num_prompts * args.num_seeds
    total = num_samples * len(ATTACKS)
    print(f"Samples per attack: {num_samples}; attacks: {len(ATTACKS)}; total detections: {total}", flush=True)

    results = {a: [] for a in ATTACKS}

    counter = 0
    for pi, prompt in enumerate(PROMPTS):
        for si in range(args.num_seeds):
            seed = pi * 100 + si
            reseed_wm(wm, seed)
            wm_img = wm.generate_watermarked_media(prompt)
            # After generation, wm.utils.sign_pattern is up to date (per-image nonce)
            for aname, attack in ATTACKS.items():
                try:
                    attacked = attack.edit(wm_img)
                    z_rec = invert_image(wm, attacked, prompt)
                    # Need to regenerate sign pattern for this image to match detector
                    if wm.config.use_per_image_nonce:
                        wm.utils.sign_pattern = wm.utils._generate_sign_pattern(
                            image_nonce=wm.config.init_latents_seed
                        )
                    p = compute_sign_agreement(wm, z_rec)
                    results[aname].append(p)
                    counter += 1
                    if counter % 10 == 0 or counter <= 5:
                        print(f"  [{counter}/{total}] pi={pi} si={si} {aname} p={p:.4f}", flush=True)
                except Exception as e:
                    print(f"  attack {aname} failed: {e}", flush=True)

    summary = {}
    for aname, ps in results.items():
        if not ps:
            continue
        summary[aname] = {
            "mean": float(np.mean(ps)),
            "std": float(np.std(ps)),
            "n": len(ps),
            "values": ps,
        }
        print(f"{aname}: mean p = {summary[aname]['mean']:.4f} (n={len(ps)})", flush=True)

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.output}", flush=True)


if __name__ == "__main__":
    main()

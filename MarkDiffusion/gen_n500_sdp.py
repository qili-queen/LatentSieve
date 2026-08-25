"""Generate N=500 watermarked images per scheme for paper evaluation at scale.

Layout: qi_eval_n500/{scheme}_wm/img_NNN.png with idx = pi * SEEDS_PER_PROMPT + si.
NUM_PROMPTS=10, SEEDS_PER_PROMPT=50 by default → 500 images per scheme.

Skips existing files (supports resume).

Usage:
  python qi_gen_n500.py --gpu 0 --scheme MIRAGE
  python qi_gen_n500.py --gpu 1 --scheme PRC,GS
  python qi_gen_n500.py --gpu 2 --scheme TR,T2SMark
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
def st(s, *a, **k): k['disable'] = True; oti(s, *a, **k)
tqdm.__init__ = st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qi_runtime_utils import build_cached_pipe, configure_qi_runtime, resolve_qi_device
from utils.diffusion_config import DiffusionConfig
from utils.media_utils import (
    get_random_latents, get_media_latents, transform_to_model_format
)
from watermark.auto_watermark import AutoWatermark

configure_qi_runtime()

MODEL = "huanzi05/stable-diffusion-2-1-base"
STEPS = 50
GS_SCALE = 7.5
SIZE = 512

from qi_sdp_n500_config import PROMPTS, OUT_BASE as _OUT_BASE
_DUMMY = [
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

OUT_BASE = _OUT_BASE; _OLD_OUT_BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "qi_eval_n500"
)


def reseed(wm, seed):
    wm.config.gen_seed = seed
    wm.config.init_latents_seed = seed
    r = torch.Generator(device=wm.config.device).manual_seed(seed)
    wm.config.init_latents = get_random_latents(
        wm.config.pipe, height=SIZE, width=SIZE, generator=r
    ).to(wm.config.device, dtype=wm.config.dtype)


def generate_markdiff_scheme(scheme_name, config_path, pipe, scheduler,
                              device, dtype, num_prompts, seeds_per_prompt):
    out_dir = os.path.join(OUT_BASE, f"{scheme_name}_wm")
    os.makedirs(out_dir, exist_ok=True)
    total = num_prompts * seeds_per_prompt

    dc = DiffusionConfig(
        scheduler=scheduler, pipe=pipe, device=device,
        image_size=(SIZE, SIZE), num_inference_steps=STEPS,
        num_inversion_steps=STEPS, guidance_scale=GS_SCALE,
        gen_seed=42, inversion_type="ddim", dtype=dtype,
    )
    config_arg = (
        f"config/MIRAGE_presence.json" if scheme_name == "MIRAGE"
        else config_path
    )
    wm = AutoWatermark.load(scheme_name, algorithm_config=config_arg,
                            diffusion_config=dc)

    metadata = []
    for pi in range(num_prompts):
        for si in range(seeds_per_prompt):
            seed = pi * 100 + si
            idx = pi * seeds_per_prompt + si
            path = os.path.join(out_dir, f"img_{idx:04d}.png")

            if os.path.exists(path):
                metadata.append({
                    "idx": idx, "seed": seed,
                    "prompt_idx": pi, "path": path,
                })
                continue

            reseed(wm, seed)
            img = wm.generate_watermarked_media(PROMPTS[pi])
            img.save(path)
            metadata.append({
                "idx": idx, "seed": seed,
                "prompt_idx": pi, "path": path,
            })
            if (idx + 1) % 10 == 0 or idx + 1 == total:
                print(f"  [{idx+1}/{total}] {scheme_name} seed={seed} done",
                      flush=True)

    meta_path = os.path.join(OUT_BASE, f"{scheme_name}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  {scheme_name}: {len(metadata)} images saved to {out_dir}",
          flush=True)
    return wm


def generate_t2smark(pipe, device, num_prompts, seeds_per_prompt):
    T2SMARK_ROOT = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir, "T2SMark"
    ))
    if T2SMARK_ROOT not in sys.path:
        sys.path.insert(0, T2SMARK_ROOT)

    from src.t2s import T2SMark
    import src.utils as t2s_utils
    from diffusers import DDIMScheduler

    out_dir = os.path.join(OUT_BASE, "T2SMark_wm")
    os.makedirs(out_dir, exist_ok=True)
    total = num_prompts * seeds_per_prompt

    KEY_LENGTH = 16
    MSG_LENGTH = 256
    TAU = 0.674
    KEY_CH = 0
    msg_ch_idx = [i for i in range(4) if i != KEY_CH]

    pipe = pipe.to(dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    t2s_key = T2SMark(m=KEY_LENGTH, tau=TAU, latent_shape=(1, 64, 64))
    t2s_msg = T2SMark(m=MSG_LENGTH, tau=TAU, latent_shape=(3, 64, 64))

    metadata = []
    for pi in range(num_prompts):
        for si in range(seeds_per_prompt):
            seed = pi * 100 + si
            idx = pi * seeds_per_prompt + si
            path = os.path.join(out_dir, f"img_{idx:04d}.png")

            base_seed = seed + pi * 1000
            t2s_utils.set_random_seed(base_seed)
            master_key = torch.randint(0, 2, (KEY_LENGTH,)).to(device)
            msg = torch.randint(0, 2, (MSG_LENGTH,)).to(device)
            key = torch.randint(0, 2, (KEY_LENGTH,)).to(device)

            if os.path.exists(path):
                metadata.append({
                    "idx": idx, "seed": seed, "prompt_idx": pi, "path": path,
                    "master_key": master_key.cpu().tolist(),
                    "key": key.cpu().tolist(),
                    "msg": msg.cpu().tolist(),
                })
                continue

            # Generate watermarked latent (T2SMark API: encode)
            z_k = t2s_key.encode(key, master_key).to(device)
            z_b = t2s_msg.encode(msg, key).to(device)

            initial_latents = torch.zeros(1, 4, 64, 64).to(device)
            initial_latents[:, KEY_CH, :, :] = z_k
            initial_latents[:, msg_ch_idx, :, :] = z_b

            output = pipe(
                PROMPTS[pi],
                guidance_scale=GS_SCALE,
                num_inference_steps=STEPS,
                height=SIZE, width=SIZE,
                latents=initial_latents.half(),
            )
            img = output.images[0]
            img.save(path)
            metadata.append({
                "idx": idx, "seed": seed, "prompt_idx": pi, "path": path,
                "master_key": master_key.cpu().tolist(),
                "key": key.cpu().tolist(),
                "msg": msg.cpu().tolist(),
            })
            if (idx + 1) % 10 == 0 or idx + 1 == total:
                print(f"  [{idx+1}/{total}] T2SMark seed={seed} done",
                      flush=True)

    meta_path = os.path.join(OUT_BASE, "T2SMark_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  T2SMark: {len(metadata)} images saved to {out_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--scheme", type=str, default="MIRAGE,PRC,GS,TR,T2SMark",
                        help="comma-separated schemes to generate")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--seeds-per-prompt", type=int, default=50)
    args = parser.parse_args()

    schemes = [s.strip() for s in args.scheme.split(",") if s.strip()]
    print(f"Generating schemes: {schemes}", flush=True)
    print(f"Total per scheme: {args.num_prompts * args.seeds_per_prompt}",
          flush=True)
    print(f"Output base: {OUT_BASE}", flush=True)
    os.makedirs(OUT_BASE, exist_ok=True)

    device, dtype = resolve_qi_device(args.gpu)
    pipe, scheduler = build_cached_pipe(MODEL, device, dtype)

    schema_configs = {
        "MIRAGE": "config/MIRAGE_presence.json",
        "PRC":    "config/PRC.json",
        "GS":     "config/GS.json",
        "TR":     "config/TR.json",
    }

    for scheme in schemes:
        print(f"\n=== Generating {scheme} ===", flush=True)
        if scheme == "T2SMark":
            generate_t2smark(pipe, device, args.num_prompts,
                              args.seeds_per_prompt)
        elif scheme in schema_configs:
            generate_markdiff_scheme(
                scheme, schema_configs[scheme], pipe, scheduler, device,
                dtype, args.num_prompts, args.seeds_per_prompt,
            )
        else:
            print(f"  Unknown scheme: {scheme}", flush=True)


if __name__ == "__main__":
    main()

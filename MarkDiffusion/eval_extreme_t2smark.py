"""T2SMark standard 8 attacks on SDP N=500.

Reads T2SMark_wm/, applies each of 8 standard image-domain attacks, detects
with T2SMark pipeline (DDIM inversion + decoder + l1 threshold).

Usage:
  CUDA_VISIBLE_DEVICES=0 python qi_eval_n500_t2smark_standard_sdp.py --gpu 0
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

T2S_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "T2SMark"
))
sys.path.insert(0, T2S_ROOT)

from evaluation.tools.image_editor import (
    AdaptiveNoiseInjection, Brightness, CrSc, GaussianBlurring,
    GaussianNoise, JPEGCompression, Mask, Overlay,
)
from qi_runtime_utils import build_cached_pipe, configure_qi_runtime, resolve_qi_device
from utils.media_utils import get_media_latents, transform_to_model_format
from src.t2s import T2SMark
from diffusers import DDIMScheduler
from inversions.ddim_inversion import DDIMInversion

configure_qi_runtime()

MODEL = "huanzi05/stable-diffusion-2-1-base"
STEPS = 50
GS_SCALE = 7.5
SIZE = 512

from qi_sdp_n500_config import PROMPTS

BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "qi_eval_n500_sdp"
)


def make_attacks():
    return {
        "JPEG-10":  JPEGCompression(quality=10),
        "Noise-50": GaussianNoise(sigma=50.0),
        "Blur-6":   GaussianBlurring(radius=6),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--output", type=str,
                        default="qi_eval_n500_sdp/extreme/T2SMark_extreme.json")
    args = parser.parse_args()

    device, dtype = resolve_qi_device(args.gpu)
    pipe, scheduler = build_cached_pipe(MODEL, device, dtype)

    KEY_LENGTH = 16
    MSG_LENGTH = 256
    TAU = 0.674
    KEY_CH = 0
    msg_ch_idx = [i for i in range(4) if i != KEY_CH]

    pipe = pipe.to(dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    inversion = DDIMInversion(pipe.scheduler, pipe.unet, device)

    t2s_key = T2SMark(m=KEY_LENGTH, tau=TAU, latent_shape=(1, 64, 64))
    t2s_msg = T2SMark(m=MSG_LENGTH, tau=TAU, latent_shape=(3, 64, 64))

    with open(os.path.join(BASE, "T2SMark_metadata.json")) as f:
        metadata = json.load(f)
    metadata = metadata[: args.max_images]

    def encode_prompt(prompt_text):
        pe, ne = pipe.encode_prompt(
            prompt=prompt_text, device=device,
            do_classifier_free_guidance=True,
            num_images_per_prompt=1,
        )
        return torch.cat([ne, pe])

    attacks = make_attacks()
    results = {a: [] for a in attacks}
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f).get("results", results)
        print(f"  Resumed; counts: "
              f"{[(a, len(results[a])) for a in attacks]}", flush=True)

    t_start = time.time()
    n = len(metadata)
    for i, info in enumerate(metadata):
        idx = info["idx"]
        pi = info["prompt_idx"]
        prompt = PROMPTS[pi]
        master_key = torch.tensor(info["master_key"]).to(device)
        key = torch.tensor(info["key"]).to(device)
        msg = torch.tensor(info["msg"]).to(device)

        wm_path = os.path.join(BASE, "T2SMark_wm", f"img_{idx:04d}.png")
        if not os.path.exists(wm_path):
            continue
        wm_img = Image.open(wm_path).convert("RGB")
        text_emb = encode_prompt(prompt)

        for atk_name, atk_fn in attacks.items():
            if any(r.get("idx") == idx for r in results[atk_name]):
                continue
            attacked = atk_fn.edit(wm_img.copy(), {}) if atk_fn else wm_img
            img_t = transform_to_model_format(
                attacked, target_size=SIZE
            ).unsqueeze(0).to(device, dtype=torch.float16)
            lat = get_media_latents(
                pipe=pipe, media=img_t, sample=False, decoder_inv=False
            )
            reversed_lat = inversion.forward_diffusion(
                latents=lat, text_embeddings=text_emb,
                guidance_scale=GS_SCALE, num_inference_steps=STEPS,
            )[-1]
            rev_key, l1 = t2s_key.decode(
                reversed_lat[0, KEY_CH, :, :], master_key, detection=True
            )
            rev_msg = t2s_msg.decode(
                reversed_lat[0, msg_ch_idx, :, :], rev_key
            )
            key_acc = (rev_key == key).float().mean().item()
            msg_acc = (rev_msg == msg).float().mean().item()
            results[atk_name].append({
                "idx": idx, "seed": info["seed"], "prompt_idx": pi,
                "l1": float(l1),
                "key_acc": float(key_acc),
                "msg_acc": float(msg_acc),
                "is_watermarked": bool(l1 > 1.0),
            })

        if (i + 1) % 25 == 0 or i + 1 == n:
            elapsed = time.time() - t_start
            print(f"  T2SMark std [{i+1}/{n}] elapsed={elapsed/60:.1f}min",
                  flush=True)
            summary = {a: {
                "n": len(recs),
                "tpr": float(np.mean([r["is_watermarked"] for r in recs])),
                "msg_acc_mean": float(np.mean([r["msg_acc"] for r in recs])),
                "l1_mean": float(np.mean([r["l1"] for r in recs])),
            } for a, recs in results.items() if recs}
            with open(args.output, "w") as f:
                json.dump({"scheme": "T2SMark", "results": results,
                           "summary": summary}, f, indent=2, default=str)

    print("\n=== T2SMark Standard SUMMARY ===", flush=True)
    for atk, recs in results.items():
        if recs:
            tpr = float(np.mean([r["is_watermarked"] for r in recs]))
            ma = float(np.mean([r["msg_acc"] for r in recs]))
            print(f"  {atk:<20} TPR={tpr*100:>5.1f}% msg_acc={ma:.3f} "
                  f"(n={len(recs)})", flush=True)


if __name__ == "__main__":
    main()

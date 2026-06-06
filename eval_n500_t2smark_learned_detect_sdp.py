"""Stage 7 — T2SMark detection on learned-attack images at N=500.

Usage:
  CUDA_VISIBLE_DEVICES=2 python eval_n500_t2smark_learned_detect.py --gpu 0
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

from runtime_utils import build_cached_pipe, configure_runtime, resolve_device
from utils.media_utils import get_media_latents, transform_to_model_format
from src.t2s import T2SMark
from diffusers import DDIMScheduler
from inversions.ddim_inversion import DDIMInversion

configure_runtime()

MODEL = "stabilityai/stable-diffusion-2-1-base"
STEPS = 50
GS_SCALE = 7.5
SIZE = 512

from sdp_n500_config import PROMPTS

BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "eval_n500_sdp"
)

ATTACKERS = ["cheng2020-anchor_3", "diff_attacker_60", "Ctrlgen"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--output", type=str,
                        default="eval_n500_sdp/learned_T2SMark_detect.json")
    args = parser.parse_args()

    device, dtype = resolve_device(args.gpu)
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

    meta_path = os.path.join(BASE, "T2SMark_metadata.json")
    with open(meta_path) as f:
        metadata = json.load(f)
    metadata = metadata[: args.max_images]

    def encode_prompt(prompt_text):
        do_cfg = GS_SCALE > 1.0
        pe, ne = pipe.encode_prompt(
            prompt=prompt_text, device=device,
            do_classifier_free_guidance=do_cfg,
            num_images_per_prompt=1,
        )
        return torch.cat([ne, pe]) if do_cfg else pe

    results = {a: [] for a in ATTACKERS}
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f).get("results", results)
        for a in ATTACKERS:
            if a not in results:
                results[a] = []
        print(f"  Resumed; counts: {[len(results[a]) for a in ATTACKERS]}",
              flush=True)

    t_start = time.time()
    n = len(metadata)
    for i, info in enumerate(metadata):
        idx = info["idx"]
        pi = info["prompt_idx"]
        prompt = PROMPTS[pi]
        master_key = torch.tensor(info["master_key"]).to(device)
        key = torch.tensor(info["key"]).to(device)
        msg = torch.tensor(info["msg"]).to(device)

        text_emb = encode_prompt(prompt)

        for attacker in ATTACKERS:
            if any(r.get("idx") == idx for r in results[attacker]):
                continue
            img_path = os.path.join(
                BASE, f"T2SMark_attacked_{attacker}/img_{idx:04d}.png"
            )
            if not os.path.exists(img_path):
                continue

            img = Image.open(img_path).convert("RGB")
            img_t = transform_to_model_format(
                img, target_size=SIZE
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

            results[attacker].append({
                "idx": idx, "seed": info["seed"], "prompt_idx": pi,
                "l1": float(l1),
                "key_acc": float(key_acc),
                "msg_acc": float(msg_acc),
                "is_watermarked": bool(l1 > 1.0),
            })

        if (i + 1) % 25 == 0 or i + 1 == n:
            elapsed = time.time() - t_start
            print(f"  T2SMark [{i+1}/{n}] elapsed={elapsed/60:.1f}min",
                  flush=True)
            summary = {}
            for a, recs in results.items():
                if recs:
                    summary[a] = {
                        "n": len(recs),
                        "tpr": float(np.mean(
                            [r["is_watermarked"] for r in recs]
                        )),
                        "msg_acc_mean": float(np.mean(
                            [r["msg_acc"] for r in recs]
                        )),
                    }
            output = {"scheme": "T2SMark", "results": results,
                       "summary": summary}
            with open(args.output, "w") as f:
                json.dump(output, f, indent=2, default=str)

    print("\n=== T2SMark Learned-attack SUMMARY ===", flush=True)
    for a, recs in results.items():
        if recs:
            tpr = float(np.mean([r["is_watermarked"] for r in recs]))
            ma = float(np.mean([r["msg_acc"] for r in recs]))
            print(f"  {a:<24} TPR={tpr*100:>5.1f}% msg_acc={ma:.3f} (n={len(recs)})",
                  flush=True)


if __name__ == "__main__":
    main()

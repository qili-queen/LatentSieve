"""Stage 6 — T2SMark UnMarker detection at N=200.

Uses eval_n500/T2SMark_metadata.json + attacked imgs at
eval_n500/T2SMark_attacked_s1/, _s2/.

Usage:
  python eval_n500_t2smark_unmarker_detect.py --gpu 1
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--output", type=str,
                        default="eval_n500_sdp/unmarker_T2SMark_detect.json")
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

    results = {"clean": [], "s1": [], "s2": []}
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f).get("results", results)
        print(f"  Resumed; counts: "
              f"{[len(v) for v in results.values()]}", flush=True)

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
        s1_path = os.path.join(BASE, "T2SMark_attacked_s1",
                                f"img_{idx:04d}.png")
        s2_path = os.path.join(BASE, "T2SMark_attacked_s2",
                                f"img_{idx:04d}.png")

        text_emb = encode_prompt(prompt)

        for stage, img_path in [
            ("clean", wm_path), ("s1", s1_path), ("s2", s2_path)
        ]:
            if any(r.get("idx") == idx for r in results[stage]):
                continue
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

            results[stage].append({
                "idx": idx, "seed": info["seed"], "prompt_idx": pi,
                "l1": float(l1),
                "key_acc": float(key_acc),
                "msg_acc": float(msg_acc),
                "is_watermarked": bool(l1 > 1.0),  # paper threshold
            })

        if (i + 1) % 10 == 0 or i + 1 == n:
            elapsed = time.time() - t_start
            print(f"  T2SMark [{i+1}/{n}] elapsed={elapsed/60:.1f}min",
                  flush=True)
            summary = {}
            for s, recs in results.items():
                if recs:
                    summary[s] = {
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

    print("\n=== T2SMark UnMarker SUMMARY ===", flush=True)
    for stage, recs in results.items():
        if recs:
            tpr = float(np.mean([r["is_watermarked"] for r in recs]))
            ma = float(np.mean([r["msg_acc"] for r in recs]))
            print(f"  {stage:<6} TPR={tpr*100:>5.1f}% msg_acc={ma:.3f} "
                  f"(n={len(recs)})", flush=True)


if __name__ == "__main__":
    main()

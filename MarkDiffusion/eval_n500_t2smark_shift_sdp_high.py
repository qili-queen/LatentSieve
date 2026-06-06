"""T2SMark SHIFT attack at 4 strengths × N=500.

Applies SHIFT trajectory deflection to T2SMark watermarked images, then
runs T2SMark detection (DDIM inversion + decoder + l1 threshold).

Usage:
  CUDA_VISIBLE_DEVICES=1 python eval_n500_t2smark_shift_sdp.py --gpu 0
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
LAMBDAS = [0.3, 0.5, 0.6, 0.7]

from sdp_n500_config import PROMPTS

BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "eval_n500_sdp"
)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--output", type=str,
                        default="eval_n500_sdp/T2SMark_shift.json")
    parser.add_argument("--lambdas", type=str,
                        default="0.75,0.80,0.85")
    args = parser.parse_args()

    lambdas = [float(x) for x in args.lambdas.split(",")]

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

    results = {str(l): [] for l in lambdas}
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f).get("results", results)
        for l in lambdas:
            results.setdefault(str(l), [])
        print(f"  Resumed; counts: "
              f"{[(l, len(results[str(l)])) for l in lambdas]}", flush=True)

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

        for lam in lambdas:
            key_str = str(lam)
            if any(r.get("idx") == idx for r in results[key_str]):
                continue
            attacked = shift_attack(pipe, wm_img, text_emb, lam, device,
                                     torch.float16)
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
            results[key_str].append({
                "idx": idx, "seed": info["seed"], "prompt_idx": pi,
                "l1": float(l1),
                "key_acc": float(key_acc),
                "msg_acc": float(msg_acc),
                "is_watermarked": bool(l1 > 1.0),
            })

        if (i + 1) % 25 == 0 or i + 1 == n:
            elapsed = time.time() - t_start
            print(f"  T2SMark SHIFT [{i+1}/{n}] elapsed={elapsed/60:.1f}min",
                  flush=True)
            summary = {l: {
                "n": len(results[l]),
                "tpr": float(np.mean([r["is_watermarked"] for r in results[l]])),
                "msg_acc_mean": float(np.mean([r["msg_acc"] for r in results[l]])),
                "l1_mean": float(np.mean([r["l1"] for r in results[l]])),
            } for l in results if results[l]}
            with open(args.output, "w") as f:
                json.dump({"scheme": "T2SMark", "results": results,
                           "summary": summary}, f, indent=2, default=str)

    print("\n=== T2SMark SHIFT SUMMARY ===", flush=True)
    for l in sorted(results, key=float):
        recs = results[l]
        if recs:
            tpr = float(np.mean([r["is_watermarked"] for r in recs]))
            ma = float(np.mean([r["msg_acc"] for r in recs]))
            print(f"  lambda={l} TPR={tpr*100:>5.1f}% msg_acc={ma:.3f} "
                  f"(n={len(recs)})", flush=True)


if __name__ == "__main__":
    main()

"""Stage 5: SHIFT attack at 4 strengths × 4 markdiff schemes × N=500.

Reuses cached watermarked images from eval_n500/{scheme}_wm/, applies
SHIFT trajectory deflection at lambda in {0.3, 0.5, 0.6, 0.7}, and detects.

Skips T2SMark (separate pipeline). Run per-scheme; designed for parallelism
across GPUs.

Usage:
  # GPU 0: MIRAGE
  python eval_n500_shift.py --gpu 0 --scheme MIRAGE
  # GPU 1: PRC + GS
  python eval_n500_shift.py --gpu 1 --scheme PRC,GS
  # GPU 2: TR
  python eval_n500_shift.py --gpu 2 --scheme TR
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
    get_random_latents, get_media_latents, transform_to_model_format,
)
from watermark.auto_watermark import AutoWatermark

configure_runtime()

MODEL = "stabilityai/stable-diffusion-2-1-base"
STEPS = 50
GS_SCALE = 7.5
SIZE = 512

from sdp_n500_config import PROMPTS

WM_DIR_BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "eval_n500_sdp"
)

SCHEME_CONFIGS = {
    "MIRAGE": "config/MIRAGE_presence.json",
    "PRC":    "config/PRC.json",
    "GS":     "config/GS.json",
    "TR":     "config/TR.json",
}

LAMBDAS = [0.3, 0.5, 0.6, 0.7]


def reseed(wm, seed):
    wm.config.gen_seed = seed
    wm.config.init_latents_seed = seed
    r = torch.Generator(device=wm.config.device).manual_seed(seed)
    wm.config.init_latents = get_random_latents(
        wm.config.pipe, height=SIZE, width=SIZE, generator=r
    ).to(wm.config.device, dtype=wm.config.dtype)


def encode_prompt(pipe, prompt, device):
    pe, ne = pipe.encode_prompt(
        prompt=prompt, device=device,
        do_classifier_free_guidance=True, num_images_per_prompt=1,
    )
    return torch.cat([ne, pe])


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


def detect_framework_scheme(wm, image, prompt, scheme_name):
    """Run scheme detector; return dict with 'detected' and 'ba'."""
    result = wm.detect_watermark_in_media(image, prompt=prompt)
    if scheme_name == "TR":
        return {
            "detected": bool(result.get("is_watermarked", False)),
            "ba": 1.0 if result.get("is_watermarked", False) else 0.0,
            "l1": float(result.get("l1_distance", 999)),
        }
    elif scheme_name == "MIRAGE":
        return {
            "detected": bool(result.get("is_watermarked", False)),
            "ba": float(result.get("weighted_full_match_rate", 0.5)),
        }
    elif scheme_name == "GS":
        return {
            "detected": bool(result.get("is_watermarked", False)),
            "ba": float(result.get("bit_acc", 0.5)),
        }
    elif scheme_name == "PRC":
        return {
            "detected": bool(result.get("is_watermarked", False)),
            "ba": 1.0 if result.get("is_watermarked", False) else 0.0,
        }
    return {"detected": False, "ba": 0.5}


def evaluate_scheme(scheme, gpu, num_prompts, seeds_per_prompt, output_dir,
                     lambdas):
    device, dtype = resolve_device(gpu)
    pipe, scheduler = build_cached_pipe(MODEL, device, dtype)
    dc = DiffusionConfig(
        scheduler=scheduler, pipe=pipe, device=device,
        image_size=(SIZE, SIZE), num_inference_steps=STEPS,
        num_inversion_steps=STEPS, guidance_scale=GS_SCALE,
        gen_seed=42, inversion_type="ddim", dtype=dtype,
    )
    wm = AutoWatermark.load(scheme, algorithm_config=SCHEME_CONFIGS[scheme],
                             diffusion_config=dc)

    wm_dir = os.path.join(WM_DIR_BASE, f"{scheme}_wm")
    if not os.path.isdir(wm_dir):
        raise FileNotFoundError(f"WM dir missing: {wm_dir}")

    out_path = os.path.join(output_dir, f"{scheme}_shift.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f).get("results_per_lambda", {})
            print(f"  {scheme}: resumed from existing JSON", flush=True)
    else:
        results = {f"{lam:.2f}": [] for lam in lambdas}

    for lam in lambdas:
        if f"{lam:.2f}" not in results:
            results[f"{lam:.2f}"] = []

    total = num_prompts * seeds_per_prompt
    print(f"  {scheme}: evaluating {total} images × {len(lambdas)} lambdas",
          flush=True)
    t_start = time.time()

    counter = 0
    for pi in range(num_prompts):
        prompt = PROMPTS[pi]
        text_emb = encode_prompt(pipe, prompt, device)

        for si in range(seeds_per_prompt):
            seed = pi * 100 + si
            idx = pi * seeds_per_prompt + si
            counter += 1
            img_path = os.path.join(wm_dir, f"img_{idx:04d}.png")

            if not os.path.exists(img_path):
                print(f"  WARN missing {img_path}", flush=True)
                continue

            wm_img = Image.open(img_path).convert("RGB")

            for lam in lambdas:
                key = f"{lam:.2f}"
                # skip if already done
                if any(r.get("idx") == idx for r in results[key]):
                    continue

                try:
                    atk_img = shift_attack(pipe, wm_img, text_emb, lam,
                                            device, dtype)
                    reseed(wm, seed)
                    det = detect_framework_scheme(
                        wm, atk_img, prompt, scheme,
                    )
                    rec = {
                        "idx": idx, "seed": seed, "prompt_idx": pi,
                        "lambda": lam,
                        "detected": det["detected"],
                        "ba": det["ba"],
                    }
                    if "l1" in det:
                        rec["l1"] = det["l1"]
                    results[key].append(rec)
                except Exception as e:
                    print(f"    {scheme} idx={idx} lam={lam}: FAILED {e}",
                          flush=True)
                    results[key].append({
                        "idx": idx, "seed": seed, "prompt_idx": pi,
                        "lambda": lam, "detected": False, "ba": 0.0,
                        "error": str(e),
                    })

            if counter % 25 == 0 or counter == total:
                elapsed = time.time() - t_start
                rate = counter / max(elapsed, 1)
                eta_s = (total - counter) / max(rate, 0.01)
                print(
                    f"    [{counter}/{total}] {scheme} elapsed={elapsed/60:.1f}min "
                    f"ETA={eta_s/60:.1f}min", flush=True
                )

                # Persist
                summary = {}
                for k, recs in results.items():
                    if recs:
                        summary[k] = {
                            "n": len(recs),
                            "tpr": float(np.mean(
                                [r["detected"] for r in recs]
                            )),
                            "ba_mean": float(np.mean(
                                [r["ba"] for r in recs]
                            )),
                        }
                output = {
                    "scheme": scheme,
                    "num_prompts": num_prompts,
                    "seeds_per_prompt": seeds_per_prompt,
                    "lambdas": lambdas,
                    "results_per_lambda": results,
                    "summary": summary,
                }
                with open(out_path, "w") as f:
                    json.dump(output, f, indent=2, default=str)

    summary = {}
    for k, recs in results.items():
        if recs:
            summary[k] = {
                "n": len(recs),
                "tpr": float(np.mean([r["detected"] for r in recs])),
                "ba_mean": float(np.mean([r["ba"] for r in recs])),
            }
    output = {
        "scheme": scheme,
        "num_prompts": num_prompts,
        "seeds_per_prompt": seeds_per_prompt,
        "lambdas": lambdas,
        "results_per_lambda": results,
        "summary": summary,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n=== {scheme} SHIFT SUMMARY ({total} images) ===", flush=True)
    print(f"  {'lambda':>8} {'TPR':>8} {'BA_mean':>10}", flush=True)
    for lam in lambdas:
        key = f"{lam:.2f}"
        if key in summary:
            d = summary[key]
            print(f"  {key:>8} {d['tpr']*100:>7.1f}% {d['ba_mean']:>10.4f}",
                  flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--scheme", type=str,
                        default="MIRAGE",
                        help="comma-separated schemes")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--seeds-per-prompt", type=int, default=5)
    parser.add_argument("--lambdas", type=str, default="0.75,0.80,0.85")
    parser.add_argument("--output-dir", type=str,
                        default="eval_n500_sdp/shift_results_high")
    args = parser.parse_args()

    schemes = [s.strip() for s in args.scheme.split(",") if s.strip()]
    lambdas = [float(x) for x in args.lambdas.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Schemes: {schemes}", flush=True)
    print(f"Lambdas: {lambdas}", flush=True)
    print(f"Total per scheme: {args.num_prompts * args.seeds_per_prompt}",
          flush=True)
    print(f"GPU: {args.gpu}", flush=True)

    for scheme in schemes:
        if scheme not in SCHEME_CONFIGS:
            print(f"SKIP unknown scheme: {scheme}", flush=True)
            continue
        print(f"\n=== Evaluating {scheme} (SHIFT) ===", flush=True)
        evaluate_scheme(
            scheme, args.gpu, args.num_prompts, args.seeds_per_prompt,
            args.output_dir, lambdas,
        )


if __name__ == "__main__":
    main()

"""Stage 8: 15 unseen + combo attacks on MIRAGE × N=500.

9 unseen single attacks + 6 combo attacks (matches paper Table 5).
Uses cached qi_eval_n500/MIRAGE_wm/ images.

Usage:
  python qi_eval_n500_unseen_combo.py --gpu 2
"""

import argparse
import functools
import io
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

oti = tqdm.__init__
@functools.wraps(oti)
def st(s, *a, **k): k['disable'] = True; oti(s, *a, **k)
tqdm.__init__ = st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.tools.image_editor import (
    CrSc, GaussianBlurring, GaussianNoise, JPEGCompression,
)
from qi_runtime_utils import build_cached_pipe, configure_qi_runtime, resolve_qi_device
from utils.diffusion_config import DiffusionConfig
from utils.media_utils import get_random_latents
from watermark.auto_watermark import AutoWatermark

configure_qi_runtime()

MODEL = "huanzi05/stable-diffusion-2-1-base"
STEPS = 50
GS_SCALE = 7.5
SIZE = 512

from qi_sdp_n500_config import PROMPTS

WM_DIR_BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "qi_eval_n500_sdp"
)


# ============================================================================
# Unseen + combo attacks
# ============================================================================

def webp(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB").copy()


def color_jitter(img):
    # Brightness 1.1, Contrast 1.1, Saturation 1.1, Hue +5deg
    enh = ImageEnhance.Brightness(img).enhance(1.1)
    enh = ImageEnhance.Contrast(enh).enhance(1.1)
    enh = ImageEnhance.Color(enh).enhance(1.1)
    return enh


def resize_back(img, scale):
    """Resize down then back up."""
    w, h = img.size
    nw, nh = int(w * scale), int(h * scale)
    small = img.resize((nw, nh), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def contrast_change(img, factor):
    return ImageEnhance.Contrast(img).enhance(factor)


def saturation_change(img, factor):
    return ImageEnhance.Color(img).enhance(factor)


def median_filter(img, size):
    return img.filter(ImageFilter.MedianFilter(size=size))


# ============================================================================
# Attack pipeline (apply multiple attacks in sequence)
# ============================================================================

def jpeg_then_noise(img, q, sigma):
    img = JPEGCompression(quality=q).edit(img.copy(), {})
    img = GaussianNoise(sigma=sigma).edit(img.copy(), {})
    return img


def blur_then_jpeg(img, r, q):
    img = GaussianBlurring(radius=r).edit(img.copy(), {})
    img = JPEGCompression(quality=q).edit(img.copy(), {})
    return img


def noise_then_webp(img, sigma, q):
    img = GaussianNoise(sigma=sigma).edit(img.copy(), {})
    return webp(img, q)


def jpeg_blur_noise(img, q, r, sigma):
    img = JPEGCompression(quality=q).edit(img.copy(), {})
    img = GaussianBlurring(radius=r).edit(img.copy(), {})
    img = GaussianNoise(sigma=sigma).edit(img.copy(), {})
    return img


def cj_jpeg(img, q):
    img = color_jitter(img)
    return JPEGCompression(quality=q).edit(img.copy(), {})


def rb_jpeg(img, scale, q):
    img = resize_back(img, scale)
    return JPEGCompression(quality=q).edit(img.copy(), {})


ATTACKS = {
    # 9 unseen single attacks
    "WebP-50":       lambda img: webp(img, 50),
    "WebP-30":       lambda img: webp(img, 30),
    "ColorJitter":   lambda img: color_jitter(img),
    "ResizeBack-0.5":  lambda img: resize_back(img, 0.5),
    "ResizeBack-0.25": lambda img: resize_back(img, 0.25),
    "Contrast-1.5":  lambda img: contrast_change(img, 1.5),
    "Contrast-0.5":  lambda img: contrast_change(img, 0.5),
    "Saturation-0.3": lambda img: saturation_change(img, 0.3),
    "MedianFilter-5": lambda img: median_filter(img, 5),
    # 6 combo attacks
    "JPEG75+Noise15":   lambda img: jpeg_then_noise(img, 75, 15.0),
    "Blur2+JPEG75":     lambda img: blur_then_jpeg(img, 2, 75),
    "Noise15+WebP50":   lambda img: noise_then_webp(img, 15.0, 50),
    "JPEG50+Blur2+N10": lambda img: jpeg_blur_noise(img, 50, 2, 10.0),
    "CJ+JPEG75":        lambda img: cj_jpeg(img, 75),
    "RB0.5+JPEG75":     lambda img: rb_jpeg(img, 0.5, 75),
}


def reseed(wm, seed):
    wm.config.gen_seed = seed
    wm.config.init_latents_seed = seed
    r = torch.Generator(device=wm.config.device).manual_seed(seed)
    wm.config.init_latents = get_random_latents(
        wm.config.pipe, height=SIZE, width=SIZE, generator=r
    ).to(wm.config.device, dtype=wm.config.dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--seeds-per-prompt", type=int, default=5)
    parser.add_argument("--output", type=str,
                        default="qi_eval_n500_sdp/unseen_combo_mirage.json")
    args = parser.parse_args()

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

    wm_dir = os.path.join(WM_DIR_BASE, "MIRAGE_wm")
    total = args.num_prompts * args.seeds_per_prompt
    print(f"MIRAGE unseen+combo attacks, N={total}", flush=True)
    print(f"Attacks: {list(ATTACKS.keys())}", flush=True)

    results = {a: [] for a in ATTACKS}
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f).get("results_per_attack", results)
        for a in ATTACKS:
            if a not in results:
                results[a] = []
        print(f"  Resumed; counts: "
              f"{[len(results[a]) for a in ATTACKS]}", flush=True)

    t_start = time.time()
    counter = 0

    for pi in range(args.num_prompts):
        prompt = PROMPTS[pi]
        for si in range(args.seeds_per_prompt):
            seed = pi * 100 + si
            idx = pi * args.seeds_per_prompt + si
            counter += 1
            img_path = os.path.join(wm_dir, f"img_{idx:04d}.png")
            if not os.path.exists(img_path):
                continue
            wm_img = Image.open(img_path).convert("RGB")

            for atk_name, atk_fn in ATTACKS.items():
                if any(r.get("idx") == idx for r in results[atk_name]):
                    continue
                try:
                    attacked = atk_fn(wm_img.copy())
                    reseed(wm, seed)
                    r = wm.detect_watermark_in_media(
                        attacked, prompt=prompt
                    )
                    results[atk_name].append({
                        "idx": idx, "seed": seed, "prompt_idx": pi,
                        "is_watermarked": bool(
                            r.get("is_watermarked", False)
                        ),
                        "ba": float(
                            r.get("weighted_full_match_rate", 0.5)
                        ),
                    })
                except Exception as e:
                    print(f"  WARN {atk_name} idx={idx}: {e}", flush=True)
                    results[atk_name].append({
                        "idx": idx, "seed": seed, "prompt_idx": pi,
                        "is_watermarked": False, "ba": 0.5,
                        "error": str(e),
                    })

            if counter % 25 == 0 or counter == total:
                elapsed = time.time() - t_start
                rate = counter / max(elapsed, 1)
                eta_s = (total - counter) / max(rate, 0.01)
                print(f"  [{counter}/{total}] elapsed={elapsed/60:.1f}min "
                      f"ETA={eta_s/60:.1f}min", flush=True)

                # Save progress
                summary = {}
                for atk_name, recs in results.items():
                    if recs:
                        summary[atk_name] = {
                            "n": len(recs),
                            "tpr": float(np.mean(
                                [r["is_watermarked"] for r in recs]
                            )),
                            "ba_mean": float(np.mean(
                                [r["ba"] for r in recs]
                            )),
                            "ba_min": float(np.min(
                                [r["ba"] for r in recs]
                            )),
                        }
                output = {
                    "scheme": "MIRAGE",
                    "num_prompts": args.num_prompts,
                    "seeds_per_prompt": args.seeds_per_prompt,
                    "results_per_attack": results,
                    "summary": summary,
                }
                with open(args.output, "w") as f:
                    json.dump(output, f, indent=2, default=str)

    print("\n=== SUMMARY ===", flush=True)
    print(f"  {'attack':<22} {'TPR':>8} {'BA_mean':>10} {'BA_min':>10}",
          flush=True)
    for atk_name in ATTACKS.keys():
        recs = results.get(atk_name, [])
        if recs:
            tpr = float(np.mean([r["is_watermarked"] for r in recs]))
            ba = float(np.mean([r["ba"] for r in recs]))
            ba_min = float(np.min([r["ba"] for r in recs]))
            print(f"  {atk_name:<22} {tpr*100:>7.1f}% {ba:>10.4f} "
                  f"{ba_min:>10.4f}", flush=True)


if __name__ == "__main__":
    main()

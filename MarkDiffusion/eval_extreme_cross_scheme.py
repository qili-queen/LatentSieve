"""Cross-scheme extreme-strength attacks (Noise-50, JPEG-10, Blur-6) × 4 schemes.

Adapts qi_eval_n500_standard_attacks_sdp.py to use extreme strengths instead of
the calibration nominals. Runs MIRAGE / PRC / GS / TR on cached *_wm/ images.

T2SMark is handled by qi_eval_extreme_t2smark.py (separate code path).

Usage:
  CUDA_VISIBLE_DEVICES=1 python qi_eval_extreme_cross_scheme.py --gpu 0
"""
import argparse, functools, json, os, sys, time
import numpy as np, torch
from PIL import Image

oti = __import__("tqdm").tqdm.__init__
@functools.wraps(oti)
def st(s, *a, **k): k['disable'] = True; oti(s, *a, **k)
__import__("tqdm").tqdm.__init__ = st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.tools.image_editor import (
    GaussianBlurring, GaussianNoise, JPEGCompression,
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

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qi_eval_n500_sdp")

CONFIGS = {
    "MIRAGE": "config/MIRAGE_presence.json",
    "PRC":    "config/PRC.json",
    "GS":     "config/GS.json",
    "TR":     "config/TR.json",
}


def make_attacks():
    return {
        "JPEG-10":  JPEGCompression(quality=10),
        "Noise-50": GaussianNoise(sigma=50.0),
        "Blur-6":   GaussianBlurring(radius=6),
    }


def reseed(wm, seed):
    wm.config.gen_seed = seed
    wm.config.init_latents_seed = seed
    r = torch.Generator(device=wm.config.device).manual_seed(seed)
    wm.config.init_latents = get_random_latents(
        wm.config.pipe, height=SIZE, width=SIZE, generator=r
    ).to(wm.config.device, dtype=wm.config.dtype)


def detect_score(wm, image, prompt):
    r = wm.detect_watermark_in_media(image, prompt=prompt)
    is_wm = bool(r.get("is_watermarked", False))
    if "weighted_full_match_rate" in r:
        s = float(r["weighted_full_match_rate"])
    elif "bit_acc" in r:
        s = float(r["bit_acc"])
    elif "l1_distance" in r:
        s = float(r["l1_distance"])
    elif "score" in r:
        s = float(r["score"])
    else:
        s = 0.0
    return is_wm, s


def evaluate(scheme, gpu, num_images, out_dir):
    device, dtype = resolve_qi_device(gpu)
    pipe, sch = build_cached_pipe(MODEL, device, dtype)
    dc = DiffusionConfig(scheduler=sch, pipe=pipe, device=device,
                          image_size=(SIZE, SIZE), num_inference_steps=STEPS,
                          num_inversion_steps=STEPS, guidance_scale=GS_SCALE,
                          gen_seed=42, inversion_type="ddim", dtype=dtype)
    wm = AutoWatermark.load(scheme, algorithm_config=CONFIGS[scheme],
                              diffusion_config=dc)

    wm_dir = os.path.join(BASE, f"{scheme}_wm")
    attacks = make_attacks()
    results = {n: [] for n in attacks}

    out_path = os.path.join(out_dir, f"{scheme}_extreme.json")
    if os.path.exists(out_path):
        results = json.load(open(out_path)).get("results_per_attack", results)
        n_done = sum(len(v) for v in results.values())
        print(f"  {scheme}: resumed from {n_done} records", flush=True)

    t0 = time.time()
    for i in range(num_images):
        pi = i // 5
        si = i % 5
        seed = pi * 100 + si
        prompt = PROMPTS[pi]
        img_path = os.path.join(wm_dir, f"img_{i:04d}.png")
        if not os.path.exists(img_path):
            continue
        wm_img = Image.open(img_path).convert("RGB")

        for an, atk in attacks.items():
            if any(r.get("idx") == i for r in results[an]):
                continue
            attacked = atk.edit(wm_img.copy(), {})
            reseed(wm, seed)
            is_wm, sc = detect_score(wm, attacked, prompt)
            results[an].append({
                "idx": i, "seed": seed, "prompt_idx": pi,
                "is_watermarked": is_wm, "score": sc,
            })

        if (i + 1) % 25 == 0 or i + 1 == num_images:
            el = (time.time() - t0) / 60
            tprs = {a: f"{np.mean([r['is_watermarked'] for r in v])*100:.1f}%" if v else "n/a"
                    for a, v in results.items()}
            print(f"  {scheme} [{i+1}/{num_images}] {tprs} el={el:.1f}min", flush=True)
            json.dump({"scheme": scheme, "results_per_attack": results},
                      open(out_path, "w"), indent=2)

    json.dump({"scheme": scheme, "results_per_attack": results},
              open(out_path, "w"), indent=2)
    summary = {a: {"n": len(v),
                    "tpr": float(np.mean([r["is_watermarked"] for r in v])) if v else 0,
                    "ba_mean": float(np.mean([r["score"] for r in v])) if v else 0}
                for a, v in results.items()}
    print(f"\n=== {scheme} ===")
    for a, s in summary.items():
        print(f"  {a}: TPR={s['tpr']*100:.1f}%  BA={s['ba_mean']:.3f}  n={s['n']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num-images", type=int, default=500)
    ap.add_argument("--schemes", default="MIRAGE,PRC,GS,TR")
    args = ap.parse_args()

    out_dir = os.path.join(BASE, "extreme")
    os.makedirs(out_dir, exist_ok=True)

    for scheme in args.schemes.split(","):
        scheme = scheme.strip()
        if scheme not in CONFIGS:
            print(f"  WARN: unknown scheme {scheme}, skipping")
            continue
        evaluate(scheme, args.gpu, args.num_images, out_dir)


if __name__ == "__main__":
    main()

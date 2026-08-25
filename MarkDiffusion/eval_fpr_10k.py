"""Large-scale empirical FPR null study for MIRAGE.

Generates UNWATERMARKED SD-2.1 images from MS-COCO captions, runs MIRAGE
detector on each, records BA score + flagged decision. Designed to tighten
the empirical Wilson upper bound on FAR from 2.6e-3 (N=1400) toward 4e-4
(N=10000) at 95% confidence.

Two GPUs in parallel:
  CUDA_VISIBLE_DEVICES=1 python qi_eval_fpr_10k.py --gpu 0 --offset 0     --num-images 5000
  CUDA_VISIBLE_DEVICES=2 python qi_eval_fpr_10k.py --gpu 0 --offset 5000  --num-images 5000

Checkpoint every 50 images to qi_eval_n500_sdp/fpr_null_10k_{offset}.json.
"""
import argparse, functools, json, os, sys, time
import numpy as np, torch
import pandas as pd

oti = __import__("tqdm").tqdm.__init__
@functools.wraps(oti)
def st(s, *a, **k): k['disable'] = True; oti(s, *a, **k)
__import__("tqdm").tqdm.__init__ = st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qi_runtime_utils import build_cached_pipe, configure_qi_runtime, resolve_qi_device
from utils.diffusion_config import DiffusionConfig
from utils.media_utils import get_random_latents
from watermark.auto_watermark import AutoWatermark

configure_qi_runtime()

MODEL = "huanzi05/stable-diffusion-2-1-base"
STEPS = 50              # inversion steps (must match detector pipeline)
GEN_STEPS = 25          # generation steps (cheaper; null samples don't need 50)
GS_SCALE = 7.5
SIZE = 512
THRESHOLD = 0.67
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qi_eval_n500_sdp")


def reseed(wm, seed):
    wm.config.gen_seed = seed
    wm.config.init_latents_seed = seed
    r = torch.Generator(device=wm.config.device).manual_seed(seed)
    wm.config.init_latents = get_random_latents(
        wm.config.pipe, height=SIZE, width=SIZE, generator=r
    ).to(wm.config.device, dtype=wm.config.dtype)


def extract_score(det):
    return float(det.get("presence_score", det["raw_full_match_rate"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--offset", type=int, required=True,
                    help="Starting index into MS-COCO caption pool (for sharding across GPUs).")
    ap.add_argument("--num-images", type=int, default=5000)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    args = ap.parse_args()

    device, dtype = resolve_qi_device(args.gpu)
    pipe, sch = build_cached_pipe(MODEL, device, dtype)
    pipe.set_progress_bar_config(disable=True)
    dc = DiffusionConfig(scheduler=sch, pipe=pipe, device=device,
                          image_size=(SIZE, SIZE), num_inference_steps=GEN_STEPS,
                          num_inversion_steps=STEPS, guidance_scale=GS_SCALE,
                          gen_seed=42, inversion_type="ddim", dtype=dtype)
    wm = AutoWatermark.load("MIRAGE", algorithm_config="config/MIRAGE_presence.json",
                              diffusion_config=dc)

    # Load MS-COCO captions with fixed deterministic ordering
    df = pd.read_parquet("dataset/mscoco/mscoco.parquet")
    captions = df["TEXT"].astype(str).tolist()
    rng = np.random.default_rng(seed=20260603)
    perm = rng.permutation(len(captions))
    captions = [captions[i] for i in perm.tolist()]
    prompts = captions[args.offset : args.offset + args.num_images]

    out_path = os.path.join(OUT_DIR, f"fpr_null_10k_offset{args.offset}.json")

    # Resume
    results = []
    done_idx = set()
    if os.path.exists(out_path):
        prev = json.load(open(out_path))
        results = prev.get("results", [])
        done_idx = {r["idx"] for r in results}
        print(f"  Resumed from {len(results)} records", flush=True)

    t0 = time.time()
    flag_count = sum(1 for r in results if r.get("flagged"))
    score_max = max((r["score"] for r in results), default=0.0)

    for i, prompt in enumerate(prompts):
        idx = args.offset + i
        if idx in done_idx:
            continue
        seed = (idx * 7 + 13) & 0x7FFFFFFF  # deterministic per-sample seed
        try:
            reseed(wm, seed)
            img = wm.generate_unwatermarked_media(prompt)
            det = wm.detect_watermark_in_media(img, prompt=prompt)
            score = extract_score(det)
            flagged = bool(det["is_watermarked"])
        except Exception as e:
            print(f"  [skip] idx={idx} err={e}", flush=True)
            continue
        results.append({"idx": idx, "seed": seed, "score": score, "flagged": flagged})
        flag_count += int(flagged)
        if score > score_max:
            score_max = score

        if (len(results)) % args.checkpoint_every == 0:
            el = (time.time() - t0) / 60
            n = len(results)
            mean_s = np.mean([r["score"] for r in results])
            print(f"  [{n}/{args.num_images}] flagged={flag_count} max={score_max:.4f} "
                  f"mean={mean_s:.4f} el={el:.1f}min", flush=True)
            json.dump({"offset": args.offset, "threshold": THRESHOLD,
                        "results": results,
                        "summary": {"n": n, "flagged": flag_count, "max_score": score_max,
                                     "mean_score": float(mean_s)}},
                      open(out_path, "w"), indent=2)

    n = len(results)
    mean_s = float(np.mean([r["score"] for r in results])) if results else 0.0
    summary = {"n": n, "flagged": flag_count, "max_score": score_max,
                "mean_score": mean_s,
                "fpr_point_estimate": flag_count / max(n, 1)}
    json.dump({"offset": args.offset, "threshold": THRESHOLD,
                "results": results, "summary": summary},
              open(out_path, "w"), indent=2)
    print(f"\n=== Final (offset={args.offset}) ===", flush=True)
    print(f"  n={n}  flagged={flag_count}  max={score_max:.4f}  mean={mean_s:.4f}", flush=True)


if __name__ == "__main__":
    main()

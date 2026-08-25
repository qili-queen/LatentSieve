"""f=0 round-trip baseline for the scheme-aware adaptive attack table.

Motivation
----------
Table 10 reports the image quality of the scheme-aware adaptive attack as an
absolute PSNR/SSIM against the original watermarked image, and the paper reads
"adaptive-high breaks LatentSieve only at PSNR 8.3 dB" as evidence that removal
requires leaving the usable perceptual range.

But every strategy in that table first DDIM-inverts the watermarked image and
then regenerates from the modified latent, and the invert-then-regenerate round
trip degrades the image on its own, before a single sign is flipped: the
lowmag f=0.05 cell already sits at PSNR 13.2 / SSIM 0.47. Without the f=0 row
there is no way to tell how much of the reported quality cost belongs to the
attack and how much belongs to the measurement pipeline.

This script measures exactly that: same generation, same inversion, same
regeneration, no sign flips. It reuses the helpers from
qi_eval_adaptive_attack.py so the code path is identical to the table it
extends, and additionally reports LPIPS, which that script does not compute.

At f=0 the three strategies (adaptive_high / random / lowmag) are identical by
construction -- each returns z_rec.clone() -- so only one pass is run.

Usage
-----
  CUDA_VISIBLE_DEVICES=0 python qi_eval_adaptive_roundtrip_baseline.py \
      --gpu 0 --prompt-start 0 --prompt-end 5 --seeds-per-prompt 50 \
      --output sp2027_proposal/adaptive_roundtrip_p0_4.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qi_eval_adaptive_attack import (  # noqa: E402
    MODEL, STEPS, GS_SCALE, SIZE, PROMPTS,
    reseed, invert_image, regenerate_from_latent,
    compute_psnr, compute_ssim,
)
from qi_runtime_utils import build_cached_pipe, resolve_qi_device  # noqa: E402
from utils.diffusion_config import DiffusionConfig  # noqa: E402
from watermark.auto_watermark import AutoWatermark  # noqa: E402


def to_lpips_tensor(img, device):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t * 2.0 - 1.0  # LPIPS expects [-1, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--prompt-start", type=int, default=0)
    ap.add_argument("--prompt-end", type=int, default=len(PROMPTS))
    ap.add_argument("--seeds-per-prompt", type=int, default=50)
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

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

    import lpips
    lpips_fn = lpips.LPIPS(net="alex").to(device)

    records = []
    if os.path.exists(args.output):
        records = json.load(open(args.output)).get("results", [])
        print(f"resumed from {len(records)} records", flush=True)
    done = {(r["prompt_idx"], r["seed"]) for r in records}

    total = (args.prompt_end - args.prompt_start) * args.seeds_per_prompt
    counter = 0
    for pi in range(args.prompt_start, args.prompt_end):
        prompt = PROMPTS[pi]
        for si in range(args.seeds_per_prompt):
            seed = pi * 100 + si
            counter += 1
            if (pi, seed) in done:
                continue

            # Same generation as Table 10.
            reseed(wm, seed)
            wm_img = wm.generate_watermarked_media(prompt)

            reseed(wm, seed)
            r_clean = wm.detect_watermark_in_media(wm_img, prompt=prompt)
            ba_clean = float(r_clean.get("weighted_full_match_rate", 0.5))

            # Round trip with no sign flips: this is the f=0 cell.
            z_rec = invert_image(wm, wm_img, prompt)
            rt_img = regenerate_from_latent(wm, z_rec.clone(), prompt)

            psnr = compute_psnr(wm_img, rt_img)
            ssim = compute_ssim(wm_img, rt_img)
            with torch.no_grad():
                lp = float(lpips_fn(to_lpips_tensor(wm_img, device),
                                    to_lpips_tensor(rt_img, device)).item())

            reseed(wm, seed)
            r_rt = wm.detect_watermark_in_media(rt_img, prompt=prompt)
            ba_rt = float(r_rt.get("weighted_full_match_rate", 0.5))
            det_rt = bool(r_rt.get("is_watermarked", False))

            records.append({
                "prompt_idx": pi, "seed": seed,
                "ba_clean": ba_clean,
                "ba_rt": ba_rt, "det_rt": det_rt,
                "psnr": psnr, "ssim": ssim, "lpips": lp,
            })

            if counter % 10 == 0:
                n = len(records)
                print(f"  [{counter}/{total}] n={n} "
                      f"PSNR={np.mean([r['psnr'] for r in records]):.2f} "
                      f"SSIM={np.mean([r['ssim'] for r in records]):.3f} "
                      f"LPIPS={np.mean([r['lpips'] for r in records]):.3f} "
                      f"TPR={np.mean([r['det_rt'] for r in records])*100:.1f}%",
                      flush=True)
                os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
                json.dump({"flip_fraction": 0.0, "results": records},
                          open(args.output, "w"), indent=2)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump({"flip_fraction": 0.0, "results": records},
              open(args.output, "w"), indent=2)

    n = max(len(records), 1)
    print(f"\n=== round-trip baseline (f=0), n={len(records)} ===")
    print(f"  TPR        = {np.mean([r['det_rt'] for r in records])*100:.1f}%")
    print(f"  BA (clean) = {np.mean([r['ba_clean'] for r in records]):.4f}")
    print(f"  BA (rt)    = {np.mean([r['ba_rt'] for r in records]):.4f}")
    print(f"  PSNR       = {np.mean([r['psnr'] for r in records]):.2f} dB")
    print(f"  SSIM       = {np.mean([r['ssim'] for r in records]):.4f}")
    print(f"  LPIPS      = {np.mean([r['lpips'] for r in records]):.4f}")


if __name__ == "__main__":
    main()

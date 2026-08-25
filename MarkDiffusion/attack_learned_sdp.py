"""Stage 7 — Learned attacks (cheng2020-anchor + DiffAttacker) on N=500 watermarked images.

Adapted from /home/nisp/Jie/security26/WatermarkAttacker/attack.py.
Reads from qi_eval_n500/{scheme}_wm/, attacks first MAX images, saves to
qi_eval_n500/{scheme}_attacked_{attacker_name}/.

Usage:
  CUDA_VISIBLE_DEVICES=0 python qi_attack_learned.py --attackers cheng2020-anchor_3
  CUDA_VISIBLE_DEVICES=1 python qi_attack_learned.py --attackers diff_attacker_60
"""
import argparse
import os
import sys

# Path setup for WatermarkAttacker module
WATERMARKER_ROOT = "/home/nisp/Jie/security26/WatermarkAttacker"
sys.path.insert(0, WATERMARKER_ROOT)

import torch
from PIL import Image
from tqdm import tqdm

from regen_pipe import ReSDPipeline
from wmattacker import DiffWMAttacker, VAEWMAttacker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attackers", type=str,
                        default="cheng2020-anchor_3,diff_attacker_60",
                        help="comma-separated attacker names")
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--schemes", type=str,
                        default="MIRAGE,PRC,GS,TR,T2SMark")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print("Loading model...", flush=True)
    pipe = ReSDPipeline.from_pretrained(
        "sd-legacy/stable-diffusion-v1-5", torch_dtype=torch.float16
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)
    print("Model loaded.", flush=True)

    attacker_pool = {
        "cheng2020-anchor_3": lambda: VAEWMAttacker(
            "cheng2020-anchor", quality=3, metric="mse", device=device,
        ),
        "diff_attacker_60": lambda: DiffWMAttacker(
            pipe, batch_size=5, noise_step=60, captions={},
        ),
    }
    selected = [a.strip() for a in args.attackers.split(",") if a.strip()]
    attackers = {a: attacker_pool[a]() for a in selected if a in attacker_pool}

    BASE = "/home/nisp/Qili/MarkDiff_research/MarkDiffusion/qi_eval_n500_sdp"
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]

    for attacker_name in attackers:
        for scheme in schemes:
            wm_dir = os.path.join(BASE, f"{scheme}_wm")
            if not os.path.isdir(wm_dir):
                print(f"  SKIP {scheme}: {wm_dir} missing", flush=True)
                continue
            out_dir = os.path.join(BASE, f"{scheme}_attacked_{attacker_name}")
            os.makedirs(out_dir, exist_ok=True)

            img_files = sorted(os.listdir(wm_dir))[: args.max_images]
            test_paths, save_paths = [], []
            for img_name in img_files:
                src = os.path.join(wm_dir, img_name)
                dst = os.path.join(out_dir, img_name)
                if os.path.exists(dst):
                    continue
                test_paths.append(src)
                save_paths.append(dst)

            print(f"\n  === {attacker_name} on {scheme}: "
                  f"{len(test_paths)} images (skipped {len(img_files) - len(test_paths)}) "
                  f"==={out_dir}", flush=True)
            if not test_paths:
                continue

            attackers[attacker_name].attack(test_paths, save_paths)
            print(f"  {attacker_name}/{scheme} DONE", flush=True)


if __name__ == "__main__":
    main()

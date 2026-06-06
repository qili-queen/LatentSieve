"""Stage 6 — Multi-scheme UnMarker attack at N=200.

Attacks cached eval_n500/{scheme}_wm/ images with UnMarker (stage1 and
stage1+stage2). Saves to eval_n500/{scheme}_attacked_s1/ and _s2/.

Usage:
  python eval_n500_unmarker_attack.py --gpu 2 --schemes MIRAGE --num-images 100
  python eval_n500_unmarker_attack.py --gpu 0 --schemes PRC,GS --num-images 100
"""

import argparse
import functools
import os
import sys
import time

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

oti = tqdm.__init__
@functools.wraps(oti)
def st(s, *a, **k): k['disable'] = True; oti(s, *a, **k)
tqdm.__init__ = st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

UNMARKER_ROOT = "/path/to/Jie/ai-watermark"
sys.path.insert(0, UNMARKER_ROOT)
sys.path.insert(0, os.path.join(UNMARKER_ROOT, "systems"))

# Stub out watermarkers module to short-circuit transitive imports
# (PtW, StableSignature, etc. require many extra deps we don't need)
import types
_stub_wm = types.ModuleType('watermarkers')
_stub_wm.init_watermarker = lambda *a, **k: None
sys.modules['watermarkers'] = _stub_wm

from runtime_utils import configure_runtime, resolve_device

configure_runtime()

WM_DIR_BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "eval_n500"
)


def build_attacker(stage_selector, device):
    """Build UnMarker attacker with given stage selector."""
    # Need to be in UNMARKER_ROOT for relative imports
    cwd = os.getcwd()
    os.chdir(UNMARKER_ROOT)
    try:
        with open(os.path.join(UNMARKER_ROOT, "attack_configs",
                                "Prc.yaml")) as f:
            conf = yaml.load(f, Loader=yaml.Loader)
        unmarker_conf = conf["UnMarker"]
        from modules.attack.unmark import UnMark
        attacker = UnMark(
            evaluator=None,
            stage_selector=stage_selector,
            preprocess_args=unmarker_conf.get("preprocess_args", {}),
            stage1_args=unmarker_conf.get("stage1_args"),
            stage2_args=unmarker_conf.get("stage2_args"),
            input_dir=None,
            output_dir=None,
            batch_size=1,
            device=str(device),
        )
    finally:
        os.chdir(cwd)
    return attacker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--schemes", type=str, default="MIRAGE",
                        help="comma-separated schemes")
    parser.add_argument("--start-idx", type=int, default=100,
                        help="start image index (inclusive)")
    parser.add_argument("--end-idx", type=int, default=200,
                        help="end image index (exclusive)")
    parser.add_argument("--stages", type=str, default="s1,s2",
                        help="comma-separated stages: s1, s2 (s2 includes "
                             "stage1+stage2 chain)")
    args = parser.parse_args()

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    device, dtype = resolve_device(args.gpu)
    print(f"GPU: {device}", flush=True)
    print(f"Schemes: {schemes}", flush=True)
    print(f"Stages: {stages}", flush=True)
    print(f"Image idx range: [{args.start_idx}, {args.end_idx})", flush=True)

    # Build attackers
    attackers = {}
    if "s1" in stages:
        print("Building S1 attacker...", flush=True)
        attackers["s1"] = build_attacker(
            ["preprocess", "stage1"], device
        )
    if "s2" in stages:
        print("Building S1+S2 attacker (produces S2)...", flush=True)
        attackers["s2"] = build_attacker(
            ["preprocess", "stage1", "stage2"], device
        )

    # Per-scheme attack loop
    for scheme in schemes:
        wm_dir = os.path.join(WM_DIR_BASE, f"{scheme}_wm")
        if not os.path.isdir(wm_dir):
            print(f"SKIP {scheme}: {wm_dir} not found", flush=True)
            continue

        for stage_name, attacker in attackers.items():
            out_dir = os.path.join(WM_DIR_BASE,
                                    f"{scheme}_attacked_{stage_name}")
            os.makedirs(out_dir, exist_ok=True)

            t_start = time.time()
            for idx in range(args.start_idx, args.end_idx):
                src = os.path.join(wm_dir, f"img_{idx:04d}.png")
                dst = os.path.join(out_dir, f"img_{idx:04d}.png")
                if not os.path.exists(src):
                    print(f"  {scheme}/{stage_name} idx={idx}: src missing",
                          flush=True)
                    continue
                if os.path.exists(dst):
                    continue

                try:
                    os.chdir(UNMARKER_ROOT)
                    attacker.attack_single(src, save_path=dst)
                    os.chdir(os.path.dirname(os.path.abspath(__file__)))
                except Exception as e:
                    print(f"  {scheme}/{stage_name} idx={idx}: FAILED {e}",
                          flush=True)
                    continue

                done = idx - args.start_idx + 1
                total = args.end_idx - args.start_idx
                elapsed = time.time() - t_start
                if done % 5 == 0 or done == total:
                    rate = done / max(elapsed, 1)
                    eta_s = (total - done) / max(rate, 0.01)
                    print(f"  {scheme}/{stage_name} [{done}/{total}] "
                          f"elapsed={elapsed/60:.1f}min "
                          f"ETA={eta_s/60:.1f}min", flush=True)

            print(f"\n{scheme}/{stage_name} DONE", flush=True)


if __name__ == "__main__":
    main()

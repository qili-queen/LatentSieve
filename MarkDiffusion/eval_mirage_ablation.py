"""MIRAGE ablation study: block count, block dimension, erasure, selection strategy.

Tests each ablation variant against a fixed set of attacks and records
TPR, FPR, and watermark score statistics.
"""
import argparse
import csv
import json
import tempfile
from pathlib import Path
from statistics import mean

import torch

from evaluation.tools.image_editor import (
    CrSc,
    GaussianBlurring,
    GaussianNoise,
    JPEGCompression,
    Overlay,
)
from qi_runtime_utils import build_cached_pipe, configure_qi_runtime, resolve_qi_device
from utils.diffusion_config import DiffusionConfig
from utils.media_utils import get_random_latents
from watermark.auto_watermark import AutoWatermark

configure_qi_runtime()

PROMPTS = [
    "A beautiful sunset over the ocean with golden clouds",
    "A cinematic mountain lake at dawn with pine trees and fog",
    "A close-up portrait of an elderly fisherman with weathered skin",
    "A rainy Tokyo street at night with neon reflections and umbrellas",
    "A red vintage car parked in a wide desert under blue sky",
    "A cozy independent bookstore interior with warm lamps and wooden shelves",
    "A snow-covered cabin in a dense pine forest at twilight",
    "A macro photograph of a bee on a bright sunflower",
    "An aerial view of a coastal city at sunrise with soft haze",
    "A futuristic laboratory with glass walls, white robots, and holograms",
]

# ---------------------------------------------------------------------------
# Ablation axis definitions
# ---------------------------------------------------------------------------
ABLATION_AXES = {
    "block_count": {
        "config_key": "num_blocks",
        "values": [64, 128, 256, 512],
        "default": 256,
    },
    "block_dim": {
        "config_key": "block_dim",
        "values": [8, 16, 32, 64],
        "default": 64,
    },
    "erasure": {
        "config_key": "presence_erasure_bits",
        "values": [0, 32],              # 0 = without erasure, 32 = with erasure
        "labels": ["no_erasure", "erasure_32"],
        "default": 32,
    },
    "selection_strategy": {
        "config_key": "selection_strategy",
        "values": ["topk", "random"],
        "default": "topk",
    },
}


def instantiate_attacks():
    return {
        "Clean": None,
        "JPEG-75": JPEGCompression(quality=75),
        "Noise-15": GaussianNoise(sigma=15.0),
        "Blur-2": GaussianBlurring(radius=2),
        "CropPadBlack-0.8": CrSc(crop_ratio=0.8, resize_back=False, pad_back=True),
        "Overlay": Overlay(num_strokes=10),
    }


# ---------------------------------------------------------------------------
# Helpers (same pattern as the robustness scripts)
# ---------------------------------------------------------------------------
def to_jsonable(value):
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def safe_mean(values):
    cleaned = [v for v in values if v is not None]
    return mean(cleaned) if cleaned else None


def summarize(values):
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return None
    return {
        "count": len(cleaned),
        "mean": float(sum(cleaned) / len(cleaned)),
        "std": float(torch.tensor(cleaned, dtype=torch.float32).std(unbiased=False).item()),
        "min": float(min(cleaned)),
        "max": float(max(cleaned)),
    }


def reseed_watermark(watermark, seed):
    watermark.config.gen_seed = seed
    watermark.config.init_latents_seed = seed
    latents_rng = torch.Generator(device=watermark.config.device)
    latents_rng.manual_seed(seed)
    latent_kwargs = {
        "height": watermark.config.image_size[0],
        "width": watermark.config.image_size[1],
        "generator": latents_rng,
    }
    if watermark.config.num_frames >= 1:
        latent_kwargs["num_frames"] = watermark.config.num_frames
    watermark.config.init_latents = get_random_latents(
        watermark.config.pipe, **latent_kwargs,
    ).to(watermark.config.device, dtype=watermark.config.dtype)


def resolve_seed_values(seed_start, num_seeds, seeds):
    if seeds:
        return [int(item.strip()) for item in seeds.split(",") if item.strip()]
    return list(range(seed_start, seed_start + num_seeds))


# ---------------------------------------------------------------------------
# Config manipulation
# ---------------------------------------------------------------------------
def make_temp_config(base_config_path, overrides):
    """Load the base MIRAGE config, apply *overrides*, write to a temp file,
    and return the path to the temp config file."""
    with open(base_config_path, "r") as f:
        cfg = json.load(f)
    cfg.update(overrides)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="mirage_ablation_", delete=False
    )
    json.dump(cfg, tmp, indent=2)
    tmp.close()
    return tmp.name


def variant_label(axis_name, axis_info, value):
    """Human-readable label for an ablation variant."""
    if "labels" in axis_info:
        idx = axis_info["values"].index(value)
        return f"{axis_name}={axis_info['labels'][idx]}"
    return f"{axis_name}={value}"


# ---------------------------------------------------------------------------
# Evaluation of a single variant
# ---------------------------------------------------------------------------
def evaluate_variant(watermark, prompts, seed_values, attacks):
    """Run embed + detect for all prompts x seeds x attacks.

    Returns a dict mapping attack_name -> metrics dict with:
        wm_detect_rate, unwm_fp_rate,
        wm_score_summary, unwm_score_summary
    """
    attack_flags = {n: {"wm": [], "unwm": []} for n in attacks}
    attack_scores = {n: {"wm": [], "unwm": []} for n in attacks}

    total = len(prompts) * len(seed_values)
    for pi, prompt in enumerate(prompts):
        for si, seed in enumerate(seed_values):
            idx = pi * len(seed_values) + si + 1
            print(f"      [{idx}/{total}] prompt={pi}, seed={seed}")

            reseed_watermark(watermark, seed)
            wm_img = watermark.generate_watermarked_media(prompt)
            reseed_watermark(watermark, seed)
            unwm_img = watermark.generate_unwatermarked_media(prompt)

            for aname, attack in attacks.items():
                if attack is None:
                    # Clean -- no distortion
                    test_wm = wm_img
                    test_unwm = unwm_img
                else:
                    test_wm = attack.edit(wm_img)
                    test_unwm = attack.edit(unwm_img)

                wm_det = watermark.detect_watermark_in_media(test_wm, prompt=prompt)
                unwm_det = watermark.detect_watermark_in_media(test_unwm, prompt=prompt)

                attack_flags[aname]["wm"].append(bool(wm_det["is_watermarked"]))
                attack_flags[aname]["unwm"].append(bool(unwm_det["is_watermarked"]))
                attack_scores[aname]["wm"].append(
                    float(wm_det.get("presence_score", wm_det["raw_full_match_rate"]))
                )
                attack_scores[aname]["unwm"].append(
                    float(unwm_det.get("presence_score", unwm_det["raw_full_match_rate"]))
                )

    results = {}
    for aname in attacks:
        results[aname] = {
            "wm_detect_rate": safe_mean(
                [1.0 if f else 0.0 for f in attack_flags[aname]["wm"]]
            ),
            "unwm_fp_rate": safe_mean(
                [1.0 if f else 0.0 for f in attack_flags[aname]["unwm"]]
            ),
            "wm_score_summary": summarize(attack_scores[aname]["wm"]),
            "unwm_score_summary": summarize(attack_scores[aname]["unwm"]),
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="MIRAGE ablation: block count, block dim, erasure, selection strategy"
    )
    parser.add_argument("--algorithm", default="MIRAGE")
    parser.add_argument("--model", default="huanzi05/stable-diffusion-2-1-base")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--config", default="config/MIRAGE_presence.json")
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--output", default="qi_mirage_ablation.json")
    parser.add_argument("--csv-output", default="qi_mirage_ablation.csv")
    args = parser.parse_args()

    device, dtype = resolve_qi_device(args.gpu)
    pipe, scheduler = build_cached_pipe(args.model, device, dtype)

    seed_values = resolve_seed_values(args.seed_start, args.num_seeds, args.seeds)
    prompts = list(PROMPTS)
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    attacks = instantiate_attacks()

    print("=== MIRAGE Ablation Study ===")
    print(f"  prompts={len(prompts)}, seeds={len(seed_values)}, "
          f"samples_per_variant={len(prompts)*len(seed_values)}")
    print(f"  attacks={list(attacks.keys())}")
    print(f"  axes: {list(ABLATION_AXES.keys())}")

    all_results = {}  # variant_label -> {attack_name -> metrics}

    for axis_name, axis_info in ABLATION_AXES.items():
        for value in axis_info["values"]:
            vlabel = variant_label(axis_name, axis_info, value)
            print(f"\n--- Variant: {vlabel} ---")

            overrides = {axis_info["config_key"]: value}
            tmp_config = make_temp_config(args.config, overrides)

            try:
                diffusion_config = DiffusionConfig(
                    scheduler=scheduler,
                    pipe=pipe,
                    device=device,
                    image_size=(args.size, args.size),
                    num_inference_steps=args.steps,
                    num_inversion_steps=args.steps,
                    guidance_scale=7.5,
                    gen_seed=args.seed_start,
                    inversion_type="ddim",
                    dtype=dtype,
                )
                watermark = AutoWatermark.load(
                    args.algorithm,
                    algorithm_config=tmp_config,
                    diffusion_config=diffusion_config,
                )
                variant_metrics = evaluate_variant(
                    watermark, prompts, seed_values, attacks
                )
                all_results[vlabel] = variant_metrics

                # Print quick summary for this variant
                for aname, m in variant_metrics.items():
                    tpr = m["wm_detect_rate"]
                    fpr = m["unwm_fp_rate"]
                    smean = m["wm_score_summary"]["mean"] if m["wm_score_summary"] else None
                    print(f"    {aname:25s}  TPR={tpr:.4f}  FPR={fpr:.4f}  score_mean={smean}")
            finally:
                Path(tmp_config).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Write JSON output
    # ------------------------------------------------------------------
    output_data = {
        "algorithm": args.algorithm,
        "model": args.model,
        "device": device,
        "steps": args.steps,
        "size": args.size,
        "base_config": args.config,
        "num_prompts": len(prompts),
        "num_seeds": len(seed_values),
        "num_samples_per_variant": len(prompts) * len(seed_values),
        "seeds": seed_values,
        "prompts": prompts,
        "ablation_axes": {
            axis_name: {
                "config_key": info["config_key"],
                "values": info["values"],
                "default": info["default"],
            }
            for axis_name, info in ABLATION_AXES.items()
        },
        "variants": to_jsonable(all_results),
    }
    Path(args.output).write_text(
        json.dumps(to_jsonable(output_data), indent=2, ensure_ascii=False)
    )

    # ------------------------------------------------------------------
    # Write CSV output
    # ------------------------------------------------------------------
    attack_names = list(attacks.keys())
    with Path(args.csv_output).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "variant", "attack",
            "TPR", "FPR",
            "wm_score_mean", "wm_score_min",
            "unwm_score_mean", "unwm_score_max",
        ])
        for vlabel, variant_metrics in all_results.items():
            for aname in attack_names:
                m = variant_metrics[aname]
                ws = m["wm_score_summary"]
                us = m["unwm_score_summary"]
                writer.writerow([
                    vlabel,
                    aname,
                    f"{m['wm_detect_rate']:.4f}",
                    f"{m['unwm_fp_rate']:.4f}",
                    f"{ws['mean']:.4f}" if ws else "",
                    f"{ws['min']:.4f}" if ws else "",
                    f"{us['mean']:.4f}" if us else "",
                    f"{us['max']:.4f}" if us else "",
                ])

    print(f"\nResults written to {args.output} and {args.csv_output}")


if __name__ == "__main__":
    main()

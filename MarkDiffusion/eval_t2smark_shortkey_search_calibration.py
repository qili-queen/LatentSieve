"""Blind-search calibration for T2SMark's default 16-bit regime.

This experiment extends the earlier positive-only brute-force study by also
measuring the max-over-keys score distribution on unwatermarked images under
the current attack family used in the REK-BW paper.

Outputs:
  - rank-1 success and score separation on watermarked images
  - max-over-keys score distribution on unwatermarked images
  - empirical 0%-FPR threshold and resulting TPR on watermarked images
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean

import torch
from PIL import Image
from tqdm import tqdm


T2SMARK_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "T2SMark"
)
T2SMARK_ROOT = os.path.normpath(T2SMARK_ROOT)
if T2SMARK_ROOT not in sys.path:
    sys.path.insert(0, T2SMARK_ROOT)

MARKDIFFUSION_ROOT = os.path.dirname(os.path.abspath(__file__))
if MARKDIFFUSION_ROOT not in sys.path:
    sys.path.insert(0, MARKDIFFUSION_ROOT)

from src.t2s import T2SMark  # noqa: E402
import src.utils as t2s_utils  # noqa: E402

from utils.media_utils import get_media_latents, transform_to_model_format  # noqa: E402
from evaluation.tools.image_editor import (  # noqa: E402
    GaussianBlurring,
    GaussianNoise,
    JPEGCompression,
)


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


TAU = 0.674
KEY_LENGTH = 16
MSG_LENGTH = 256
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 50
NUM_INVERSION_STEPS = 50
KEY_CHANNEL_IDX = 0
MODEL_KEY = "stabilityai/stable-diffusion-2-1-base"

ATTACKS = {
    "Clean": lambda: None,
    "JPEG-10": lambda: JPEGCompression(quality=10),
    "Noise-50": lambda: GaussianNoise(sigma=50.0),
    "Blur-6": lambda: GaussianBlurring(radius=6),
}


def to_jsonable(value):
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def summarize(values):
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return None
    t = torch.tensor(cleaned, dtype=torch.float32)
    return {
        "count": len(cleaned),
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
    }


def aggregate_results(per_attack, prompt_count, seeds_per_prompt):
    aggregated = {}
    for attack_name, buckets in per_attack.items():
        wm = buckets["watermarked"]
        unwm = buckets["unwatermarked"]
        if not wm or not unwm:
            continue

        max_unwm = max(x["best_l1"] for x in unwm)
        tpr_at_empirical_0fpr = mean(
            [1.0 if x["best_l1"] > max_unwm else 0.0 for x in wm]
        )
        rank1_success = mean([1.0 if x["correct_rank"] == 1 else 0.0 for x in wm])

        aggregated[attack_name] = {
            "samples": len(wm),
            "rank1_success_rate": float(rank1_success),
            "empirical_threshold_0fpr": float(max_unwm),
            "tpr_at_empirical_0fpr": float(tpr_at_empirical_0fpr),
            "watermarked_best_l1_summary": summarize([x["best_l1"] for x in wm]),
            "watermarked_correct_l1_summary": summarize([x["correct_l1"] for x in wm]),
            "watermarked_best_wrong_l1_summary": summarize([x["best_wrong_l1"] for x in wm]),
            "watermarked_l1_gap_summary": summarize([x["l1_gap"] for x in wm]),
            "unwatermarked_best_l1_summary": summarize([x["best_l1"] for x in unwm]),
            "avg_wm_bruteforce_time_sec": float(mean([x["bruteforce_time_sec"] for x in wm])),
            "avg_unwm_bruteforce_time_sec": float(mean([x["bruteforce_time_sec"] for x in unwm])),
        }

    return {
        "metadata": {
            "key_length": KEY_LENGTH,
            "prompt_count": prompt_count,
            "seeds_per_prompt": seeds_per_prompt,
            "total_samples_per_attack": prompt_count * seeds_per_prompt,
        },
        "aggregated": aggregated,
        "per_attack": per_attack,
    }


def save_checkpoint(output_path: Path, per_attack, prompt_count, seeds_per_prompt):
    payload = aggregate_results(per_attack, prompt_count, seeds_per_prompt)
    payload["metadata"]["checkpoint_complete"] = False
    output_path.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False))


def load_checkpoint(output_path: Path):
    if not output_path.exists():
        return None
    try:
        with output_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or "per_attack" not in payload:
            return None
        return payload
    except Exception:
        return None


def apply_attack(image: Image.Image, attack_editor):
    if attack_editor is None:
        return image
    return attack_editor.edit(image)


def int_to_binary_tensor(val, length, device):
    bits = []
    for i in range(length - 1, -1, -1):
        bits.append((val >> i) & 1)
    return torch.tensor(bits, dtype=torch.long, device=device)


def main():
    parser = argparse.ArgumentParser(description="Blind-search calibration for T2SMark 16-bit.")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--prompt-count", type=int, default=10)
    parser.add_argument("--seeds-per-prompt", type=int, default=1)
    parser.add_argument(
        "--output",
        type=str,
        default="t2smark_shortkey_search_calibration.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing partial JSON output if present",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from diffusers import StableDiffusionPipeline, DDIMScheduler
    from inversions.ddim_inversion import DDIMInversion

    print("Loading StableDiffusionPipeline ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_KEY, torch_dtype=torch.float16
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    inversion = DDIMInversion(pipe.scheduler, pipe.unet, device)

    def encode_prompt_for_inv(prompt_text):
        do_cfg = GUIDANCE_SCALE > 1.0
        prompt_embeds, neg_embeds = pipe.encode_prompt(
            prompt=prompt_text,
            device=device,
            do_classifier_free_guidance=do_cfg,
            num_images_per_prompt=1,
        )
        return torch.cat([neg_embeds, prompt_embeds]) if do_cfg else prompt_embeds

    def brute_force_l1(reversed_key_ch, correct_key_int=None):
        l1_norms = []
        for candidate_int in range(2 ** KEY_LENGTH):
            candidate_key = int_to_binary_tensor(candidate_int, KEY_LENGTH, device)
            _, l1 = t2s_key.decode(reversed_key_ch.float(), candidate_key, detection=True)
            l1_norms.append(l1)

        best_idx, best_l1 = max(enumerate(l1_norms), key=lambda x: x[1])
        result = {
            "best_key_int": int(best_idx),
            "best_l1": float(best_l1),
            "mean_l1": float(mean(l1_norms)),
        }
        if correct_key_int is not None:
            sorted_l1 = sorted(enumerate(l1_norms), key=lambda x: -x[1])
            correct_l1 = l1_norms[correct_key_int]
            correct_rank = None
            best_wrong_l1 = None
            for rank_idx, (cand_idx, l1_val) in enumerate(sorted_l1):
                if cand_idx == correct_key_int and correct_rank is None:
                    correct_rank = rank_idx + 1
                if cand_idx != correct_key_int and best_wrong_l1 is None:
                    best_wrong_l1 = l1_val
                if correct_rank is not None and best_wrong_l1 is not None:
                    break
            result.update({
                "correct_key_int": int(correct_key_int),
                "correct_l1": float(correct_l1),
                "correct_rank": int(correct_rank),
                "best_wrong_l1": float(best_wrong_l1),
                "l1_gap": float(correct_l1 - best_wrong_l1),
            })
        return result

    t2s_key = T2SMark(m=KEY_LENGTH, tau=TAU, latent_shape=(1, 64, 64))
    t2s_msg = T2SMark(m=MSG_LENGTH, tau=TAU, latent_shape=(3, 64, 64))
    msg_channel_idx = [i for i in range(4) if i != KEY_CHANNEL_IDX]

    output_path = Path(args.output)
    results = {
        "experiment": "t2smark_shortkey_search_calibration",
        "key_length": KEY_LENGTH,
        "prompt_count": args.prompt_count,
        "seeds_per_prompt": args.seeds_per_prompt,
        "attacks": list(ATTACKS.keys()),
        "per_attack": {name: {"watermarked": [], "unwatermarked": []} for name in ATTACKS},
    }

    completed_pairs = set()
    if args.resume:
        checkpoint = load_checkpoint(output_path)
        if checkpoint is not None:
            for attack_name in ATTACKS:
                if attack_name in checkpoint.get("per_attack", {}):
                    results["per_attack"][attack_name]["watermarked"] = checkpoint["per_attack"][attack_name].get("watermarked", [])
                    results["per_attack"][attack_name]["unwatermarked"] = checkpoint["per_attack"][attack_name].get("unwatermarked", [])
            for attack_name, buckets in results["per_attack"].items():
                wm_pairs = {(x["prompt_idx"], x["seed_idx"]) for x in buckets["watermarked"]}
                unwm_pairs = {(x["prompt_idx"], x["seed_idx"]) for x in buckets["unwatermarked"]}
                for pair in wm_pairs & unwm_pairs:
                    completed_pairs.add((attack_name, pair[0], pair[1]))
            print(f"Loaded checkpoint with {len(completed_pairs)} completed attack-sample pairs from {output_path}")

    total = args.prompt_count * args.seeds_per_prompt
    sample_idx = 0
    for pi in range(args.prompt_count):
        prompt = PROMPTS[pi % len(PROMPTS)]
        for sj in range(args.seeds_per_prompt):
            base_seed = 1600000 + pi * 1000 + sj
            sample_idx += 1
            print(f"[{sample_idx}/{total}] prompt={pi}, seed={sj}")

            t2s_utils.set_random_seed(base_seed)
            master_key = torch.randint(0, 2, (KEY_LENGTH,)).to(device)
            msg = torch.randint(0, 2, (MSG_LENGTH,)).to(device)
            key = torch.randint(0, 2, (KEY_LENGTH,)).to(device)

            master_key_int = 0
            for b in master_key:
                master_key_int = master_key_int * 2 + b.item()

            z_k = t2s_key.encode(key, master_key)
            z_b = t2s_msg.encode(msg, key)
            initial_latents_wm = torch.zeros(1, 4, 64, 64).to(device)
            initial_latents_wm[:, KEY_CHANNEL_IDX, :, :] = z_k
            initial_latents_wm[:, msg_channel_idx, :, :] = z_b

            wm_output = pipe(
                prompt,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=NUM_INFERENCE_STEPS,
                height=512,
                width=512,
                latents=initial_latents_wm.half(),
            )
            wm_image = wm_output.images[0]

            t2s_utils.set_random_seed(base_seed)
            initial_latents_unwm = torch.randn(1, 4, 64, 64).to(device)
            unwm_output = pipe(
                prompt,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=NUM_INFERENCE_STEPS,
                height=512,
                width=512,
                latents=initial_latents_unwm.half(),
            )
            unwm_image = unwm_output.images[0]

            text_emb = encode_prompt_for_inv(prompt)

            for attack_name, attack_factory in ATTACKS.items():
                if (attack_name, pi, sj) in completed_pairs:
                    print(f"  {attack_name:<8} already complete; skipping")
                    continue

                attack_editor = attack_factory()

                attacked_wm = apply_attack(wm_image, attack_editor)
                wm_tensor = transform_to_model_format(attacked_wm, target_size=512).unsqueeze(0)
                wm_tensor = wm_tensor.to(device, dtype=torch.float16)
                wm_latents = get_media_latents(pipe=pipe, media=wm_tensor, sample=False, decoder_inv=False)
                wm_reversed = inversion.forward_diffusion(
                    latents=wm_latents,
                    text_embeddings=text_emb,
                    guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=NUM_INVERSION_STEPS,
                )[-1]
                wm_reversed_key_ch = wm_reversed[0, KEY_CHANNEL_IDX, :, :]

                attacked_unwm = apply_attack(unwm_image, attack_editor)
                unwm_tensor = transform_to_model_format(attacked_unwm, target_size=512).unsqueeze(0)
                unwm_tensor = unwm_tensor.to(device, dtype=torch.float16)
                unwm_latents = get_media_latents(pipe=pipe, media=unwm_tensor, sample=False, decoder_inv=False)
                unwm_reversed = inversion.forward_diffusion(
                    latents=unwm_latents,
                    text_embeddings=text_emb,
                    guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=NUM_INVERSION_STEPS,
                )[-1]
                unwm_reversed_key_ch = unwm_reversed[0, KEY_CHANNEL_IDX, :, :]

                t_start = time.time()
                wm_result = brute_force_l1(wm_reversed_key_ch, correct_key_int=master_key_int)
                wm_result["bruteforce_time_sec"] = time.time() - t_start
                wm_result.update({"prompt_idx": pi, "seed_idx": sj, "base_seed": base_seed})
                results["per_attack"][attack_name]["watermarked"].append(wm_result)

                t_start = time.time()
                unwm_result = brute_force_l1(unwm_reversed_key_ch, correct_key_int=None)
                unwm_result["bruteforce_time_sec"] = time.time() - t_start
                unwm_result.update({"prompt_idx": pi, "seed_idx": sj, "base_seed": base_seed})
                results["per_attack"][attack_name]["unwatermarked"].append(unwm_result)

                save_checkpoint(output_path, results["per_attack"], args.prompt_count, args.seeds_per_prompt)

                print(
                    f"  {attack_name:<8} wm_rank={wm_result['correct_rank']} "
                    f"wm_gap={wm_result['l1_gap']:.1f} "
                    f"unwm_max={unwm_result['best_l1']:.1f}"
                )

    output = aggregate_results(results["per_attack"], args.prompt_count, args.seeds_per_prompt)
    output["metadata"]["checkpoint_complete"] = True
    output_path.write_text(json.dumps(to_jsonable(output), indent=2, ensure_ascii=False))
    print(f"\nResults written to {args.output}")
    for attack_name, agg in output["aggregated"].items():
        print(
            f"{attack_name:<8} rank1={agg['rank1_success_rate']:.3f} "
            f"TPR@0FPR={agg['tpr_at_empirical_0fpr']:.3f} "
            f"thr={agg['empirical_threshold_0fpr']:.1f}"
        )


if __name__ == "__main__":
    main()

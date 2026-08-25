"""Runtime performance measurement for MIRAGE, PRC, and GS watermarking.

Measures:
  1. Calibration time (one-time setup cost)
  2. Embedding time per image (sign-flip / codeword generation only, excluding diffusion)
  3. Generation time (with and without watermark)
  4. Detection time per image (detector.eval_watermark only, excluding inversion)
  5. Inversion time per image (DDIM inversion only)
  6. Total end-to-end time (embed + generate + invert + detect)

All timings over N images; reports mean +/- std.
"""
import argparse
import time
from statistics import mean, stdev

import torch
from PIL import Image

from qi_runtime_utils import build_cached_pipe, configure_qi_runtime, resolve_qi_device
from utils.diffusion_config import DiffusionConfig
from utils.media_utils import get_media_latents, get_random_latents, transform_to_model_format
from utils.utils import set_random_seed
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


def reseed_watermark(watermark, seed):
    """Reset seed and init_latents on the watermark config (same as robustness script)."""
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


def fmt(vals):
    """Format a list of floats as mean +/- std."""
    if len(vals) < 2:
        return f"{vals[0]:.4f}s"
    m = mean(vals)
    s = stdev(vals)
    return f"{m:.4f} +/- {s:.4f}s"


# ---------------------------------------------------------------------------
# Inversion helper (shared by all algorithms -- mirrors _detect logic)
# ---------------------------------------------------------------------------
def do_inversion(watermark, image, prompt):
    """Run DDIM inversion only, return reversed_latents."""
    cfg = watermark.config
    guidance = cfg.guidance_scale
    do_cfg = guidance > 1.0
    prompt_embeds, neg_embeds = cfg.pipe.encode_prompt(
        prompt=prompt,
        device=cfg.device,
        do_classifier_free_guidance=do_cfg,
        num_images_per_prompt=1,
    )
    text_embeddings = torch.cat([neg_embeds, prompt_embeds]) if do_cfg else prompt_embeds

    image_tensor = transform_to_model_format(
        image, target_size=cfg.image_size[0],
    ).unsqueeze(0).to(cfg.device, dtype=text_embeddings.dtype)
    image_latents = get_media_latents(
        pipe=cfg.pipe, media=image_tensor, sample=False, decoder_inv=False,
    )
    reversed_latents = cfg.inversion.forward_diffusion(
        latents=image_latents,
        text_embeddings=text_embeddings,
        guidance_scale=guidance,
        num_inference_steps=cfg.num_inference_steps,
    )[-1]
    return reversed_latents.detach()


def do_detect_only(watermark, reversed_latents):
    """Run only the detector on already-inverted latents."""
    return watermark.detector.eval_watermark(reversed_latents)


# ---------------------------------------------------------------------------
# Main measurement
# ---------------------------------------------------------------------------
def measure_algorithm(
    algorithm: str,
    config_path: str,
    model: str,
    device: str,
    dtype: torch.dtype,
    pipe,
    scheduler,
    num_images: int,
    steps: int,
    size: int,
):
    print(f"\n{'='*60}")
    print(f"  Measuring {algorithm}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Calibration time (includes model init + calibration cache build)
    # ------------------------------------------------------------------
    print(f"  [1/6] Calibration (AutoWatermark.load) ...")
    torch.cuda.synchronize() if "cuda" in device else None
    t0 = time.perf_counter()
    diffusion_config = DiffusionConfig(
        scheduler=scheduler, pipe=pipe, device=device,
        image_size=(size, size),
        num_inference_steps=steps, num_inversion_steps=steps,
        guidance_scale=7.5, gen_seed=42, inversion_type="ddim", dtype=dtype,
    )
    watermark = AutoWatermark.load(
        algorithm,
        algorithm_config=config_path,
        diffusion_config=diffusion_config,
    )
    torch.cuda.synchronize() if "cuda" in device else None
    calibration_time = time.perf_counter() - t0
    print(f"        calibration_time = {calibration_time:.4f}s")

    # Collect per-image timings
    embed_times = []
    gen_wm_times = []
    gen_nowm_times = []
    inversion_times = []
    detect_only_times = []
    e2e_times = []

    for i in range(num_images):
        prompt = PROMPTS[i % len(PROMPTS)]
        seed = 42 + i
        print(f"  [{i+1}/{num_images}] prompt={i % len(PROMPTS)}, seed={seed}")

        reseed_watermark(watermark, seed)

        # ------------------------------------------------------------------
        # 2. Embed time (sign-flip / codeword only, no diffusion)
        # ------------------------------------------------------------------
        torch.cuda.synchronize() if "cuda" in device else None
        t0 = time.perf_counter()
        wm_latents = watermark.utils.inject_watermark()
        torch.cuda.synchronize() if "cuda" in device else None
        embed_time = time.perf_counter() - t0
        embed_times.append(embed_time)

        # ------------------------------------------------------------------
        # 3a. Generation with watermark
        # ------------------------------------------------------------------
        set_random_seed(seed)
        gen_params = {
            "num_images_per_prompt": 1,
            "guidance_scale": watermark.config.guidance_scale,
            "num_inference_steps": watermark.config.num_inference_steps,
            "height": watermark.config.image_size[0],
            "width": watermark.config.image_size[1],
            "latents": wm_latents,
        }
        torch.cuda.synchronize() if "cuda" in device else None
        t0 = time.perf_counter()
        wm_img = watermark.config.pipe(prompt, **gen_params).images[0]
        torch.cuda.synchronize() if "cuda" in device else None
        gen_wm_time = time.perf_counter() - t0
        gen_wm_times.append(gen_wm_time)

        # ------------------------------------------------------------------
        # 3b. Generation without watermark (same seed, original latents)
        # ------------------------------------------------------------------
        reseed_watermark(watermark, seed)
        set_random_seed(seed)
        gen_params_nowm = {
            "num_images_per_prompt": 1,
            "guidance_scale": watermark.config.guidance_scale,
            "num_inference_steps": watermark.config.num_inference_steps,
            "height": watermark.config.image_size[0],
            "width": watermark.config.image_size[1],
            "latents": watermark.config.init_latents,
        }
        torch.cuda.synchronize() if "cuda" in device else None
        t0 = time.perf_counter()
        nowm_img = watermark.config.pipe(prompt, **gen_params_nowm).images[0]
        torch.cuda.synchronize() if "cuda" in device else None
        gen_nowm_time = time.perf_counter() - t0
        gen_nowm_times.append(gen_nowm_time)

        # ------------------------------------------------------------------
        # 5. Inversion time
        # ------------------------------------------------------------------
        torch.cuda.synchronize() if "cuda" in device else None
        t0 = time.perf_counter()
        reversed_latents = do_inversion(watermark, wm_img, prompt)
        torch.cuda.synchronize() if "cuda" in device else None
        inv_time = time.perf_counter() - t0
        inversion_times.append(inv_time)

        # ------------------------------------------------------------------
        # 4. Detection time (detector only, no inversion)
        # ------------------------------------------------------------------
        torch.cuda.synchronize() if "cuda" in device else None
        t0 = time.perf_counter()
        det_result = do_detect_only(watermark, reversed_latents)
        torch.cuda.synchronize() if "cuda" in device else None
        det_time = time.perf_counter() - t0
        detect_only_times.append(det_time)

        # ------------------------------------------------------------------
        # 6. End-to-end: embed + generate + invert + detect
        # ------------------------------------------------------------------
        e2e = embed_time + gen_wm_time + inv_time + det_time
        e2e_times.append(e2e)

        print(f"        embed={embed_time:.4f}s  gen_wm={gen_wm_time:.4f}s  "
              f"gen_nowm={gen_nowm_time:.4f}s  inversion={inv_time:.4f}s  "
              f"detect={det_time:.6f}s  e2e={e2e:.4f}s")

    results = {
        "algorithm": algorithm,
        "calibration_time": calibration_time,
        "embed_time": embed_times,
        "gen_wm_time": gen_wm_times,
        "gen_nowm_time": gen_nowm_times,
        "inversion_time": inversion_times,
        "detect_only_time": detect_only_times,
        "e2e_time": e2e_times,
    }

    print(f"\n  --- {algorithm} Summary ({num_images} images) ---")
    print(f"  Calibration (one-time):       {calibration_time:.4f}s")
    print(f"  Embed per image:              {fmt(embed_times)}")
    print(f"  Generation (with WM):         {fmt(gen_wm_times)}")
    print(f"  Generation (no WM):           {fmt(gen_nowm_times)}")
    print(f"  Inversion per image:          {fmt(inversion_times)}")
    print(f"  Detection per image:          {fmt(detect_only_times)}")
    print(f"  End-to-end (embed+gen+inv+det): {fmt(e2e_times)}")
    return results


def print_comparison_table(all_results):
    """Print a nicely formatted comparison table."""
    print("\n" + "=" * 100)
    print("  RUNTIME COMPARISON TABLE (seconds, mean +/- std)")
    print("=" * 100)
    header = f"{'Metric':<32}"
    for r in all_results:
        header += f" {'  ' + r['algorithm']:<22}"
    print(header)
    print("-" * 100)

    metrics = [
        ("Calibration (one-time)", "calibration_time"),
        ("Embed per image", "embed_time"),
        ("Generation (with WM)", "gen_wm_time"),
        ("Generation (no WM)", "gen_nowm_time"),
        ("DDIM inversion per image", "inversion_time"),
        ("Detection per image", "detect_only_time"),
        ("End-to-end per image", "e2e_time"),
    ]
    for label, key in metrics:
        row = f"{label:<32}"
        for r in all_results:
            v = r[key]
            if isinstance(v, list):
                row += f"  {fmt(v):<22}"
            else:
                row += f"  {v:<22.4f}"
        print(row)
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Measure runtime of watermarking algorithms")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--model", default="huanzi05/stable-diffusion-2-1-base")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--algorithms", nargs="+", default=["MIRAGE", "PRC", "GS"],
                        help="Algorithms to benchmark")
    args = parser.parse_args()

    device, dtype = resolve_qi_device(args.gpu)
    print(f"Device: {device}, dtype: {dtype}")
    print(f"Steps: {args.steps}, Size: {args.size}, Num images: {args.num_images}")

    # Load the shared pipeline once
    pipe, scheduler = build_cached_pipe(args.model, device, dtype)

    config_map = {
        "MIRAGE": "config/MIRAGE_presence.json",
        "PRC": "config/PRC.json",
        "GS": "config/GS.json",
    }

    all_results = []
    for alg in args.algorithms:
        if alg not in config_map:
            print(f"  WARNING: No config for {alg}, skipping")
            continue
        results = measure_algorithm(
            algorithm=alg,
            config_path=config_map[alg],
            model=args.model,
            device=device,
            dtype=dtype,
            pipe=pipe,
            scheduler=scheduler,
            num_images=args.num_images,
            steps=args.steps,
            size=args.size,
        )
        all_results.append(results)

    if len(all_results) > 1:
        print_comparison_table(all_results)

    print("\nDone.")


if __name__ == "__main__":
    main()

"""MIRAGE-on-FLUX N=500 SDP cross-backbone evaluation.

Sharded version of flux_mirage_sdp_n100.py for 3-GPU parallel execution.
Each shard processes a disjoint subset of the SDP prompt set with deterministic
per-image seeds, so shards never overlap.

Usage (3-GPU parallel):
  CUDA_VISIBLE_DEVICES=0 python flux_mirage_sdp_n500.py --offset 0   --n 167 --output flux_n500_off0.json
  CUDA_VISIBLE_DEVICES=1 python flux_mirage_sdp_n500.py --offset 167 --n 167 --output flux_n500_off167.json
  CUDA_VISIBLE_DEVICES=2 python flux_mirage_sdp_n500.py --offset 334 --n 166 --output flux_n500_off334.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift
from torchvision import transforms

MODELS = {
    "flux-dev": "black-forest-labs/FLUX.1-dev",
    "flux-schnell": "black-forest-labs/FLUX.1-schnell",
}

DEFAULTS = {
    "flux-dev": {"steps": 28, "guidance": 3.5},
    "flux-schnell": {"steps": 4, "guidance": 0.0},
}

LATENT_C = 16
BLOCK_DIM = 64
ERASURE_RATIO = 32 / 256


def make_sign_pattern(n_coords, seed):
    g = torch.Generator().manual_seed(seed)
    bits = (torch.rand(n_coords, generator=g) > 0.5).float()
    return bits * 2 - 1


def make_watermarked_initial_latent(sign_pattern, shape, magnitude_seed):
    g = torch.Generator().manual_seed(magnitude_seed)
    n = int(np.prod(shape))
    mag = torch.abs(torch.randn(n, generator=g))
    z = sign_pattern * mag
    return z.reshape(shape)


def pack_latents(latents):
    bs, c, h, w = latents.shape
    latents = latents.view(bs, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(bs, (h // 2) * (w // 2), c * 4)
    return latents


def unpack_latents(packed, h, w):
    bs, _, _ = packed.shape
    packed = packed.view(bs, h // 2, w // 2, LATENT_C, 2, 2)
    packed = packed.permute(0, 3, 1, 4, 2, 5)
    return packed.reshape(bs, LATENT_C, h, w)


@torch.no_grad()
def flux_invert_image(pipe, image_pil, prompt, num_steps, size, device, dtype,
                      guidance=3.5):
    img_tensor = (transforms.ToTensor()(image_pil.convert("RGB")) * 2 - 1
                   ).unsqueeze(0).to(device, dtype=dtype)
    img_tensor = transforms.functional.resize(img_tensor, [size, size],
                                              antialias=True)
    latents = pipe.vae.encode(img_tensor).latent_dist.sample()
    latents = ((latents - pipe.vae.config.shift_factor)
               * pipe.vae.config.scaling_factor)

    (prompt_embeds, pooled_prompt_embeds,
     text_ids) = pipe.encode_prompt(
        prompt=prompt, prompt_2=prompt, device=device,
        num_images_per_prompt=1, max_sequence_length=512,
    )
    h, w = latents.shape[-2], latents.shape[-1]
    packed = pack_latents(latents)
    latent_image_ids = pipe._prepare_latent_image_ids(
        1, h // 2, w // 2, device, dtype,
    )

    image_seq_len = packed.shape[1]
    if getattr(pipe.scheduler.config, "use_dynamic_shifting", False):
        mu = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.15),
        )
        pipe.scheduler.set_timesteps(num_steps, device=device, mu=mu)
    else:
        pipe.scheduler.set_timesteps(num_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    sigmas = pipe.scheduler.sigmas

    z = packed.clone()
    guidance_t = torch.tensor([guidance], device=device, dtype=dtype)

    rev_sigmas = list(reversed(sigmas.tolist()))
    rev_timesteps = list(reversed(timesteps.tolist()))
    for i in range(len(rev_timesteps) - 1):
        t_cur = rev_timesteps[i]
        sigma_cur = rev_sigmas[i]
        sigma_next = rev_sigmas[i + 1]
        t_batch = torch.tensor([t_cur / 1000.0], device=device, dtype=dtype)
        noise_pred = pipe.transformer(
            hidden_states=z, timestep=t_batch,
            guidance=guidance_t,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids, img_ids=latent_image_ids,
            joint_attention_kwargs=None, return_dict=False,
        )[0]
        z = z + (sigma_next - sigma_cur) * noise_pred

    recovered = unpack_latents(z, h, w)
    return recovered.cpu()


@torch.no_grad()
def flux_generate_with_latent(pipe, prompt, init_latent, num_steps, size,
                              device, dtype, guidance=3.5):
    h, w = init_latent.shape[-2], init_latent.shape[-1]
    packed = pack_latents(init_latent.to(device, dtype=dtype))
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        height=h * 8, width=w * 8,
        latents=packed,
        output_type="pil",
    ).images[0]
    return image


def apply_attack(image_pil, attack):
    if attack == "clean":
        return image_pil
    if attack == "jpeg75":
        import io
        buf = io.BytesIO()
        image_pil.save(buf, format="JPEG", quality=75)
        return Image.open(buf).convert("RGB")
    if attack == "blur2":
        return image_pil.filter(ImageFilter.GaussianBlur(radius=2))
    if attack == "noise15":
        arr = np.array(image_pil).astype(np.float32)
        noise = np.random.normal(0, 15.0, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    raise ValueError(f"unknown attack: {attack}")


def detect_per_block_sign(z_rec, sign_pattern, block_dim=BLOCK_DIM,
                           erasure_ratio=ERASURE_RATIO):
    z_flat = z_rec.flatten().to(torch.float32)
    sp_flat = sign_pattern.to(torch.float32)
    n_blocks = z_flat.numel() // block_dim
    z_blocks = z_flat[: n_blocks * block_dim].reshape(n_blocks, block_dim)
    sp_blocks = sp_flat[: n_blocks * block_dim].reshape(n_blocks, block_dim)
    block_corr = (z_blocks * sp_blocks).sum(dim=1)
    block_match = (block_corr > 0).float()
    block_conf = torch.abs(block_corr)
    n_erase = int(erasure_ratio * n_blocks)
    if n_erase > 0:
        keep_idx = torch.argsort(block_conf, descending=True)[: n_blocks - n_erase]
        kept_match = block_match[keep_idx]
    else:
        kept_match = block_match
    ba = float(kept_match.mean().item())
    return ba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", type=int, default=0,
                        help="Starting index into SDP prompt list")
    parser.add_argument("--n", type=int, default=167,
                        help="Number of images to process from offset")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--model", default="flux-dev",
                        choices=list(MODELS.keys()))
    parser.add_argument("--attacks", default="clean,jpeg75,blur2,noise15")
    parser.add_argument("--threshold", type=float, default=0.67)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    args = parser.parse_args()

    steps = args.steps or DEFAULTS[args.model]["steps"]
    guidance = (args.guidance if args.guidance is not None
                else DEFAULTS[args.model]["guidance"])
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"[offset={args.offset}, n={args.n}] Loading {MODELS[args.model]}...",
          flush=True)
    pipe = FluxPipeline.from_pretrained(
        MODELS[args.model], torch_dtype=dtype, local_files_only=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    print(f"  Loaded. steps={steps} guidance={guidance} size={args.size}",
          flush=True)

    from sdp_n500_config import PROMPTS as ALL_PROMPTS
    # Global image index global_idx in [args.offset, args.offset+args.n) maps to
    # (prompt_idx = global_idx // 5, seed_idx = global_idx % 5) so 100 prompts x
    # 5 seeds = 500 unique images, consistent with the SDP protocol used
    # elsewhere in this paper.
    SEEDS_PER_PROMPT = 5
    assert len(ALL_PROMPTS) * SEEDS_PER_PROMPT >= args.offset + args.n, \
        f"need >= {args.offset + args.n} images but have {len(ALL_PROMPTS)} x {SEEDS_PER_PROMPT}"

    latent_hw = args.size // 8
    n_coords = LATENT_C * latent_hw * latent_hw
    print(f"  Latent: 1 x {LATENT_C} x {latent_hw} x {latent_hw} = {n_coords}",
          flush=True)

    attacks = [a.strip() for a in args.attacks.split(",")]
    results = {a: [] for a in attacks}
    done_idx = set()

    # Resume support
    if os.path.exists(args.output):
        try:
            prev = json.load(open(args.output))
            results = prev.get("results", results)
            for a in attacks:
                done_idx.update(r["idx"] for r in results.get(a, []))
            print(f"  Resumed: {len(done_idx)} indices already done", flush=True)
        except Exception as e:
            print(f"  Resume failed: {e}; starting fresh", flush=True)

    t_start = time.time()
    for i in range(args.n):
        global_idx = args.offset + i
        if global_idx in done_idx:
            continue
        prompt_idx = global_idx // SEEDS_PER_PROMPT
        seed_idx = global_idx % SEEDS_PER_PROMPT
        prompt = ALL_PROMPTS[prompt_idx]
        sign_pattern = make_sign_pattern(n_coords, seed=10000 + global_idx)
        init_latent = make_watermarked_initial_latent(
            sign_pattern, (1, LATENT_C, latent_hw, latent_hw),
            magnitude_seed=20000 + global_idx,
        )

        try:
            gen_img = flux_generate_with_latent(
                pipe, prompt, init_latent, steps, args.size, device, dtype,
                guidance=guidance,
            )
        except Exception as e:
            print(f"  [skip] idx={global_idx} gen-err: {e}", flush=True)
            continue

        for atk in attacks:
            try:
                attacked = apply_attack(gen_img, atk)
                z_rec = flux_invert_image(
                    pipe, attacked, prompt, steps, args.size, device, dtype,
                    guidance=guidance,
                )
                ba = detect_per_block_sign(z_rec, sign_pattern)
                is_wm = bool(ba > args.threshold)
                results[atk].append({
                    "idx": global_idx, "prompt": prompt, "ba": ba, "is_wm": is_wm,
                })
            except Exception as e:
                print(f"  [skip] idx={global_idx} {atk} err: {e}", flush=True)

        elapsed = (time.time() - t_start) / 60
        print(f"  [{i+1}/{args.n}] idx={global_idx} done  ({elapsed:.1f}min)",
              flush=True)

        if (i + 1) % args.checkpoint_every == 0:
            summary = {}
            for a in attacks:
                recs = results[a]
                if recs:
                    summary[a] = {
                        "n": len(recs),
                        "tpr": float(np.mean([r["is_wm"] for r in recs])),
                        "ba_mean": float(np.mean([r["ba"] for r in recs])),
                        "ba_min": float(min(r["ba"] for r in recs)),
                    }
            with open(args.output, "w") as f:
                json.dump({
                    "model": args.model, "steps": steps, "size": args.size,
                    "offset": args.offset, "n": args.n,
                    "results": results, "summary": summary,
                }, f, indent=2, default=str)

    # Final save
    summary = {}
    for a in attacks:
        recs = results[a]
        if recs:
            summary[a] = {
                "n": len(recs),
                "tpr": float(np.mean([r["is_wm"] for r in recs])),
                "ba_mean": float(np.mean([r["ba"] for r in recs])),
                "ba_min": float(min(r["ba"] for r in recs)),
            }
    with open(args.output, "w") as f:
        json.dump({
            "model": args.model, "steps": steps, "size": args.size,
            "offset": args.offset, "n": args.n,
            "results": results, "summary": summary,
        }, f, indent=2, default=str)

    print(f"\n=== FLUX shard offset={args.offset} done ===", flush=True)
    for a in attacks:
        recs = results[a]
        if recs:
            print(f"  {a:<10}  n={len(recs)}  TPR={100*np.mean([r['is_wm'] for r in recs]):.1f}%  "
                  f"BA_mean={np.mean([r['ba'] for r in recs]):.4f}", flush=True)


if __name__ == "__main__":
    main()

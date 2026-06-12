"""SDXL cross-backbone transfer for MIRAGE-CCS (Channel A only).

CCS MIRAGE construction:
  - Embedding: z'_T[j] = |z_T[j]| * s[j], where s = PRF(K)-derived ±1 pattern
  - Detection: per-coord sign-match BA against key-derived s
  - SDXL latent: 4×128×128 = 65,536 coords (vs 4×64×64=16,384 for SD-2.1)

Steps:
  1. Generate n watermarked images with sign-flipped z_T
  2. DDIM-invert each attacked image (clean / JPEG / blur / noise)
  3. Compute per-coord BSMR_A against ground-truth sign pattern
  4. Report null (unwatermarked) BSMR_A for FPR calibration

Usage:
  CUDA_VISIBLE_DEVICES=0 python sdxl_mirage.py --gpu 0 --n 30 --attacks clean,jpeg75,noise15,blur2
"""
import argparse, functools, json, sys, time, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import (
    StableDiffusionXLPipeline, DDIMScheduler, DDIMInverseScheduler,
    AutoencoderKL,
)
from tqdm import tqdm
oti = tqdm.__init__
@functools.wraps(oti)
def _st(s, *a, **k): k['disable']=True; oti(s, *a, **k)
tqdm.__init__ = _st

ROOT = Path(__file__).resolve().parent
MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
SIZE = 1024
LATENT_H = LATENT_W = SIZE // 8  # 128
N_CHANNELS = 4
N_COORDS = N_CHANNELS * LATENT_H * LATENT_W  # 65,536
STEPS = 30
GUIDANCE = 5.0
NEG_PROMPT = ""

PROMPTS = [
    "A beautiful sunset over the ocean with golden clouds",
    "A cinematic mountain lake at dawn with pine trees and fog",
    "Portrait of an elderly fisherman with a weathered face",
    "A rainy Tokyo street at night with neon reflections",
    "A red vintage car parked in a desert under blue sky",
    "A cozy independent bookstore interior with warm lamps",
    "A snow-covered cabin in a dense pine forest at twilight",
    "A macro photograph of a bee on a bright sunflower",
    "An aerial view of a coastal city at sunrise with haze",
    "A futuristic laboratory with glass walls and holograms",
]


def derive_sign_pattern(key_seed: int, n_coords: int, device, dtype) -> torch.Tensor:
    """Derive per-coord ±1 sign pattern from a key seed via SHA256-CTR."""
    key_bytes = key_seed.to_bytes(32, byteorder="big", signed=False)
    # Generate ceil(n/8) bytes deterministically
    n_bytes = (n_coords + 7) // 8
    out = bytearray()
    ctr = 0
    while len(out) < n_bytes:
        h = hashlib.sha256(key_bytes + ctr.to_bytes(8, byteorder="big")).digest()
        out.extend(h)
        ctr += 1
    bits = np.unpackbits(np.frombuffer(bytes(out[:n_bytes]), dtype=np.uint8))[:n_coords]
    signs = 2.0 * bits.astype(np.float32) - 1.0
    return torch.from_numpy(signs).to(device, dtype).reshape(N_CHANNELS, LATENT_H, LATENT_W)


@torch.no_grad()
def encode_prompts_sdxl(pipe, prompt, device, batch_size=1):
    """Returns (prompt_embeds [B,77,2048], pooled [B,1280], neg_embeds, neg_pooled)."""
    out = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, device=device, num_images_per_prompt=1,
        do_classifier_free_guidance=True, negative_prompt=NEG_PROMPT,
        negative_prompt_2=None,
    )
    pe, ne, pp, np_ = out
    return pe, ne, pp, np_


@torch.no_grad()
def construct_watermarked_zT(sign_pattern, device, dtype):
    """z'_T = |G| * sign_pattern, marginal N(0, I)."""
    G = torch.randn((1, N_CHANNELS, LATENT_H, LATENT_W), device=device, dtype=torch.float32)
    z = torch.abs(G) * sign_pattern.float().unsqueeze(0)
    return z.to(dtype)


@torch.no_grad()
def construct_unwm_zT(device, dtype):
    return torch.randn((1, N_CHANNELS, LATENT_H, LATENT_W), device=device, dtype=dtype)


@torch.no_grad()
def generate_latent(pipe, latents, prompt, device, dtype):
    """Use SDXL with provided initial latent, return final z_0 (NOT PIL).
    Avoids the pipe's internal VAE decode (which is fp16/fp32-buggy on SDXL)."""
    out = pipe(
        prompt=prompt, prompt_2=None, latents=latents,
        num_inference_steps=STEPS, guidance_scale=GUIDANCE,
        negative_prompt=NEG_PROMPT, output_type="latent",
        original_size=(SIZE, SIZE), target_size=(SIZE, SIZE),
        crops_coords_top_left=(0, 0),
    )
    # out.images is the latent z_0 in fp16 with shape [1, 4, 128, 128]
    return out.images


@torch.no_grad()
def decode_latent_to_pil(pipe, z_0):
    """Decode z_0 → PIL via VAE in fp32 (SDXL VAE is unstable in fp16)."""
    z = (z_0.to(torch.float32) / pipe.vae.config.scaling_factor)
    x = pipe.vae.decode(z).sample  # fp32, in [-1, 1] range
    x = (x / 2 + 0.5).clamp(0, 1)
    arr = (x[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


@torch.no_grad()
def encode_image_vae(pipe, image, device, dtype):
    """Image (PIL) → z_0 latent via VAE in fp32 (SDXL VAE numerically unstable in fp16)."""
    if isinstance(image, Image.Image):
        image = image.resize((SIZE, SIZE))
        arr = np.array(image).astype(np.float32) / 127.5 - 1.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, torch.float32)
    else:
        x = image.to(torch.float32)
    z = pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor
    return z.to(dtype)


@torch.no_grad()
def ddim_invert_sdxl(pipe, z_0, prompt, device, dtype):
    """DDIM-invert z_0 → z_T using SDXL with provided prompt for conditioning."""
    pe, ne, pp, npp = encode_prompts_sdxl(pipe, prompt, device)
    # For inversion we use the empty/positive prompt (no CFG)
    text_embeds = pp  # pooled prompt
    add_time_ids = torch.tensor(
        [[SIZE, SIZE, 0, 0, SIZE, SIZE]], device=device, dtype=dtype
    )
    inv_sched = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    inv_sched.set_timesteps(STEPS, device=device)
    z = z_0.clone().to(dtype)
    for t in inv_sched.timesteps:
        noise_pred = pipe.unet(
            z, t,
            encoder_hidden_states=pe.to(dtype),
            added_cond_kwargs={"text_embeds": text_embeds.to(dtype),
                                "time_ids":    add_time_ids},
        ).sample
        z = inv_sched.step(noise_pred, t, z).prev_sample
    return z.to(torch.float32)


def apply_attack(img: Image.Image, attack: str) -> Image.Image:
    """Apply a standard image-space attack."""
    if attack == "clean":
        return img
    if attack.startswith("jpeg"):
        q = int(attack[4:])
        from io import BytesIO
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).copy()
    if attack.startswith("noise"):
        sigma = float(attack[5:])
        arr = np.array(img).astype(np.float32)
        arr = arr + np.random.normal(0, sigma, arr.shape)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if attack.startswith("blur"):
        from PIL import ImageFilter
        radius = float(attack[4:])
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    raise ValueError(f"unknown attack: {attack}")


BLOCK_DIM = 64  # MIRAGE-CCS default block dim
ERASURE = 128   # 1/8 of 1024 blocks (proportional to CCS's 32/256)


def mirage_BA(z_inv: torch.Tensor, sign_pattern: torch.Tensor,
              block_dim: int = BLOCK_DIM,
              erasure: int = ERASURE) -> tuple[float, float]:
    """MIRAGE per-block bit-accuracy with low-confidence erasure (no
    reliability weighting since SDXL is uncalibrated).

    Returns (BA, BA_no_erasure).
    BA: uniform-weighted bit accuracy on the highest-confidence blocks.
    BA_no_erasure: uniform-weighted bit accuracy on all blocks."""
    s_inv = torch.sign(z_inv.float()).view(-1)
    s_ref = sign_pattern.float().view(-1)
    n = s_inv.numel()
    assert n % block_dim == 0, f"latent {n} not divisible by block_dim {block_dim}"
    n_blocks = n // block_dim
    # Per-block correlation S_i = (1/d) sum_j sign(z_j) * s_j
    prod = (s_inv * s_ref).view(n_blocks, block_dim)
    S = prod.mean(dim=1)  # in [-1, +1]
    confidence = S.abs()
    matches = (S > 0).float()
    BA_no_erasure = float(matches.mean().item())
    if erasure > 0 and erasure < n_blocks:
        # Drop the `erasure` lowest-confidence blocks
        thresh = torch.topk(confidence, n_blocks - erasure, largest=True).values.min()
        keep = confidence >= thresh
        BA = float(matches[keep].mean().item())
    else:
        BA = BA_no_erasure
    return BA, BA_no_erasure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--attacks", type=str,
                    default="clean,jpeg75,noise15,blur2")
    ap.add_argument("--key-seed", type=int, default=42)
    ap.add_argument("--out", type=str,
                    default="/overnight_logs/sdxl_mirage_results.json")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    attacks = args.attacks.split(",")

    print(f"[sdxl] device={device} latent={N_CHANNELS}x{LATENT_H}x{LATENT_W}={N_COORDS} "
          f"n={args.n} attacks={attacks}", flush=True)

    print(f"[sdxl] loading {MODEL} ...", flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL, torch_dtype=dtype, use_safetensors=True,
        local_files_only=True,
    ).to(device)
    # SDXL VAE is numerically unstable in fp16 → upcast to fp32 (HF recommended fix)
    pipe.vae.to(torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    try: pipe.enable_xformers_memory_efficient_attention()
    except Exception: pass
    pipe.set_progress_bar_config(disable=True)
    print(f"[sdxl] pipeline loaded ({sum(p.numel() for p in pipe.unet.parameters())/1e9:.2f}B unet params)",
          flush=True)

    sign_pattern = derive_sign_pattern(args.key_seed, N_COORDS, device, dtype)
    print(f"[sdxl] sign pattern derived from key_seed={args.key_seed}", flush=True)

    # === Generate watermarked images + invert + attack ===
    results = {atk: {"wm": [], "unwm": []} for atk in attacks}
    out_dir = ROOT / "/overnight_logs" / "sdxl_mirage_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        prompt = PROMPTS[i % len(PROMPTS)]
        # === WATERMARKED: generate to LATENT (not PIL) ===
        t0 = time.time()
        z_T_wm = construct_watermarked_zT(sign_pattern, device, dtype)
        z0_wm_clean = generate_latent(pipe, z_T_wm, prompt, device, dtype)
        gen_t = time.time() - t0

        # === UNWATERMARKED (null) ===
        t1 = time.time()
        z_T_un = construct_unwm_zT(device, dtype)
        z0_un_clean = generate_latent(pipe, z_T_un, prompt, device, dtype)
        gen_un_t = time.time() - t1

        # Decode to PIL for attacks + visual check (in fp32)
        img_wm = decode_latent_to_pil(pipe, z0_wm_clean)
        img_un = decode_latent_to_pil(pipe, z0_un_clean)
        if i < 2:
            img_wm.save(out_dir / f"wm_{i:03d}.png")
            img_un.save(out_dir / f"un_{i:03d}.png")

        # === ATTACK + INVERT + BSMR (per attack) ===
        for atk in attacks:
            ti = time.time()
            if atk == "clean":
                # Skip VAE round-trip; invert the generated z_0 directly
                z0_wm = z0_wm_clean
                z0_un = z0_un_clean
            else:
                atk_wm = apply_attack(img_wm, atk)
                atk_un = apply_attack(img_un, atk)
                z0_wm = encode_image_vae(pipe, atk_wm, device, dtype)
                z0_un = encode_image_vae(pipe, atk_un, device, dtype)
            zT_wm = ddim_invert_sdxl(pipe, z0_wm, prompt, device, dtype)
            zT_un = ddim_invert_sdxl(pipe, z0_un, prompt, device, dtype)
            ba_wm, ba_wm_no_er = mirage_BA(zT_wm, sign_pattern)
            ba_un, ba_un_no_er = mirage_BA(zT_un, sign_pattern)
            inv_t = time.time() - ti
            results[atk]["wm"].append({"i": i, "ba": ba_wm, "ba_no_er": ba_wm_no_er,
                                         "prompt_idx": i % len(PROMPTS)})
            results[atk]["unwm"].append({"i": i, "ba": ba_un, "ba_no_er": ba_un_no_er,
                                          "prompt_idx": i % len(PROMPTS)})
            print(f"[sdxl {i+1}/{args.n} {atk:10s}] BA_wm={ba_wm:.4f} BA_unwm={ba_un:.4f} "
                  f"(no-er wm={ba_wm_no_er:.4f}/un={ba_un_no_er:.4f}) "
                  f"(gen {gen_t:.0f}+{gen_un_t:.0f}s, attack+invert {inv_t:.0f}s)",
                  flush=True)

    # === Summary ===
    summary = {}
    for atk in attacks:
        wm = np.array([r["ba"] for r in results[atk]["wm"]])
        un = np.array([r["ba"] for r in results[atk]["unwm"]])
        # MIRAGE uses 0.67 threshold (CCS paper). For SDXL with more coords, recalibrate.
        tau = 0.67  # nominal CCS threshold; may need re-cal for SDXL
        tpr = float((wm > tau).mean())
        fpr = float((un > tau).mean())
        summary[atk] = {
            "n": len(wm),
            "BA_wm_mean":  float(wm.mean()),  "BA_wm_std": float(wm.std()),
            "BA_wm_min":   float(wm.min()),
            "BA_unwm_mean":float(un.mean()),  "BA_unwm_std": float(un.std()),
            "BA_unwm_max": float(un.max()),
            "TPR@0.67": tpr, "FPR@0.67": fpr,
            "gap": float(wm.mean() - un.mean()),
        }
        print(f"\n[{atk:10s}] wm: {wm.mean():.4f}±{wm.std():.4f} min={wm.min():.4f} | "
              f"un: {un.mean():.4f}±{un.std():.4f} max={un.max():.4f} | "
              f"TPR={tpr:.4f} FPR={fpr:.4f}", flush=True)

    out_path = ROOT / args.out
    out_path.parent.mkdir(exist_ok=True)
    json.dump({"summary": summary, "per_image": results,
                "config": {"n": args.n, "attacks": attacks, "key_seed": args.key_seed,
                            "size": SIZE, "latent_h": LATENT_H, "steps": STEPS}},
               open(out_path, "w"), indent=2)
    print(f"\n[sdxl] saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()

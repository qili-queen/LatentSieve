"""PRC sign-correlation BA under SHIFT high-lambda.

For each PRC-watermarked image, applies MIRAGE-style sign-correlation
decoder (block correlation + per-image low-confidence erasure) to PRC's
codeword. This is the cross-scheme counterfactual extended to high SHIFT λ
— answers "what is PRC's true sign-level BA under SHIFT high-lambda?"

Output: per-image (idx, lambda, ba, detected at tau=0.67).

Run (3-GPU parallel, shard by global image idx):
  CUDA_VISIBLE_DEVICES=0 python eval_prc_sign_ba_shift_high.py \
      --gpu 0 --offset 0   --n 167 \
      --out eval_n500_sdp/prc_signba_high_off0.json
  CUDA_VISIBLE_DEVICES=1 python eval_prc_sign_ba_shift_high.py \
      --gpu 0 --offset 167 --n 167 \
      --out eval_n500_sdp/prc_signba_high_off167.json
  CUDA_VISIBLE_DEVICES=2 python eval_prc_sign_ba_shift_high.py \
      --gpu 0 --offset 334 --n 166 \
      --out eval_n500_sdp/prc_signba_high_off334.json
"""

import argparse, functools, json, os, sys, time
import numpy as np, torch
from PIL import Image
from diffusers import DDIMScheduler

oti = __import__("tqdm").tqdm.__init__
@functools.wraps(oti)
def st(s, *a, **k): k['disable'] = True; oti(s, *a, **k)
__import__("tqdm").tqdm.__init__ = st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runtime_utils import build_cached_pipe, configure_runtime, resolve_device
from utils.diffusion_config import DiffusionConfig
from utils.media_utils import (
    get_media_latents, get_random_latents, transform_to_model_format,
)
from watermark.auto_watermark import AutoWatermark
from sdp_n500_config import PROMPTS

configure_runtime()

MODEL = "stabilityai/stable-diffusion-2-1-base"
STEPS = 50
GS_SCALE = 7.5
SIZE = 512
BLOCK_DIM = 64
ERASE_K = 32
TAU = 0.67
SEEDS_PER_PROMPT = 5


def reseed_full(wm, seed):
    """Reseed PRC pseudogaussian_seed so each (prompt, seed) gives distinct codeword."""
    wm.config.gen_seed = seed
    wm.config.init_latents_seed = seed
    wm.config.pseudogaussian_seed = 49 + seed
    r = torch.Generator(device=wm.config.device).manual_seed(seed)
    wm.config.init_latents = get_random_latents(
        wm.config.pipe, height=SIZE, width=SIZE, generator=r
    ).to(wm.config.device, dtype=wm.config.dtype)


def encode_prompt(pipe, prompt, device):
    pe, ne = pipe.encode_prompt(
        prompt=prompt, device=device,
        do_classifier_free_guidance=True, num_images_per_prompt=1,
    )
    return torch.cat([ne, pe])


def vae_encode(pipe, image, device, dtype):
    t = transform_to_model_format(image, target_size=SIZE).unsqueeze(0).to(
        device, dtype=dtype)
    return get_media_latents(pipe=pipe, media=t, sample=False, decoder_inv=False)


def vae_decode(pipe, latents):
    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor,
                             return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    arr = (image[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


@torch.no_grad()
def shift_attack(pipe, image, text_emb, lam, device, dtype):
    """SHIFT trajectory-deflection attack at deflection strength lam."""
    z0 = vae_encode(pipe, image, device, dtype)
    sched = DDIMScheduler.from_pretrained(MODEL, subfolder="scheduler",
                                            local_files_only=True)
    t_l = int(lam * 1000)
    at = sched.alphas_cumprod[min(t_l, 999)]
    zt = torch.sqrt(at) * z0 + torch.sqrt(1 - at) * torch.randn_like(z0)
    pipe.scheduler = DDIMScheduler.from_pretrained(MODEL, subfolder="scheduler",
                                                     local_files_only=True)
    pipe.scheduler.set_timesteps(STEPS)
    ts = pipe.scheduler.timesteps
    si = next(i for i, t in enumerate(ts) if t.item() <= t_l)
    la = zt
    for t in ts[si:]:
        li = torch.cat([la] * 2)
        li = pipe.scheduler.scale_model_input(li, t)
        np_ = pipe.unet(li, t, encoder_hidden_states=text_emb).sample
        nu, nt = np_.chunk(2)
        ng = nu + GS_SCALE * (nt - nu)
        apt = pipe.scheduler.alphas_cumprod[t]
        pt = t - 1000 // STEPS
        ap = pipe.scheduler.alphas_cumprod[max(pt, 0)] if pt >= 0 else torch.tensor(1.0, device=device)
        x0p = (la - torch.sqrt(1 - apt) * ng) / torch.sqrt(apt)
        sig = 0.5 * torch.sqrt((1 - ap) / (1 - apt) * (1 - apt / ap)).clamp(min=0)
        pd = torch.sqrt(torch.clamp(1 - ap - sig**2, min=0)) * ng
        la = torch.sqrt(ap) * x0p + pd + sig * torch.randn_like(la)
    return vae_decode(pipe, la)


@torch.no_grad()
def invert_image(wm, image, prompt):
    """DDIM-invert image back to initial latent z_T."""
    do_cfg = wm.config.guidance_scale > 1.0
    pe, ne = wm.config.pipe.encode_prompt(
        prompt=prompt, device=wm.config.device,
        do_classifier_free_guidance=do_cfg, num_images_per_prompt=1,
    )
    text_emb = torch.cat([ne, pe]) if do_cfg else pe
    img_t = transform_to_model_format(
        image, target_size=wm.config.image_size[0]
    ).unsqueeze(0).to(wm.config.device, dtype=text_emb.dtype)
    img_lat = get_media_latents(
        pipe=wm.config.pipe, media=img_t, sample=False, decoder_inv=False,
    )
    rev = wm.config.inversion.forward_diffusion(
        latents=img_lat, text_embeddings=text_emb,
        guidance_scale=wm.config.guidance_scale,
        num_inference_steps=wm.config.num_inference_steps,
    )[-1]
    return rev.detach()


def compute_mirage_decoder_ba(z_rec, sign_pattern, block_dim=BLOCK_DIM,
                                erase_k=ERASE_K, tau=TAU):
    """MIRAGE-style decoder: per-block sign correlation + per-image
    low-confidence erasure, applied to ANY codeword's sign_pattern."""
    z_flat = z_rec.detach().to(torch.float32).flatten().cpu()
    sp_flat = sign_pattern.detach().to(torch.float32).flatten().cpu()
    n_total = min(z_flat.numel(), sp_flat.numel())
    n_blocks = n_total // block_dim
    z_blocks = z_flat[:n_blocks*block_dim].reshape(n_blocks, block_dim)
    s_blocks = sp_flat[:n_blocks*block_dim].reshape(n_blocks, block_dim)
    block_corr = (z_blocks * s_blocks).sum(dim=1)
    block_match = (block_corr > 0).float()
    block_conf = torch.abs(block_corr)
    if erase_k > 0 and n_blocks > erase_k:
        keep_idx = torch.argsort(block_conf, descending=True)[:n_blocks-erase_k]
        kept_match = block_match[keep_idx]
    else:
        kept_match = block_match
    ba = float(kept_match.mean().item())
    return ba, bool(ba > tau)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0,
                    help="Starting global image index")
    ap.add_argument("--n", type=int, default=167,
                    help="Number of images this shard processes")
    ap.add_argument("--lambdas", type=str, default="0.75,0.80,0.85")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    args = ap.parse_args()

    device, dtype = resolve_device(args.gpu)
    pipe, sch = build_cached_pipe(MODEL, device, dtype)
    pipe.set_progress_bar_config(disable=True)
    dc = DiffusionConfig(scheduler=sch, pipe=pipe, device=device,
                          image_size=(SIZE, SIZE), num_inference_steps=STEPS,
                          num_inversion_steps=STEPS, guidance_scale=GS_SCALE,
                          gen_seed=42, inversion_type="ddim", dtype=dtype)
    wm = AutoWatermark.load("PRC", algorithm_config="config/PRC.json",
                              diffusion_config=dc)

    lambdas = [float(x) for x in args.lambdas.split(",")]
    print(f"[off={args.offset} n={args.n}] PRC sign-BA SHIFT lambdas={lambdas}",
          flush=True)

    # Resume
    results = {f"{lam:.2f}": [] for lam in lambdas}
    done = {f"{lam:.2f}": set() for lam in lambdas}
    if os.path.exists(args.out):
        try:
            prev = json.load(open(args.out))
            results = {**results, **prev.get("results_per_lambda", {})}
            for lk in results:
                done[lk] = {r["idx"] for r in results[lk]}
            print(f"  resumed", flush=True)
        except Exception as e:
            print(f"  resume err: {e}", flush=True)

    t0 = time.time()
    for i in range(args.n):
        gidx = args.offset + i
        prompt_idx = gidx // SEEDS_PER_PROMPT
        seed_idx = gidx % SEEDS_PER_PROMPT
        if prompt_idx >= len(PROMPTS):
            print(f"  out of prompts at gidx={gidx}", flush=True)
            break
        prompt = PROMPTS[prompt_idx]
        text_emb = encode_prompt(pipe, prompt, device)

        # Each (prompt, seed) maps to a unique PRC codeword via pseudogaussian_seed
        seed = prompt_idx * 100 + seed_idx
        reseed_full(wm, seed)
        try:
            wm_img = wm.generate_watermarked_media(prompt)
            init_lat = wm.config.init_latents.clone().detach()
            sign_pattern = torch.sign(init_lat).to(torch.float32)
        except Exception as e:
            print(f"  [skip] gidx={gidx} gen err={e}", flush=True)
            continue

        for lam in lambdas:
            lk = f"{lam:.2f}"
            if gidx in done[lk]:
                continue
            try:
                atk_img = shift_attack(pipe, wm_img, text_emb, lam, device, dtype)
                z_rec = invert_image(wm, atk_img, prompt)
                ba, detected = compute_mirage_decoder_ba(z_rec, sign_pattern)
                results[lk].append({
                    "idx": gidx, "prompt_idx": prompt_idx, "seed_idx": seed_idx,
                    "lambda": lam, "ba": ba, "detected": detected,
                })
            except Exception as e:
                print(f"  [skip] gidx={gidx} lam={lam} err={e}", flush=True)

        if (i + 1) % args.checkpoint_every == 0:
            el = (time.time() - t0) / 60
            for lk in sorted(results.keys()):
                n = len(results[lk])
                if n > 0:
                    tpr = sum(1 for r in results[lk] if r["detected"]) / n
                    ba_mean = np.mean([r["ba"] for r in results[lk]])
                    print(f"  [{i+1}/{args.n}] lam={lk} n={n} "
                          f"TPR={tpr*100:.1f}% BA_mean={ba_mean:.4f} "
                          f"el={el:.1f}min", flush=True)
            summary = {lk: {"n": len(results[lk]),
                            "tpr": (sum(1 for r in results[lk] if r["detected"])
                                    / max(len(results[lk]), 1)),
                            "ba_mean": float(np.mean([r["ba"] for r in results[lk]])
                                              if results[lk] else 0)}
                       for lk in results}
            json.dump({"offset": args.offset, "n": args.n,
                        "results_per_lambda": results,
                        "summary": summary},
                      open(args.out, "w"), indent=2, default=str)

    # Final
    summary = {lk: {"n": len(results[lk]),
                    "tpr": (sum(1 for r in results[lk] if r["detected"])
                            / max(len(results[lk]), 1)),
                    "ba_mean": float(np.mean([r["ba"] for r in results[lk]])
                                      if results[lk] else 0)}
               for lk in results}
    json.dump({"offset": args.offset, "n": args.n,
                "results_per_lambda": results,
                "summary": summary},
              open(args.out, "w"), indent=2, default=str)

    print(f"\n=== PRC sign-BA SHIFT shard off={args.offset} done ===", flush=True)
    for lk in sorted(results.keys()):
        n = len(results[lk])
        if n > 0:
            tpr = sum(1 for r in results[lk] if r["detected"]) / n
            ba_mean = np.mean([r["ba"] for r in results[lk]])
            print(f"  lam={lk}: n={n} TPR={tpr*100:.1f}% BA_mean={ba_mean:.4f}",
                  flush=True)


if __name__ == "__main__":
    main()

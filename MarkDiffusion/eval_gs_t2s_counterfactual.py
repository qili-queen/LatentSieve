"""GS / T2SMark counterfactual: apply MIRAGE-style sign-correlation detector
to GS- and T2SMark-watermarked images under SHIFT lambda=0.7.

This tests whether the per-block correlation + per-image confidence erasure
decoder structure is the universal trick (in which case GS and T2SMark
counterfactuals should also recover) or whether MIRAGE's PRF embedding
co-design is also load-bearing.

Usage:
  CUDA_VISIBLE_DEVICES=0 python qi_eval_gs_t2s_counterfactual.py --scheme GS
  CUDA_VISIBLE_DEVICES=1 python qi_eval_gs_t2s_counterfactual.py --scheme T2SMark
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

from qi_runtime_utils import build_cached_pipe, configure_qi_runtime, resolve_qi_device
from utils.media_utils import (
    get_random_latents, get_media_latents, transform_to_model_format,
)

configure_qi_runtime()

MODEL = "huanzi05/stable-diffusion-2-1-base"
STEPS = 50
GS_SCALE = 7.5
SIZE = 512

from qi_sdp_n500_config import PROMPTS

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qi_eval_n500_sdp")


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


def encode_prompt(pipe, prompt, device):
    pe, ne = pipe.encode_prompt(prompt=prompt, device=device,
                                  do_classifier_free_guidance=True,
                                  num_images_per_prompt=1)
    return torch.cat([ne, pe])


@torch.no_grad()
def shift_attack(pipe, image, text_emb, lam, device, dtype):
    z0 = vae_encode(pipe, image, device, dtype)
    sched = DDIMScheduler.from_pretrained(MODEL, subfolder="scheduler", local_files_only=True)
    t_l = int(lam * 1000)
    at = sched.alphas_cumprod[min(t_l, 999)]
    zt = torch.sqrt(at) * z0 + torch.sqrt(1 - at) * torch.randn_like(z0)
    pipe.scheduler = DDIMScheduler.from_pretrained(MODEL, subfolder="scheduler", local_files_only=True)
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
def ddim_invert(pipe, image, text_emb, device, dtype):
    z0 = vae_encode(pipe, image, device, dtype)
    pipe.scheduler = DDIMScheduler.from_pretrained(MODEL, subfolder="scheduler", local_files_only=True)
    pipe.scheduler.set_timesteps(STEPS)
    ts = list(reversed(pipe.scheduler.timesteps.tolist()))
    la = z0
    for i in range(len(ts) - 1):
        t_cur, t_next = ts[i], ts[i + 1]
        li = torch.cat([la] * 2)
        li = pipe.scheduler.scale_model_input(li, t_cur)
        np_ = pipe.unet(li, t_cur, encoder_hidden_states=text_emb).sample
        nu, nt = np_.chunk(2)
        ng = nu + GS_SCALE * (nt - nu)
        apt = pipe.scheduler.alphas_cumprod[t_cur]
        ap = pipe.scheduler.alphas_cumprod[t_next]
        x0p = (la - torch.sqrt(1 - apt) * ng) / torch.sqrt(apt)
        la = torch.sqrt(ap) * x0p + torch.sqrt(1 - ap) * ng
    return la


def block_sign_correlation(z_rec, z_ref, block_dim=64, n_select=256, n_erase=32, threshold=0.67):
    """MIRAGE-style detection: per-block sign correlation + bottom-erase + threshold.
    z_rec: recovered latent (1D flat)
    z_ref: reference latent (sign pattern source — DDIM-inverted clean wm image)
    """
    z_rec_flat = z_rec.detach().flatten().to(torch.float32).cpu().numpy()
    z_ref_flat = z_ref.detach().flatten().to(torch.float32).cpu().numpy()
    n_total = z_rec_flat.size
    n_blocks_all = n_total // block_dim

    s_rec = np.sign(z_rec_flat[: n_blocks_all * block_dim]).reshape(n_blocks_all, block_dim)
    s_ref = np.sign(z_ref_flat[: n_blocks_all * block_dim]).reshape(n_blocks_all, block_dim)

    block_corr = (s_rec * s_ref).sum(axis=1)  # in [-block_dim, +block_dim]
    block_match = (block_corr > 0).astype(np.float32)
    block_conf = np.abs(block_corr)

    # Top n_select blocks by reference magnitude
    ref_mag = np.abs(z_ref_flat[: n_blocks_all * block_dim]).reshape(
        n_blocks_all, block_dim).mean(axis=1)
    sel_idx = np.argsort(-ref_mag)[:n_select]
    block_match_sel = block_match[sel_idx]
    block_conf_sel = block_conf[sel_idx]

    keep_idx = np.argsort(-block_conf_sel)[: n_select - n_erase]
    ba = block_match_sel[keep_idx].mean()
    return float(ba), bool(ba > threshold), float(block_corr.mean() / block_dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--scheme", required=True, choices=["GS", "T2SMark"])
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--num-images", type=int, default=500)
    args = ap.parse_args()

    device, dtype = resolve_qi_device(args.gpu)
    pipe, _ = build_cached_pipe(MODEL, device, dtype)
    pipe.set_progress_bar_config(disable=True)

    wm_dir = os.path.join(BASE, f"{args.scheme}_wm")
    out_path = os.path.join(BASE, f"{args.scheme.lower()}_counterfactual.json")

    results = []
    if os.path.exists(out_path):
        results = json.load(open(out_path)).get("results", [])
        print(f"resumed from {len(results)} records", flush=True)

    done = {r["idx"] for r in results}
    t0 = time.time()
    for i in range(args.num_images):
        if i in done:
            continue
        pi = i // 5
        prompt = PROMPTS[pi]
        img_path = os.path.join(wm_dir, f"img_{i:04d}.png")
        if not os.path.exists(img_path):
            continue
        wm_img = Image.open(img_path).convert("RGB")
        text_emb = encode_prompt(pipe, prompt, device)
        # Reference: invert clean watermarked image
        z_ref = ddim_invert(pipe, wm_img, text_emb, device, dtype)
        # SHIFT attack, then invert
        attacked = shift_attack(pipe, wm_img, text_emb, args.lam, device, dtype)
        z_rec = ddim_invert(pipe, attacked, text_emb, device, dtype)
        # MIRAGE-style detection
        ba, detected, mean_corr = block_sign_correlation(z_rec, z_ref)
        results.append({"idx": i, "prompt_idx": pi, "lambda": args.lam,
                          "ba": ba, "detected": detected, "mean_corr": mean_corr})

        if (i + 1) % 20 == 0:
            tpr = sum(r["detected"] for r in results) / len(results)
            el = (time.time() - t0) / 60
            print(f"  {args.scheme}-cf [{i+1}/{args.num_images}] "
                  f"TPR={tpr*100:.1f}%  el={el:.1f}min", flush=True)
            json.dump({"scheme": args.scheme, "lambda": args.lam, "results": results},
                      open(out_path, "w"), indent=2)

    json.dump({"scheme": args.scheme, "lambda": args.lam, "results": results},
              open(out_path, "w"), indent=2)
    n = len(results)
    tpr = sum(r["detected"] for r in results) / max(n, 1)
    ba_mean = np.mean([r["ba"] for r in results])
    corr_mean = np.mean([r["mean_corr"] for r in results])
    print(f"\n=== {args.scheme} counterfactual lambda={args.lam} ===")
    print(f"  n={n}  TPR={tpr*100:.1f}%  BA={ba_mean:.4f}  mean_corr={corr_mean:.4f}",
          flush=True)


if __name__ == "__main__":
    main()

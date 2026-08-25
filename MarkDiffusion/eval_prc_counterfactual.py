"""PRC counterfactual at N=500 SDP, λ=0.7: apply MIRAGE's sign-correlation
detector to PRC-watermarked images under SHIFT.

For each PRC-watermarked image at PRC_wm/:
  1. DDIM-invert the clean PRC_wm image to get reference sign pattern.
  2. Apply SHIFT(λ) attack to the PRC_wm image.
  3. DDIM-invert the SHIFT-attacked image to get test latent.
  4. Compute per-coordinate sign agreement vs reference.
  5. Apply MIRAGE-style threshold for detection.

Usage:
  CUDA_VISIBLE_DEVICES=2 python qi_eval_prc_counterfactual.py --gpu 0 --lambdas 0.7 --num-images 500
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
    get_media_latents, transform_to_model_format, get_random_latents,
)

configure_qi_runtime()

MODEL = "huanzi05/stable-diffusion-2-1-base"
STEPS = 50
GS_SCALE = 7.5
SIZE = 512

from qi_sdp_n500_config import PROMPTS

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qi_eval_n500_sdp")


def encode_prompt(pipe, prompt, device):
    pe, ne = pipe.encode_prompt(prompt=prompt, device=device,
                                  do_classifier_free_guidance=True,
                                  num_images_per_prompt=1)
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
def ddim_invert(pipe, image, text_emb, device, dtype):
    """Manual DDIM inversion to z_T."""
    z0 = vae_encode(pipe, image, device, dtype)
    pipe.scheduler = DDIMScheduler.from_pretrained(MODEL, subfolder="scheduler",
                                                     local_files_only=True)
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
    """MIRAGE-style detection on PRC latent: sign correlation vs reference."""
    z_rec_flat = z_rec.detach().flatten().to(torch.float32).cpu().numpy()
    z_ref_flat = z_ref.detach().flatten().to(torch.float32).cpu().numpy()
    n_total = z_rec_flat.size
    n_blocks_all = n_total // block_dim

    s_rec = np.sign(z_rec_flat[: n_blocks_all * block_dim]).reshape(n_blocks_all, block_dim)
    s_ref = np.sign(z_ref_flat[: n_blocks_all * block_dim]).reshape(n_blocks_all, block_dim)

    block_corr = (s_rec * s_ref).sum(axis=1)  # in [-block_dim, +block_dim]
    block_match = (block_corr > 0).astype(np.float32)
    block_conf = np.abs(block_corr)

    # Use top n_select blocks by reference magnitude (mimics MIRAGE block selection)
    ref_mag = np.abs(z_ref_flat[: n_blocks_all * block_dim]).reshape(
        n_blocks_all, block_dim).mean(axis=1)
    sel_idx = np.argsort(-ref_mag)[:n_select]
    block_match_sel = block_match[sel_idx]
    block_conf_sel = block_conf[sel_idx]

    # Erase n_erase lowest-confidence
    keep_idx = np.argsort(-block_conf_sel)[: n_select - n_erase]
    ba = block_match_sel[keep_idx].mean()
    return float(ba), bool(ba > threshold), float(block_corr.mean() / block_dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--lambdas", default="0.7", help="comma-separated")
    ap.add_argument("--num-images", type=int, default=500)
    ap.add_argument("--output", default="qi_eval_n500_sdp/prc_counterfactual.json")
    args = ap.parse_args()
    lambdas = [float(x) for x in args.lambdas.split(",")]

    device, dtype = resolve_qi_device(args.gpu)
    pipe, _ = build_cached_pipe(MODEL, device, dtype)
    pipe.scheduler = DDIMScheduler.from_pretrained(MODEL, subfolder="scheduler",
                                                     local_files_only=True)
    pipe.set_progress_bar_config(disable=True)

    wm_dir = os.path.join(BASE, "PRC_wm")

    results = {}
    if os.path.exists(args.output):
        results = json.load(open(args.output)).get("results", {})

    t0 = time.time()
    for lam in lambdas:
        lk = str(lam)
        results.setdefault(lk, [])
        done = {r["idx"] for r in results[lk]}

        # Pre-compute reference sign patterns from clean PRC_wm
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

            # Reference: invert clean PRC_wm
            z_ref = ddim_invert(pipe, wm_img, text_emb, device, dtype)

            # SHIFT-attacked PRC image, then invert
            attacked = shift_attack(pipe, wm_img, text_emb, lam, device, dtype)
            z_rec = ddim_invert(pipe, attacked, text_emb, device, dtype)

            # MIRAGE-style sign correlation
            ba, detected, mean_corr = block_sign_correlation(z_rec, z_ref)

            results[lk].append({
                "idx": i, "prompt_idx": pi, "lambda": lam,
                "ba": ba, "detected": detected, "mean_corr": mean_corr,
            })

            if (i + 1) % 20 == 0:
                tpr = sum(r["detected"] for r in results[lk]) / len(results[lk])
                el = (time.time() - t0) / 60
                print(f"  PRC-cf λ={lam} [{i+1}/{args.num_images}] "
                      f"TPR={tpr*100:.1f}%  mean_corr={np.mean([r['mean_corr'] for r in results[lk]]):.4f}  el={el:.1f}min",
                      flush=True)
                json.dump({"results": results}, open(args.output, "w"), indent=2)

        json.dump({"results": results}, open(args.output, "w"), indent=2)
        n = len(results[lk])
        tpr = sum(r["detected"] for r in results[lk]) / max(n, 1)
        ba_mean = np.mean([r["ba"] for r in results[lk]])
        corr_mean = np.mean([r["mean_corr"] for r in results[lk]])
        print(f"\n=== PRC counterfactual λ={lam} ===")
        print(f"  n={n}  TPR={tpr*100:.1f}%  BA={ba_mean:.4f}  mean per-coord sign agreement={corr_mean:.4f}",
              flush=True)


if __name__ == "__main__":
    main()

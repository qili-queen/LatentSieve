"""LPIPS for adaptive attack: proper diffusion-pipeline version.

For each (strategy, flip_fraction), runs full attack pipeline:
1. Generate watermarked image x' from MIRAGE init_latents
2. DDIM-invert x' to get z_rec
3. Flip coords by strategy
4. Regenerate via full diffusion (50 steps) + VAE decode -> x''
5. LPIPS(x', x'')

Run on 20 images, key strategies/fractions for sanity LPIPS column.
"""
import argparse, functools, json, os, sys, time
import numpy as np, torch
from PIL import Image

oti = __import__("tqdm").tqdm.__init__
@functools.wraps(oti)
def st(s, *a, **k): k['disable'] = True; oti(s, *a, **k)
__import__("tqdm").tqdm.__init__ = st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lpips
from qi_runtime_utils import build_cached_pipe, configure_qi_runtime, resolve_qi_device
from utils.diffusion_config import DiffusionConfig
from utils.media_utils import (
    get_random_latents, get_media_latents, transform_to_model_format,
)
from watermark.auto_watermark import AutoWatermark

configure_qi_runtime()

MODEL = "huanzi05/stable-diffusion-2-1-base"
STEPS = 50
GS_SCALE = 7.5
SIZE = 512

from qi_sdp_n500_config import PROMPTS

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qi_eval_n500_sdp")


def reseed(wm, seed):
    wm.config.gen_seed = seed
    wm.config.init_latents_seed = seed
    r = torch.Generator(device=wm.config.device).manual_seed(seed)
    wm.config.init_latents = get_random_latents(
        wm.config.pipe, height=SIZE, width=SIZE, generator=r
    ).to(wm.config.device, dtype=wm.config.dtype)


def invert_image(wm, image, prompt):
    """DDIM invert image -> z_rec, mirror of qi_eval_adaptive_attack."""
    img_t = transform_to_model_format(image, target_size=SIZE).unsqueeze(0).to(
        wm.config.device, dtype=wm.config.dtype)
    z0 = get_media_latents(pipe=wm.config.pipe, media=img_t, sample=False, decoder_inv=False)
    # Use diffusion config's invert (matches MIRAGE's detector pipeline)
    from diffusers import DDIMScheduler
    sch = DDIMScheduler.from_pretrained(MODEL, subfolder="scheduler", local_files_only=True)
    sch.set_timesteps(STEPS)
    ts = list(reversed(sch.timesteps.tolist()))
    pe, ne = wm.config.pipe.encode_prompt(prompt=prompt, device=wm.config.device,
                                            do_classifier_free_guidance=True,
                                            num_images_per_prompt=1)
    text_emb = torch.cat([ne, pe])
    la = z0
    with torch.no_grad():
        for i in range(len(ts) - 1):
            t_cur, t_next = ts[i], ts[i + 1]
            li = torch.cat([la] * 2)
            np_ = wm.config.pipe.unet(li, t_cur, encoder_hidden_states=text_emb).sample
            nu, nt_ = np_.chunk(2)
            ng = nu + GS_SCALE * (nt_ - nu)
            apt = sch.alphas_cumprod[t_cur]
            ap = sch.alphas_cumprod[t_next]
            x0p = (la - torch.sqrt(1 - apt) * ng) / torch.sqrt(apt)
            la = torch.sqrt(ap) * x0p + torch.sqrt(1 - ap) * ng
    return la


def regenerate(wm, latent, prompt):
    """Re-run diffusion pipeline from latent."""
    out = wm.config.pipe(prompt, latents=latent.to(dtype=wm.config.dtype),
                           guidance_scale=GS_SCALE, num_inference_steps=STEPS,
                           height=SIZE, width=SIZE)
    return out.images[0]


def flip_coords(z, strategy, frac, seed=0):
    z_flat = z.flatten()
    n = z_flat.numel()
    n_flip = int(frac * n)
    if n_flip == 0:
        return z.clone()
    if strategy == "random":
        g = torch.Generator(device=z.device).manual_seed(seed)
        idx = torch.randperm(n, generator=g, device=z.device)[:n_flip]
    elif strategy == "lowmag":
        _, idx = torch.topk(-torch.abs(z_flat), n_flip)
    elif strategy == "adaptive_high":
        _, idx = torch.topk(torch.abs(z_flat), n_flip)
    z_mod = z_flat.clone()
    z_mod[idx] = -z_mod[idx]
    return z_mod.reshape_as(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num-images", type=int, default=20)
    args = ap.parse_args()

    # Subset of fractions/strategies for table column (matches tab:adaptive)
    fractions = [0.05, 0.20, 0.50]
    strategies = ["adaptive_high", "random", "lowmag"]

    device, dtype = resolve_qi_device(args.gpu)
    pipe, sch = build_cached_pipe(MODEL, device, dtype)
    dc = DiffusionConfig(scheduler=sch, pipe=pipe, device=device,
                          image_size=(SIZE, SIZE), num_inference_steps=STEPS,
                          num_inversion_steps=STEPS, guidance_scale=GS_SCALE,
                          gen_seed=42, inversion_type="ddim", dtype=dtype)
    wm = AutoWatermark.load("MIRAGE", algorithm_config="config/MIRAGE_presence.json",
                              diffusion_config=dc)

    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    def to_lpips_tensor(img):
        a = np.array(img).astype(np.float32) / 127.5 - 1.0
        return torch.tensor(a).permute(2, 0, 1).unsqueeze(0).to(device)

    wm_dir = os.path.join(BASE, "MIRAGE_wm")
    results = {}
    t0 = time.time()
    for i in range(args.num_images):
        pi = i // 5
        si = i % 5
        seed = pi * 100 + si
        prompt = PROMPTS[pi]
        img_path = os.path.join(wm_dir, f"img_{i:04d}.png")
        if not os.path.exists(img_path):
            continue
        wm_img = Image.open(img_path).convert("RGB")
        # Invert -> z_rec
        reseed(wm, seed)
        z_rec = invert_image(wm, wm_img, prompt)
        for strategy in strategies:
            for f in fractions:
                z_atk = flip_coords(z_rec, strategy, f, seed=seed)
                atk_img = regenerate(wm, z_atk, prompt)
                # LPIPS
                with torch.no_grad():
                    t1 = to_lpips_tensor(wm_img)
                    t2 = to_lpips_tensor(atk_img)
                    d = lpips_fn(t1, t2).item()
                key = f"{strategy}_{f:.2f}"
                results.setdefault(key, []).append({"idx": i, "lpips": float(d)})
        el = (time.time() - t0) / 60
        print(f"  [{i+1}/{args.num_images}] el={el:.1f}min", flush=True)
        # Save partial
        summary = {k: {"mean": float(np.mean([r["lpips"] for r in v])),
                        "std": float(np.std([r["lpips"] for r in v])),
                        "n": len(v)} for k, v in results.items()}
        json.dump({"results": results, "summary": summary},
                  open("sp2027_proposal/adaptive_lpips_v2.json", "w"), indent=2)

    print("\n=== Final LPIPS ===")
    for k in sorted(results.keys()):
        v = results[k]
        m = np.mean([r["lpips"] for r in v])
        s = np.std([r["lpips"] for r in v])
        print(f"  {k}: LPIPS = {m:.3f} +/- {s:.3f}  n={len(v)}")


if __name__ == "__main__":
    main()

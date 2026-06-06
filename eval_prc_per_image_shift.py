"""PRC per-image-varied SHIFT eval.

Address reviewer W6 concern: original PRC SDP run has 500 records corresponding
to 100 unique watermarked images (PRC uses fixed pseudogaussian_seed per key).
Here we vary pseudogaussian_seed per (prompt, seed) pair so 500 records =
500 truly distinct watermarked images, then run SHIFT lambda=0.7 + PRC native
detection.

Run: CUDA_VISIBLE_DEVICES=0 python eval_prc_per_image_shift.py
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
from utils.media_utils import get_random_latents, get_media_latents, transform_to_model_format
from watermark.auto_watermark import AutoWatermark

configure_runtime()

MODEL = "stabilityai/stable-diffusion-2-1-base"
STEPS = 50
GS_SCALE = 7.5
SIZE = 512

from sdp_n500_config import PROMPTS

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "eval_n500_sdp")


def reseed_full(wm, seed):
    """Reseed both gen_seed AND PRC's pseudogaussian_seed so latent differs."""
    wm.config.gen_seed = seed
    wm.config.init_latents_seed = seed
    # Vary PRC's internal pseudogaussian seed to get distinct codeword latents
    wm.config.pseudogaussian_seed = 49 + seed  # base 49 + offset
    r = torch.Generator(device=wm.config.device).manual_seed(seed)
    wm.config.init_latents = get_random_latents(
        wm.config.pipe, height=SIZE, width=SIZE, generator=r
    ).to(wm.config.device, dtype=wm.config.dtype)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num-prompts", type=int, default=100)
    ap.add_argument("--seeds-per-prompt", type=int, default=5)
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--out",
                    default="eval_n500_sdp/prc_per_image_shift_lam07.json")
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

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    results = []
    done_idx = set()
    if os.path.exists(out_path):
        prev = json.load(open(out_path))
        results = prev.get("results", [])
        done_idx = {r["idx"] for r in results}
        print(f"  resumed from {len(results)} records", flush=True)

    total = args.num_prompts * args.seeds_per_prompt
    t0 = time.time()

    for pi in range(args.num_prompts):
        prompt = PROMPTS[pi]
        text_emb = encode_prompt(pipe, prompt, device)
        for si in range(args.seeds_per_prompt):
            idx = pi * args.seeds_per_prompt + si
            if idx in done_idx:
                continue
            seed = pi * 100 + si
            # 1) Generate watermarked image with varied pseudogaussian seed
            reseed_full(wm, seed)
            try:
                wm_img = wm.generate_watermarked_media(prompt)
            except Exception as e:
                print(f"  [skip] idx={idx} gen err={e}", flush=True)
                continue
            # 2) Apply SHIFT
            try:
                atk_img = shift_attack(pipe, wm_img, text_emb, args.lam,
                                         device, dtype)
            except Exception as e:
                print(f"  [skip] idx={idx} shift err={e}", flush=True)
                continue
            # 3) Detect with PRC's native detector (LDPC)
            try:
                det = wm.detect_watermark_in_media(atk_img, prompt=prompt)
            except Exception as e:
                print(f"  [skip] idx={idx} det err={e}", flush=True)
                continue
            results.append({
                "idx": idx, "seed": seed, "prompt_idx": pi,
                "pseudogaussian_seed": 49 + seed,
                "is_watermarked": bool(det.get("is_watermarked", False)),
            })

            if len(results) % 25 == 0:
                tpr = sum(r["is_watermarked"] for r in results) / len(results)
                el = (time.time() - t0) / 60
                print(f"  [{len(results)}/{total}] PRC-per-image lam={args.lam} "
                      f"TPR={tpr*100:.1f}%  el={el:.1f}min", flush=True)
                json.dump({"results": results,
                            "summary": {"n": len(results),
                                         "tpr": tpr}},
                          open(out_path, "w"), indent=2)

    n = len(results)
    tpr = sum(r["is_watermarked"] for r in results) / max(n, 1)
    json.dump({"results": results,
                "summary": {"n": n, "tpr": tpr}},
              open(out_path, "w"), indent=2)
    print(f"\n=== Final PRC per-image SHIFT lam={args.lam} ===", flush=True)
    print(f"  n={n}  TPR={tpr*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()

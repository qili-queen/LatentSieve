"""Key-derived-reference cross-scheme counterfactual under SHIFT.

Motivation
----------
The counterfactual in the paper (Table 7) applies LatentSieve's sign-correlation
decoder to GS / PRC / T2SMark codewords, but it takes the reference sign pattern
from the DDIM inversion of the *clean watermarked image*:

    qi_eval_gs_t2s_counterfactual.py:174
        z_ref = ddim_invert(pipe, wm_img, ...)   # per-image stored artifact

That is a per-image stored reference, not something a key-holding verifier has.
It is also a reference that has already been through the round-trip channel, so
it may be easier to correlate against than the codeword actually embedded.

This script recomputes the same counterfactual with the reference each scheme's
verifier can actually derive from its key, and reports both side by side on the
same images so the two are paired.

It also decomposes the decoder, on each baseline's own codeword, into
    M1  = per-block sign correlation, no erasure
    M2  = M1 + per-image erasure of the 32 lowest-confidence blocks
so that the lift can be attributed to block aggregation (generic) versus
reliability-aware erasure (the paper's claimed contribution).

Output per image therefore has 2 references x 2 decoders = 4 numbers.

SHIFT-attacked images are written to disk so downstream runs can reuse them
(the existing scripts recompute the attack every time and never save it).

Usage
-----
  CUDA_VISIBLE_DEVICES=0 python qi_eval_keyref_counterfactual.py --scheme GS
  CUDA_VISIBLE_DEVICES=1 python qi_eval_keyref_counterfactual.py --scheme PRC
  CUDA_VISIBLE_DEVICES=0 python qi_eval_keyref_counterfactual.py --scheme T2SMark
"""
import argparse
import functools
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from diffusers import DDIMScheduler

oti = __import__("tqdm").tqdm.__init__


@functools.wraps(oti)
def st(s, *a, **k):
    k['disable'] = True
    oti(s, *a, **k)


__import__("tqdm").tqdm.__init__ = st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qi_runtime_utils import build_cached_pipe, configure_qi_runtime, resolve_qi_device
from utils.media_utils import (
    get_random_latents, get_media_latents, transform_to_model_format,
)
from utils.diffusion_config import DiffusionConfig
from utils.utils import set_random_seed
from watermark.auto_watermark import AutoWatermark

configure_qi_runtime()

MODEL = "huanzi05/stable-diffusion-2-1-base"
STEPS = 50
GS_SCALE = 7.5
SIZE = 512
BLOCK_DIM = 64
N_ERASE = 32
THRESHOLD = 0.67

from qi_sdp_n500_config import PROMPTS

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qi_eval_n500_sdp")


# --------------------------------------------------------------------------
# Shared latent-space helpers (kept byte-identical in behaviour to
# qi_eval_gs_t2s_counterfactual.py so the two are comparable)
# --------------------------------------------------------------------------

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
        ap = (pipe.scheduler.alphas_cumprod[max(pt, 0)] if pt >= 0
              else torch.tensor(1.0, device=device))
        x0p = (la - torch.sqrt(1 - apt) * ng) / torch.sqrt(apt)
        sig = 0.5 * torch.sqrt((1 - ap) / (1 - apt) * (1 - apt / ap)).clamp(min=0)
        pd = torch.sqrt(torch.clamp(1 - ap - sig ** 2, min=0)) * ng
        la = torch.sqrt(ap) * x0p + pd + sig * torch.randn_like(la)
    return vae_decode(pipe, la)


@torch.no_grad()
def ddim_invert(pipe, image, text_emb, device, dtype):
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


# --------------------------------------------------------------------------
# Decoder: block sign correlation, with and without low-confidence erasure
# --------------------------------------------------------------------------

def block_scores(z_rec_flat, ref_sign_flat, block_dim=BLOCK_DIM):
    """Per-block signed correlation against a reference sign pattern.

    Returns (block_match, block_conf). Coordinates whose reference sign is 0
    (T2SMark leaves part of the latent untouched in some configurations)
    contribute 0 to the correlation, which is the correct neutral behaviour.
    """
    n_blocks = z_rec_flat.size // block_dim
    s_rec = np.sign(z_rec_flat[: n_blocks * block_dim]).reshape(n_blocks, block_dim)
    s_ref = ref_sign_flat[: n_blocks * block_dim].reshape(n_blocks, block_dim)
    block_corr = (s_rec * s_ref).sum(axis=1)
    return (block_corr > 0).astype(np.float32), np.abs(block_corr)


def ba_no_erasure(block_match):
    """M1: plain block-majority aggregation."""
    return float(block_match.mean())


def ba_with_erasure(block_match, block_conf, n_erase=N_ERASE):
    """M1 + M2: drop the n_erase lowest-confidence blocks, then aggregate."""
    keep = np.argsort(-block_conf)[: block_match.size - n_erase]
    return float(block_match[keep].mean())


# --------------------------------------------------------------------------
# Key-derived reference sign patterns, one builder per scheme
# --------------------------------------------------------------------------

def build_keyref_gs(device, dtype, pipe, scheduler):
    """GS: sign(z'_T) = 2m-1 where m is the ChaCha20-encrypted watermark.

    _truncSampling maps bit 0 to truncnorm(-inf,0) and bit 1 to truncnorm(0,inf),
    so the seed only affects magnitudes. The sign pattern is therefore fixed by
    the key and identical for every image.
    """
    dc = DiffusionConfig(scheduler=scheduler, pipe=pipe, device=device,
                         image_size=(SIZE, SIZE), num_inference_steps=STEPS,
                         num_inversion_steps=STEPS, guidance_scale=GS_SCALE,
                         gen_seed=42, inversion_type="ddim", dtype=dtype)
    wm = AutoWatermark.load("GS", algorithm_config="config/GS.json",
                            diffusion_config=dc)

    def _ref(idx, seed, prompt_idx):
        wm.config.gen_seed = seed
        set_random_seed(seed)
        z = wm.utils.inject_watermark()
        return np.sign(z.detach().flatten().to(torch.float32).cpu().numpy())

    return _ref


def build_keyref_prc(device, dtype, pipe, scheduler):
    """PRC: sign(z'_T) = the PRC codeword.

    _sample_prc_codeword multiplies the codeword by |randn|, so the
    pseudogaussian seed changes magnitudes only, never signs.
    """
    dc = DiffusionConfig(scheduler=scheduler, pipe=pipe, device=device,
                         image_size=(SIZE, SIZE), num_inference_steps=STEPS,
                         num_inversion_steps=STEPS, guidance_scale=GS_SCALE,
                         gen_seed=42, inversion_type="ddim", dtype=dtype)
    wm = AutoWatermark.load("PRC", algorithm_config="config/PRC.json",
                            diffusion_config=dc)

    def _ref(idx, seed, prompt_idx):
        z = wm.utils.inject_watermark()
        return np.sign(z.detach().flatten().to(torch.float32).cpu().numpy())

    return _ref


def build_keyref_t2smark(device, dtype, pipe, scheduler):
    """T2SMark: replay the generation RNG to rebuild z'_T exactly.

    qi_gen_n500_sdp.py draws master_key, msg, key in that order after
    set_random_seed(seed + prompt_idx*1000); encode() then consumes further
    randomness, so the whole sequence has to be replayed in the same order.
    Only the support coordinates carry key-determined signs; the rest get
    random signs from the same replayed stream, so replay reproduces both.
    """
    t2s_root = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir, "T2SMark"))
    if t2s_root not in sys.path:
        sys.path.insert(0, t2s_root)
    from src.t2s import T2SMark
    import src.utils as t2s_utils

    KEY_LENGTH, MSG_LENGTH, TAU = 16, 256, 0.674
    KEY_CH = 0
    msg_ch_idx = [i for i in range(4) if i != KEY_CH]
    t2s_key = T2SMark(m=KEY_LENGTH, tau=TAU, latent_shape=(1, 64, 64))
    t2s_msg = T2SMark(m=MSG_LENGTH, tau=TAU, latent_shape=(3, 64, 64))

    def _ref(idx, seed, prompt_idx):
        base_seed = seed + prompt_idx * 1000
        t2s_utils.set_random_seed(base_seed)
        master_key = torch.randint(0, 2, (KEY_LENGTH,)).cuda()
        msg = torch.randint(0, 2, (MSG_LENGTH,)).cuda()
        key = torch.randint(0, 2, (KEY_LENGTH,)).cuda()
        z_k = t2s_key.encode(key, master_key)
        z_b = t2s_msg.encode(msg, key)
        lat = torch.zeros(1, 4, 64, 64).cuda()
        lat[:, KEY_CH, :, :] = z_k
        lat[:, msg_ch_idx, :, :] = z_b
        return np.sign(lat.detach().flatten().to(torch.float32).cpu().numpy())

    return _ref


BUILDERS = {"GS": build_keyref_gs, "PRC": build_keyref_prc,
            "T2SMark": build_keyref_t2smark}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--scheme", required=True, choices=list(BUILDERS))
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--num-images", type=int, default=500)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out-suffix", type=str, default="",
                    help="appended to the output filename, for sharded runs")
    ap.add_argument("--save-attacked", action="store_true", default=True)
    args = ap.parse_args()

    device, dtype = resolve_qi_device(args.gpu)
    pipe, scheduler = build_cached_pipe(MODEL, device, dtype)
    pipe.set_progress_bar_config(disable=True)

    wm_dir = os.path.join(BASE, f"{args.scheme}_wm")
    atk_dir = os.path.join(BASE, f"{args.scheme}_attacked_shift{args.lam}")
    os.makedirs(atk_dir, exist_ok=True)
    out_path = os.path.join(
        BASE, f"{args.scheme.lower()}_keyref_counterfactual{args.out_suffix}.json")

    meta_path = os.path.join(BASE, f"{args.scheme}_metadata.json")
    meta = {m["idx"]: m for m in json.load(open(meta_path))}

    keyref_fn = BUILDERS[args.scheme](device, dtype, pipe, scheduler)

    results = []
    if os.path.exists(out_path):
        results = json.load(open(out_path)).get("results", [])
        print(f"resumed from {len(results)} records", flush=True)
    done = {r["idx"] for r in results}

    # Record whether the key-derived reference is image-independent. For GS and
    # PRC it should be constant (fixed key, fixed payload); for T2SMark it is
    # per-image. Worth stating explicitly in the paper.
    ref0 = keyref_fn(0, meta[0]["seed"], meta[0]["prompt_idx"])
    ref1 = keyref_fn(1, meta[1]["seed"], meta[1]["prompt_idx"])
    ref_is_constant = bool(np.array_equal(ref0, ref1))
    print(f"key-derived reference constant across images: {ref_is_constant}",
          flush=True)

    t0 = time.time()
    for i in range(args.start, args.num_images):
        if i in done or i not in meta:
            continue
        img_path = os.path.join(wm_dir, f"img_{i:04d}.png")
        if not os.path.exists(img_path):
            continue
        m = meta[i]
        pi = m["prompt_idx"]
        prompt = PROMPTS[pi]

        wm_img = Image.open(img_path).convert("RGB")
        text_emb = encode_prompt(pipe, prompt, device)

        # Reference A: DDIM inversion of the clean watermarked image (paper).
        z_ref_inv = ddim_invert(pipe, wm_img, text_emb, device, dtype)
        ref_inv = np.sign(
            z_ref_inv.detach().flatten().to(torch.float32).cpu().numpy())

        # Reference B: derived from the scheme's own key.
        ref_key = keyref_fn(i, m["seed"], pi)

        # Attack, then invert.
        attacked = shift_attack(pipe, wm_img, text_emb, args.lam, device, dtype)
        if args.save_attacked:
            attacked.save(os.path.join(atk_dir, f"img_{i:04d}.png"))
        z_rec = ddim_invert(pipe, attacked, text_emb, device, dtype)
        z_rec_flat = z_rec.detach().flatten().to(torch.float32).cpu().numpy()

        rec = {"idx": i, "prompt_idx": pi, "seed": m["seed"], "lambda": args.lam}
        for tag, ref in (("inv", ref_inv), ("key", ref_key)):
            bm, bc = block_scores(z_rec_flat, ref)
            ba_m1 = ba_no_erasure(bm)
            ba_m2 = ba_with_erasure(bm, bc)
            rec[f"ba_{tag}_m1"] = ba_m1
            rec[f"ba_{tag}_m2"] = ba_m2
            rec[f"det_{tag}_m1"] = bool(ba_m1 > THRESHOLD)
            rec[f"det_{tag}_m2"] = bool(ba_m2 > THRESHOLD)
        results.append(rec)

        if (i + 1) % 20 == 0:
            n = len(results)
            el = (time.time() - t0) / 60
            inv2 = sum(r["det_inv_m2"] for r in results) / n
            key2 = sum(r["det_key_m2"] for r in results) / n
            print(f"  {args.scheme} [{i+1}/{args.num_images}] "
                  f"inv+erase={inv2*100:.1f}%  key+erase={key2*100:.1f}%  "
                  f"el={el:.1f}min", flush=True)
            json.dump({"scheme": args.scheme, "lambda": args.lam,
                       "ref_is_constant": ref_is_constant, "results": results},
                      open(out_path, "w"), indent=2)

    json.dump({"scheme": args.scheme, "lambda": args.lam,
               "ref_is_constant": ref_is_constant, "results": results},
              open(out_path, "w"), indent=2)

    n = max(len(results), 1)
    print(f"\n=== {args.scheme} key-ref counterfactual, SHIFT lambda={args.lam} "
          f"(n={len(results)}) ===")
    for tag, label in (("inv", "inverted-clean ref"), ("key", "key-derived ref")):
        for dec in ("m1", "m2"):
            tpr = sum(r[f"det_{tag}_{dec}"] for r in results) / n
            ba = np.mean([r[f"ba_{tag}_{dec}"] for r in results])
            name = "no erasure" if dec == "m1" else "erase 32"
            print(f"  {label:22s} {name:11s}  TPR={tpr*100:5.1f}%  BA={ba:.4f}")


if __name__ == "__main__":
    main()

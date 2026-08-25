"""Null distribution of LatentSieve's decoder against a key-derived reference.

Why this is needed
------------------
qi_eval_keyref_counterfactual.py scores each baseline's codeword against a
reference the verifier derives from its key. For GS and PRC that reference is a
single fixed sign pattern, the same for every image. Comparing TPR at the shared
tau=0.67 is only fair if the null stays centred at 0.5 under such a reference.

Sign symmetry says it should: for z ~ N(0, I) the products sign(z_j) s_j are
i.i.d. Rademacher against any fixed s. But the detector never sees z. It sees
the DDIM inversion of a generated image, which is not exactly N(0, I) and does
carry spatial structure, which is why the paper measures its own null rather
than relying on the symmetry argument. This script measures it.

Unwatermarked images are generated the same way as in
qi_eval_far_cross_scheme.py (seed 7_000_000 + i, disjoint from every
watermarked seed) and cached in the same directory, so the two studies share
the same null images.

  CUDA_VISIBLE_DEVICES=0 python qi_eval_keyref_null.py --num-images 200
"""
import argparse
import functools
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

oti = __import__("tqdm").tqdm.__init__


@functools.wraps(oti)
def st(s, *a, **k):
    k['disable'] = True
    oti(s, *a, **k)


__import__("tqdm").tqdm.__init__ = st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import qi_eval_keyref_counterfactual as K
from qi_runtime_utils import build_cached_pipe, configure_qi_runtime, resolve_qi_device
from utils.media_utils import get_random_latents
from qi_sdp_n500_config import PROMPTS

configure_qi_runtime()

BASE = K.BASE
REF_POOL = 5          # mismatched inverted-clean references per scheme
UNWM_DIR = os.path.join(BASE, "UNWM_sdp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num-images", type=int, default=200)
    ap.add_argument("--schemes", type=str, default="GS,PRC,T2SMark")
    ap.add_argument("--output", type=str,
                    default=os.path.join(BASE, "keyref_null.json"))
    args = ap.parse_args()

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    device, dtype = resolve_qi_device(args.gpu)
    pipe, scheduler = build_cached_pipe(K.MODEL, device, dtype)
    pipe.set_progress_bar_config(disable=True)
    os.makedirs(UNWM_DIR, exist_ok=True)

    refs = {}
    for s in schemes:
        fn = K.BUILDERS[s](device, dtype, pipe, scheduler)
        # For T2SMark the reference is per-image; the null only needs *a* valid
        # key-derived pattern, so index 0's is representative.
        meta_path = os.path.join(BASE, f"{s}_metadata.json")
        m0 = json.load(open(meta_path))[0]
        refs[s] = fn(0, m0["seed"], m0["prompt_idx"])

    # Mismatched inverted-clean references. The counterfactual in the paper
    # takes its reference from the DDIM inversion of a clean watermarked image,
    # and such a latent carries generic round-trip structure. If that structure
    # correlates with the round-trip structure of *any* inverted latent, then
    # the inverted-clean reference inflates the null as well as the signal, and
    # its higher TPR is partly an artifact. Test it by scoring unwatermarked
    # images against inverted-clean references taken from unrelated images.
    inv_refs = {}
    for s in schemes:
        meta = json.load(open(os.path.join(BASE, f"{s}_metadata.json")))
        pats = []
        for m in meta[:REF_POOL]:
            p = os.path.join(BASE, f"{s}_wm", f"img_{m['idx']:04d}.png")
            if not os.path.exists(p):
                continue
            te = K.encode_prompt(pipe, PROMPTS[m["prompt_idx"]], device)
            z = K.ddim_invert(pipe, Image.open(p).convert("RGB"), te, device, dtype)
            pats.append(np.sign(
                z.detach().flatten().to(torch.float32).cpu().numpy()))
        inv_refs[s] = pats
        print(f"  {s}: {len(pats)} mismatched inverted-clean references",
              flush=True)

    rows = []
    if os.path.exists(args.output):
        rows = json.load(open(args.output)).get("results", [])
        print(f"resumed from {len(rows)} records", flush=True)
    done = {r["idx"] for r in rows}

    for i in range(args.num_images):
        if i in done:
            continue
        pi = i % len(PROMPTS)
        prompt = PROMPTS[pi]
        path = os.path.join(UNWM_DIR, f"img_{i:04d}.png")
        if not os.path.exists(path):
            seed = 7_000_000 + i          # disjoint from every watermarked seed
            g = torch.Generator(device=device).manual_seed(seed)
            latents = get_random_latents(pipe, height=K.SIZE, width=K.SIZE,
                                         generator=g).to(device, dtype=dtype)
            with torch.no_grad():
                img = pipe(prompt, num_inference_steps=K.STEPS,
                           guidance_scale=K.GS_SCALE, latents=latents,
                           height=K.SIZE, width=K.SIZE).images[0]
            img.save(path)
        else:
            img = Image.open(path).convert("RGB")

        te = K.encode_prompt(pipe, prompt, device)
        z = K.ddim_invert(pipe, img, te, device, dtype)
        zf = z.detach().flatten().to(torch.float32).cpu().numpy()

        rec = {"idx": i, "prompt_idx": pi}
        for s in schemes:
            bm, bc = K.block_scores(zf, refs[s])
            rec[f"ba_{s}_key_m1"] = K.ba_no_erasure(bm)
            rec[f"ba_{s}_key_m2"] = K.ba_with_erasure(bm, bc)
            # Same unwatermarked latent, but scored against inverted-clean
            # references from unrelated images.
            m1s, m2s = [], []
            for ref in inv_refs[s]:
                bm2, bc2 = K.block_scores(zf, ref)
                m1s.append(K.ba_no_erasure(bm2))
                m2s.append(K.ba_with_erasure(bm2, bc2))
            if m1s:
                rec[f"ba_{s}_inv_m1"] = float(np.mean(m1s))
                rec[f"ba_{s}_inv_m2"] = float(np.mean(m2s))
                rec[f"ba_{s}_inv_m2_max"] = float(np.max(m2s))
        rows.append(rec)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{args.num_images}] " + "  ".join(
                f"{s}: key_max={max(r[f'ba_{s}_key_m2'] for r in rows):.3f}"
                for s in schemes), flush=True)
            json.dump({"results": rows}, open(args.output, "w"), indent=2)

    json.dump({"results": rows}, open(args.output, "w"), indent=2)

    print(f"\n=== null distributions, n={len(rows)} unwatermarked images ===")
    print(f"  {'scheme':<9} {'reference':<16} {'decoder':<11} {'mean':>7} {'sd':>7} "
          f"{'max':>7}  {'FP@0.67':>9}")
    for s in schemes:
        for tag, label in (("key", "key-derived"), ("inv", "inverted-clean")):
            for dec in ("m1", "m2"):
                col = f"ba_{s}_{tag}_{dec}"
                if col not in rows[0]:
                    continue
                v = np.array([r[col] for r in rows])
                fp = int((v > K.THRESHOLD).sum())
                name = "no erasure" if dec == "m1" else "erase 32"
                print(f"  {s:<9} {label:<16} {name:<11} {v.mean():7.4f} "
                      f"{v.std():7.4f} {v.max():7.4f}  {fp:>4}/{len(v)}")
    print("\n  A null mean above 0.5 for the inverted-clean reference would mean "
          "that\n  reference inflates the null as well as the signal.")


if __name__ == "__main__":
    main()

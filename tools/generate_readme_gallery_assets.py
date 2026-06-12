"""Generate paired README gallery assets.

This script generates the paired unwatermarked originals used by
``make_readme_gallery.py`` and saves a compact set of SHIFT-0.7 attacked
examples for the same SDP prompt/seed indices.
"""

from pathlib import Path
import os
import sys

import torch
from PIL import Image
from diffusers import DDIMScheduler


RELEASE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = RELEASE_ROOT.parent
MARKDIFF_ROOT = WORKSPACE_ROOT / "MarkDiffusion"
EVAL_ROOT = MARKDIFF_ROOT / "qi_eval_n500_sdp"
GALLERY_SOURCE = RELEASE_ROOT / "assets" / "gallery" / "source"

sys.path.insert(0, str(MARKDIFF_ROOT))

from qi_eval_n500_shift_sdp import encode_prompt, shift_attack  # noqa: E402
from qi_gen_n500_sdp import reseed  # noqa: E402
from qi_runtime_utils import build_cached_pipe, configure_qi_runtime, resolve_qi_device  # noqa: E402
from qi_sdp_n500_config import PROMPTS  # noqa: E402
from utils.diffusion_config import DiffusionConfig  # noqa: E402
from watermark.auto_watermark import AutoWatermark  # noqa: E402


MODEL = "huanzi05/stable-diffusion-2-1-base"
STEPS = 50
GS_SCALE = 7.5
SIZE = 512
SAMPLE_IDXS = [0, 5, 10, 15, 20]
SCHEMES = ["MIRAGE", "GS", "PRC", "T2SMark", "TR"]
SCHEME_CONFIGS = {
    "MIRAGE": "config/MIRAGE_presence.json",
    "PRC": "config/PRC.json",
    "GS": "config/GS.json",
    "TR": "config/TR.json",
}


def prompt_seed(idx: int) -> tuple[int, int, str]:
    prompt_idx = idx // 5
    seed_idx = idx % 5
    seed = prompt_idx * 100 + seed_idx
    return prompt_idx, seed, PROMPTS[prompt_idx]


def build_watermark(pipe, scheduler, device, dtype):
    dc = DiffusionConfig(
        scheduler=scheduler,
        pipe=pipe,
        device=device,
        image_size=(SIZE, SIZE),
        num_inference_steps=STEPS,
        num_inversion_steps=STEPS,
        guidance_scale=GS_SCALE,
        gen_seed=42,
        inversion_type="ddim",
        dtype=dtype,
    )
    return AutoWatermark.load(
        "MIRAGE",
        algorithm_config=str(MARKDIFF_ROOT / "config" / "MIRAGE_presence.json"),
        diffusion_config=dc,
    )


def generate_originals(wm) -> None:
    out_dir = GALLERY_SOURCE / "original"
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in SAMPLE_IDXS:
        _, seed, prompt = prompt_seed(idx)
        out = out_dir / f"img_{idx:04d}.png"
        if out.exists():
            continue
        reseed(wm, seed)
        image = wm.generate_unwatermarked_media(prompt)
        image.save(out)
        print(f"saved {out}", flush=True)


@torch.no_grad()
def generate_shift(pipe, device, dtype) -> None:
    out_root = GALLERY_SOURCE / "shift_lambda_0.7"
    out_root.mkdir(parents=True, exist_ok=True)

    for idx in SAMPLE_IDXS:
        _, _, prompt = prompt_seed(idx)
        text_emb = encode_prompt(pipe, prompt, device)
        for scheme in SCHEMES:
            src = EVAL_ROOT / f"{scheme}_wm" / f"img_{idx:04d}.png"
            if not src.exists():
                raise FileNotFoundError(src)
            out_dir = out_root / scheme
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"img_{idx:04d}.png"
            if out.exists():
                continue
            wm_img = Image.open(src).convert("RGB")
            pipe.scheduler = DDIMScheduler.from_pretrained(
                MODEL, subfolder="scheduler", local_files_only=True
            )
            attacked = shift_attack(pipe, wm_img, text_emb, 0.7, device, dtype)
            attacked.save(out)
            print(f"saved {out}", flush=True)


def main() -> None:
    configure_qi_runtime()
    os.chdir(MARKDIFF_ROOT)
    device, dtype = resolve_qi_device(0)
    pipe, scheduler = build_cached_pipe(MODEL, device, dtype)
    wm = build_watermark(pipe, scheduler, device, dtype)
    generate_originals(wm)
    generate_shift(pipe, device, dtype)


if __name__ == "__main__":
    main()

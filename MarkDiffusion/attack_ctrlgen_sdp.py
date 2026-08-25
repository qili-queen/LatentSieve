"""Stage 7 — Ctrlgen attack on N=500 watermarked images.

Adapted from /home/nisp/Jie/security26/CtrlRegen/ctrlgen.py.

Usage:
  CUDA_VISIBLE_DEVICES=2 python qi_attack_ctrlgen.py
"""
import argparse
import os
import sys

# Path setup for ctrlregen module
CTRLGEN_ROOT = "/home/nisp/Jie/security26/CtrlRegen"
sys.path.insert(0, CTRLGEN_ROOT)

import torch
from PIL import Image
from tqdm import tqdm

import torchvision.transforms as transforms
from diffusers import ControlNetModel, UniPCMultistepScheduler, AutoencoderKL
from custom_i2i_pipeline import CustomStableDiffusionControlNetImg2ImgPipeline
from controlnet_aux import CannyDetector
from transformers import AutoModel, AutoImageProcessor
from utils import color_match


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--schemes", type=str,
                        default="MIRAGE,PRC,GS,TR,T2SMark")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    transform_size_to_512 = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),
    ])

    DIFFUSION_MODEL = "SG161222/Realistic_Vision_V4.0_noVAE"
    SPATIAL_CONTROL_PATH = os.path.join(
        CTRLGEN_ROOT, "ctrlregen/spatialnet_ckp/spatial_control_ckp_14000"
    )
    SEMANTIC_CONTROL_PATH = os.path.join(
        CTRLGEN_ROOT, "ctrlregen/semanticnet_ckp"
    )
    SEMANTIC_CONTROL_NAME = "semantic_control_ckp_435000.bin"
    IMAGE_ENCODER = "facebook/dinov2-giant"
    VAE = "stabilityai/sd-vae-ft-mse"

    print("Loading models...", flush=True)
    spatialnet = [ControlNetModel.from_pretrained(
        SPATIAL_CONTROL_PATH, torch_dtype=torch.float16
    )]
    pipe = CustomStableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        DIFFUSION_MODEL,
        controlnet=spatialnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.costum_load_ip_adapter(
        SEMANTIC_CONTROL_PATH, subfolder="models",
        weight_name=SEMANTIC_CONTROL_NAME,
    )
    pipe.image_encoder = AutoModel.from_pretrained(IMAGE_ENCODER).to(
        device, dtype=torch.float16
    )
    pipe.feature_extractor = AutoImageProcessor.from_pretrained(IMAGE_ENCODER)
    pipe.vae = AutoencoderKL.from_pretrained(VAE).to(dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_ip_adapter_scale(1.0)
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)
    processor = CannyDetector()
    print("Loaded.", flush=True)

    def ctrl_regen_plus(input_img, step, seed=0):
        generator = torch.manual_seed(seed)
        input_img = transform_size_to_512(input_img)
        processed_img = processor(
            input_img, low_threshold=100, high_threshold=150
        )
        prompt = "best quality, high quality"
        negative_prompt = (
            "monochrome, lowres, bad anatomy, worst quality, low quality"
        )
        output_img = pipe(
            prompt,
            negative_prompt=negative_prompt,
            image=[input_img],
            control_image=[processed_img],
            ip_adapter_image=[input_img],
            strength=step,
            generator=generator,
            num_inference_steps=50,
            controlnet_conditioning_scale=1.0,
            guidance_scale=2.0,
            control_guidance_start=0,
            control_guidance_end=1,
        ).images[0]
        return color_match(input_img, output_img)

    BASE = "/home/nisp/Qili/MarkDiff_research/MarkDiffusion/qi_eval_n500_sdp"
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]

    for scheme in schemes:
        wm_dir = os.path.join(BASE, f"{scheme}_wm")
        if not os.path.isdir(wm_dir):
            print(f"SKIP {scheme}: {wm_dir} missing", flush=True)
            continue
        out_dir = os.path.join(BASE, f"{scheme}_attacked_Ctrlgen")
        os.makedirs(out_dir, exist_ok=True)

        img_files = sorted(os.listdir(wm_dir))[: args.max_images]
        print(f"\n=== Ctrlgen on {scheme}: {len(img_files)} images ===",
              flush=True)
        for i, img_name in enumerate(img_files):
            src = os.path.join(wm_dir, img_name)
            dst = os.path.join(out_dir, img_name)
            if os.path.exists(dst):
                continue
            image = Image.open(src).convert("RGB")
            att_img = ctrl_regen_plus(input_img=image, step=0.5, seed=1)
            att_img.save(dst)
            if (i + 1) % 25 == 0 or i + 1 == len(img_files):
                print(f"  [{i+1}/{len(img_files)}] {scheme} done",
                      flush=True)

        print(f"{scheme} DONE", flush=True)


if __name__ == "__main__":
    main()

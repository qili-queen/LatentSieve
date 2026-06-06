"""MIRAGE: Measure-preserving Indistinguishable Reliability-Adaptive Gaussian Embedding.

A distribution-preserving watermarking scheme for latent diffusion models.
Unlike RSWST (which modifies z_T via directional transport, breaking N(0,I)),
MIRAGE uses keyed sign encoding: each coordinate in a selected reliable block
has its sign set according to a keyed pseudorandom pattern XORed with the
watermark bit. Since N(0,1) is symmetric, flipping the sign of any coordinate
preserves the marginal distribution. Without the key, the sign pattern is
computationally indistinguishable from random, so z'_T is computationally
indistinguishable from N(0,I).

Key properties:
  - Computational indistinguishability: without key K, Poly-time adversary
    cannot distinguish watermarked z'_T from z_T ~ N(0,I)
  - Reliability-shaped block selection (from RSWST): concentrate watermark
    budget in blocks that survive generation + attack + inversion
  - Weighted detection with low-confidence erasure (from RSWST)
  - No norm modification, no directional transport, no whitening at embedding
"""
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from Crypto.Cipher import ChaCha20
from PIL import Image

from ..base import BaseConfig, BaseWatermark
from detection.mirage.mirage_detection import MIRAGEDetector
from evaluation.tools.image_editor import (
    AdaptiveNoiseInjection,
    CrSc,
    GaussianBlurring,
    GaussianNoise,
    JPEGCompression,
    Overlay,
)
from utils.media_utils import get_media_latents, get_random_latents, transform_to_model_format
from utils.utils import set_random_seed


class MIRAGEConfig(BaseConfig):
    def initialize_parameters(self) -> None:
        self.num_blocks = int(self.config_dict["num_blocks"])
        self.block_dim = int(self.config_dict.get("block_dim", 16))
        self.selection_pool_size = int(self.config_dict["selection_pool_size"])
        self.selection_seed = int(self.config_dict["selection_seed"])
        self.selection_strategy = self.config_dict.get("selection_strategy", "topk")
        self.permutation_seed = int(self.config_dict["permutation_seed"])

        # Key for sign pattern generation
        self.sign_key_seed = int(self.config_dict.get("sign_key_seed", 314159))
        self.watermark_seed = int(self.config_dict.get("watermark_seed", 271828))

        # Detection parameters
        self.presence_bitacc_threshold = float(self.config_dict.get("presence_bitacc_threshold", 0.67))
        self.presence_erasure_bits = int(self.config_dict.get("presence_erasure_bits", 32))

        # Reliability weights
        self.weight_min = float(self.config_dict.get("weight_min", 1.0))
        self.weight_max = float(self.config_dict.get("weight_max", 2.4))

        # Whitening (for detection only, not embedding)
        self.whitening_eps = float(self.config_dict.get("whitening_eps", 1e-4))
        self.whitening_shrinkage = float(self.config_dict.get("whitening_shrinkage", 0.1))

        # Calibration
        self.calibration_prompts = list(self.config_dict.get("calibration_prompts", []))
        self.reliability_attack_names = list(self.config_dict.get("reliability_attack_names", []))
        self.reliability_attack_weights = dict(self.config_dict.get("reliability_attack_weights", {}))
        self.reliability_aggregation = self.config_dict.get("reliability_aggregation", "harmonic")
        self.reliability_attack_focus = float(self.config_dict.get("reliability_attack_focus", 2.0))
        self.reliability_attack_repeats = {
            str(name): max(1, int(repeats))
            for name, repeats in self.config_dict.get("reliability_attack_repeats", {}).items()
        }
        self.whitening_include_clean_recovered = bool(
            self.config_dict.get("whitening_include_clean_recovered", True)
        )
        self.whitening_include_attacked_recovered = bool(
            self.config_dict.get("whitening_include_attacked_recovered", True)
        )
        self.cache_dir = self.config_dict.get("cache_dir", "watermark/mirage/cache")
        self.fpr = float(self.config_dict.get("fpr", 1e-9))

        # Per-image nonce for multi-image undetectability
        self.use_per_image_nonce = bool(self.config_dict.get("use_per_image_nonce", True))

        # Latent dimensions
        self.latents_height = self.image_size[0] // self.pipe.vae_scale_factor
        self.latents_width = self.image_size[1] // self.pipe.vae_scale_factor
        self.latents_channel = self.pipe.unet.config.in_channels
        self.total_latent_dim = self.latents_height * self.latents_width * self.latents_channel
        if self.total_latent_dim % self.block_dim != 0:
            raise ValueError(
                f"Latent dim {self.total_latent_dim} must be divisible by block_dim={self.block_dim}"
            )
        self.total_blocks = self.total_latent_dim // self.block_dim
        self.selection_pool_size = min(self.selection_pool_size, self.total_blocks)
        if self.selection_pool_size < self.num_blocks:
            self.selection_pool_size = self.num_blocks
        if self.num_blocks > self.total_blocks:
            raise ValueError(
                f"num_blocks={self.num_blocks} exceeds available blocks={self.total_blocks}"
            )

    @property
    def algorithm_name(self) -> str:
        return "MIRAGE"


class MIRAGEUtils:
    """Calibration, block selection, and keyed sign pattern generation.

    Reuses RSWST's reliability calibration pipeline. The critical difference
    is in inject_watermark: sign-flip encoding instead of directional transport.
    """

    def __init__(self, config: MIRAGEConfig):
        self.config = config
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.reliability_attacks = self._build_reliability_attacks()

        self.channel_stats = self._build_channel_statistics()
        self.reliability_scores = self.channel_stats["reliability_scores"].to(
            device=self.config.device, dtype=torch.float32,
        )
        self.whitening_matrix = self.channel_stats["whitening_matrix"].to(
            device=self.config.device, dtype=torch.float32,
        )
        self.selected_block_indices, self.selected_weights = self._select_blocks()
        self.watermark_bits = self._generate_watermark_bits()
        self.sign_pattern = self._generate_sign_pattern()

    # ---- Reliability attacks (same as RSWST) ----

    def _build_reliability_attacks(self) -> List[Tuple[str, object]]:
        attack_map = {
            "JPEG-75": JPEGCompression(quality=75),
            "Blur-2": GaussianBlurring(radius=2),
            "Noise-15": GaussianNoise(sigma=15.0),
            "Overlay": Overlay(num_strokes=10),
            "CropPadBlack-0.8": CrSc(crop_ratio=0.8, resize_back=False, pad_back=True),
            "AdaptiveNoise": AdaptiveNoiseInjection(intensity=0.5),
        }
        attacks = []
        for name in self.config.reliability_attack_names:
            if name not in attack_map:
                if name.startswith("LatentDecorrelation"):
                    continue  # handled separately in _build_channel_statistics
                raise ValueError(f"Unsupported reliability attack: {name}")
            attacks.append((name, attack_map[name]))
        return attacks

    # ---- Real MarkNull attack for calibration ----

    def _marknull_attack(
        self, gen_image: Image.Image, prompt: str,
        epsilon: float = 1.0, lr: float = 0.05, num_steps: int = 50,
        lambda_sim: float = 3.0, lambda_lpips: float = 1.0,
        lambda_ssim: float = 1.5, lambda_mse: float = 20.0,
    ) -> Image.Image:
        """Real MarkNull attack: Adam optimizer + geodesic loss + AMP.

        Uses the actual MarkNull optimization (not a linear proxy):
        1. VAE encode image → z_0
        2. DDIM inversion → z_T
        3. Optimize z_0_adv to minimize geodesic_loss(z_0_adv, z_T)
           subject to image quality constraints and L_inf budget
        4. VAE decode z_0_adv → attacked image

        Blocks whose sign survives this attack are genuinely MarkNull-robust.
        """
        import torch.nn.functional as F
        from torchvision import transforms
        from diffusers import DDIMInverseScheduler

        vae = self.config.pipe.vae
        device = self.config.device
        dtype = self.config.dtype
        scaling = 0.18215

        # 1. VAE encode
        to_tensor = transforms.ToTensor()
        x_orig = to_tensor(gen_image).unsqueeze(0).to(device, dtype=dtype)
        with torch.no_grad():
            x_input = x_orig.clone() * 2.0 - 1.0
            z0_orig = vae.encode(x_input).latent_dist.sample() * scaling

        # 2. DDIM inversion → z_T
        inv_scheduler = DDIMInverseScheduler.from_config(self.config.pipe.scheduler.config)
        inv_scheduler.set_timesteps(50, device=device)
        do_cfg = self.config.guidance_scale > 1.0
        prompt_embeds, neg_embeds = self.config.pipe.encode_prompt(
            prompt=prompt, device=device,
            do_classifier_free_guidance=do_cfg, num_images_per_prompt=1,
        )
        text_emb = torch.cat([neg_embeds, prompt_embeds]) if do_cfg else prompt_embeds

        xt = z0_orig.clone()
        with torch.no_grad():
            for t in inv_scheduler.timesteps:
                model_in = torch.cat([xt, xt]) if do_cfg else xt
                noise_pred = self.config.pipe.unet(model_in, t, encoder_hidden_states=text_emb).sample
                if do_cfg:
                    eps_u, eps_t = noise_pred.chunk(2)
                    noise_pred = eps_u + self.config.guidance_scale * (eps_t - eps_u)
                inv_out = inv_scheduler.step(noise_pred, t, xt)
                xt = inv_out.prev_sample
        zT_orig = xt.detach()

        # 3. Optimize z_0_adv
        z0_adv = z0_orig.clone().detach().to(dtype=torch.float32)
        z0_adv.requires_grad = True
        optimizer = torch.optim.Adam([z0_adv], lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.3)
        scaler = torch.cuda.amp.GradScaler()

        batch_size = z0_orig.shape[0]
        zT_flat = zT_orig.view(batch_size, -1).float()

        for i in range(num_steps):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                # Geodesic loss
                z_adv_flat = F.normalize(z0_adv.view(batch_size, -1), p=2, dim=1)
                z_T_flat = F.normalize(zT_flat, p=2, dim=1)
                cos_sim = (z_adv_flat * z_T_flat).sum(dim=1)
                cos_sim = torch.clamp(cos_sim, -1 + 1e-5, 1 - 1e-5)
                theta = torch.acos(cos_sim)
                loss_contrast = ((theta - torch.pi / 2) ** 2).mean()

                # Image quality losses
                z_decode = z0_adv / scaling
                x_adv = vae.decode(z_decode.to(dtype)).sample
                x_adv = (x_adv / 2 + 0.5).clamp(0, 1)

                loss_mse = F.mse_loss(x_adv, x_orig, reduction='none').view(batch_size, -1).mean()
                loss = lambda_sim * loss_contrast + lambda_mse * loss_mse

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([z0_adv], max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            with torch.no_grad():
                diff = z0_adv - z0_orig.float()
                diff = torch.clamp(diff, -epsilon, epsilon)
                z0_adv.copy_(z0_orig.float() + diff)

        # 4. Decode attacked image
        with torch.no_grad():
            z_final = z0_adv.detach().to(dtype) / scaling
            x_out = vae.decode(z_final).sample
            x_out = (x_out / 2 + 0.5).clamp(0, 1)
        out_np = (x_out[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
        return Image.fromarray(out_np)

    # ---- Watermark bit generation ----

    def _generate_watermark_bits(self) -> torch.Tensor:
        """Generate the fixed watermark bit pattern (1 bit per block) from key."""
        rng = np.random.default_rng(self.config.watermark_seed)
        bits = rng.integers(0, 2, size=self.config.num_blocks, dtype=np.int64)
        return torch.tensor(bits, device=self.config.device, dtype=torch.long)

    # ---- Keyed sign pattern ----

    def _generate_sign_pattern(self, image_nonce: int = None) -> torch.Tensor:
        """Generate the keyed sign pattern for each coordinate in each selected block.

        sign_pattern[i, j] ∈ {-1, +1}:
          target_sign = keyed_random_bit(block_i, coord_j) XOR watermark_bit[i]
          sign_pattern[i, j] = +1 if target_sign == 1 else -1

        Args:
            image_nonce: per-image nonce for multi-image undetectability.
                If provided, the ChaCha20 nonce is derived from
                SHA256(permutation_seed || image_nonce), decorrelating
                sign patterns across images.
        """
        B = self.config.num_blocks
        d = self.config.block_dim
        block_indices = self.selected_block_indices.detach().cpu().numpy()
        wm_bits = self.watermark_bits.detach().cpu().numpy()

        # Use ChaCha20 to generate keyed pseudorandom bits
        key_bytes = self.config.sign_key_seed.to_bytes(32, byteorder="big", signed=False)
        if image_nonce is not None:
            # Derive per-image nonce: SHA256(permutation_seed || image_nonce)[:12]
            nonce_input = (
                self.config.permutation_seed.to_bytes(8, byteorder="big", signed=True)
                + int(image_nonce).to_bytes(8, byteorder="big", signed=True)
            )
            nonce_bytes = hashlib.sha256(nonce_input).digest()[:12]
        else:
            nonce_bytes = self.config.permutation_seed.to_bytes(12, byteorder="big", signed=True)
        cipher = ChaCha20.new(key=key_bytes[:32], nonce=nonce_bytes[:12])
        # Generate enough bytes: B * d bits → ceil(B*d/8) bytes
        num_bytes = (B * d + 7) // 8
        keystream = cipher.encrypt(bytes(num_bytes))
        key_bits = np.unpackbits(np.frombuffer(keystream, dtype=np.uint8))[: B * d]
        key_bits = key_bits.reshape(B, d)

        # XOR with watermark bits (broadcast: wm_bits[i] applies to all coords in block i)
        target_bits = np.bitwise_xor(key_bits, wm_bits[:, None])

        # Convert {0, 1} → {-1, +1}: sign = 2*bit - 1
        sign_pattern = 2.0 * target_bits.astype(np.float32) - 1.0
        return torch.tensor(sign_pattern, device=self.config.device, dtype=torch.float32)

    # ---- Embedding: sign-flip encoding (distribution-preserving) ----

    def inject_watermark(self) -> torch.Tensor:
        """Embed watermark by flipping signs of coordinates to match the keyed pattern.

        This is MEASURE-PRESERVING: since N(0,1) is symmetric, flipping the sign
        of any coordinate preserves the marginal distribution. The joint distribution
        z'_T is still N(0, I) coordinate-wise (and computationally indistinguishable
        from N(0, I) jointly, since the sign pattern is keyed by a CSPRNG).

        When use_per_image_nonce is enabled, the sign pattern is regenerated per
        image using init_latents_seed as nonce, providing multi-image undetectability.
        """
        # Regenerate sign pattern with per-image nonce if enabled
        if self.config.use_per_image_nonce:
            self.sign_pattern = self._generate_sign_pattern(
                image_nonce=self.config.init_latents_seed
            )

        latents = self.config.init_latents.clone().to(self.config.device, dtype=self.config.dtype)
        flat = latents.detach().to(dtype=torch.float32).flatten()
        flat_blocks = flat.reshape(-1, self.config.block_dim)

        selected_blocks = flat_blocks[self.selected_block_indices].clone()

        # For each coordinate: if current sign doesn't match target, flip it
        # This is equivalent to: z'[i,j] = abs(z[i,j]) * sign_pattern[i,j]
        # Which preserves |z[i,j]| (half-normal) and sets the sign deterministically
        selected_blocks = torch.abs(selected_blocks) * self.sign_pattern

        flat_blocks[self.selected_block_indices] = selected_blocks
        watermarked = flat_blocks.reshape_as(latents.flatten()).reshape_as(latents)
        return watermarked.to(self.config.device, dtype=self.config.dtype)

    # ---- Calibration (reused from RSWST, simplified) ----

    def _cache_path(self) -> Path:
        model_name = getattr(self.config.pipe, "name_or_path", None)
        if model_name is None:
            model_name = getattr(self.config.pipe, "_name_or_path", None)
        if model_name is None:
            model_name = getattr(
                getattr(self.config.pipe, "config", None), "_name_or_path", "unknown-model"
            )
        cache_signature = hashlib.sha1(
            json.dumps(
                {
                    "algorithm": "MIRAGE",
                    "model_name": model_name,
                    "calibration_prompts": self.config.calibration_prompts,
                    "attack_names": self.config.reliability_attack_names,
                    "attack_weights": self.config.reliability_attack_weights,
                    "attack_repeats": self.config.reliability_attack_repeats,
                    "whitening_eps": self.config.whitening_eps,
                    "whitening_shrinkage": self.config.whitening_shrinkage,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:12]
        return self.cache_dir / (
            f"mirage_stats_h{self.config.image_size[0]}_w{self.config.image_size[1]}"
            f"_steps{self.config.num_inference_steps}_bd{self.config.block_dim}"
            f"_nb{self.config.num_blocks}_pool{self.config.selection_pool_size}"
            f"_{cache_signature}.pt"
        )

    def _build_generation_params(self, latents: torch.Tensor) -> Dict:
        params = {
            "num_images_per_prompt": self.config.num_images,
            "guidance_scale": self.config.guidance_scale,
            "num_inference_steps": self.config.num_inference_steps,
            "height": self.config.image_size[0],
            "width": self.config.image_size[1],
            "latents": latents.to(self.config.device, dtype=self.config.dtype),
        }
        if hasattr(self.config, "gen_kwargs") and self.config.gen_kwargs:
            for key, value in self.config.gen_kwargs.items():
                params.setdefault(key, value)
        return params

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        do_cfg = self.config.guidance_scale > 1.0
        prompt_embeds, neg_embeds = self.config.pipe.encode_prompt(
            prompt=prompt,
            device=self.config.device,
            do_classifier_free_guidance=do_cfg,
            num_images_per_prompt=1,
        )
        return torch.cat([neg_embeds, prompt_embeds]) if do_cfg else prompt_embeds

    def _generate_image_from_latents(self, prompt: str, latents: torch.Tensor) -> Image.Image:
        set_random_seed(self.config.gen_seed)
        return self.config.pipe(prompt, **self._build_generation_params(latents)).images[0]

    def _invert_image(self, image: Image.Image, prompt: str) -> torch.Tensor:
        text_embeddings = self._encode_prompt(prompt)
        image_tensor = transform_to_model_format(
            image, target_size=self.config.image_size[0]
        ).unsqueeze(0)
        image_tensor = image_tensor.to(self.config.device, dtype=text_embeddings.dtype)
        image_latents = get_media_latents(
            pipe=self.config.pipe, media=image_tensor, sample=False, decoder_inv=False,
        )
        reversed_latents = self.config.inversion.forward_diffusion(
            latents=image_latents,
            text_embeddings=text_embeddings,
            guidance_scale=self.config.guidance_scale,
            num_inference_steps=self.config.num_inference_steps,
        )[-1]
        return reversed_latents.detach()

    def _sample_calibration_latents(self, offset: int) -> torch.Tensor:
        gen = torch.Generator(device=self.config.device)
        gen.manual_seed(self.config.init_latents_seed + offset)
        latents = get_random_latents(
            self.config.pipe,
            height=self.config.image_size[0],
            width=self.config.image_size[1],
            generator=gen,
        )
        return latents.to(self.config.device, dtype=self.config.dtype)

    def _reshape_blocks(self, latents: torch.Tensor) -> torch.Tensor:
        return latents.detach().to(dtype=torch.float32).flatten().reshape(-1, self.config.block_dim)

    def _apply_attack(self, image, attack_name, attack, seed):
        py_state = random.getstate()
        np_state = np.random.get_state()
        random.seed(seed)
        np.random.seed(seed % (2**32))
        try:
            return attack.edit(image, prompt=attack_name)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)

    def _block_reliability(self, original: torch.Tensor, recovered: torch.Tensor) -> torch.Tensor:
        orig_norms = torch.linalg.norm(original, dim=1)
        rec_norms = torch.linalg.norm(recovered, dim=1)
        cosine = torch.nn.functional.cosine_similarity(original, recovered, dim=1, eps=1e-8).clamp(min=0.0)
        norm_ratio = torch.minimum(orig_norms, rec_norms) / torch.maximum(orig_norms, rec_norms).clamp_min(1e-8)
        strength = torch.tanh(orig_norms / math.sqrt(self.config.block_dim))
        score = cosine * norm_ratio * strength
        return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    def _aggregate_condition_scores(self, condition_scores: Dict[str, torch.Tensor]) -> torch.Tensor:
        clean = condition_scores.get("clean")
        if clean is None:
            raise ValueError("Must include clean scores")
        attack_scores = []
        attack_weights = []
        for name in self.config.reliability_attack_names:
            if name in condition_scores:
                attack_scores.append(condition_scores[name])
                attack_weights.append(float(self.config.reliability_attack_weights.get(name, 1.0)))
        if not attack_scores:
            return clean
        stacked = torch.stack(attack_scores, dim=0).clamp_min(1e-8)
        w = torch.tensor(attack_weights, dtype=stacked.dtype)[:, None]
        wsum = w.sum().clamp_min(1e-8)
        if self.config.reliability_aggregation == "harmonic":
            robust = wsum / (w / stacked).sum(dim=0).clamp_min(1e-8)
        elif self.config.reliability_aggregation == "geometric":
            robust = torch.exp((torch.log(stacked) * w).sum(dim=0) / wsum)
        elif self.config.reliability_aggregation == "min":
            robust = torch.min(stacked, dim=0).values
        else:
            robust = (stacked * w).sum(dim=0) / wsum
        clean_w = float(self.config.reliability_attack_weights.get("clean", 1.0))
        focus = max(self.config.reliability_attack_focus, 1e-8)
        return (clean_w * clean + focus * robust) / (clean_w + focus)

    def _estimate_whitening(self, samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        centered = samples - samples.mean(dim=0, keepdim=True)
        denom = max(samples.shape[0] - 1, 1)
        cov = (centered.T @ centered) / denom
        identity = torch.eye(self.config.block_dim, dtype=cov.dtype)
        cov = (1.0 - self.config.whitening_shrinkage) * cov + self.config.whitening_shrinkage * identity
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = eigvals.clamp_min(self.config.whitening_eps)
        W = eigvecs @ torch.diag(torch.rsqrt(eigvals)) @ eigvecs.T
        C = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T
        return W, C

    def _build_channel_statistics(self) -> Dict[str, torch.Tensor]:
        cache_path = self._cache_path()
        if cache_path.exists():
            return torch.load(cache_path, map_location="cpu")

        original_block_sets = []
        recovered_by_condition: Dict[str, List[torch.Tensor]] = {"clean": []}
        scores_by_condition: Dict[str, List[torch.Tensor]] = {"clean": []}
        for name, _ in self.reliability_attacks:
            recovered_by_condition[name] = []
            scores_by_condition[name] = []

        # Check if latent decorrelation attacks are configured
        latent_decorr_configs = []
        for name in self.config.reliability_attack_names:
            if name.startswith("LatentDecorrelation"):
                # Parse alpha from name, e.g. "LatentDecorrelation-0.8"
                parts = name.split("-")
                alpha = float(parts[1]) if len(parts) > 1 else 0.8
                latent_decorr_configs.append((name, alpha))
                recovered_by_condition[name] = []
                scores_by_condition[name] = []

        for offset, prompt in enumerate(self.config.calibration_prompts):
            orig_latents = self._sample_calibration_latents(offset)
            gen_image = self._generate_image_from_latents(prompt, orig_latents)
            orig_blocks = self._reshape_blocks(orig_latents).cpu()
            original_block_sets.append(orig_blocks)

            rec_latents = self._invert_image(gen_image, prompt)
            clean_blocks = self._reshape_blocks(rec_latents).cpu()
            recovered_by_condition["clean"].append(clean_blocks)
            scores_by_condition["clean"].append(self._block_reliability(orig_blocks, clean_blocks))

            # Image-space attacks
            for ai, (aname, attack) in enumerate(self.reliability_attacks):
                repeats = int(self.config.reliability_attack_repeats.get(aname, 1))
                rep_blocks, rep_scores = [], []
                for ri in range(repeats):
                    att_img = self._apply_attack(
                        gen_image, aname, attack,
                        seed=17_000 + offset * 101 + ai * 997 + ri * 65_537,
                    )
                    att_latents = self._invert_image(att_img, prompt)
                    att_blocks = self._reshape_blocks(att_latents).cpu()
                    rep_blocks.append(att_blocks)
                    rep_scores.append(self._block_reliability(orig_blocks, att_blocks))
                recovered_by_condition[aname].append(torch.stack(rep_blocks).mean(0))
                scores_by_condition[aname].append(torch.stack(rep_scores).mean(0))

            # Latent-space decorrelation attacks (real MarkNull)
            for ld_name, ld_alpha in latent_decorr_configs:
                repeats = int(self.config.reliability_attack_repeats.get(ld_name, 1))
                rep_blocks, rep_scores = [], []
                for ri in range(repeats):
                    # ld_alpha is used as epsilon for MarkNull
                    decorr_img = self._marknull_attack(
                        gen_image, prompt,
                        epsilon=ld_alpha, lr=0.05, num_steps=50,
                    )
                    decorr_latents = self._invert_image(decorr_img, prompt)
                    decorr_blocks = self._reshape_blocks(decorr_latents).cpu()
                    rep_blocks.append(decorr_blocks)
                    rep_scores.append(self._block_reliability(orig_blocks, decorr_blocks))
                recovered_by_condition[ld_name].append(torch.stack(rep_blocks).mean(0))
                scores_by_condition[ld_name].append(torch.stack(rep_scores).mean(0))

        cond_scores = {
            n: torch.stack(s).mean(0) for n, s in scores_by_condition.items() if s
        }
        reliability = self._aggregate_condition_scores(cond_scores)
        pool_size = min(self.config.selection_pool_size, reliability.numel())
        top_pool = torch.topk(reliability, k=pool_size, largest=True).indices

        # Whitening (for detection only)
        wh_samples = []
        for si, orig_blocks in enumerate(original_block_sets):
            wh_samples.append(orig_blocks[top_pool])
            if self.config.whitening_include_clean_recovered:
                wh_samples.append(recovered_by_condition["clean"][si][top_pool])
            if self.config.whitening_include_attacked_recovered:
                for cn, rsets in recovered_by_condition.items():
                    if cn != "clean":
                        wh_samples.append(rsets[si][top_pool])
        W, C = self._estimate_whitening(torch.cat(wh_samples, dim=0))

        if reliability.max() > 0:
            reliability = reliability / reliability.max()

        stats = {
            "reliability_scores": reliability.to(torch.float32),
            "whitening_matrix": W.to(torch.float32),
            "coloring_matrix": C.to(torch.float32),
        }
        torch.save(stats, cache_path)
        return stats

    def _select_blocks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.config.selection_seed)
        if self.config.selection_strategy == "topk":
            cands = torch.topk(self.reliability_scores, k=self.config.num_blocks, largest=True).indices.cpu()
            selected = cands[torch.randperm(cands.numel(), generator=gen)]
        else:
            total = int(self.reliability_scores.numel())
            selected = torch.randperm(total, generator=gen)[: self.config.num_blocks]
        weights = self.reliability_scores[selected.to(self.config.device)]
        weights = weights / weights.max().clamp_min(1e-6)
        weights = (
            self.config.weight_min
            + (self.config.weight_max - self.config.weight_min) * weights
        ).clamp(self.config.weight_min, self.config.weight_max)
        return (
            selected.to(self.config.device, dtype=torch.long),
            weights.to(self.config.device, dtype=torch.float32),
        )


class MIRAGE(BaseWatermark):
    """MIRAGE watermark: distribution-preserving, reliability-adaptive.

    Embedding: keyed sign-flip encoding (measure-preserving on N(0,I))
    Detection: sign-correlation with reliability-weighted aggregation
    """

    def __init__(self, watermark_config: MIRAGEConfig, *args, **kwargs):
        self.config = watermark_config
        self.utils = MIRAGEUtils(self.config)
        self.detector = MIRAGEDetector(
            selected_block_indices=self.utils.selected_block_indices,
            selected_weights=self.utils.selected_weights,
            sign_pattern=self.utils.sign_pattern,
            whitening_matrix=self.utils.whitening_matrix,
            num_blocks=self.config.num_blocks,
            block_dim=self.config.block_dim,
            presence_bitacc_threshold=self.config.presence_bitacc_threshold,
            presence_erasure_bits=self.config.presence_erasure_bits,
            fpr=self.config.fpr,
            device=self.config.device,
        )

    def _generate_watermarked_image(self, prompt: str, *args, **kwargs) -> Image.Image:
        set_random_seed(self.config.gen_seed)
        watermarked_latents = self.utils.inject_watermark()
        self.set_orig_watermarked_latents(watermarked_latents)
        params = self.utils._build_generation_params(watermarked_latents)
        for key, value in kwargs.items():
            params[key] = value
        params["latents"] = watermarked_latents
        return self.config.pipe(prompt, **params).images[0]

    def _detect_watermark_in_image(
        self, image: Image.Image, prompt: str = "", *args, **kwargs,
    ) -> Dict[str, float]:
        guidance = kwargs.get("guidance_scale", self.config.guidance_scale)
        num_steps = kwargs.get("num_inference_steps", self.config.num_inference_steps)

        # Regenerate sign pattern with per-image nonce if enabled
        if self.config.use_per_image_nonce:
            image_nonce = kwargs.get("image_nonce", self.config.init_latents_seed)
            sign_pattern = self.utils._generate_sign_pattern(image_nonce=image_nonce)
            self.detector.update_sign_pattern(sign_pattern)

        do_cfg = guidance > 1.0
        prompt_embeds, neg_embeds = self.config.pipe.encode_prompt(
            prompt=prompt,
            device=self.config.device,
            do_classifier_free_guidance=do_cfg,
            num_images_per_prompt=1,
        )
        text_embeddings = torch.cat([neg_embeds, prompt_embeds]) if do_cfg else prompt_embeds

        image_tensor = transform_to_model_format(
            image, target_size=self.config.image_size[0]
        ).unsqueeze(0)
        image_tensor = image_tensor.to(self.config.device, dtype=text_embeddings.dtype)
        image_latents = get_media_latents(
            pipe=self.config.pipe, media=image_tensor, sample=False,
            decoder_inv=kwargs.get("decoder_inv", False),
        )

        inv_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ["decoder_inv", "guidance_scale", "num_inference_steps", "image_nonce"]
        }
        reversed_latents = self.config.inversion.forward_diffusion(
            latents=image_latents,
            text_embeddings=text_embeddings,
            guidance_scale=guidance,
            num_inference_steps=num_steps,
            **inv_kwargs,
        )[-1]

        return self.detector.eval_watermark(reversed_latents)

    def get_data_for_visualize(self, media, *args, **kwargs):
        raise NotImplementedError("Visualization is not implemented for MIRAGE")

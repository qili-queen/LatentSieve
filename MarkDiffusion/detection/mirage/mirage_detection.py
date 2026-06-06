"""MIRAGE Detector: sign-correlation based presence detection.

Detects watermark presence by computing per-block correlation between
the recovered latent signs and the keyed sign pattern. Uses reliability-
weighted aggregation and low-confidence erasure from RSWST.
"""
import math

import numpy as np
import torch
from scipy.stats import binom

from detection.base import BaseDetector


class MIRAGEDetector(BaseDetector):
    def __init__(
        self,
        selected_block_indices: torch.Tensor,
        selected_weights: torch.Tensor,
        sign_pattern: torch.Tensor,
        whitening_matrix: torch.Tensor,
        num_blocks: int,
        block_dim: int,
        presence_bitacc_threshold: float,
        presence_erasure_bits: int,
        fpr: float,
        device: torch.device,
    ):
        super().__init__(threshold=fpr, device=device)
        self.selected_block_indices = selected_block_indices.to(device=device, dtype=torch.long)
        self.selected_weights = selected_weights.to(device=device, dtype=torch.float32)
        self.sign_pattern = sign_pattern.to(device=device, dtype=torch.float32)
        self.whitening_matrix = whitening_matrix.to(device=device, dtype=torch.float32)
        self.num_blocks = num_blocks
        self.block_dim = block_dim
        self.presence_bitacc_threshold = float(presence_bitacc_threshold)
        self.presence_erasure_bits = max(0, int(presence_erasure_bits))

    def update_sign_pattern(self, sign_pattern: torch.Tensor) -> None:
        """Update the sign pattern for per-image nonce detection."""
        self.sign_pattern = sign_pattern.to(device=self.selected_block_indices.device, dtype=torch.float32)

    def eval_watermark(
        self,
        reversed_latents: torch.Tensor,
        reference_latents: torch.Tensor = None,
        detector_type: str = "soft",
    ):
        del reference_latents, detector_type

        flat = reversed_latents.detach().to(dtype=torch.float32).flatten()
        flat_blocks = flat.reshape(-1, self.block_dim)
        selected_blocks = flat_blocks[self.selected_block_indices]

        # --- Per-block sign correlation ---
        # sign_pattern has shape (num_blocks, block_dim) with values in {-1, +1}
        # For the correct watermark bit, the correlation should be positive (signs match)
        # sign_pattern encodes the expected signs under bit=0; for bit=1 it's flipped
        #
        # block_corr_i = sum_j sign_pattern[i,j] * z_recovered[i,j]
        # If watermark bit was correctly embedded, block_corr_i > 0
        # The magnitude reflects confidence

        block_correlations = (selected_blocks * self.sign_pattern).sum(dim=1)

        # Normalize by block norm for stability
        block_norms = torch.linalg.norm(selected_blocks, dim=1).clamp_min(1e-8)
        normalized_corr = block_correlations / (block_norms * math.sqrt(self.block_dim))

        # Per-block match: correlation > 0 means correct bit
        block_matches = (block_correlations > 0).float()
        block_confidences = torch.abs(block_correlations)

        # Expand to per-bit level for compatibility with RSWST-style aggregation
        # Each block carries 1 bit, so block-level = bit-level
        weights_np = self.selected_weights.detach().cpu().numpy()
        matches_np = block_matches.detach().cpu().numpy()
        confidences_np = block_confidences.detach().cpu().numpy()

        # Low-confidence erasure
        keep_mask = np.ones(self.num_blocks, dtype=bool)
        if self.presence_erasure_bits > 0 and self.num_blocks > self.presence_erasure_bits:
            erased_indices = np.argsort(confidences_np)[: self.presence_erasure_bits]
            keep_mask[erased_indices] = False

        kept_matches = matches_np[keep_mask]
        kept_weights = weights_np[keep_mask]
        kept_count = int(kept_matches.size)
        match_count = int(kept_matches.sum())

        # Raw (unweighted) match rate
        raw_match_rate = float(matches_np.mean())

        # Weighted match rate
        weight_sum = float(np.sum(kept_weights))
        if weight_sum > 0:
            weighted_match_rate = float(np.sum(kept_weights * kept_matches) / weight_sum)
        else:
            weighted_match_rate = raw_match_rate

        presence_score = weighted_match_rate
        p_value = float(binom.sf(match_count - 1, kept_count, 0.5)) if kept_count > 0 else 1.0
        is_watermarked = bool(presence_score >= self.presence_bitacc_threshold)

        # Mean correlation (useful diagnostic)
        mean_corr = float(normalized_corr.mean().item())
        min_corr = float(normalized_corr.min().item())

        return {
            "is_watermarked": is_watermarked,
            "decode_success": True,
            "p_value": p_value,
            "test_bits_ok": None,
            "crc_ok": None,
            "test_matches": match_count,
            "num_test_bits": self.num_blocks,
            "crc_matches": None,
            "decoded_message": None,
            "message_bit_matches": 0,
            "message_bit_length": 0,
            "message_match_rate": None,
            "presence_score": presence_score,
            "raw_full_match_rate": raw_match_rate,
            "weighted_full_match_rate": weighted_match_rate,
            "raw_codeword_match_rate": raw_match_rate,
            "decoded_codeword_match_rate": raw_match_rate,
            "tail_match_rate": 1.0,
            "decode_strategy": "mirage_sign_correlation",
            "erasure_count": 0,
            "corrected_errors": 0,
            "codeword_alignment_score": float(block_correlations.abs().sum().item()),
            "candidate_trials": 1,
            "mean_block_margin": mean_corr,
            "mean_whitened_norm": 0.0,
            "presence_bitacc_threshold": self.presence_bitacc_threshold,
            "presence_score_type": "weighted_bitacc",
            "presence_erasure_bits": self.presence_erasure_bits,
            "effective_bits": kept_count,
            "mean_sign_correlation": mean_corr,
            "min_sign_correlation": min_corr,
        }

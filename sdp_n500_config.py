"""Shared config for the SDP-prompts re-run of the N=500 cross-scheme experiments.

100 SDP-derived prompts (from a separate selection script) x 5 seeds = N=500.

Use:
    from sdp_n500_config import PROMPTS, OUT_BASE, NUM_PROMPTS, SEEDS_PER_PROMPT
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROMPTS_JSON = ROOT / "sdp_100" / "prompts_sdp_100.json"

with open(PROMPTS_JSON) as f:
    PROMPTS = json.load(f)

NUM_PROMPTS = 100
SEEDS_PER_PROMPT = 5
TOTAL_N = NUM_PROMPTS * SEEDS_PER_PROMPT  # = 500

# Cache dir (separate from eval_n500/ so we don't collide with the old prompts)
OUT_BASE = str(ROOT / "eval_n500_sdp")

assert len(PROMPTS) == NUM_PROMPTS, f"Expected {NUM_PROMPTS} prompts, got {len(PROMPTS)}"

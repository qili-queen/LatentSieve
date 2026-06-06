# MIRAGE: Reliability-Aware Image Watermarking for Diffusion Models

Anonymous code release accompanying the IEEE S&P 2027 submission
**MIRAGE: Reliability-Aware Image Watermarking for Diffusion Models**.

This repository contains the MIRAGE implementation, the four baseline schemes
the paper compares against (Tree-Ring, Gaussian Shading, PRC, T2SMark),
and the evaluation scripts that produce every table and figure in the paper.

> *Submission under double-blind review. Do not contact or attempt to
> deanonymize the authors.*

---

## 1. Repository Layout

```
mirage_release/
├── README.md                     # This file
├── MarkDiffusion/                # MIRAGE scheme + baselines + evaluation
│   ├── watermark/                # Embedding implementations
│   │   ├── mirage/               # MIRAGE: PRF sign-flip + reliability-aware
│   │   ├── gs/                   # Gaussian Shading baseline
│   │   ├── prc/                  # PRC baseline
│   │   └── tr/                   # Tree-Ring baseline
│   ├── detection/                # Decoder implementations (matched pairs)
│   ├── config/                   # Per-scheme JSON configurations
│   ├── utils/                    # Shared utilities
│   ├── evaluation/               # Image attack pipelines
│   ├── runtime_utils.py       # Runtime helpers (pipe cache, GPU resolve)
│   ├── sdp_n500_config.py     # SDP prompt slice used in cross-scheme tables
│   ├── eval_n500_*.py         # Per-table evaluation scripts
│   ├── requirements.txt          # Python dependencies
│   └── LICENSE                   # Apache 2.0 (inherited from MarkDiffusion)
└── T2SMark/                      # T2SMark baseline (upstream, Apache 2.0)
```

## 2. Installation

```bash
conda create -n mirage python=3.10 -y
conda activate mirage
cd MarkDiffusion
pip install -r requirements.txt
```

Stable Diffusion 2.1-base is the default backbone; the scripts download
weights from Hugging Face on first use. SDXL and FLUX.1-dev are also
supported (used in the cross-backbone tables) but require additional
disk space and longer inversion.

## 3. Quick Start

Embed a watermark and run the standard attack suite on a few images:

```bash
cd MarkDiffusion
python eval_n500_standard_attacks.py --gpu 0 --num-images 100
```

Run the SHIFT cross-scheme comparison at the headline strength
$\lambda = 0.7$:

```bash
python eval_n500_shift_sdp.py --gpu 0 --scheme MIRAGE --lambdas 0.7
python eval_n500_shift_sdp.py --gpu 0 --scheme GS    --lambdas 0.7
python eval_n500_shift_sdp.py --gpu 0 --scheme PRC   --lambdas 0.7
python eval_n500_shift_sdp.py --gpu 0 --scheme TR    --lambdas 0.7
```

The cross-scheme counterfactual (MIRAGE decoder on other schemes'
codewords, paper Table 12) is in
`eval_n500_gzs_shift_sdp.py`.

## 4. Mapping from Paper Tables to Scripts

| Paper section                       | Script                                              |
|-------------------------------------|-----------------------------------------------------|
| Table: MS-COCO robustness suite     | `eval_n500_standard_attacks.py`                  |
| Table: SDP standard attacks         | `eval_n500_standard_attacks_sdp.py`              |
| Table: SHIFT $\lambda \in [0.3, 0.7]$ | `eval_n500_shift_sdp.py`                       |
| Table: SHIFT $\lambda \in [0.75, 0.85]$ | `eval_n500_shift_sdp_high.py`                |
| Table: Learned removal (cheng2020, DiffAttacker, Ctrlgen) | `eval_n500_learned_detect_sdp.py` |
| Table: UnMarker S1/S2               | `eval_n500_unmarker_detect_sdp.py`               |
| Table: Cross-scheme counterfactual  | `eval_n500_gzs_shift_sdp.py`                     |
| Table: PRC sign-level BA            | `eval_prc_sign_ba_shift_high.py`                 |
| Table: Held-out calibration         | `eval_n500_heldout.py`                           |
| Table: Sigma table                  | `eval_n500_sigma_table.py`                       |
| Table: SD 1.5 cross-backbone        | `eval_n500_sd15.py`                              |
| Table: Unseen + combo attacks       | `eval_n500_unseen_combo_sdp.py`                  |
| Figure: Attack strength sweep       | `eval_n500_strength_sweep_sdp.py`                |

The T2SMark variants of each table use the `eval_n500_t2smark_*`
counterparts, because T2SMark requires a separate inversion pipeline
backed by its own checkpoint.

## 5. Reproducibility Notes

- All scripts seed PyTorch and NumPy via `runtime_utils.configure_runtime()`.
- The cross-scheme SDP slice is fixed: 100 prompts × 5 seeds = 500
  images, identical across schemes for paired statistical tests.
- We log raw per-image scores (one JSON per scheme per attack); the
  aggregation scripts under `eval_n500_*` recompute paper numbers
  from these logs.
- Reported confidence intervals use the Wilson score; McNemar paired
  tests use Edwards' continuity correction.

## 6. Licensing

Code is released under Apache 2.0 (`MarkDiffusion/LICENSE`,
`T2SMark/LICENSE`).

## 7. Notes on Anonymization

- HuggingFace prefixes have been changed to upstream-official
  organizations (`stabilityai/...`) so the reviewer does not need our
  private mirror.
- Absolute file paths in script docstrings have been rewritten to
  `/path/to/...` placeholders.

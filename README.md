# MIRAGE: Reliability-Aware Image Watermarking for Diffusion Models

Anonymous code release for the IEEE S&P 2027 submission **MIRAGE: Reliability-Aware Image Watermarking for Diffusion Models**.

Contains the MIRAGE implementation, the four baseline schemes (Tree-Ring, Gaussian Shading, PRC, T2SMark), and the evaluation scripts that produce the experimental results in the paper.

> *Submission under double-blind review. Do not contact or attempt to deanonymize the authors.*

---

## 1. Repository Layout

```
mirage_release/
├── README.md
├── MarkDiffusion/                # MIRAGE + baselines + evaluation
│   ├── watermark/{mirage,gs,prc,tr}/   # Embedding implementations
│   ├── detection/{mirage,gs,prc,tr}/   # Matched decoders
│   ├── config/                         # Per-scheme JSON configurations
│   ├── utils/, evaluation/             # Shared utilities + attack pipelines
│   ├── runtime_utils.py, sdp_n500_config.py
│   ├── eval_*.py                       # Evaluation scripts
│   ├── requirements.txt
│   └── LICENSE                         # Apache 2.0
└── T2SMark/                            # T2SMark baseline (upstream, Apache 2.0)
```

## 2. Installation

```bash
conda create -n mirage python=3.10 -y
conda activate mirage
cd MarkDiffusion
pip install -r requirements.txt
```

Stable Diffusion 2.1-base is the default backbone; weights download from Hugging Face on first use. SDXL and FLUX.1-dev are also supported but require additional disk space and longer inversion.

## 3. Quick Start

Embed a watermark and run the standard attack suite:

```bash
cd MarkDiffusion
python eval_n500_standard_attacks.py --gpu 0 --num-images 100
```

Run the SHIFT cross-scheme comparison at $\lambda = 0.7$:

```bash
python eval_n500_shift_sdp.py --gpu 0 --scheme MIRAGE --lambdas 0.7
python eval_n500_shift_sdp.py --gpu 0 --scheme GS    --lambdas 0.7
python eval_n500_shift_sdp.py --gpu 0 --scheme PRC   --lambdas 0.7
python eval_n500_shift_sdp.py --gpu 0 --scheme TR    --lambdas 0.7
```

Run the erasure-aware white-box adaptive attack (worst-case key compromise):

```bash
python eval_n200_adaptive_whitebox_lpips.py --gpu 0 --num-images 100 \
    --latent-budgets 5.0,20.0,50.0,100.0,200.0
```

## 4. Mapping from Paper Results to Scripts

Table and figure numbers below match the compiled paper. Image quality (Table 3,
FID/KID/IS) is computed with standard `torchmetrics` estimators over generated
images and has no dedicated script.

| Paper item                                        | Script                                              |
|---------------------------------------------------|-----------------------------------------------------|
| Table 2  (MS-COCO robustness, N=1000)             | `eval_n500_standard_attacks.py`                     |
| Table 4  (TPR at extreme single-attack)           | `eval_n500_strength_sweep_sdp.py`                   |
| Table 5  (Regeneration img2img attack)            | `eval_n500_standard_attacks_sdp.py` (regen mode)    |
| Table 6  (Advanced removal: SHIFT 0.3-0.85, learned, UnMarker) | `eval_n500_shift_sdp.py`, `eval_n500_shift_sdp_high.py`, `eval_n500_learned_detect_sdp.py`, `eval_n500_unmarker_detect_sdp.py` |
| Table 7  (Cross-scheme counterfactual at SHIFT 0.7) | `eval_n500_gzs_shift_sdp.py`                      |
| Table 8  (McNemar paired tests)                   | aggregation over outputs of Table 6 scripts         |
| Table 9  (Threshold-free / fixed-FPR AUC)         | aggregation over OOD null + Table 6 outputs         |
| Table 10 (Scheme-aware adaptive attacks)          | `eval_n500_shift_sdp.py` (adaptive variants)        |
| §6.4 (Erasure-aware white-box adaptive, inline)   | `eval_n200_adaptive_whitebox_lpips.py`              |
| Table 11 (OOD false positive study)               | `eval_n500_standard_attacks.py` (OOD null mode)     |
| Table 12 (MIRAGE ablation)                        | `eval_n500_standard_attacks.py` (config sweep)      |
| §B (Held-out calibration, prose)                  | `eval_n500_heldout.py`                              |
| Table 13 (Mechanism disentangling M1-M4)          | `eval_n500_gzs_shift_sdp.py`                        |
| Table 14 (Unseen and combo attacks)               | `eval_n500_unseen_combo_sdp.py`                     |
| Table 15 (SD 1.5 cross-backbone)                  | `eval_n500_sd15.py`                                 |
| Table 16 (SDXL and FLUX cross-backbone)           | `sdxl_mirage.py`, `flux_mirage_sdp_n500.py`         |
| Table 17 (Stratified per-magnitude flip rate)     | `eval_n500_sigma_table.py`                          |
| Table 18 (Empirical undetectability test battery) | `eval_undetectability.py`                           |
| Table 19 (Runtime per stage)                      | `measure_runtime.py`                                |
| Figure 4 (Sheppard bound validation / sigma_eff)  | `eval_sign_agreement_attacks.py`                    |
| Figure 6 (SHIFT TPR across schemes)               | `eval_n500_shift_sdp.py`, `eval_n500_shift_sdp_high.py` |
| §6.6 (T2SMark key-length security, text)          | `eval_t2smark_shortkey_search_calibration.py`       |

T2SMark variants use `eval_n500_t2smark_*` counterparts (T2SMark requires a separate inversion pipeline with its own checkpoint).

## 5. Reproducibility

- All scripts seed PyTorch and NumPy via `runtime_utils.configure_runtime()`.
- The cross-scheme SDP slice is fixed: 100 prompts × 5 seeds = 500 images, identical across schemes for paired statistical tests.
- Raw per-image scores are logged as JSON; aggregation recomputes paper numbers from these logs.
- Confidence intervals use the Wilson score; McNemar tests use Edwards' continuity correction.

## 6. Licensing

Apache 2.0 (`MarkDiffusion/LICENSE`, `T2SMark/LICENSE`).

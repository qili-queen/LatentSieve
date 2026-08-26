# LatentSieve: Reliability-Aware Decoding for Noise-Space Diffusion Watermarks

Anonymous code release for the USENIX Security 2027 submission **LatentSieve: Reliability-Aware Decoding for Noise-Space Diffusion Watermarks**.

Contains the LatentSieve implementation, the four baseline schemes (Tree-Ring, Gaussian Shading, PRC, T2SMark), and the evaluation scripts that produce the experimental results in the paper.

> *Submission under double-blind review. Do not contact or attempt to deanonymize the authors.*

---

> **Naming.** The scheme is called LatentSieve in the paper. The code predates the
> rename and still uses `MIRAGE` for the algorithm name, config files
> (`config/MIRAGE_presence.json`), and output paths. They refer to the same scheme.

---

## 1. Repository Layout

```
latentsieve_release/
├── README.md
├── assets/gallery/              # README visual examples
├── tools/                       # Release helper scripts
├── MarkDiffusion/                # LatentSieve + baselines + evaluation
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
conda create -n latentsieve python=3.10 -y
conda activate latentsieve
cd MarkDiffusion
pip install -r requirements.txt
```

Stable Diffusion 2.1-base is the default backbone; weights download from Hugging Face on first use. SDXL and FLUX.1-dev are also supported but require additional disk space and longer inversion.

## 3. Visual Examples

Each row below is one SDP prompt/seed pair. Columns compare the paired unwatermarked original with LatentSieve and four baseline watermarks.

**Clean watermarked outputs**

![Clean watermarked outputs](assets/gallery/clean_watermarks.png)

**After Ctrlgen removal attack**

![Ctrlgen-attacked watermarked outputs](assets/gallery/ctrlgen_attacked_watermarks.png)

**After UnMarker-S2 removal attack**

![UnMarker-S2-attacked watermarked outputs](assets/gallery/unmarker_s2_attacked_watermarks.png)

**SHIFT trajectory deflection ($\lambda = 0.5$)**

![SHIFT-attacked watermarked outputs at lambda 0.5](assets/gallery/shift_attacked_lambda05_watermarks.png)

**SHIFT trajectory deflection ($\lambda = 0.7$)**

![SHIFT-attacked watermarked outputs at lambda 0.7](assets/gallery/shift_attacked_watermarks.png)

The checked-in gallery images were built from cached evaluation outputs. The helper script `tools/make_readme_gallery.py` is for release preparation and expects those cached outputs to be present locally.

## 4. Quick Start

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


## 5. Reproducibility

- All scripts seed PyTorch and NumPy via `runtime_utils.configure_runtime()`.
- The cross-scheme SDP slice is fixed: 100 prompts × 5 seeds = 500 images, identical across schemes for paired statistical tests.
- Raw per-image scores are logged as JSON; aggregation recomputes paper numbers from these logs.
- Confidence intervals use the Wilson score; McNemar tests use Edwards' continuity correction.

## 6. Licensing

Apache 2.0 (`MarkDiffusion/LICENSE`, `T2SMark/LICENSE`).

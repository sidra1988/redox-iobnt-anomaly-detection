# Anomaly Detection at Redox-Based Bio-Cyber Interfaces

Code accompanying *Hybrid Deep Learning Techniques for Securing Redox Based
Interfaces in Internet of Bio Nano Things*.

The corpus used in the paper is synthetic and generated deterministically from
an electrochemical forward model. Rather than distributing a data dump, this
repository contains the generator, so the exact corpus used in every reported
experiment can be reproduced from a fixed seed.

## Contents

| File | Purpose |
|---|---|
| `rbi_all.py` | Signal generation, fidelity checks, six deep architectures, six baselines, cross-validation protocol, resampling ablation |
| `rbi_report.py` | Builds the results tables and figures from a completed run |
| `rbi_eda.py` | Exploratory analysis of the window-level feature representation |

## Requirements

```
python >= 3.9
numpy, pandas, scipy, scikit-learn, matplotlib
tensorflow >= 2.10   (GPU strongly recommended)
```

```bash
pip install numpy pandas scipy scikit-learn matplotlib tensorflow
```

## Reproducing the reported results

**Dataset fidelity analysis (Section 4.1.5, Figure 4).** Under a minute, no GPU
needed.

```bash
python rbi_all.py --mode fidelity
```

Expected: Randles–Sevcik R² = 1.0000 with slope within 2.13% of theory; Tafel
transfer coefficient 0.5001 against an input of 0.5000; Cottrell R² = 1.0000;
peak separation 61.6 mV against the 59 mV reversible limit.

**Main experiment (Tables 8–9, Figures 11–14).** Roughly 1–2 hours on an
NVIDIA T4.

```bash
python rbi_all.py --mode experiment --n 6000 --folds 10 --out results
```

**Resampling ablation (Table 7, Figure 10).** Roughly 9–12 hours on a T4.

```bash
python rbi_all.py --mode ablation --n 6000 --folds 10 --out ablation
```

**Exploratory analysis (Table 5, Figures 5–9).**

```bash
python rbi_eda.py --n 6000 --out paper_assets
```

**Tables and figures from a completed run.**

```bash
python rbi_report.py --results results --ablation ablation --out paper_assets
```

Every run checkpoints after each fold and resumes if interrupted; rerun the
same command to continue.

## Reproducibility

The generator is deterministic. `generate_corpus(n_windows=6000, seed=42)`
returns the identical corpus on any platform, so the dataset, the fidelity
checks, and all classical and statistical baselines reproduce exactly.

The deep architectures do not reproduce bit-for-bit. GPU-accelerated training
with cuDNN uses non-deterministic reduction ordering in convolution and
recurrent kernels, so independent runs of identical code with an identical seed
differ by roughly one to three percentage points in mean F1. This is discussed
in Section 5.6 of the paper and is the reason results are reported as
cross-validated means with confidence intervals rather than single point
estimates. The ordering of architectures was preserved across independent
executions on separate hardware.

## Resampling conditions

`--resampling` selects one of three conditions, used for the ablation in
Table 7:

- `none` — no resampling for any model
- `smote_raw` — SMOTE interpolated between raw sequences (deep models) and
  between feature vectors (classical baselines)
- `window` — window-based oversampling with jitter and circular shift (deep
  models), feature-space SMOTE (classical baselines)

`window` is the condition used for the primary results in Table 8. Resampling
is applied only to the fit split of each training fold, never to the
calibration split or the test fold.

## Attack classes

Six adversarial classes (A1–A6) and two benign perturbation classes (B1–B2)
are generated, following the taxonomy in Table 3 of the paper. B1 and B2 are
assigned to the negative class, so a detector cannot succeed by flagging any
departure from the quiescent baseline.

## Citation

```bibtex
@article{zafar2026redox,
  title   = {Hybrid Deep Learning Techniques for Securing Redox Based
             Interfaces in Internet of Bio Nano Things},
  author  = {Zafar, Sidra and Iqbal, Saba and Shah, Asghar Ali and
             Jabbar, Sohail and Rehman, Zia Ul},
  journal = {Engineering Reports},
  year    = {2026}
}
```

## License

[Choose one — MIT and Apache-2.0 are both common for research code.]

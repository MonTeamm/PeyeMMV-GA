[![DOI](https://zenodo.org/badge/1249778361.svg)](https://doi.org/10.5281/zenodo.21725842)

# PeyeMMV Synthetic Gaze Generation — User Guide (English)

> **Research project**: Synthetic gaze data generation for dyslexia detection using fractional Brownian motion (fBm) optimised by a Genetic Algorithm (GA2), compared against Stochastic Gaussian Generation (SGG) and Deterministic Centroid Minimisation (DCM) baselines.
> **Dataset**: ETDD70 (70 subjects, tasks T4 and T5).

---

## Directory structure

```
project_root/
├── engine.py              ← GA engine, fBm synthesis, PeyeMMV fixation detector
├── Utils.py               ← Shared utilities (sigma estimation, metrics, I/O)
├── find_theta_ga2.py      ← Phase 1 : find optimal θ* per subject-task
├── ga2_generate.py        ← Phase 2 : generate one GA2 set (for Phase 3 comparison)
├── ga2x20.py              ← Phase 4 : generate 20 GA2 sets (TSTR)
├── sgg.py                 ← Phase 2 : generate one SGG set
├── sggx20.py              ← Phase 4 : generate 20 SGG sets (TSTR)
├── dcm.py                 ← Phase 2 : generate one DCM set
├── dcmx20.py              ← Phase 4 : generate 20 DCM sets (TSTR)
├── compare_generators.py  ← Phase 3 : compare the three methods
├── evaluate_tstr.py       ← Phase 5 : TSTR evaluation
├── data/                  ← ETDD70 dataset (download separately — see below)
│   ├── Subject_{sid}_{task}_raw.csv
│   ├── Subject_{sid}_{task}_fixations.csv
│   ├── Subject_{sid}_{task}_metrics.csv
│   └── dyslexia_class_label.csv
└── rois/                  ← Region-of-interest definitions
    ├── Meaningful_Text_rois.csv
    └── Pseudo_Text_rois.csv
```

---

## Data download

Two separate downloads are required before running any script.

### 1. ETDD70 dataset (~500 MB) — original eye-tracking data

The raw ETDD70 data is **not included** in this repository. Download it and place all files inside the `data/` folder.

**Download**: [ETDD70 dataset (~500 MB)](https://drive.google.com/drive/folders/15tfAwfem7B489nL317xOkOsJdb5EtDNN?usp=sharing)

## Requirements

Install all required libraries before running any script:

```bash
pip install numpy pandas scipy tqdm scikit-learn matplotlib statsmodels
```

Optional classifiers (needed only for `--classifier xgb` or `--classifier catboost`):

```bash
pip install xgboost catboost
```

Python **3.10 or later** is required. Tested on Python 3.11 and 3.12.

---

## Quick Start & Reproducing the full pipeline

Since the project is packaged as a standard Python software, you can execute the pipeline using the installed CLI commands.

````bash
# 1. Install the package
pip install peyemmv-ga-Mon

# 2. Optimize subject-task parameters
peyemmv-find-theta \
  --data_dir ./data \
  --output_root ./syn_output \
  --pop 100 \
  --gens 50 \
  --w1 8 \
  --w2 2 \
  --w3 10

# 3. Generate one GA2 dataset per subject-task
peyemmv-ga2-generate \
  --data_dir ./data \
  --output_dir ./ga2_output \
  --theta_csv ./syn_output/ga2_theta_star.csv

# 4. Generate repeated sets for TSTR
peyemmv-ga2x20 \
  --theta_csv ./syn_output/ga2_theta_star.csv \
  --output_root ./ga2_tstr_output \
  --n_sets 20

peyemmv-sggx20 \
  --data_dir ./data \
  --output_dir ./sgg_output \
  --tstr_output_dir ./sgg_tstr_output \
  --n_sets 20

peyemmv-dcmx20 \
  --data_dir ./data \
  --output_dir ./dcm_output \
  --tstr_output_dir ./dcm_tstr_output \
  --n_sets 20

# 5. Compare generation methods
peyemmv-compare \
  --data_dir ./data \
  --ga2_dir ./ga2_output \
  --sgg_dir ./sgg_output \
  --dcm_dir ./dcm_output \
  --output_root ./phase1_results

# 6. Evaluate GA2 with TSTR
peyemmv-evaluate-tstr \
  --syn_root ./ga2_tstr_output \
  --data_dir ./data \
  --output_root ./tstr_results/ga2 \
  --classifier all \
  --n_sets 20

---

## Checkpoint and progress monitoring

```bash
find ./ga2_tstr_output -name "*_metrics_syn_*.csv" | wc -l
find ./sgg_tstr_output -name "*_metrics_syn_*.csv" | wc -l
find ./dcm_tstr_output -name "*_metrics_syn_*.csv" | wc -l
````

Target: **2,800 files** per method.

Remove empty files if a run was interrupted:

```bash
find ./ga2_tstr_output -name "*_metrics_syn_*.csv" -empty -delete
find ./sgg_tstr_output -name "*_metrics_syn_*.csv" -empty -delete
find ./dcm_tstr_output -name "*_metrics_syn_*.csv" -empty -delete
```

All scripts support **automatic checkpointing** — re-running the same command resumes from where it stopped.

---

## Reproducibility

All random seeds are derived deterministically from subject ID, task name, and a global seed (`GLOBAL_SEED = 42`). Re-running any phase with the same arguments produces identical output files.

---

## Citation and Code Availability

The Python package `peyemmv-ga-Mon` implementing the proposed framework is openly available via `pip install peyemmv-ga-Mon`.

- **PyPI Package**: [https://pypi.org/project/peyemmv-ga-Mon/](https://pypi.org/project/peyemmv-ga-Mon/)
- **Software Archive (DOI)**: [https://doi.org/10.5281/zenodo.21725842](https://doi.org/10.5281/zenodo.21725842)
- **Dataset Archive (ETDD70)**: [https://doi.org/10.5281/zenodo.17513247](https://doi.org/10.5281/zenodo.17513247)
- **Version used in the paper**: `v1.0.0`

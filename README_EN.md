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

## Running the full pipeline

> **Note — command formatting**: All commands below are written on a single line for maximum compatibility. The `\` line-continuation syntax works on Linux/macOS. On **Windows**, replace `\` with `` ` `` (PowerShell) or remove line breaks entirely and type as one line.

### Phase 1 — Find optimal θ\* parameters

```bash
python find_theta_ga2.py --data_dir ./data --output_root ./syn_output --pop 100 --gens 50 --w1 8 --w2 2 --w3 10 --max_workers 4
```

**Outputs**:

- `syn_output/ga2_theta_star.csv` — optimal parameters per subject-task
- `syn_output/ga2_theta_summary.json` — summary statistics
- `syn_output/ga2_theta_checkpoint.json` — checkpoint (resume-safe if interrupted)

| Argument        | Default        | Description               |
| --------------- | -------------- | ------------------------- |
| `--data_dir`    | `./data`       | ETDD70 data directory     |
| `--output_root` | `./syn_output` | Output directory          |
| `--pop`         | 100            | GA population size        |
| `--gens`        | 50             | Number of GA generations  |
| `--w1`          | 8.0            | Detection penalty weight  |
| `--w2`          | 2.0            | Outlier penalty weight    |
| `--w3`          | 10.0           | Spectral error weight     |
| `--max_workers` | 4              | Parallel worker processes |

---

### Phase 2 — Generate one synthetic set (for comparison)

```bash
python ga2_generate.py --data_dir ./data --output_dir ./ga2_output --theta_csv ./syn_output/ga2_theta_star.csv --max_workers 4

python sgg.py --data_dir ./data --output_dir ./sgg_output

python dcm.py --data_dir ./data --output_dir ./dcm_output
```

---

### Phase 3 — Compare the three methods

```bash
python compare_generators.py --data_dir ./data --ga2_dir ./ga2_output --sgg_dir ./sgg_output --dcm_dir ./dcm_output --output_root ./phase3_results
```

---

### Phase 4 — Generate 20 sets each (TSTR)

```bash
python ga2x20.py --theta_csv ./syn_output/ga2_theta_star.csv --output_root ./ga2_tstr_output --n_sets 20 --max_workers 4

python sggx20.py --data_dir ./data --output_dir ./sgg_output --tstr_output_dir ./sgg_tstr_output --n_sets 20

python dcmx20.py --data_dir ./data --output_dir ./dcm_output --tstr_output_dir ./dcm_tstr_output --n_sets 20
```

**Expected output**: 2,800 files per method (140 subject-task pairs × 20 sets).

---

### Phase 5 — TSTR evaluation

```bash
python evaluate_tstr.py --syn_root ./ga2_tstr_output --data_dir ./data --output_root ./tstr_results/ga2 --classifier all

python evaluate_tstr.py --syn_root ./sgg_tstr_output --data_dir ./data --output_root ./tstr_results/sgg --classifier all

python evaluate_tstr.py --syn_root ./dcm_tstr_output --data_dir ./data --output_root ./tstr_results/dcm --classifier all
```

---

## Checkpoint and progress monitoring

```bash
find ./ga2_tstr_output -name "*_metrics_syn_*.csv" | wc -l
find ./sgg_tstr_output -name "*_metrics_syn_*.csv" | wc -l
find ./dcm_tstr_output -name "*_metrics_syn_*.csv" | wc -l
```

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

## Citation

If you use this code, the ETDD70 dataset, or the pre-computed outputs in your work, please cite the paper:

```
[Citation will be added after publication]
```

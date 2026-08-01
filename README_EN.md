[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21730320.svg)](https://doi.org/10.5281/zenodo.21730320)
[![PyPI](https://img.shields.io/pypi/v/peyemmv-ga-Mon.svg)](https://pypi.org/project/peyemmv-ga-Mon/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# PeyeMMV Synthetic Gaze Generation — User Guide (English)

> **Research project**: Synthetic gaze data generation for dyslexia detection using fractional Brownian motion (fBm) optimised by a Genetic Algorithm (GA2), compared against Stochastic Gaussian Generation (SGG) and Deterministic Centroid Minimisation (DCM) baselines.
> **Dataset**: ETDD70 (70 subjects, tasks T4 and T5).

---

## Directory structure

```
project_root/
├── src/peyemmv_ga/
│   ├── engine.py              ← GA engine, fBm synthesis, PeyeMMV fixation detector
│   ├── utils.py                ← Shared utilities (sigma estimation, metrics, I/O)
│   └── cli/
│       ├── find_theta_ga2.py   ← Phase 1 : find optimal θ* per subject-task
│       ├── ga2_generate.py     ← Phase 2 : generate one GA2 set (for Phase 3 comparison)
│       ├── ga2x20.py           ← Phase 4 : generate 20 GA2 sets (TSTR)
│       ├── sgg.py               ← Phase 2 : generate one SGG set
│       ├── sggx20.py            ← Phase 4 : generate 20 SGG sets (TSTR)
│       ├── dcm.py               ← Phase 2 : generate one DCM set
│       ├── dcmx20.py            ← Phase 4 : generate 20 DCM sets (TSTR)
│       ├── compare_generators.py ← Phase 3 : compare the three methods
│       └── evaluate_tstr.py     ← Phase 5 : TSTR evaluation
├── examples/                    ← Example scripts, sample data, reproduction notebook
│   ├── sample_data/              ← Small sample subjects (no full ETDD70 download needed)
│   ├── example_01_find_theta.py
│   └── reproduce_paper_results.ipynb
├── docs/                        ← Documentation source (see REPOSITORY_STRUCTURE.md below)
├── data/                        ← ETDD70 dataset (download separately — see below)
│   ├── Subject_{sid}_{task}_raw.csv
│   ├── Subject_{sid}_{task}_fixations.csv
│   ├── Subject_{sid}_{task}_metrics.csv
│   └── dyslexia_class_label.csv
├── rois/                        ← Region-of-interest definitions
│   ├── Meaningful_Text_rois.csv
│   └── Pseudo_Text_rois.csv
├── pyproject.toml
├── environment.yml
└── requirements.txt
```

For a comprehensive breakdown of all source files, command-line interfaces, and data schema specifications, see [REPOSITORY_STRUCTURE.md](./docs/REPOSITORY_STRUCTURE.md).

---

## Data download

Two separate downloads are required before running any script.

### 1. ETDD70 dataset (~500 MB) — original eye-tracking data

The raw ETDD70 data is **not included** in this repository. Download it and place all files inside the `data/` folder.

**Download**: [ETDD70 dataset (~500 MB)](https://drive.google.com/drive/folders/15tfAwfem7B489nL317xOkOsJdb5EtDNN?usp=sharing)

## Quick test with sample data (no need to download full ETDD70)

The repository includes sample subjects at `examples/sample_data/`, so you can try the pipeline in seconds before downloading the full dataset:

```bash
python examples/example_01_find_theta.py
```

## Using your own data instead of ETDD70

The pipeline works with other eye-tracking datasets as long as the format matches:

- `Subject_{sid}_{task}_raw.csv` — required columns: see `load_raw_gaze()` in `src/peyemmv_ga/engine.py`
- `Subject_{sid}_{task}_fixations.csv` — required columns: see `load_fixations()` in `src/peyemmv_ga/engine.py`
- `dyslexia_class_label.csv` — required columns: `subject_id, class_id` (0/1)

Place your files with matching names/format in any folder, then point `--data_dir` to that folder — no code changes needed.

## Requirements

Install all required libraries before running any script:

```bash
pip install numpy pandas scipy tqdm scikit-learn matplotlib statsmodels
```

Optional classifiers (needed only for `--classifier xgb` or `--classifier catboost`):

```bash
pip install xgboost catboost
```

Python **3.10 or later** is required. Tested on Python 3.13.

Alternatively, using conda:

```bash
conda env create -f environment.yml
conda activate peyemmv-ga
```

---

## Quick Start & Reproducing the full pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MonTeamm/PeyeMMV-GA/blob/main/examples/reproduce_paper_results.ipynb)

Since the project is packaged as a standard Python software, you can execute the pipeline using the installed CLI commands after `pip install peyemmv-ga-Mon`.

### Phase 1 — Find optimal θ\* parameters

```bash
peyemmv-find-theta --data_dir ./data --output_root ./syn_output --pop 100 --gens 50 --w1 8 --w2 2 --w3 10 --max_workers 4
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
peyemmv-ga2-generate --data_dir ./data --output_dir ./ga2_output --theta_csv ./syn_output/ga2_theta_star.csv --max_workers 4

peyemmv-sgg --data_dir ./data --output_dir ./sgg_output

peyemmv-dcm --data_dir ./data --output_dir ./dcm_output
```

---

### Phase 3 — Compare the three methods

```bash
peyemmv-compare --data_dir ./data --ga2_dir ./ga2_output --sgg_dir ./sgg_output --dcm_dir ./dcm_output --output_root ./phase3_results
```

---

### Phase 4 — Generate 20 sets each (TSTR)

```bash
peyemmv-ga2x20 --theta_csv ./syn_output/ga2_theta_star.csv --output_root ./ga2_tstr_output --n_sets 20 --max_workers 4

peyemmv-sggx20 --data_dir ./data --output_dir ./sgg_output --tstr_output_dir ./sgg_tstr_output --n_sets 20

peyemmv-dcmx20 --data_dir ./data --output_dir ./dcm_output --tstr_output_dir ./dcm_tstr_output --n_sets 20
```

**Expected output**: 2,800 files per method (140 subject-task pairs × 20 sets).

---

### Phase 5 — TSTR evaluation

```bash
peyemmv-evaluate-tstr --syn_root ./ga2_tstr_output --data_dir ./data --output_root ./tstr_results/ga2 --classifier all

peyemmv-evaluate-tstr --syn_root ./sgg_tstr_output --data_dir ./data --output_root ./tstr_results/sgg --classifier all

peyemmv-evaluate-tstr --syn_root ./dcm_tstr_output --data_dir ./data --output_root ./tstr_results/dcm --classifier all
```

> **Note**: On Windows PowerShell, replace the multi-line `\` continuations shown above with a single line, or use `` ` `` as the line-continuation character.

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

Dependency versions are pinned in `requirements.txt`, `pyproject.toml`, and `environment.yml`, matching the exact environment used to generate the results reported in the paper (Python 3.13).

---

## Documentation & Examples

- Repository structure reference: [`docs/REPOSITORY_STRUCTURE.md`](./docs/REPOSITORY_STRUCTURE.md)
- Example scripts demonstrating individual functions: [`examples/`](examples/)
- End-to-end reproduction notebook (runnable on Google Colab, badge above): [`examples/reproduce_paper_results.ipynb`](examples/reproduce_paper_results.ipynb)
- Changelog: [`CHANGELOG.md`](CHANGELOG.md)

---

## Citation and Code Availability

The Python package `peyemmv-ga-Mon` implementing the proposed framework is openly available under the MIT license and can be installed via `pip install peyemmv-ga-Mon`.

- **PyPI Package**: https://pypi.org/project/peyemmv-ga-Mon/
- **Software Archive (DOI, packaged release)**: https://doi.org/10.5281/zenodo.21730320
- **Dataset Archive (ETDD70)**: https://doi.org/10.5281/zenodo.17513247
- **Version referenced**: `v1.0.1`

If you use this software, please cite:

**Software:**

```
Nguyen, T.Q.H., Nguyen, D.H., Tran, G.L., Le, T.H., Ngo, T.D. (2026).
PeyeMMV-GA (v1.0.1) [Software]. Zenodo. https://doi.org/10.5281/zenodo.21730320
```

**Paper:**

```
[Citation will be added after publication]
```

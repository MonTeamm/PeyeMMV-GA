# PeyeMMV-GA Code Documentation

**Package:** `peyemmv-ga`  
**Repository:** `MonTeamm/PeyeMMV-GA`  
**Documentation scope:** current `main` branch, reviewed on 2026-07-31  
**Primary package directory:** `src/peyemmv_ga/`

---

## 1. Purpose and scope

PeyeMMV-GA is a Python package for generating synthetic eye-tracking gaze data from ETDD70 fixation data. The proposed generator models within-fixation gaze displacement using fractional Brownian motion (fBm). A genetic algorithm, referred to in the command-line tools as **GA2**, searches for subject-task-specific parameters that balance:

- successful fixation detection by the PeyeMMV validation procedure;
- a low proportion of spatial outliers;
- similarity between synthetic and real spectral and dispersion characteristics.

The repository also implements two comparison baselines:

- **SGG — Stochastic Gaussian Generation:** independent Gaussian gaze samples around each fixation centroid;
- **DCM — Deterministic Centroid Minimization:** almost static gaze around the fixation centroid with near-zero noise and drift.

The generated data can be evaluated in two complementary ways:

1. **Generator-level comparison:** ACR, CLE, VPC, JSD, and PSD slope;
2. **TSTR evaluation:** Train on Synthetic, Test on Real for dyslexia classification.

This document describes the package architecture, processing workflow, main data structures, modules, functions, command-line programs, inputs, and outputs.

---

## 2. Package architecture

```text
src/
└── peyemmv_ga/
    ├── __init__.py
    ├── engine.py
    ├── utils.py
    └── cli/
        ├── __init__.py
        ├── find_theta_ga2.py
        ├── ga2_generate.py
        ├── ga2x20.py
        ├── sgg.py
        ├── sggx20.py
        ├── dcm.py
        ├── dcmx20.py
        ├── compare_generators.py
        └── evaluate_tstr.py
```

### Module responsibilities

| Module                      | Responsibility                                                                                                                                               |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `engine.py`                 | Core fBm model, baseline estimation, PeyeMMV validation, fitness computation, genetic algorithm, parameter experiments, and the main GA generation pipeline. |
| `utils.py`                  | Shared data loading, normalization, sigma estimation, sampling-rate selection, ROI handling, and ETDD70-compatible metric computation.                       |
| `cli/find_theta_ga2.py`     | Optimizes and stores the best GA2 parameter vector for every subject-task pair.                                                                              |
| `cli/ga2_generate.py`       | Generates one GA2 synthetic gaze dataset per subject-task, using saved parameters or online optimization.                                                    |
| `cli/ga2x20.py`             | Generates multiple independent GA2 datasets for TSTR experiments.                                                                                            |
| `cli/sgg.py`                | Generates one SGG baseline dataset per subject-task.                                                                                                         |
| `cli/sggx20.py`             | Generates multiple SGG datasets and optional TSTR-compatible metrics.                                                                                        |
| `cli/dcm.py`                | Generates one near-static DCM baseline dataset per subject-task.                                                                                             |
| `cli/dcmx20.py`             | Generates multiple DCM datasets and optional TSTR-compatible metrics.                                                                                        |
| `cli/compare_generators.py` | Compares GA2, SGG, and DCM with generator-level metrics, statistical tests, and figures.                                                                     |
| `cli/evaluate_tstr.py`      | Evaluates synthetic data using Base%, CVSyn%, and TSTR% classification protocols.                                                                            |

---

## 3. End-to-end processing workflow

```text
ETDD70 raw gaze + fixation files
                │
                ▼
        Load and normalize data
                │
                ├── Detect sampling rate
                ├── Estimate H_base
                ├── Estimate sigma_base
                └── Estimate real PSD target
                │
                ▼
       Optimize θ = (H, σ, φ, seed)
                │
                ├── Generate fBm displacement
                ├── Apply PeyeMMV validation
                ├── Compute D, O, and E
                └── Evolve population with GA
                │
                ▼
       Save θ* for each subject-task
                │
                ▼
       Generate synthetic gaze datasets
                │
                ├── GA2 proposed method
                ├── SGG noisy baseline
                └── DCM near-static baseline
                │
                ├──────────────────────────┐
                ▼                          ▼
 Generator-level comparison          TSTR evaluation
 ACR, CLE, VPC, JSD, alpha      Base%, CVSyn%, TSTR%
 ANOVA, Tukey HSD, figures      Accuracy, F1, AUC, MCC
```

---

## 4. Expected input data

### 4.1 Raw gaze file

Expected naming pattern:

```text
Subject_{subject_id}_{task}_raw.csv
```

The GA2 engine expects the following ETDD70 columns:

| Column         | Description                                                               |
| -------------- | ------------------------------------------------------------------------- |
| `time`         | Raw timestamp, interpreted as microseconds and converted to milliseconds. |
| `gaze_x_left`  | Horizontal gaze coordinate from the left eye.                             |
| `gaze_y_left`  | Vertical gaze coordinate from the left eye.                               |
| `gaze_x_right` | Horizontal gaze coordinate from the right eye.                            |
| `gaze_y_right` | Vertical gaze coordinate from the right eye.                              |

The shared utilities also accept several alternative timestamp and gaze-column names. When both eyes are available, binocular coordinates are formed by averaging valid left- and right-eye samples.

### 4.2 Fixation file

Expected naming pattern:

```text
Subject_{subject_id}_{task}_fixations.csv
```

Canonical fields used by `engine.py`:

| Column        | Description                                                        |
| ------------- | ------------------------------------------------------------------ |
| `start_ms`    | Fixation start time in milliseconds.                               |
| `end_ms`      | Fixation end time in milliseconds.                                 |
| `duration_ms` | Fixation duration.                                                 |
| `fix_x`       | Reference fixation centroid, horizontal coordinate.                |
| `fix_y`       | Reference fixation centroid, vertical coordinate.                  |
| `id`          | Optional fixation identifier; generated automatically when absent. |

`utils.normalize_fixation_dataframe()` maps alternative names such as `x_fix`, `y_fix`, `start_time`, and `end_time` to the normalized representation used by the baseline generators.

### 4.3 Class-label file

```text
dyslexia_class_label.csv
```

At minimum, the file must contain:

- `subject_id`;
- a binary class column such as `class_id`, `label`, `dyslexia`, `class`, `diagnosis`, or `target`.

### 4.4 ROI files

ROI files are used when AOI-level TSTR metrics are requested. The repository maps the two main tasks to their corresponding ROI definitions through `ROI_MAP`.

---

## 5. Core data structures in `engine.py`

### 5.1 `Individual`

`Individual` is a dataclass representing one genetic-algorithm candidate.

| Field     | Meaning                                                                                       |
| --------- | --------------------------------------------------------------------------------------------- |
| `H`       | Hurst exponent controlling temporal correlation in the fast fBm component.                    |
| `sigma`   | Spatial scale of the fast displacement component.                                             |
| `phi`     | Rotation angle controlling the orientation of the two-dimensional displacement/drift process. |
| `seed`    | Random seed encoded as part of the genotype.                                                  |
| `fitness` | Cached objective-function value.                                                              |
| `D`       | Detection component of the objective.                                                         |
| `O`       | Outlier component of the objective.                                                           |
| `E`       | Spectral and dispersion error component.                                                      |

Methods:

| Method      | Description                                                                      |
| ----------- | -------------------------------------------------------------------------------- |
| `copy()`    | Creates an independent copy of an individual, including cached objective values. |
| `to_dict()` | Converts the dataclass to a serializable dictionary for CSV/JSON output.         |

### 5.2 `SubjectTaskBaseline`

This dataclass stores parameters estimated from the real data of one subject-task pair.

| Field              | Meaning                                                       |
| ------------------ | ------------------------------------------------------------- |
| `H_base`           | Subject-task baseline Hurst exponent.                         |
| `sigma_base`       | Baseline within-fixation spatial dispersion.                  |
| `target_psd_slope` | Reference spectral slope derived from real gaze.              |
| `sampling_rate`    | Detected sampling rate in Hz.                                 |
| `n_fixations_used` | Number of valid fixations used during estimation.             |
| `H_source`         | Text label recording the H estimation method or fallback.     |
| `sigma_source`     | Text label recording the sigma estimation method or fallback. |

---

# 6. `engine.py` function reference

## 6.1 Logging, deterministic seeds, and numerical helpers

| Function                                                              | Inputs                                                    | Return                | Functionality                                                                                                                              |
| --------------------------------------------------------------------- | --------------------------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `setup_logger(name="ga_synthetic")`                                   | Logger name                                               | `logging.Logger`      | Creates a standard console logger using the configured log level and formatting. Existing handlers are reused to avoid duplicate messages. |
| `stable_hash_int(text, mod=10_000_000)`                               | Text key and modulus                                      | Deterministic integer | Converts an MD5 digest into a stable integer. It is used instead of Python's process-dependent `hash()` function.                          |
| `combine_seed(base_seed, fixation_id)`                                | Candidate seed and fixation ID                            | Integer seed          | Produces a deterministic fixation-specific seed while avoiding collisions caused by simple seed addition.                                  |
| `_angular_velocity_from_xy_time(x, y, t_ms=None, sampling_rate=None)` | Gaze coordinates and either timestamps or a sampling rate | NumPy array           | Computes sample-to-sample angular-velocity magnitude. It is reused by fitness and comparison procedures.                                   |
| `_histogram_distribution(values, bin_edges)`                          | Values and histogram bins                                 | Probability vector    | Builds and normalizes a histogram. Empty histograms are handled safely.                                                                    |
| `_jensen_shannon_divergence_normalized(p, q)`                         | Two discrete distributions                                | Float                 | Computes normalized Jensen-Shannon divergence after safe normalization and epsilon protection.                                             |
| `ensure_dir(path)`                                                    | Directory path                                            | `None`                | Creates a directory and all missing parent directories.                                                                                    |
| `reflect_into(value, low, high)`                                      | Value and lower/upper bounds                              | Float                 | Reflects an out-of-range continuous gene back into its interval instead of clipping it at a boundary.                                      |

## 6.2 File discovery and data loading

| Function                                 | Inputs                       | Return                 | Functionality                                                                                                                                                                                                        |
| ---------------------------------------- | ---------------------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `find_file(subject_id, task, file_type)` | Subject, task, and file type | File path or `None`    | Locates a matching ETDD70 raw/fixation file using repository naming conventions.                                                                                                                                     |
| `discover_subjects()`                    | Uses `CFG["ETDD70_DIR"]`     | List of subject IDs    | Discovers available subjects from the configured dataset directory.                                                                                                                                                  |
| `load_raw_gaze(raw_path)`                | Raw ETDD70 CSV path          | Normalized `DataFrame` | Validates required binocular columns, converts microseconds to milliseconds, treats zero-coded blink samples as missing, averages both eyes, removes invalid rows, sorts timestamps, and drops duplicate timestamps. |
| `load_fixations(fix_path)`               | Fixation CSV path            | Normalized `DataFrame` | Validates mandatory fixation columns, converts them to numeric form, adds an ID when absent, removes invalid or non-positive-duration rows, and sorts fixations chronologically.                                     |
| `detect_sampling_rate(raw_df)`           | Normalized raw gaze frame    | Sampling rate in Hz    | Estimates the sampling frequency from the median positive inter-sample interval, while filtering implausibly long intervals.                                                                                         |

## 6.3 Baseline and spectral estimation

| Function                                                             | Inputs                            | Return                | Functionality                                                                                                                                                                                                                         |
| -------------------------------------------------------------------- | --------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hurst_higuchi(series, k_max=None)`                                  | One-dimensional time series       | H estimate or `NaN`   | Estimates the Hurst exponent using Higuchi fractal dimension. It is retained as an alternative or fallback estimator.                                                                                                                 |
| `hurst_from_position_psd(series, sampling_rate)`                     | Position residuals and Hz         | H estimate or `NaN`   | Estimates H from the log-log PSD slope of position residuals under the fBm relationship.                                                                                                                                              |
| `hurst_from_velocity_psd(series, sampling_rate)`                     | Position series and Hz            | H estimate or `NaN`   | Differences the input into velocity increments and estimates H from the fGn spectral relationship.                                                                                                                                    |
| `_compute_psd_slope(x, y, sampling_rate)`                            | Two-dimensional gaze and Hz       | PSD slope             | Computes a combined spectral slope for horizontal and vertical gaze.                                                                                                                                                                  |
| `_compute_psd_slope_within_fixations(raw_df, fix_df, sampling_rate)` | Raw gaze, fixations, Hz           | PSD slope             | Restricts spectral analysis to real fixation windows so that saccades and between-fixation movement do not dominate the target.                                                                                                       |
| `estimate_baseline(raw_df, fix_df, sampling_rate)`                   | Real raw gaze, fixation table, Hz | `SubjectTaskBaseline` | Estimates `H_base`, `sigma_base`, and the target PSD slope. H is aggregated from within-fixation estimates; sigma is based on local residual dispersion consistent with the PeyeMMV window. Fallback sources are recorded explicitly. |

## 6.4 Fractional Brownian motion and displacement generation

| Function                                                                            | Inputs                                           | Return                       | Functionality                                                                                                                                                                                                           |
| ----------------------------------------------------------------------------------- | ------------------------------------------------ | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fbm_spectral(n, H, rng)`                                                           | Number of samples, Hurst exponent, RNG           | Standardized fBm-like series | Generates a real-valued correlated series in the frequency domain using power-law amplitudes, random phases, and conjugate symmetry.                                                                                    |
| `_standardize(x)`                                                                   | Numeric array                                    | Standardized array           | Removes the mean and divides by the standard deviation with protection against near-zero variance.                                                                                                                      |
| `generate_displacement(n_points, H, sigma, phi, sampling_rate, seed, H_drift=None)` | Genotype, sample count, Hz, and optional drift H | `(dx, dy)` arrays            | Generates two-dimensional within-fixation displacement. A fast fBm component models tremor/jitter; a slower correlated drift component models wandering movement. The result is scaled by `sigma` and rotated by `phi`. |

## 6.5 PeyeMMV validation and fitness

| Function                                                 | Inputs                                       | Return                | Functionality                                                                                                                                                                                                                        |
| -------------------------------------------------------- | -------------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `peyemmv_check(x, y, duration_ms)`                       | Synthetic fixation coordinates and duration  | Validation dictionary | Applies the repository's two-stage PeyeMMV-inspired spatial validation. It reports detection status, spatial range, remaining points, removed outliers, duration pass, and threshold passes.                                         |
| `_allowed_radius_px(fixation_row, deg_to_px)`            | Fixation metadata and conversion factor      | Allowed radius        | Computes an allowed spatial radius used during outlier/error evaluation.                                                                                                                                                             |
| `_spectral_error(velocity, target_slope, sampling_rate)` | Synthetic velocity, real target slope, Hz    | Non-negative error    | Measures mismatch between the synthetic spectral slope and the real subject-task target.                                                                                                                                             |
| `evaluate_individual(individual, fixations, baseline)`   | Candidate, fitness fixation subset, baseline | Updated `Individual`  | Generates candidate displacements across selected fixations, executes PeyeMMV checks, and computes the objective components: detection `D`, outlier penalty `O`, and error `E`. The cached fitness follows `F = w1·D − w2·O − w3·E`. |

`E` combines spectral mismatch and spatial-dispersion mismatch. A larger fitness is better.

## 6.6 Genetic algorithm

| Function                                            | Inputs                                       | Return               | Functionality                                                                                                                                                                                                                   |
| --------------------------------------------------- | -------------------------------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `initialize_population(baseline, rng)`              | Subject-task baseline and RNG                | List of `Individual` | Samples the initial population within subject-specific H and sigma bounds and global phi/seed bounds.                                                                                                                           |
| `select_elites(population)`                         | Evaluated population                         | Elite list           | Sorts candidates by fitness and retains the configured elitism proportion.                                                                                                                                                      |
| `tournament_select(population, rng)`                | Population and RNG                           | Parent `Individual`  | Randomly samples a tournament and returns its highest-fitness member.                                                                                                                                                           |
| `crossover(p1, p2, rng)`                            | Two parents and RNG                          | Child `Individual`   | Uses arithmetic crossover for H and sigma, circular averaging for phi, and parent selection for the seed gene.                                                                                                                  |
| `mutate(child, baseline, rng)`                      | Child, baseline, RNG                         | Mutated child        | Applies Gaussian mutation to continuous genes and optional seed resampling. Reflection enforces H and sigma bounds, while phi wraps around the full circle. Cached fitness is cleared.                                          |
| `run_ga(fixations_for_fitness, baseline, rng_seed)` | Fitness subset, baseline, deterministic seed | `(best, history)`    | Runs the complete GA loop: initialize, evaluate, rank, preserve elites, select parents, crossover, mutate, and stop at the generation limit or convergence criterion. It returns the best candidate and per-generation history. |

## 6.7 Synthetic output and subject-task execution

| Function                                                                   | Inputs                                             | Return                      | Functionality                                                                                                                                                                                                                                   |
| -------------------------------------------------------------------------- | -------------------------------------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `generate_synthetic_dataset(fixations, theta, baseline, subject_id, task)` | All fixations, optimized individual, baseline, IDs | Synthetic gaze `DataFrame`  | Generates sample-level synthetic gaze for every fixation using the optimized parameter vector and deterministic fixation-specific seeds.                                                                                                        |
| `process_subject_task(subject_id, task)`                                   | Subject and task                                   | Result dictionary or `None` | Executes the complete GA workflow for one subject-task pair: locate files, load data, detect Hz, estimate baseline, select a fitness subset, run GA, generate all synthetic samples, and save parameters, history, summary, and gaze CSV files. |

## 6.8 Serialization, configuration, and parameter experiments

| Function                                                                                  | Functionality                                                                                                                   |
| ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `atomic_write_json(obj, path)`                                                            | Writes JSON through a temporary file and replaces the target atomically to reduce checkpoint corruption.                        |
| `_json_default(obj)`                                                                      | Converts NumPy and other non-standard values to JSON-compatible Python objects.                                                 |
| `parse_number_list(text, cast=float)`                                                     | Parses comma-separated numeric command-line values.                                                                             |
| `parse_weight_candidates(text)`                                                           | Parses candidate fitness-weight combinations used by parameter tests.                                                           |
| `config_snapshot(keys=None)`                                                              | Copies selected or all current configuration values.                                                                            |
| `set_cfg_values(values)`                                                                  | Temporarily patches the global configuration and returns the previous values.                                                   |
| `restore_cfg_values(old)`                                                                 | Restores a previous configuration snapshot.                                                                                     |
| `make_subject_split_for_parameter_test(subjects, calib_subjects, heldout_subjects, seed)` | Creates deterministic calibration and held-out subject sets for hyperparameter testing.                                         |
| `build_subject_task_pairs(subjects, tasks)`                                               | Produces the Cartesian list of subject-task jobs.                                                                               |
| `evaluate_subject_task_no_save(subject_id, task, config_id, synth_out_dir=None)`          | Runs one subject-task evaluation without executing the standard full-output pipeline; optionally saves selected synthetic data. |
| `aggregate_trial_results(results)`                                                        | Aggregates subject-task records into mean, standard deviation, count, and failure summaries.                                    |
| `choose_best_trial(trials, min_detection)`                                                | Selects the best valid candidate trial while enforcing a minimum detection level.                                               |
| `load_checkpoint(path)`                                                                   | Loads an existing JSON checkpoint or returns an empty state.                                                                    |
| `save_checkpoint(ckpt, path)`                                                             | Persists experiment progress.                                                                                                   |
| `run_candidate_trial(...)`                                                                | Executes one hyperparameter candidate over the calibration/held-out jobs and returns aggregated performance.                    |
| `append_phase_summary(...)`                                                               | Appends one parameter-search phase to the cumulative summary.                                                                   |
| `save_candidate_results_csv(...)`                                                         | Saves detailed candidate-level results to CSV.                                                                                  |
| `save_phase_comparison_csv(...)`                                                          | Saves a phase-level comparison table.                                                                                           |
| `save_top_synthetic_for_phase(...)`                                                       | Regenerates/saves synthetic outputs for the best candidate of an experiment phase.                                              |
| `run_parameter_test_only(args)`                                                           | Runs the repository's multi-phase parameter-test mode without the standard all-subject generation path.                         |
| `main()`                                                                                  | Parses command-line options and dispatches either parameter testing or the standard engine pipeline.                            |

---

# 7. `utils.py` function reference

`utils.py` centralizes operations shared by the SGG/DCM repeated-generation and TSTR workflows.

## 7.1 Paths, values, and identifiers

| Function                                                              | Functionality                                                                     |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `ensure_dir(path)`                                                    | Creates the requested output directory recursively.                               |
| `safe_float(value, default=NaN)`                                      | Converts a value to `float`; returns a safe default for missing or invalid input. |
| `stable_seed_from_string(seed_str)`                                   | Creates an MD5-based deterministic integer seed.                                  |
| `normalize_task_name(task)`                                           | Normalizes task aliases such as `TASK4` to `T4`.                                  |
| `infer_subject_task_from_path(path)`                                  | Extracts the subject identifier and task name from a fixation filename.           |
| `list_fixation_files(data_dir, fixation_pattern, exclude_tasks=None)` | Recursively discovers fixation files and excludes configured tasks.               |
| `find_raw_file(fixation_path)`                                        | Locates the raw gaze file corresponding to a fixation file.                       |

## 7.2 Data normalization and raw-gaze reading

| Function                                        | Functionality                                                                                                                                                                    |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `normalize_fixation_dataframe(df, config=None)` | Detects alternative column names, converts fixation centroids/times/durations to a canonical schema, removes invalid rows, and optionally drops short fixations.                 |
| `_find_raw_columns(raw_df)`                     | Maps supported timestamp and monocular/binocular gaze-column aliases to actual columns.                                                                                          |
| `_normalize_time_to_ms(t_raw)`                  | Uses the median timestamp interval to infer whether timestamps are in microseconds or milliseconds and normalizes them to milliseconds.                                          |
| `read_raw_gaze(raw_path, config=None)`          | Reads raw gaze, selects direct x/y or combines available eyes, normalizes time, removes invalid samples, and returns time, x, y, detected Hz, median interval, and a source tag. |

## 7.3 Sigma and sampling-rate estimation

| Function                                                     | Functionality                                                                                                                                                                                                     |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `compute_sigma_from_raw(fix_df, raw_path, config=None)`      | Collects raw residuals relative to the fixation centroid inside every fixation window and calculates `sigma_base = (std_x + std_y)/2`. It also returns sampling-rate metadata and the number of raw samples used. |
| `compute_sigma_from_existing_column(fix_df)`                 | Uses the median positive sigma value already stored in the fixation file when raw estimation is unavailable.                                                                                                      |
| `clip_sigma(sigma, config)`                                  | Restricts sigma to configured empirical bounds when those bounds are valid.                                                                                                                                       |
| `compute_global_sigma_stats(fixation_files, config)`         | Estimates sigma across the full dataset and returns the median, lower/upper percentiles, descriptive statistics, and per-file details.                                                                            |
| `select_sampling_rate(fixation_path, config, raw_info=None)` | Chooses a sampling rate using this priority: configured fixed rate, cached raw metadata, direct raw-file estimation, then an explicitly enabled fallback.                                                         |

## 7.4 ETDD70-compatible metrics

| Function                                                                                   | Functionality                                                                                                                                                                                                   |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `compute_metrics(fix_df, sacc_df, roi_df, sid, task, trial_id=12)`                         | Calculates trial-level and AOI-level ETDD70-style fixation/saccade metrics, including counts, durations, amplitude, progress/regression measures, dwell time, skipped AOIs, first-visit measures, and revisits. |
| `compute_tstr_metrics_from_gaze(syn_df, orig_fix_df, sid, task, trial_id=12, roi_df=None)` | Converts sample-level synthetic gaze into fixation and inferred saccade representations, then calls `compute_metrics()` to produce files compatible with `evaluate_tstr.py`.                                    |

---

# 8. Command-line modules

The package entry points declared in `pyproject.toml` expose the following commands:

```text
peyemmv-find-theta
peyemmv-ga2-generate
peyemmv-ga2x20
peyemmv-sgg
peyemmv-sggx20
peyemmv-dcm
peyemmv-dcmx20
peyemmv-compare
peyemmv-evaluate-tstr
```

## 8.1 `find_theta_ga2.py`

### Purpose

Optimizes one parameter vector

```text
θ* = (H, sigma, phi, seed)
```

for every available subject-task pair. This script does **not** generate the final full synthetic datasets.

### Main functions

| Function                                 | Description                                                                                                                                                                                                                                                                |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `worker_theta_ga2(args_tuple)`           | Worker for one subject-task. It patches GA settings in the subprocess, loads raw/fixation data, filters invalid fixations, detects Hz, estimates the baseline, selects the fitness subset, runs `run_ga()`, and returns the optimized parameters and objective components. |
| `load_checkpoint(cp_file)`               | Loads completed subject-task results from the JSON checkpoint so interrupted jobs can resume.                                                                                                                                                                              |
| `save_checkpoint(cp_file, records_dict)` | Safely stores the current optimized-parameter records.                                                                                                                                                                                                                     |
| `main()`                                 | Parses GA hyperparameters and paths, reads subject labels, creates all subject-task jobs, runs workers sequentially or in parallel, updates the checkpoint, and writes `ga2_theta_star.csv` and `ga2_theta_summary.json`.                                                  |

### Primary outputs

```text
<output_root>/ga2_theta_star.csv
<output_root>/ga2_theta_summary.json
<output_root>/ga2_theta_checkpoint.json
```

---

## 8.2 `ga2_generate.py`

### Purpose

Generates one complete GA2 sample-level synthetic dataset per subject-task. Parameters can come from `ga2_theta_star.csv`; when the table is unavailable and online optimization is allowed, the script runs GA directly.

### Main functions

| Function                                             | Description                                                                                                                                                                |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `load_ga2_theta_csv(theta_csv_path)`                 | Loads the optimized parameter table and builds a lookup indexed by `(subject_id, task)`.                                                                                   |
| `resolve_theta_csv_path(user_theta_csv, output_dir)` | Resolves an explicitly provided parameter file or searches the standard candidate locations.                                                                               |
| `worker_ga2(args_tuple)`                             | Executes generation for one subject-task: patch config, load data, estimate baseline, obtain theta from CSV or GA, generate synthetic gaze, save CSV, and return metadata. |
| `main()`                                             | Parses paths and GA options, reads the subject list, resolves theta, builds jobs, runs workers, and writes the generation summary.                                         |

### Output schema

The generated CSV includes identifiers, timestamps, synthetic x/y coordinates, fixation centroids, PeyeMMV status, optimized parameters, baseline H, sampling rate, interval, and fixation duration.

Primary files:

```text
ga2_output/GA2_subject_{sid}_task_{task}_Subject_{sid}_{task}_fixations.csv
ga2_output/ga2_generate_summary.json
```

---

## 8.3 `ga2x20.py`

### Purpose

Generates `N` independent GA2 datasets, normally 20, per subject-task for repeated TSTR experiments.

### Main functions

| Function                                      | Description                                                                                                                                                                                               |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `resolve_sampling_rate(sid, task, data_root)` | Reads the corresponding raw file and estimates subject-task Hz; returns the configured fallback when estimation is impossible.                                                                            |
| `generate_syn_set(...)`                       | Generates one independent synthetic set using optimized H, sigma, phi, seed, baseline H, sampling rate, ROI data, and class metadata. It constructs gaze-, fixation-, saccade-, and metric-level outputs. |
| `main()`                                      | Loads `ga2_theta_star.csv`, reads labels and ROIs, skips already completed checkpoints, generates all requested sets, and saves TSTR-compatible files.                                                    |

### Outputs per subject-task-set

```text
Subject_{sid}_{task}_metrics_syn_{i}.csv
Subject_{sid}_{task}_fixations_syn_{i}.csv
Subject_{sid}_{task}_saccades_syn_{i}.csv
raw/Subject_{sid}_{task}_raw_syn_{i}.csv
```

---

## 8.4 `sgg.py`

### Purpose

Creates the **extreme noisy baseline**. For each fixation:

```text
x_i ~ Normal(x_fix, sigma_base²)
y_i ~ Normal(y_fix, sigma_base²)
```

Samples are independent, so the method produces white noise without fBm, temporal autocorrelation, or explicit drift.

### Main functions

| Function                                                                                               | Description                                                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `generate_sgg_fixation(...)`                                                                           | Generates one fixation as IID Gaussian samples around the supplied centroid at the specified sampling rate and duration.                                                                                 |
| `generate_sgg_for_file(fixation_path, output_dir, config, global_sigma=None, global_sigma_stats=None)` | Handles one fixation file. It resolves sigma according to subject-task/global/fixed mode, detects sampling rate, generates all fixation clusters, saves the synthetic CSV, and returns summary metadata. |
| `run_sgg_pipeline(config)`                                                                             | Discovers all fixation files, optionally computes global empirical sigma statistics, executes per-file generation, aggregates success/error and sigma/Hz statistics, and saves `sgg_summary.json`.       |
| `parse_args()`                                                                                         | Defines CLI options for data paths, sigma mode, sampling-rate mode, duration handling, and file saving.                                                                                                  |
| `main()`                                                                                               | Maps parsed options into the SGG configuration and starts the pipeline.                                                                                                                                  |

The file contains local copies of several path, normalization, raw-reading, sigma, and sampling helpers. Their behavior corresponds to the shared functions documented under `utils.py`.

---

## 8.5 `sggx20.py`

### Purpose

Generates multiple independent SGG datasets per subject-task and optionally produces TSTR-compatible metrics.

### Main functions

| Function                                                                   | Description                                                                                                                                                       |
| -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `generate_sgg_fixation(...)`                                               | Same IID Gaussian core as `sgg.py`.                                                                                                                               |
| `generate_sgg_for_file(..., set_idx=0, tstr_output_dir=None, roi_df=None)` | Generates one specified synthetic set, saves gaze data, computes ETDD70-compatible TSTR metrics when requested, and records set-specific seed and metadata.       |
| `run_sgg_pipeline(config)`                                                 | Iterates through every fixation file and set index, supports output-based checkpoint skipping, loads ROIs, aggregates all runs, and writes `sggx20_summary.json`. |
| `parse_args()`                                                             | Adds repeated-set, TSTR-output, ROI, and metrics-control options to the SGG arguments.                                                                            |
| `main()`                                                                   | Builds configuration and runs the repeated SGG pipeline.                                                                                                          |

---

## 8.6 `dcm.py`

### Purpose

Creates the **extreme noiseless baseline**:

```text
x(t) = x_fix + drift_x(t) + noise_x(t)
y(t) = y_fix + drift_y(t) + noise_y(t)
```

Noise and random-walk drift are fixed constants rather than parameters estimated from dyslexia labels. The method intentionally keeps gaze extremely close to the fixation centroid.

### Main functions

| Function                                                   | Description                                                                                                       |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `estimate_sampling_rate_from_raw(raw_path, config=None)`   | Reads timestamps and estimates Hz from their median positive interval.                                            |
| `select_sampling_rate(fixation_path, config)`              | Selects raw-derived Hz or an explicitly enabled fallback.                                                         |
| `generate_dcm_fixation(...)`                               | Generates one near-static fixation using tiny IID noise and cumulative small random-walk drift.                   |
| `generate_dcm_for_file(fixation_path, output_dir, config)` | Normalizes one fixation file, selects Hz, generates all clusters, saves the DCM gaze CSV, and returns metadata.   |
| `run_dcm_pipeline(config)`                                 | Discovers files, generates DCM data, aggregates status and sampling-rate statistics, and writes the summary JSON. |
| `parse_args()`                                             | Defines DCM CLI arguments.                                                                                        |
| `main()`                                                   | Builds the runtime configuration and runs DCM generation.                                                         |

As in `sgg.py`, common file and normalization helpers are local duplicates of the shared utilities.

---

## 8.7 `dcmx20.py`

### Purpose

Generates multiple independent DCM datasets per subject-task and optionally produces TSTR-compatible metric files.

### Main functions

| Function                                                                   | Description                                                                                                                                                     |
| -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `generate_dcm_fixation(...)`                                               | Near-static DCM generation for one fixation.                                                                                                                    |
| `generate_dcm_for_file(..., set_idx=0, tstr_output_dir=None, roi_df=None)` | Generates one DCM set for a file, saves gaze output, optionally computes TSTR metrics, and records the set index.                                               |
| `run_dcm_pipeline(config)`                                                 | Executes all fixation-file × set-index jobs, skips completed metric files, loads ROI data when necessary, aggregates results, and writes `dcmx20_summary.json`. |
| `parse_args()`                                                             | Defines repeated-set, TSTR, ROI, and DCM generation options.                                                                                                    |
| `main()`                                                                   | Creates the final configuration and starts the repeated DCM pipeline.                                                                                           |

---

## 8.8 `compare_generators.py`

### Purpose

Compares generated GA2, SGG, and DCM gaze with real ETDD70 fixation behavior.

### Metrics

| Metric  | Meaning                                                                                             |
| ------- | --------------------------------------------------------------------------------------------------- |
| `ACR`   | Acceptance/detection rate: percentage of synthetic fixations accepted by `peyemmv_check()`.         |
| `CLE`   | Centroid-location error between the synthetic cluster centroid and the reference fixation centroid. |
| `VPC`   | Mean sample-to-sample step length, used as a velocity proxy.                                        |
| `JSD`   | Jensen-Shannon divergence between real and synthetic angular-velocity distributions.                |
| `alpha` | Negative PSD slope used to characterize power-law temporal structure.                               |

### Main functions

| Function                                                                              | Description                                                                                                                     |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `build_omega_real(raw_df, fix_df)`                                                    | Extracts angular velocities inside real fixation windows and builds the reference probability distribution and bin edges.       |
| `_eval_synthetic_file(args_tuple)`                                                    | Worker that reads one synthetic file, locates its real raw/fixation counterparts, and calculates ACR, CLE, VPC, JSD, and alpha. |
| `run_statistics(df, output_root)`                                                     | Performs one-way ANOVA for each task/metric, calculates eta-squared, and optionally runs Tukey HSD post-hoc comparisons.        |
| `_color(method)`                                                                      | Returns the fixed plotting color assigned to a generation method.                                                               |
| `make_spatial_scatter(syn_files, output_root, n_sample=3000, seed=42)`                | Produces the three-method two-dimensional spatial scatter figure.                                                               |
| `make_psd_and_velocity(df, syn_files, data_dir, output_root, sr=1000, n_subjects=10)` | Produces a log-log PSD comparison and a real-versus-GA2 angular-velocity KDE figure.                                            |
| `make_boxplots(df, output_root)`                                                      | Creates task-separated boxplots for all five generator-level metrics.                                                           |
| `print_and_save_summary(df, output_root)`                                             | Computes and saves mean and standard deviation by method and task.                                                              |
| `collect_synthetic_files(dcm_dir, sgg_dir, ga2_dir=None)`                             | Discovers generated CSV files in the method-specific output directories.                                                        |
| `main()`                                                                              | Loads labels, collects files, evaluates them in parallel, writes result tables, performs statistics, and generates figures.     |

### Main outputs

```text
comparison_results.csv
summary_by_method.csv
anova_summary.csv
tukey_hsd.csv
boxplot_ACR.png
boxplot_CLE.png
boxplot_VPC.png
boxplot_JSD.png
boxplot_alpha.png
fig1_spatial_scatter.png
fig2a_loglog_psd.png
fig2b_velocity_kde.png
```

---

## 8.9 `evaluate_tstr.py`

### Purpose

Evaluates whether synthetic features retain information that generalizes to real dyslexia classification.

### Evaluation protocols

| Protocol | Training data               | Test data               | Interpretation                                                                          |
| -------- | --------------------------- | ----------------------- | --------------------------------------------------------------------------------------- |
| `Base%`  | Real folds                  | Held-out real fold      | Reference performance obtainable from real data.                                        |
| `CVSyn%` | Synthetic folds             | Held-out synthetic fold | Measures whether the synthetic data preserve internally discriminative class structure. |
| `TSTR%`  | All selected synthetic sets | Real observations       | Main test of whether synthetic data generalize to unseen real data.                     |

### Main functions

| Function                                                                                  | Description                                                                                                                                                                          |
| ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `compute_mmd(X_syn, X_real, gamma=1.0)`                                                   | Computes RBF-kernel Maximum Mean Discrepancy between standardized synthetic and real feature matrices, using bounded subsamples for memory efficiency.                               |
| `distribution_shift_penalty(mmd, alpha=0.15, beta=10.0)`                                  | Converts MMD into a bounded multiplicative score penalty used by the standard TSTR routine.                                                                                          |
| `extract_trial_features(df)`                                                              | Reduces repeated AOI rows to one trial-level record per subject-task.                                                                                                                |
| `extract_aoi_features(df)`                                                                | Aggregates AOI features by mean and standard deviation and merges them with trial-level features.                                                                                    |
| `load_real_metrics(data_dir, subject_ids, tasks, use_aoi=False)`                          | Loads and concatenates real ETDD70 metric files.                                                                                                                                     |
| `load_syn_set(syn_root, subject_ids, tasks, i_set, use_aoi=False)`                        | Loads all available synthetic metric files for one set index.                                                                                                                        |
| `build_Xy(df, label_map, feat_cols)`                                                      | Maps subject labels, inserts absent features as zeros, converts feature columns to numeric arrays, and returns `X, y`.                                                               |
| `normalize_binary_label(value)`                                                           | Converts common textual and numeric binary-label formats to 0 or 1.                                                                                                                  |
| `make_clf(name, n_estimators=100, seed=42)`                                               | Instantiates Random Forest, RBF-SVM, MLP, XGBoost, or CatBoost with reproducible settings.                                                                                           |
| `run_tstr(...)`                                                                           | Legacy/per-set TSTR routine: trains on each synthetic set and evaluates stratified folds of the real data, returning per-set scores and the best model.                              |
| `run_base(X_real, y_real, clf_name, n_estimators, n_folds=5, seed=42)`                    | Runs stratified cross-validation exclusively on real data.                                                                                                                           |
| `run_cv_syn(...)`                                                                         | Runs stratified cross-validation separately on each synthetic set and aggregates the scores across successful sets.                                                                  |
| `run_tstr_standard(...)`                                                                  | Concatenates all synthetic sets, standardizes train/test features, computes MMD and its penalty, performs repeated bootstrap model training, and evaluates on the complete real set. |
| `plot_boxplot(scores_df, output_root)`                                                    | Plots score distributions across synthetic sets.                                                                                                                                     |
| `plot_confusion(y_real, y_pred, output_root)`                                             | Saves the confusion matrix for the selected TSTR prediction.                                                                                                                         |
| `plot_feature_importance(clf_scaler, feat_cols, output_root)`                             | Saves the top feature importances for classifiers exposing `feature_importances_`.                                                                                                   |
| `plot_tstr_grouped_bar(all_summaries, output_root, metric="accuracy", task_label="both")` | Compares Base% and TSTR% by classifier for one metric.                                                                                                                               |
| `plot_tstr_grouped_bar_all_metrics(all_summaries, output_root, task_label="both")`        | Produces a combined 2×2 Accuracy/F1/AUC/MCC figure and individual metric figures.                                                                                                    |
| `main()`                                                                                  | Parses evaluation settings, loads labels and metrics, selects features/tasks/classifiers, executes Base%, CVSyn%, and TSTR%, writes result tables, and creates plots.                |

### Primary evaluation measures

- Accuracy;
- binary F1 score;
- ROC-AUC;
- Matthews correlation coefficient (MCC);
- mean and standard deviation across folds/sets/bootstrap repetitions;
- retention percentage and percentage-point difference relative to Base%;
- MMD and the applied distribution penalty.

---

# 9. Reproducibility mechanisms

The code includes the following reproducibility controls:

1. **Stable MD5-derived seeds** rather than process-dependent hashes;
2. **Subject-task-specific seeds**;
3. **Fixation-specific combined seeds** in the GA generator;
4. **Set-index-specific seeds** in the repeated generators;
5. **Explicit global seed values** in configuration;
6. **Deterministic cross-validation seeds**;
7. **JSON checkpoints** for expensive optimization/generation tasks;
8. **Pinned execution parameters** recorded in output summaries;
9. **Detected sampling-rate source tags**;
10. **Parameter-source tags** for H, sigma, and theta.

To reproduce paper results exactly, the publication should additionally record:

- the Git release/tag and Zenodo software DOI;
- Python version;
- operating system;
- dependency versions;
- the exact ETDD70 dataset DOI/version;
- all GA parameters and fitness weights;
- all random seeds;
- the selected subject/task list;
- the number of synthetic sets;
- the classifier and evaluation options.

---

# 10. Recommended documentation conventions for source functions

The reviewer requests complete docstrings for each function and class. Public functions should preferably follow the NumPy format already used in parts of `engine.py`.

Example:

```python
def generate_displacement(
    n_points: int,
    H: float,
    sigma: float,
    phi: float,
    sampling_rate: float,
    seed: int,
    H_drift: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a two-dimensional synthetic within-fixation displacement.

    Parameters
    ----------
    n_points
        Number of gaze samples to generate.
    H
        Hurst exponent of the fast fractional Brownian component.
    sigma
        Spatial scale in pixels.
    phi
        Rotation angle in radians.
    sampling_rate
        Sampling frequency in Hz.
    seed
        Deterministic random seed.
    H_drift
        Optional Hurst exponent for the slow drift component.

    Returns
    -------
    dx, dy
        Horizontal and vertical displacement arrays.

    Raises
    ------
    ValueError
        If the number of points, sampling rate, or model parameters are invalid.
    """
```

Each docstring should state:

- purpose;
- parameter meaning, type, and unit;
- return type and field meaning;
- raised exceptions;
- file-writing side effects;
- deterministic/random behavior;
- important formulas or methodological assumptions.

---

# 11. Recommended separation between public and internal API

Functions beginning with `_`, such as `_standardize()` and `_spectral_error()`, are internal implementation helpers. They may still be documented for maintainers, but the public API should emphasize:

```text
load_raw_gaze
load_fixations
detect_sampling_rate
estimate_baseline
generate_displacement
peyemmv_check
evaluate_individual
run_ga
generate_synthetic_dataset
compute_metrics
compute_tstr_metrics_from_gaze
```

The command-line programs are the preferred user-facing interface.

---

# 12. Known documentation and maintainability considerations

1. `sgg.py` and `dcm.py` contain utility logic that overlaps with `utils.py`. Future releases should import the shared functions consistently to avoid behavior divergence.
2. File naming distinguishes `GA`, `GA2`, and the package name `PeyeMMV-GA`. The documentation should define these names once and use them consistently.
3. The executable commands defined in `pyproject.toml` should be used in the README instead of only showing `python script.py`.
4. Units should be stated explicitly: timestamps in milliseconds, coordinates and sigma in pixels, sampling rate in Hz, angles in radians or degrees as applicable.
5. The software DOI must identify the exact released source-code version, separately from the ETDD70 dataset DOI.
6. The final paper-reproduction workflow should be provided as a notebook or an executable script that calls the CLI stages in the required order.

---

# 13. Suggested reproducible command sequence

```bash
# 1. Install the package
pip install peyemmv-ga

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
```

Before publication, verify all options against the tagged release and store the exact commands in `docs/REPRODUCIBILITY.md` or `examples/reproduce_paper_results.sh`.

---

## 14. Citation and availability placeholders

Replace the placeholders below after publishing the release:

```text
Source code: https://github.com/MonTeamm/PeyeMMV-GA
PyPI: [PYPI_URL]
Software archive: [ZENODO_SOFTWARE_DOI]
Dataset archive: [ETDD70_DATASET_DOI]
License: MIT
Version used in the paper: [RELEASE_TAG]
```

---

## 15. Summary

The codebase separates the proposed fBm-GA generator from two controlled baselines and provides both low-level physiological/statistical comparison and downstream TSTR evaluation. The central scientific implementation is located in `engine.py`; reusable data and metric operations are in `utils.py`; and each experimental stage is exposed through a dedicated command-line program.

This file is intended to be stored as:

```text
docs/CODE_DOCUMENTATION.md
```

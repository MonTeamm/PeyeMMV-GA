# ============================================================
# DCM - Deterministic Centroid Minimization
# Baseline 2 in the paper — Extreme Noiseless Baseline
# ============================================================
#
# Method description (Section 4.7 of the paper):
#   DCM generates "near-static" gaze points with variance ≈ 0.
#   Coordinates are generated from a tightly constrained random walk to
#   maintained within an extremely small radius (ε ≈ 0) around the centroid.
#
#       x(t) = x_fix + drift_x(t) + noise_x(t)
#       y(t) = y_fix + drift_y(t) + noise_y(t)
#
#   With near-zero sigma (≈ 0):
#       noise ~ N(0, NOISE_SIGMA^2),   NOISE_SIGMA = 0.1 px  (IID)
#       drift = cumsum(N(0, DRIFT_STEP^2)),  DRIFT_STEP ≈ 0 px/step
#
#   Does not use empirical sigma from dataset (unlike SGG): DCM always uses
#   fixed near-zero sigma to maximise detection rate while eliminating
#   physiological micro-movements.
#
# Three-way comparison from the paper:
#   DCM  : near-static (variance ≈ 0) -> very high detection rate, non-physiological
#   SGG  : IID Gaussian with empirical sigma -> white noise, no autocorrelation
#   GA2  : fBm with optimised H -> pink 1/f noise, physiological
#
# NO data leakage in DCM:
#   - x_fix, y_fix taken from subject fixation file (not from labels)
#   - sampling_rate estimated from raw timestamps (not from labels)
#   - sigma_noise and sigma_drift are FIXED CONSTANTS, not data-derived
#
# ============================================================

import json
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# 1. CONFIG
# ============================================================

CONFIG = {
    # Directory containing fixation and raw files
    "DATA_DIR": "./data",

    # Output directory
    "OUTPUT_DIR": "./dcm_output",

    # Pattern file fixation
    "FIXATION_PATTERN": "*_fixations.csv",

    # Excluded tasks (T1 not used in this study)
    "EXCLUDE_TASKS": ["T1"],

    # --------------------------------------------------------
    # Sampling rate: always estimated from raw per subject-task
    # --------------------------------------------------------
    "SAMPLING_RATE_MODE": "subject_task",
    "SAMPLING_RATE_HZ": None,           # Used only when mode="fixed" or as fallback
    "ALLOW_SAMPLING_RATE_FALLBACK": False,
    "SAMPLING_RATE_MIN_HZ": 10.0,
    "SAMPLING_RATE_MAX_HZ": 2000.0,

    # --------------------------------------------------------
    # DCM Core Parameters — FIXED CONSTANTS, not data-derived
    # --------------------------------------------------------
    # Per Section 4.7: "artificially constrained to remain within
    # within an extremely small spatial radius (ε ≈ 0)"
    #
    # NOISE_SIGMA_PX: IID Gaussian noise, < 1 pixel
    "NOISE_SIGMA_PX": 0,

    # DRIFT_SIGMA_PX_PER_SEC: random walk drift, extremely slow
    "DRIFT_SIGMA_PX_PER_SEC": 0.01,

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------
    "ENFORCE_MIN_DURATION": False,
    "MIN_DURATION_MS": 80.0,
    "DROP_SHORT_FIXATIONS": False,

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------
    "RANDOM_SEED": 42,
    "USE_DETERMINISTIC_SEED": True,

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------
    "SAVE_SYNTHETIC_FILES": True,
    "SUMMARY_FILENAME": "dcm_summary.json",
    "CSV_ENCODING": "utf-8-sig",
}


# ============================================================
# 2. UTILS
# ============================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def stable_seed_from_string(seed_str):
    """MD5-based seed, stable across runs."""
    h = hashlib.md5(seed_str.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def normalize_task_name(task):
    if task is None:
        return "unknown"
    task = str(task).upper().strip()
    mapping = {"TASK1": "T1", "TASK2": "T2", "TASK3": "T3",
               "TASK4": "T4", "TASK5": "T5"}
    return mapping.get(task, task)


def infer_subject_task_from_path(path):
    stem = Path(path).stem.replace("_fixations", "")
    parts = stem.replace("-", "_").split("_")
    subject_id = None
    task = None

    for p in parts:
        if p.isdigit():
            subject_id = p
            break

    for i, p in enumerate(parts):
        if p.lower() == "subject" and i + 1 < len(parts):
            if parts[i + 1].isdigit():
                subject_id = parts[i + 1]
                break

    for i, p in enumerate(parts):
        p_upper = p.upper()
        if p_upper in ["T1", "T2", "T3", "T4", "T5"]:
            task = "_".join([p_upper] + parts[i + 1:])
            break
        if p_upper in ["TASK1", "TASK2", "TASK3", "TASK4", "TASK5"]:
            task = "_".join([normalize_task_name(p_upper)] + parts[i + 1:])
            break

    if subject_id is None:
        subject_id = stem
    if task is None:
        task = "unknown"

    return str(subject_id), task


def list_fixation_files(data_dir, fixation_pattern, exclude_tasks=None):
    data_dir = Path(data_dir)
    exclude_set = set()
    if exclude_tasks:
        exclude_set = {normalize_task_name(t) for t in exclude_tasks}

    files = []
    for f in data_dir.rglob(fixation_pattern):
        _, task = infer_subject_task_from_path(f)
        if task.split("_")[0] in exclude_set:
            continue
        files.append(f)

    return sorted(files)


def find_raw_file(fixation_path):
    p = Path(fixation_path)
    candidates = []
    if p.name.endswith("_fixations.csv"):
        candidates.append(p.parent / p.name.replace("_fixations.csv", "_raw.csv"))
    stem = p.stem.replace("_fixations", "")
    candidates.append(p.parent / f"{stem}_raw.csv")
    for c in candidates:
        if c.exists() and c != p:
            return c
    return None


# ============================================================
# 3. NORMALIZE FIXATION FILE
# ============================================================

def normalize_fixation_dataframe(df, config=None):
    lower_map = {c.lower().strip(): c for c in df.columns}

    def find_col(candidates):
        for cand in candidates:
            if cand.lower().strip() in lower_map:
                return lower_map[cand.lower().strip()]
        return None

    col_fix_id = find_col(["fixation_id", "fix_id", "id", "fixid"])
    col_x = find_col(["x", "center_x", "centroid_x", "gaze_x",
                       "x_fix", "fix_x", "mean_x", "avg_x"])
    col_y = find_col(["y", "center_y", "centroid_y", "gaze_y",
                       "y_fix", "fix_y", "mean_y", "avg_y"])
    col_duration = find_col(["duration_ms", "duration", "dur_ms",
                              "fixation_duration", "duration_msec"])
    col_start = find_col(["start_time", "start_ms", "onset", "start",
                           "t_start", "begin_time"])
    col_end = find_col(["end_time", "end_ms", "offset", "end",
                         "t_end", "finish_time"])

    if col_x is None or col_y is None:
        raise ValueError(f"Cannot find x/y columns. Available: {list(df.columns)}")

    out = pd.DataFrame()
    out["fixation_id"] = df[col_fix_id] if col_fix_id else np.arange(len(df))
    out["x_fix"] = df[col_x].astype(float)
    out["y_fix"] = df[col_y].astype(float)

    if col_duration is not None:
        out["duration_ms"] = df[col_duration].astype(float)
    elif col_start is not None and col_end is not None:
        out["duration_ms"] = df[col_end].astype(float) - df[col_start].astype(float)
    else:
        raise ValueError("Cannot find duration_ms or start_time/end_time columns.")

    out["start_time"] = df[col_start].astype(float) if col_start else np.nan
    out["end_time"] = df[col_end].astype(float) if col_end else np.nan

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["x_fix", "y_fix", "duration_ms"])
    out = out[out["duration_ms"] > 0].copy()

    if config is not None and config.get("DROP_SHORT_FIXATIONS", False):
        min_dur = float(config.get("MIN_DURATION_MS", 80.0))
        out = out[out["duration_ms"] >= min_dur].copy()

    return out.reset_index(drop=True)


# ============================================================
# 4. RAW FILE: READ SAMPLING RATE
# ============================================================

def _find_raw_columns(raw_df):
    lower_map = {c.lower().strip(): c for c in raw_df.columns}

    def find_col(candidates):
        for c in candidates:
            if c.lower().strip() in lower_map:
                return lower_map[c.lower().strip()]
        return None

    return {
        "time": find_col(["time", "t", "timestamp", "time_us",
                           "time_ms", "timestamp_ms", "timestamp_us"]),
        "x": find_col(["x", "gaze_x", "gaze_x_mean", "mean_x"]),
        "y": find_col(["y", "gaze_y", "gaze_y_mean", "mean_y"]),
        "xl": find_col(["gaze_x_left", "x_left", "left_x", "gaze_left_x"]),
        "yl": find_col(["gaze_y_left", "y_left", "left_y", "gaze_left_y"]),
        "xr": find_col(["gaze_x_right", "x_right", "right_x", "gaze_right_x"]),
        "yr": find_col(["gaze_y_right", "y_right", "right_y", "gaze_right_y"]),
    }


def _normalize_time_to_ms(t_raw):
    t_raw = np.asarray(t_raw, dtype=float)
    t_valid = t_raw[np.isfinite(t_raw)]
    if len(t_valid) < 3:
        return t_raw, "unknown"
    dt = np.diff(t_valid)
    dt = dt[(np.isfinite(dt)) & (dt > 0)]
    if len(dt) == 0:
        return t_raw, "unknown"
    median_dt = float(np.median(dt))
    # median_dt > 100 → likely microseconds (e.g. 1000 Hz → 1000 us)
    if median_dt > 100:
        return t_raw / 1000.0, "us_to_ms"
    return t_raw, "ms"


def estimate_sampling_rate_from_raw(raw_path, config=None):
    """
    Read raw file and estimate sampling rate from timestamps.
    Returns: (sampling_rate_hz, median_dt_ms, source_str)
    """
    try:
        raw_df = pd.read_csv(raw_path)
    except Exception as e:
        return np.nan, np.nan, f"raw_read_error: {e}"

    cols = _find_raw_columns(raw_df)
    if cols["time"] is None:
        return np.nan, np.nan, "no_time_column"

    t_raw = raw_df[cols["time"]].astype(float).values
    t_ms, _ = _normalize_time_to_ms(t_raw)
    t_ms = t_ms[np.isfinite(t_ms)]

    if len(t_ms) < 3:
        return np.nan, np.nan, "not_enough_timestamps"

    dt = np.diff(t_ms)
    dt = dt[(np.isfinite(dt)) & (dt > 0)]
    if len(dt) == 0:
        return np.nan, np.nan, "no_positive_dt"

    median_dt = float(np.median(dt))
    if not np.isfinite(median_dt) or median_dt <= 0:
        return np.nan, np.nan, "invalid_median_dt"

    sr_hz = 1000.0 / median_dt

    if config is not None:
        min_hz = float(config.get("SAMPLING_RATE_MIN_HZ", 10.0))
        max_hz = float(config.get("SAMPLING_RATE_MAX_HZ", 2000.0))
        if not (min_hz <= sr_hz <= max_hz):
            return sr_hz, median_dt, f"out_of_range_{sr_hz:.1f}Hz"

    return float(sr_hz), float(median_dt), "raw_timestamp"


def select_sampling_rate(fixation_path, config):
    """Select sampling rate: prefer raw timestamp, fallback if permitted."""
    mode = config.get("SAMPLING_RATE_MODE", "subject_task")

    if mode == "fixed":
        hz = safe_float(config.get("SAMPLING_RATE_HZ"), np.nan)
        if not np.isfinite(hz) or hz <= 0:
            raise ValueError("SAMPLING_RATE_MODE=fixed but SAMPLING_RATE_HZ is invalid.")
        return hz, 1000.0 / hz, "fixed"

    raw_path = find_raw_file(fixation_path)
    if raw_path is None:
        if config.get("ALLOW_SAMPLING_RATE_FALLBACK", False):
            hz = safe_float(config.get("SAMPLING_RATE_HZ"), np.nan)
            if np.isfinite(hz) and hz > 0:
                return hz, 1000.0 / hz, "fallback_no_raw_file"
        raise ValueError(f"Raw file not found for: {fixation_path}")

    sr_hz, median_dt_ms, source = estimate_sampling_rate_from_raw(raw_path, config)

    if np.isfinite(sr_hz) and sr_hz > 0 and source == "raw_timestamp":
        return sr_hz, median_dt_ms, source

    if config.get("ALLOW_SAMPLING_RATE_FALLBACK", False):
        hz = safe_float(config.get("SAMPLING_RATE_HZ"), np.nan)
        if np.isfinite(hz) and hz > 0:
            return hz, 1000.0 / hz, f"fallback_after_{source}"

    raise ValueError(
        f"Cannot estimate sampling rate from {raw_path}: {source}. "
        "ALLOW_SAMPLING_RATE_FALLBACK=False."
    )


# ============================================================
# 5. DCM GENERATION — CORE
# ============================================================

def generate_dcm_fixation(
    fixation_id,
    x_fix,
    y_fix,
    duration_ms,
    noise_sigma_px,
    drift_sigma_px_per_sec,
    sampling_rate_hz,
    rng,
    start_time=np.nan,
    config=None,
):
    """
    Generate one fixation using DCM — Extreme Noiseless Baseline.

    Theo docx Section 4.7:
        "Coordinates are generated via a constrained random walk model,
         artificially constrained to remain within an extremely small
         spatial radius (ε ≈ 0). Variance is approximately zero."

    Formula:
        x(t) = x_fix + drift_x(t) + noise_x(t)
        y(t) = y_fix + drift_y(t) + noise_y(t)

    With:
        noise_x(t) ~ IID N(0, noise_sigma_px^2),  noise_sigma_px = 0.1 px
        drift_x(t) = cumsum(step_x(t)),  step_x ~ N(0, drift_step_sigma^2)
        drift_step_sigma = drift_sigma_px_per_sec * sqrt(dt_ms/1000)
                        ≈ 0.05 * sqrt(0.004) ≈ 0.003 px/step @ 250 Hz

    Gaze deviates < 1 pixel from centroid -> variance ≈ 0.
    """
    duration_ms = float(duration_ms)
    if config is not None and config.get("ENFORCE_MIN_DURATION", False):
        duration_ms = max(duration_ms, float(config.get("MIN_DURATION_MS", 80.0)))

    dt_ms = 1000.0 / float(sampling_rate_hz)
    n_points = max(int(round(duration_ms / dt_ms)), 1)

    t_ms = np.arange(n_points, dtype=float) * dt_ms
    t_abs = float(start_time) + t_ms if np.isfinite(start_time) else np.full(n_points, np.nan)

    # Drift: random walk with extremely small steps
    drift_step_sigma = float(drift_sigma_px_per_sec) * np.sqrt(dt_ms / 1000.0)
    if drift_step_sigma > 0:
        drift_x = np.cumsum(rng.normal(0.0, drift_step_sigma, n_points))
        drift_y = np.cumsum(rng.normal(0.0, drift_step_sigma, n_points))
    else:
        drift_x = np.zeros(n_points)
        drift_y = np.zeros(n_points)

    # Noise: extremely small IID Gaussian
    noise_sigma_f = float(noise_sigma_px)
    if noise_sigma_f > 0:
        noise_x = rng.normal(0.0, noise_sigma_f, n_points)
        noise_y = rng.normal(0.0, noise_sigma_f, n_points)
    else:
        noise_x = np.zeros(n_points)
        noise_y = np.zeros(n_points)

    return pd.DataFrame({
        "fixation_id":               fixation_id,
        "point_index":               np.arange(n_points, dtype=int),
        "t_ms":                      t_ms,
        "t_abs":                     t_abs,
        "x":                         float(x_fix) + drift_x + noise_x,
        "y":                         float(y_fix) + drift_y + noise_y,
        "x_fix":                     float(x_fix),
        "y_fix":                     float(y_fix),
        "drift_x":                   drift_x,
        "drift_y":                   drift_y,
        "noise_x":                   noise_x,
        "noise_y":                   noise_y,
        "noise_sigma_used":          noise_sigma_f,
        "drift_sigma_px_per_sec_used": float(drift_sigma_px_per_sec),
        "drift_step_sigma_px":       drift_step_sigma,
        "sampling_rate_hz":          float(sampling_rate_hz),
        "dt_ms":                     float(dt_ms),
        "duration_ms":               float(duration_ms),
    })


# ============================================================
# 6. PER-FILE PIPELINE
# ============================================================

def generate_dcm_for_file(fixation_path, output_dir, config):
    fixation_path = Path(fixation_path)
    subject_id, task = infer_subject_task_from_path(fixation_path)

    # Deterministic seed per subject-task
    if config["USE_DETERMINISTIC_SEED"]:
        seed = stable_seed_from_string(
            f"{config['RANDOM_SEED']}_{subject_id}_{task}_{fixation_path.stem}_DCM"
        )
    else:
        seed = None
    rng = np.random.default_rng(seed)

    # Load fixations
    fix_df = normalize_fixation_dataframe(pd.read_csv(fixation_path), config=config)
    if len(fix_df) == 0:
        raise ValueError("No valid fixations remaining after filtering.")

    # Sampling rate from raw file (not from empirical distribution)
    sr_hz, median_dt_ms, sr_source = select_sampling_rate(fixation_path, config)

    # DCM parameters: FIXED CONSTANTS (not data-derived)
    noise_sigma = float(config["NOISE_SIGMA_PX"])
    drift_sigma = float(config["DRIFT_SIGMA_PX_PER_SEC"])

    # Generate
    synthetic_parts = []
    for _, row in fix_df.iterrows():
        syn_fix = generate_dcm_fixation(
            fixation_id=row["fixation_id"],
            x_fix=float(row["x_fix"]),
            y_fix=float(row["y_fix"]),
            duration_ms=float(row["duration_ms"]),
            noise_sigma_px=noise_sigma,
            drift_sigma_px_per_sec=drift_sigma,
            sampling_rate_hz=sr_hz,
            rng=rng,
            start_time=safe_float(row.get("start_time", np.nan)),
            config=config,
        )
        synthetic_parts.append(syn_fix)

    syn_df = pd.concat(synthetic_parts, ignore_index=True) if synthetic_parts else pd.DataFrame()
    syn_df.insert(0, "subject_id", subject_id)
    syn_df.insert(1, "task", task)
    syn_df.insert(2, "method", "DCM")

    output_path = None
    if config["SAVE_SYNTHETIC_FILES"]:
        ensure_dir(output_dir)
        output_path = Path(output_dir) / f"DCM_{subject_id}_{task}_{fixation_path.stem}.csv"
        syn_df.to_csv(output_path, index=False, encoding=config.get("CSV_ENCODING", "utf-8-sig"))

    return {
        "method": "DCM",
        "subject_id": subject_id,
        "task": task,
        "fixation_path": str(fixation_path),
        "synthetic_path": str(output_path) if output_path else None,
        "ok": True,
        "error": None,
        "n_fixations": int(len(fix_df)),
        "n_points_total": int(len(syn_df)),
        "sampling_rate_hz": float(sr_hz),
        "sampling_rate_source": sr_source,
        "median_dt_ms": float(median_dt_ms) if np.isfinite(median_dt_ms) else None,
        "noise_sigma_px": float(noise_sigma),
        "drift_sigma_px_per_sec": float(drift_sigma),
        "random_seed": int(seed) if seed is not None else None,
        # Confirm: sigma is a fixed constant, not data-derived
        "sigma_source": "fixed_constant_not_from_dataset",
    }


# ============================================================
# 7. PIPELINE
# ============================================================

def run_dcm_pipeline(config):
    data_dir = config["DATA_DIR"]
    output_dir = config["OUTPUT_DIR"]
    ensure_dir(output_dir)

    fixation_files = list_fixation_files(
        data_dir=data_dir,
        fixation_pattern=config["FIXATION_PATTERN"],
        exclude_tasks=config.get("EXCLUDE_TASKS", []),
    )

    print("=" * 80)
    print("DCM - Deterministic Centroid Minimization (Extreme Noiseless Baseline)")
    print("=" * 80)
    print(f"Data dir             : {data_dir}")
    print(f"Output dir           : {output_dir}")
    print(f"Files found          : {len(fixation_files)}")
    print(f"Exclude tasks        : {config.get('EXCLUDE_TASKS', [])}")
    print(f"Noise sigma          : {config['NOISE_SIGMA_PX']} px (IID, fixed constant)")
    print(f"Drift sigma          : {config['DRIFT_SIGMA_PX_PER_SEC']} px/s (fixed constant)")
    print(f"Sampling rate mode   : {config['SAMPLING_RATE_MODE']}")
    print(f"NOTE: DCM sigma is FIXED CONSTANT (no data-driven estimation)")
    print(f"      → variance ≈ 0, gaze < 1px from centroid, non-physiological by design")
    print("=" * 80)

    if not fixation_files:
        raise FileNotFoundError(f"No fixation files found in: {data_dir}")

    summaries = []
    for path in tqdm(fixation_files, desc="Generating DCM"):
        try:
            summary = generate_dcm_for_file(path, output_dir, config)
            summaries.append(summary)
        except Exception as e:
            sid, task = infer_subject_task_from_path(path)
            summaries.append({
                "method": "DCM", "subject_id": sid, "task": task,
                "fixation_path": str(path), "ok": False, "error": str(e),
            })

    n_ok = sum(1 for s in summaries if s.get("ok"))
    n_err = len(summaries) - n_ok

    # Sampling rate statistics
    srs = [float(s["sampling_rate_hz"]) for s in summaries
           if s.get("ok") and s.get("sampling_rate_hz")]

    summary_obj = {
        "method": "DCM",
        "description": (
            "DCM (Deterministic Centroid Minimization): Extreme Noiseless Baseline. "
            "x(t)=x_fix+drift+noise with noise_sigma=0.1px, drift_sigma=0.05px/s (fixed constants). "
            "Variance≈0, physiological drift by design. No empirical sigma from dataset."
        ),
        "config": config,
        "n_files": len(fixation_files),
        "n_ok": n_ok,
        "n_error": n_err,
        "noise_sigma_px_used": float(config["NOISE_SIGMA_PX"]),
        "drift_sigma_px_per_sec_used": float(config["DRIFT_SIGMA_PX_PER_SEC"]),
        "sampling_rate_stats": {
            "n": len(srs),
            "min": float(np.min(srs)) if srs else None,
            "max": float(np.max(srs)) if srs else None,
            "median": float(np.median(srs)) if srs else None,
            "mean": float(np.mean(srs)) if srs else None,
        },
        "results": summaries,
    }

    out_json = Path(output_dir) / config["SUMMARY_FILENAME"]
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_obj, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"DONE  OK={n_ok}  Error={n_err}")
    print(f"Summary: {out_json}")
    if srs:
        print(f"Sampling rate range: {min(srs):.1f} - {max(srs):.1f} Hz (median {np.median(srs):.1f})")
    print("=" * 80)

    return summary_obj


# ============================================================
# 8. CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="DCM - Extreme Noiseless Baseline")

    parser.add_argument("--data_dir",      default=CONFIG["DATA_DIR"])
    parser.add_argument("--output_dir",    default=CONFIG["OUTPUT_DIR"])
    parser.add_argument("--fixation_pattern", default=CONFIG["FIXATION_PATTERN"])
    parser.add_argument("--exclude_tasks", default=",".join(CONFIG["EXCLUDE_TASKS"]))
    parser.add_argument("--sampling_rate_mode", default=CONFIG["SAMPLING_RATE_MODE"],
                        choices=["subject_task", "fixed"])
    parser.add_argument("--sampling_rate", type=float, default=CONFIG["SAMPLING_RATE_HZ"])
    parser.add_argument("--allow_sampling_rate_fallback", action="store_true")

    # DCM core params — fixed constants
    parser.add_argument("--noise_sigma_px", type=float, default=CONFIG["NOISE_SIGMA_PX"],
                        help="IID noise sigma (px), default 0.1")
    parser.add_argument("--drift_sigma_px_per_sec", type=float,
                        default=CONFIG["DRIFT_SIGMA_PX_PER_SEC"],
                        help="Drift sigma (px/s), default 0.05")

    parser.add_argument("--enforce_min_duration", action="store_true")
    parser.add_argument("--drop_short_fixations", action="store_true")
    parser.add_argument("--min_duration_ms", type=float, default=CONFIG["MIN_DURATION_MS"])
    parser.add_argument("--save_files", action="store_true")
    parser.add_argument("--no_save_files", action="store_true")
    parser.add_argument("--summary_filename", default=CONFIG["SUMMARY_FILENAME"])

    return parser.parse_args()


def main():
    args = parse_args()
    config = CONFIG.copy()

    config["DATA_DIR"] = args.data_dir
    config["OUTPUT_DIR"] = args.output_dir
    config["FIXATION_PATTERN"] = args.fixation_pattern
    config["EXCLUDE_TASKS"] = (
        [normalize_task_name(x.strip())
         for x in args.exclude_tasks.split(",") if x.strip()]
        if args.exclude_tasks.strip() else []
    )
    config["SAMPLING_RATE_MODE"] = args.sampling_rate_mode
    config["SAMPLING_RATE_HZ"] = args.sampling_rate
    config["ALLOW_SAMPLING_RATE_FALLBACK"] = args.allow_sampling_rate_fallback
    config["NOISE_SIGMA_PX"] = args.noise_sigma_px
    config["DRIFT_SIGMA_PX_PER_SEC"] = args.drift_sigma_px_per_sec
    config["MIN_DURATION_MS"] = args.min_duration_ms
    config["SUMMARY_FILENAME"] = args.summary_filename

    if args.save_files:
        config["SAVE_SYNTHETIC_FILES"] = True
    if args.no_save_files:
        config["SAVE_SYNTHETIC_FILES"] = False
    if args.enforce_min_duration:
        config["ENFORCE_MIN_DURATION"] = True
    if args.drop_short_fixations:
        config["DROP_SHORT_FIXATIONS"] = True

    run_dcm_pipeline(config)


if __name__ == "__main__":
    main()
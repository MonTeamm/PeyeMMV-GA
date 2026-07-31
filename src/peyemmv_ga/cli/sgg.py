# ============================================================
# SGG - Stochastic Gaussian Generation
# Baseline 1 in the paper — Extreme Noisy Baseline
# ============================================================
#
# Method description (Section 4.7 of the paper):
#   SGG generates gaze points by IID 2D Gaussian sampling,
#   centred at the fixation target, sigma set for PeyeMMV compatibility
#   within PeyeMMV spatial dispersion limits.
#
#       x_i ~ N(x_fix, sigma_base^2)    (IID, independent)
#       y_i ~ N(y_fix, sigma_base^2)    (IID, independent)
#
#   Core property: pure white noise (zero autocorrelation),
#   no drift, no fBm, no 1/f structure.
#
# Sigma source (NO data leakage from dyslexia labels):
#   sigma_base computed from raw gaze of the corresponding subject-task.
#   This is a technical signal property (not a pathology label).
#   Dyslexia/control labels are never used in sigma estimation.
#
# Data leakage prevention:
#   - sigma_base taken from residuals inside each fixation window (not between fixations)
#   - No test-set information is used in sigma estimation
#   - Separation of calibration and evaluation pools is handled by evaluate_tstr.py
#
# Standard pipeline:
#   For each subject-task:
#       1. Read fixation file → x_fix, y_fix, duration_ms
#       2. Find raw file → estimate sampling_rate_hz
#       3. Estimate sigma_base from raw residuals inside fixation windows
#       4. Generate IID Gaussian points per fixation
#       5. Save synthetic CSV
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
    "DATA_DIR": "./data",
    "OUTPUT_DIR": "./sgg_output",
    "FIXATION_PATTERN": "*_fixations.csv",
    "EXCLUDE_TASKS": ["T1"],

    # --------------------------------------------------------
    # Sampling rate: from raw file per subject-task
    # --------------------------------------------------------
    "SAMPLING_RATE_MODE": "subject_task",
    "SAMPLING_RATE_HZ": None,
    "ALLOW_SAMPLING_RATE_FALLBACK": False,
    "SAMPLING_RATE_MIN_HZ": 10.0,
    "SAMPLING_RATE_MAX_HZ": 2000.0,

    # --------------------------------------------------------
    # Sigma mode:
    #   "subject_task" : computed from raw residuals inside fixation windows
    #                    -> sigma reflects sensor noise characteristics per subject
    #   "global"       : median sigma across the full dataset
    #                    -> less precise but avoids inconsistency
    #   "fixed"        : fixed constant (use only when raw files are unavailable)
    # --------------------------------------------------------
    "SIGMA_MODE": "subject_task",
    "SIGMA_FIXED": None,        # Used only when mode="fixed" or as fallback
    "SIGMA_MIN": None,          # Auto-computed from dataset (P05 percentile)
    "SIGMA_MAX": None,          # Auto-computed from dataset (P95 percentile)

    "AUTO_SIGMA_PARAMS_FROM_DATASET": True,
    "REQUIRE_DATASET_SIGMA_PARAMS": True,
    "ALLOW_MANUAL_SIGMA_FALLBACK": False,
    "SIGMA_MIN_PERCENTILE": 5.0,
    "SIGMA_MAX_PERCENTILE": 95.0,
    "SIGMA_HARD_CAP_PX": 7.5,

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
    "SUMMARY_FILENAME": "sgg_summary.json",
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
    col_sigma = find_col(["sigma_base", "sigma", "std", "std_xy", "sigma_real"])

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
    out["sigma_base_existing"] = df[col_sigma].astype(float) if col_sigma else np.nan

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["x_fix", "y_fix", "duration_ms"])
    out = out[out["duration_ms"] > 0].copy()

    if config is not None and config.get("DROP_SHORT_FIXATIONS", False):
        min_dur = float(config.get("MIN_DURATION_MS", 80.0))
        out = out[out["duration_ms"] >= min_dur].copy()

    return out.reset_index(drop=True)


# ============================================================
# 4. RAW FILE UTILITIES
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
    if float(np.median(dt)) > 100:
        return t_raw / 1000.0, "us_to_ms"
    return t_raw, "ms"


def read_raw_gaze(raw_path, config=None):
    """
    Read raw file, returns (t_ms, gaze_x, gaze_y, sr_hz, median_dt_ms, sr_source).
    """
    raw_df = pd.read_csv(raw_path)
    cols = _find_raw_columns(raw_df)

    if cols["time"] is None:
        raise ValueError(f"No time column found in {raw_path}")

    t_raw = raw_df[cols["time"]].astype(float).values

    if cols["x"] is not None and cols["y"] is not None:
        gaze_x = raw_df[cols["x"]].astype(float).values
        gaze_y = raw_df[cols["y"]].astype(float).values
    elif (cols["xl"] is not None and cols["yl"] is not None and
          cols["xr"] is not None and cols["yr"] is not None):
        gaze_x = np.nanmean(np.vstack([raw_df[cols["xl"]].values.astype(float),
                                        raw_df[cols["xr"]].values.astype(float)]), axis=0)
        gaze_y = np.nanmean(np.vstack([raw_df[cols["yl"]].values.astype(float),
                                        raw_df[cols["yr"]].values.astype(float)]), axis=0)
    elif cols["xr"] is not None and cols["yr"] is not None:
        gaze_x = raw_df[cols["xr"]].astype(float).values
        gaze_y = raw_df[cols["yr"]].astype(float).values
    elif cols["xl"] is not None and cols["yl"] is not None:
        gaze_x = raw_df[cols["xl"]].astype(float).values
        gaze_y = raw_df[cols["yl"]].astype(float).values
    else:
        raise ValueError(f"No gaze x/y columns found in {raw_path}")

    t_ms, _ = _normalize_time_to_ms(t_raw)

    # Sampling rate
    t_valid = t_ms[np.isfinite(t_ms)]
    sr_hz, median_dt_ms, sr_source = np.nan, np.nan, "no_valid_timestamps"
    if len(t_valid) >= 3:
        dt = np.diff(t_valid)
        dt = dt[(np.isfinite(dt)) & (dt > 0)]
        if len(dt) > 0:
            median_dt_ms = float(np.median(dt))
            if median_dt_ms > 0:
                sr_hz = 1000.0 / median_dt_ms
                min_hz = float(config.get("SAMPLING_RATE_MIN_HZ", 10.0)) if config else 10.0
                max_hz = float(config.get("SAMPLING_RATE_MAX_HZ", 2000.0)) if config else 2000.0
                sr_source = ("raw_timestamp" if (min_hz <= sr_hz <= max_hz)
                              else f"out_of_range_{sr_hz:.1f}Hz")

    return t_ms, gaze_x, gaze_y, float(sr_hz), float(median_dt_ms), sr_source


# ============================================================
# 5. SIGMA ESTIMATION
# ============================================================

def compute_sigma_from_raw(fix_df, raw_path, config=None):
    """
    Compute sigma_base = mean(std_x, std_y) of residuals inside fixation windows.

    Method:
        - Take all raw gaze points within [start_time, end_time] of each fixation
        - Compute dx = x_raw - x_fix, dy = y_raw - y_fix
        - sigma_base = (std(dx) + std(dy)) / 2

    No data leakage:
        - Dyslexia labels are never used
        - sigma reflects only eye-tracker noise, not pathology
    """
    try:
        t_ms, gaze_x, gaze_y, sr_hz, median_dt_ms, sr_source = read_raw_gaze(
            raw_path, config=config
        )
    except Exception as e:
        return {
            "sigma_base": np.nan, "sigma_source": f"raw_read_error: {e}",
            "n_raw_points_used": 0,
            "raw_sampling_rate_hz": np.nan, "raw_sampling_rate_source": "error",
            "raw_median_dt_ms": np.nan,
        }

    dx_all, dy_all = [], []

    for _, row in fix_df.iterrows():
        st = safe_float(row.get("start_time", np.nan))
        et = safe_float(row.get("end_time", np.nan))
        x_fix = safe_float(row.get("x_fix", np.nan))
        y_fix = safe_float(row.get("y_fix", np.nan))

        if not all(np.isfinite([st, et, x_fix, y_fix])) or et <= st:
            continue

        mask = (t_ms >= st) & (t_ms <= et)
        xs = gaze_x[mask]
        ys = gaze_y[mask]
        valid = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[valid], ys[valid]

        if len(xs) < 2:
            continue

        dx_all.extend((xs - x_fix).tolist())
        dy_all.extend((ys - y_fix).tolist())

    dx_all = np.asarray(dx_all, dtype=float)
    dy_all = np.asarray(dy_all, dtype=float)
    valid = np.isfinite(dx_all) & np.isfinite(dy_all)
    dx_all, dy_all = dx_all[valid], dy_all[valid]

    if len(dx_all) < 2:
        return {
            "sigma_base": np.nan,
            "sigma_source": "not_enough_raw_points_in_fixations",
            "n_raw_points_used": int(len(dx_all)),
            "raw_sampling_rate_hz": float(sr_hz),
            "raw_sampling_rate_source": sr_source,
            "raw_median_dt_ms": float(median_dt_ms),
        }

    sigma_x = float(np.std(dx_all, ddof=1))
    sigma_y = float(np.std(dy_all, ddof=1))
    sigma_base = (sigma_x + sigma_y) / 2.0

    return {
        "sigma_base": float(sigma_base),
        "sigma_x": float(sigma_x),
        "sigma_y": float(sigma_y),
        "sigma_source": "raw_residuals_in_fixation_windows",
        "n_raw_points_used": int(len(dx_all)),
        "raw_sampling_rate_hz": float(sr_hz),
        "raw_sampling_rate_source": sr_source,
        "raw_median_dt_ms": float(median_dt_ms),
    }


def compute_sigma_from_existing_column(fix_df):
    """Fallback: compute median sigma_base from the sigma column in the fixation file."""
    if "sigma_base_existing" not in fix_df.columns:
        return np.nan
    vals = fix_df["sigma_base_existing"].dropna().astype(float).values
    vals = vals[(np.isfinite(vals)) & (vals > 0)]
    return float(np.median(vals)) if len(vals) > 0 else np.nan


def clip_sigma(sigma, config):
    sigma = safe_float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        return np.nan
    lo = safe_float(config.get("SIGMA_MIN"), np.nan)
    hi = safe_float(config.get("SIGMA_MAX"), np.nan)
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        return float(np.clip(sigma, lo, hi))
    return float(sigma)


def compute_global_sigma_stats(fixation_files, config):
    """
    Compute empirical sigma distribution across the full dataset.
    Returns: global_sigma (median), sigma_min (P05), sigma_max (P95).
    """
    sigma_values = []
    details = []

    for path in tqdm(fixation_files, desc="Computing empirical sigma (SGG)"):
        sid, task = infer_subject_task_from_path(path)
        try:
            fix_df = normalize_fixation_dataframe(pd.read_csv(path), config=config)
            raw_path = find_raw_file(path)

            if raw_path is not None:
                info = compute_sigma_from_raw(fix_df, raw_path, config=config)
                sigma = safe_float(info.get("sigma_base", np.nan))
                source = info.get("sigma_source", "unknown")
            else:
                sigma = compute_sigma_from_existing_column(fix_df)
                source = "fixation_column_median_no_raw" if np.isfinite(sigma) else "no_raw_or_sigma"
                info = {}

            used = bool(np.isfinite(sigma) and sigma > 0)
            if used:
                sigma_values.append(float(sigma))

            details.append({
                "fixation_path": str(path), "subject_id": sid, "task": task,
                "sigma_base": float(sigma) if np.isfinite(sigma) else None,
                "sigma_source": source, "used_for_empirical": used,
                "raw_path": str(raw_path) if raw_path else None,
                "raw_sampling_rate_hz": (
                    float(info.get("raw_sampling_rate_hz"))
                    if info and np.isfinite(safe_float(info.get("raw_sampling_rate_hz")))
                    else None
                ),
            })
        except Exception as e:
            details.append({
                "fixation_path": str(path), "subject_id": sid, "task": task,
                "error": str(e), "used_for_empirical": False,
            })

    if not sigma_values:
        if bool(config.get("REQUIRE_DATASET_SIGMA_PARAMS", True)):
            raise ValueError(
                "Could not compute sigma from dataset. "
                "Check raw files, time/x/y columns, and start_time/end_time."
            )
        if not bool(config.get("ALLOW_MANUAL_SIGMA_FALLBACK", False)):
            raise ValueError("No empirical sigma and manual fallback is disabled.")
        sigma_fixed = safe_float(config.get("SIGMA_FIXED"), np.nan)
        if not np.isfinite(sigma_fixed) or sigma_fixed <= 0:
            raise ValueError("SIGMA_FIXED is invalid for fallback.")
        return {
            "global_sigma": float(sigma_fixed),
            "sigma_min": float(sigma_fixed),
            "sigma_max": float(sigma_fixed),
            "source": "manual_fallback",
            "n": 0, "details": details,
        }

    vals = np.array(sigma_values)
    lower_p = float(config.get("SIGMA_MIN_PERCENTILE", 5.0))
    upper_p = float(config.get("SIGMA_MAX_PERCENTILE", 95.0))

    global_sigma = float(np.median(vals))
    sigma_min = float(np.percentile(vals, lower_p))
    sigma_max = float(np.percentile(vals, upper_p))
    sigma_max = min(sigma_max, 7.5)

    if not np.isfinite(sigma_min) or sigma_min <= 0:
        sigma_min = float(np.min(vals))
    if not np.isfinite(sigma_max) or sigma_max <= sigma_min:
        sigma_max = float(np.max(vals))

    return {
        "global_sigma": global_sigma,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "source": "dataset_empirical",
        "n": len(sigma_values),
        "stats": {
            "min": float(np.min(vals)), "p05": float(np.percentile(vals, 5)),
            "p25": float(np.percentile(vals, 25)), "median": float(np.median(vals)),
            "p75": float(np.percentile(vals, 75)), "p95": float(np.percentile(vals, 95)),
            "max": float(np.max(vals)), "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        },
        "details": details,
    }


# ============================================================
# 6. SAMPLING RATE SELECTION
# ============================================================

def select_sampling_rate(fixation_path, config, raw_info=None):
    """
    Select sampling rate.
    raw_info: dict already containing raw_sampling_rate_hz from compute_sigma_from_raw.
    """
    mode = config.get("SAMPLING_RATE_MODE", "subject_task")

    if mode == "fixed":
        hz = safe_float(config.get("SAMPLING_RATE_HZ"), np.nan)
        if not np.isfinite(hz) or hz <= 0:
            raise ValueError("SAMPLING_RATE_MODE=fixed but SAMPLING_RATE_HZ is invalid.")
        return hz, 1000.0 / hz, "fixed"

    # Use cached raw_info if available (avoids reading raw file twice)
    if raw_info is not None:
        sr_hz = safe_float(raw_info.get("raw_sampling_rate_hz", np.nan))
        sr_source = raw_info.get("raw_sampling_rate_source", "unknown")
        median_dt = safe_float(raw_info.get("raw_median_dt_ms", np.nan))
        min_hz = float(config.get("SAMPLING_RATE_MIN_HZ", 10.0))
        max_hz = float(config.get("SAMPLING_RATE_MAX_HZ", 2000.0))

        if (np.isfinite(sr_hz) and sr_hz > 0 and min_hz <= sr_hz <= max_hz
                and sr_source == "raw_timestamp"):
            return float(sr_hz), float(median_dt), sr_source

    # Read raw file directly if raw_info is not cached
    raw_path = find_raw_file(fixation_path)
    if raw_path is None:
        if config.get("ALLOW_SAMPLING_RATE_FALLBACK", False):
            hz = safe_float(config.get("SAMPLING_RATE_HZ"), np.nan)
            if np.isfinite(hz) and hz > 0:
                return hz, 1000.0 / hz, "fallback_no_raw_file"
        raise ValueError(f"Raw file not found for: {fixation_path}")

    try:
        _, _, _, sr_hz, median_dt_ms, sr_source = read_raw_gaze(raw_path, config)
        min_hz = float(config.get("SAMPLING_RATE_MIN_HZ", 10.0))
        max_hz = float(config.get("SAMPLING_RATE_MAX_HZ", 2000.0))

        if (np.isfinite(sr_hz) and sr_hz > 0 and min_hz <= sr_hz <= max_hz
                and sr_source == "raw_timestamp"):
            return float(sr_hz), float(median_dt_ms), sr_source
    except Exception as e:
        sr_source = f"raw_read_error: {e}"

    if config.get("ALLOW_SAMPLING_RATE_FALLBACK", False):
        hz = safe_float(config.get("SAMPLING_RATE_HZ"), np.nan)
        if np.isfinite(hz) and hz > 0:
            return hz, 1000.0 / hz, f"fallback_after_{sr_source}"

    raise ValueError(
        f"Cannot estimate sampling rate: {sr_source}. "
        "ALLOW_SAMPLING_RATE_FALLBACK=False."
    )


# ============================================================
# 7. SGG GENERATION — CORE
# ============================================================

def generate_sgg_fixation(
    fixation_id,
    x_fix,
    y_fix,
    duration_ms,
    sigma_base,
    sampling_rate_hz,
    rng,
    start_time=np.nan,
    config=None,
):
    """
    Generate one fixation using SGG — IID Gaussian.

    Theo docx Section 4.7:
        x_i ~ N(x_fix, sigma_base^2)    (IID, no autocorrelation)
        y_i ~ N(y_fix, sigma_base^2)    (IID, no autocorrelation)

    Property: pure white noise. No drift, no fBm, no 1/f structure.
    """
    duration_ms = float(duration_ms)
    if config is not None and config.get("ENFORCE_MIN_DURATION", False):
        duration_ms = max(duration_ms, float(config.get("MIN_DURATION_MS", 80.0)))

    dt_ms = 1000.0 / float(sampling_rate_hz)
    n_points = max(int(round(duration_ms / dt_ms)), 1)

    # IID Gaussian — core property of SGG
    xs = rng.normal(loc=float(x_fix), scale=float(sigma_base), size=n_points)
    ys = rng.normal(loc=float(y_fix), scale=float(sigma_base), size=n_points)

    t_ms = np.arange(n_points, dtype=float) * dt_ms
    t_abs = float(start_time) + t_ms if np.isfinite(start_time) else np.full(n_points, np.nan)

    return pd.DataFrame({
        "fixation_id":       fixation_id,
        "point_index":       np.arange(n_points, dtype=int),
        "t_ms":              t_ms,
        "t_abs":             t_abs,
        "x":                 xs,
        "y":                 ys,
        "x_fix":             float(x_fix),
        "y_fix":             float(y_fix),
        "sigma_used":        float(sigma_base),
        "sampling_rate_hz":  float(sampling_rate_hz),
        "dt_ms":             float(dt_ms),
        "duration_ms":       float(duration_ms),
    })


# ============================================================
# 8. PER-FILE PIPELINE
# ============================================================

def generate_sgg_for_file(fixation_path, output_dir, config, global_sigma=None,
                           global_sigma_stats=None):
    fixation_path = Path(fixation_path)
    subject_id, task = infer_subject_task_from_path(fixation_path)

    if config["USE_DETERMINISTIC_SEED"]:
        seed = stable_seed_from_string(
            f"{config['RANDOM_SEED']}_{subject_id}_{task}_{fixation_path.stem}_SGG"
        )
    else:
        seed = None
    rng = np.random.default_rng(seed)

    fix_df = normalize_fixation_dataframe(pd.read_csv(fixation_path), config=config)
    if len(fix_df) == 0:
        raise ValueError("No valid fixations remaining after filtering.")

    sigma_mode = config["SIGMA_MODE"]

    if sigma_mode == "global":
        if global_sigma is None or not np.isfinite(global_sigma):
            raise ValueError("SIGMA_MODE='global' but global_sigma has not been computed.")
        sigma_base = clip_sigma(global_sigma, config)
        raw_info = None
        sigma_source = "global_dataset_median"
        n_raw_used = 0
        sr_raw_source = None

        # Still need raw file to get sampling rate
        raw_path = find_raw_file(fixation_path)
        if raw_path is not None:
            try:
                _, _, _, sr_hz_raw, median_dt_raw, sr_source_raw = read_raw_gaze(
                    raw_path, config
                )
                raw_info = {
                    "raw_sampling_rate_hz": sr_hz_raw,
                    "raw_sampling_rate_source": sr_source_raw,
                    "raw_median_dt_ms": median_dt_raw,
                }
            except Exception:
                raw_info = None

    elif sigma_mode == "fixed":
        sigma_fixed = safe_float(config.get("SIGMA_FIXED"), np.nan)
        if not np.isfinite(sigma_fixed) or sigma_fixed <= 0:
            raise ValueError("SIGMA_MODE='fixed' but SIGMA_FIXED is invalid.")
        sigma_base = clip_sigma(sigma_fixed, config)
        raw_info = None
        sigma_source = "fixed"
        n_raw_used = 0
        sr_raw_source = None

        raw_path = find_raw_file(fixation_path)
        if raw_path is not None:
            try:
                _, _, _, sr_hz_raw, median_dt_raw, sr_source_raw = read_raw_gaze(
                    raw_path, config
                )
                raw_info = {
                    "raw_sampling_rate_hz": sr_hz_raw,
                    "raw_sampling_rate_source": sr_source_raw,
                    "raw_median_dt_ms": median_dt_raw,
                }
            except Exception:
                raw_info = None

    else:  # subject_task
        raw_path = find_raw_file(fixation_path)
        if raw_path is None:
            # Fallback: sigma_base column from fixation file
            sigma_existing = compute_sigma_from_existing_column(fix_df)
            sigma_existing = clip_sigma(sigma_existing, config)
            if np.isfinite(sigma_existing):
                sigma_base = sigma_existing
                sigma_source = "fixation_column_median_no_raw"
                n_raw_used = 0
                raw_info = None
            elif bool(config.get("ALLOW_MANUAL_SIGMA_FALLBACK", False)):
                sigma_fixed = safe_float(config.get("SIGMA_FIXED"), np.nan)
                if not np.isfinite(sigma_fixed) or sigma_fixed <= 0:
                    raise ValueError("No sigma available and SIGMA_FIXED is invalid.")
                sigma_base = clip_sigma(sigma_fixed, config)
                sigma_source = "manual_fallback_no_raw"
                n_raw_used = 0
                raw_info = None
            else:
                raise ValueError(
                    "No raw file found and no sigma in fixation file. "
                    "Set ALLOW_MANUAL_SIGMA_FALLBACK=True or provide a raw file."
                )
        else:
            info = compute_sigma_from_raw(fix_df, raw_path, config=config)
            sigma_raw = clip_sigma(info.get("sigma_base", np.nan), config)
            raw_info = info

            if np.isfinite(sigma_raw):
                sigma_base = sigma_raw
                sigma_source = info.get("sigma_source", "raw_residuals")
                n_raw_used = int(info.get("n_raw_points_used", 0))
            else:
                # Fallback: sigma_base column
                sigma_existing = compute_sigma_from_existing_column(fix_df)
                sigma_existing = clip_sigma(sigma_existing, config)
                if np.isfinite(sigma_existing):
                    sigma_base = sigma_existing
                    sigma_source = "fixation_column_median_after_raw_failed"
                    n_raw_used = 0
                elif bool(config.get("ALLOW_MANUAL_SIGMA_FALLBACK", False)):
                    sigma_fixed = safe_float(config.get("SIGMA_FIXED"), np.nan)
                    if not np.isfinite(sigma_fixed) or sigma_fixed <= 0:
                        raise ValueError("SIGMA_FIXED is invalid.")
                    sigma_base = clip_sigma(sigma_fixed, config)
                    sigma_source = "manual_fallback_after_raw_failed"
                    n_raw_used = 0
                else:
                    raise ValueError(
                        f"Cannot compute sigma from raw ({info.get('sigma_source')}). "
                        "ALLOW_MANUAL_SIGMA_FALLBACK=False."
                    )

    if not np.isfinite(sigma_base) or sigma_base <= 0:
        raise ValueError(f"sigma_base is invalid: {sigma_base}")

    # Hard cap sigma for PeyeMMV tol2 compatibility
    hard_cap = config.get("SIGMA_HARD_CAP_PX", None)
    if hard_cap is not None and sigma_base > hard_cap:
        sigma_base = float(hard_cap)
    # Sampling rate
    sr_hz, median_dt_ms, sr_source = select_sampling_rate(fixation_path, config, raw_info)

    # Generate
    synthetic_parts = []
    for _, row in fix_df.iterrows():
        syn_fix = generate_sgg_fixation(
            fixation_id=row["fixation_id"],
            x_fix=float(row["x_fix"]),
            y_fix=float(row["y_fix"]),
            duration_ms=float(row["duration_ms"]),
            sigma_base=float(sigma_base),
            sampling_rate_hz=sr_hz,
            rng=rng,
            start_time=safe_float(row.get("start_time", np.nan)),
            config=config,
        )
        synthetic_parts.append(syn_fix)

    syn_df = pd.concat(synthetic_parts, ignore_index=True) if synthetic_parts else pd.DataFrame()
    syn_df.insert(0, "subject_id", subject_id)
    syn_df.insert(1, "task", task)
    syn_df.insert(2, "method", "SGG")

    output_path = None
    if config["SAVE_SYNTHETIC_FILES"]:
        ensure_dir(output_dir)
        output_path = Path(output_dir) / f"SGG_{subject_id}_{task}_{fixation_path.stem}.csv"
        syn_df.to_csv(output_path, index=False, encoding=config.get("CSV_ENCODING", "utf-8-sig"))

    raw_sr_hz = safe_float(raw_info.get("raw_sampling_rate_hz") if raw_info else np.nan)

    return {
        "method": "SGG",
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
        "sigma_mode": sigma_mode,
        "sigma_base": float(sigma_base),
        "sigma_source": sigma_source,
        "sigma_min_clip": float(config["SIGMA_MIN"]) if config.get("SIGMA_MIN") else None,
        "sigma_max_clip": float(config["SIGMA_MAX"]) if config.get("SIGMA_MAX") else None,
        "n_raw_points_used_for_sigma": int(n_raw_used) if sigma_mode == "subject_task" else 0,
        "raw_sampling_rate_hz": float(raw_sr_hz) if np.isfinite(raw_sr_hz) else None,
        "random_seed": int(seed) if seed is not None else None,
    }


# ============================================================
# 9. PIPELINE
# ============================================================

def run_sgg_pipeline(config):
    data_dir = config["DATA_DIR"]
    output_dir = config["OUTPUT_DIR"]
    ensure_dir(output_dir)

    fixation_files = list_fixation_files(
        data_dir=data_dir,
        fixation_pattern=config["FIXATION_PATTERN"],
        exclude_tasks=config.get("EXCLUDE_TASKS", []),
    )

    print("=" * 80)
    print("SGG - Stochastic Gaussian Generation (Extreme Noisy Baseline)")
    print("=" * 80)
    print(f"Data dir             : {data_dir}")
    print(f"Output dir           : {output_dir}")
    print(f"Files found          : {len(fixation_files)}")
    print(f"Exclude tasks        : {config.get('EXCLUDE_TASKS', [])}")
    print(f"Sigma mode           : {config['SIGMA_MODE']}")
    print(f"Sampling rate mode   : {config['SAMPLING_RATE_MODE']}")
    print(f"NOTE: SGG = IID Gaussian, no autocorrelation, white noise by design")
    print("=" * 80)

    if not fixation_files:
        raise FileNotFoundError(f"No fixation files found in: {data_dir}")

    global_sigma = None
    global_sigma_stats = None

    need_global = (
        bool(config.get("AUTO_SIGMA_PARAMS_FROM_DATASET", True))
        or config["SIGMA_MODE"] == "global"
    )

    if need_global:
        result = compute_global_sigma_stats(fixation_files, config)
        global_sigma = result["global_sigma"]
        global_sigma_stats = result

        if config.get("AUTO_SIGMA_PARAMS_FROM_DATASET", True):
            config["SIGMA_FIXED"] = float(result["global_sigma"])
            config["SIGMA_MIN"] = float(result["sigma_min"])
            config["SIGMA_MAX"] = float(result["sigma_max"])

        print(f"Empirical sigma source  : {result['source']}")
        print(f"Global sigma (median)   : {global_sigma:.4f} px")
        print(f"Sigma clip range        : [{config.get('SIGMA_MIN'):.4f}, {config.get('SIGMA_MAX'):.4f}] px")
        print(f"N sigma values          : {result.get('n', 0)}")

    summaries = []
    for path in tqdm(fixation_files, desc="Generating SGG"):
        try:
            summary = generate_sgg_for_file(
                path, output_dir, config,
                global_sigma=global_sigma,
                global_sigma_stats=global_sigma_stats,
            )
            summaries.append(summary)
        except Exception as e:
            sid, task = infer_subject_task_from_path(path)
            summaries.append({
                "method": "SGG", "subject_id": sid, "task": task,
                "fixation_path": str(path), "ok": False, "error": str(e),
            })

    n_ok = sum(1 for s in summaries if s.get("ok"))
    n_err = len(summaries) - n_ok

    srs = [float(s["sampling_rate_hz"]) for s in summaries
           if s.get("ok") and s.get("sampling_rate_hz")]
    sigmas = [float(s["sigma_base"]) for s in summaries
              if s.get("ok") and s.get("sigma_base")]

    summary_obj = {
        "method": "SGG",
        "description": (
            "SGG (Stochastic Gaussian Generation): Extreme Noisy Baseline. "
            "x_i~N(x_fix, sigma^2), y_i~N(y_fix, sigma^2), IID. "
            "Pure white noise, no autocorrelation."
        ),
        "config": config,
        "n_files": len(fixation_files),
        "n_ok": n_ok,
        "n_error": n_err,
        "global_sigma_used": float(global_sigma) if global_sigma is not None else None,
        "global_sigma_stats": global_sigma_stats,
        "sampling_rate_stats": {
            "n": len(srs),
            "min": float(np.min(srs)) if srs else None,
            "max": float(np.max(srs)) if srs else None,
            "median": float(np.median(srs)) if srs else None,
        },
        "sigma_stats": {
            "n": len(sigmas),
            "min": float(np.min(sigmas)) if sigmas else None,
            "max": float(np.max(sigmas)) if sigmas else None,
            "median": float(np.median(sigmas)) if sigmas else None,
            "mean": float(np.mean(sigmas)) if sigmas else None,
        },
        "results": summaries,
    }

    out_json = Path(output_dir) / config["SUMMARY_FILENAME"]
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_obj, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"DONE  OK={n_ok}  Error={n_err}")
    print(f"Summary: {out_json}")
    if sigmas:
        print(f"Sigma range: {min(sigmas):.3f} - {max(sigmas):.3f} px (median {np.median(sigmas):.3f})")
    if srs:
        print(f"Sampling rate range: {min(srs):.1f} - {max(srs):.1f} Hz")
    print("=" * 80)

    return summary_obj


# ============================================================
# 10. CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="SGG - Stochastic Gaussian Generation")

    parser.add_argument("--data_dir",   default=CONFIG["DATA_DIR"])
    parser.add_argument("--output_dir", default=CONFIG["OUTPUT_DIR"])
    parser.add_argument("--fixation_pattern", default=CONFIG["FIXATION_PATTERN"])
    parser.add_argument("--exclude_tasks", default=",".join(CONFIG["EXCLUDE_TASKS"]))

    parser.add_argument("--sampling_rate_mode", default=CONFIG["SAMPLING_RATE_MODE"],
                        choices=["subject_task", "fixed"])
    parser.add_argument("--sampling_rate", type=float, default=CONFIG["SAMPLING_RATE_HZ"])
    parser.add_argument("--allow_sampling_rate_fallback", action="store_true")

    parser.add_argument("--sigma_mode", default=CONFIG["SIGMA_MODE"],
                        choices=["subject_task", "global", "fixed"])
    parser.add_argument("--sigma_fixed", type=float, default=CONFIG["SIGMA_FIXED"])
    parser.add_argument("--sigma_min", type=float, default=CONFIG["SIGMA_MIN"])
    parser.add_argument("--sigma_max", type=float, default=CONFIG["SIGMA_MAX"])
    parser.add_argument("--no_auto_sigma_params", action="store_true")
    parser.add_argument("--allow_manual_sigma_fallback", action="store_true")
    parser.add_argument("--sigma_min_percentile", type=float,
                        default=CONFIG["SIGMA_MIN_PERCENTILE"])
    parser.add_argument("--sigma_max_percentile", type=float,
                        default=CONFIG["SIGMA_MAX_PERCENTILE"])

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

    config["SIGMA_MODE"] = args.sigma_mode
    if args.sigma_fixed is not None:
        config["SIGMA_FIXED"] = args.sigma_fixed
    if args.sigma_min is not None:
        config["SIGMA_MIN"] = args.sigma_min
    if args.sigma_max is not None:
        config["SIGMA_MAX"] = args.sigma_max
    if args.no_auto_sigma_params:
        config["AUTO_SIGMA_PARAMS_FROM_DATASET"] = False
        config["REQUIRE_DATASET_SIGMA_PARAMS"] = False
    if args.allow_manual_sigma_fallback:
        config["ALLOW_MANUAL_SIGMA_FALLBACK"] = True
        config["REQUIRE_DATASET_SIGMA_PARAMS"] = False
    config["SIGMA_MIN_PERCENTILE"] = args.sigma_min_percentile
    config["SIGMA_MAX_PERCENTILE"] = args.sigma_max_percentile

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

    run_sgg_pipeline(config)


if __name__ == "__main__":
    main()
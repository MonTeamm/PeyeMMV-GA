"""
utils.py — Shared utilities for the PeyeMMV synthetic gaze generation pipeline.

Provides:
    - File I/O helpers
    - Fixation / raw gaze normalisation
    - Sigma estimation
    - Sampling-rate detection
    - ETDD70-compatible metrics computation (compute_metrics, compute_tstr_metrics_from_gaze)

Used by: sggx100.py, dcmx100.py, ga2x100.py, ga2_generate.py, evaluate_tstr.py
"""

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# ROI map (shared across all generators)
# ---------------------------------------------------------------------------
ROI_MAP = {
    "T4_Meaningful_Text": "Meaningful_Text_rois.csv",
    "T5_Pseudo_Text":     "Pseudo_Text_rois.csv",
}


# ===========================================================================
# 1.  GENERAL HELPERS
# ===========================================================================

def ensure_dir(path: str | Path) -> None:
    """Create directory (and parents) if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_float(value, default: float = float("nan")) -> float:
    """Safely cast *value* to float; return *default* on failure or NaN."""
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return default


def stable_seed_from_string(seed_str: str) -> int:
    """Deterministic 32-bit seed derived from an arbitrary string via MD5."""
    h = hashlib.md5(seed_str.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


# ===========================================================================
# 2.  PATH / TASK UTILITIES
# ===========================================================================

def normalize_task_name(task: str | None) -> str:
    """Normalise task identifiers to canonical short form (e.g. 'T4')."""
    if task is None:
        return "unknown"
    task = str(task).upper().strip()
    mapping = {
        "TASK1": "T1", "TASK2": "T2", "TASK3": "T3",
        "TASK4": "T4", "TASK5": "T5",
    }
    return mapping.get(task, task)


def infer_subject_task_from_path(path: str | Path) -> tuple[str, str]:
    """
    Extract (subject_id, task) from a fixation-file path.

    Examples
    --------
    ``Subject_12_T4_Meaningful_Text_fixations.csv`` → ('12', 'T4_Meaningful_Text')
    """
    stem = Path(path).stem.replace("_fixations", "")
    parts = stem.replace("-", "_").split("_")

    subject_id: str | None = None
    task: str | None = None

    # subject_id: first digit-only part, or the part after "Subject"
    for p in parts:
        if p.isdigit():
            subject_id = p
            break
    for i, p in enumerate(parts):
        if p.lower() == "subject" and i + 1 < len(parts):
            if parts[i + 1].isdigit():
                subject_id = parts[i + 1]
                break

    # task: part matching T1–T5 or TASK1–TASK5
    for i, p in enumerate(parts):
        p_upper = p.upper()
        if p_upper in {"T1", "T2", "T3", "T4", "T5"}:
            task = "_".join([p_upper] + parts[i + 1:])
            break
        if p_upper in {"TASK1", "TASK2", "TASK3", "TASK4", "TASK5"}:
            task = "_".join([normalize_task_name(p_upper)] + parts[i + 1:])
            break

    return str(subject_id or stem), task or "unknown"


def list_fixation_files(
    data_dir: str | Path,
    fixation_pattern: str,
    exclude_tasks: list[str] | None = None,
) -> list[Path]:
    """Return sorted list of fixation files, optionally excluding certain tasks."""
    data_dir = Path(data_dir)
    exclude_set: set[str] = set()
    if exclude_tasks:
        exclude_set = {normalize_task_name(t) for t in exclude_tasks}

    files = []
    for f in data_dir.rglob(fixation_pattern):
        _, task = infer_subject_task_from_path(f)
        if task.split("_")[0] in exclude_set:
            continue
        files.append(f)
    return sorted(files)


def find_raw_file(fixation_path: str | Path) -> Path | None:
    """Locate the raw gaze file corresponding to a fixation file."""
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


# ===========================================================================
# 3.  FIXATION DATAFRAME NORMALISATION
# ===========================================================================

def normalize_fixation_dataframe(
    df: pd.DataFrame,
    config: dict | None = None,
) -> pd.DataFrame:
    """
    Normalise a raw fixation CSV into a canonical schema:
    ``fixation_id, x_fix, y_fix, duration_ms, start_time, end_time, sigma_base_existing``
    """
    lower_map = {c.lower().strip(): c for c in df.columns}

    def find_col(candidates: list[str]) -> str | None:
        for cand in candidates:
            if cand.lower().strip() in lower_map:
                return lower_map[cand.lower().strip()]
        return None

    col_fix_id  = find_col(["fixation_id", "fix_id", "id", "fixid"])
    col_x       = find_col(["x", "center_x", "centroid_x", "gaze_x",
                             "x_fix", "fix_x", "mean_x", "avg_x"])
    col_y       = find_col(["y", "center_y", "centroid_y", "gaze_y",
                             "y_fix", "fix_y", "mean_y", "avg_y"])
    col_dur     = find_col(["duration_ms", "duration", "dur_ms",
                             "fixation_duration", "duration_msec"])
    col_start   = find_col(["start_time", "start_ms", "onset", "start",
                             "t_start", "begin_time"])
    col_end     = find_col(["end_time", "end_ms", "offset", "end",
                             "t_end", "finish_time"])
    col_sigma   = find_col(["sigma_base", "sigma", "std", "std_xy", "sigma_real"])

    if col_x is None or col_y is None:
        raise ValueError(
            f"Cannot find x/y columns. Available: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["fixation_id"]        = df[col_fix_id] if col_fix_id else np.arange(len(df))
    out["x_fix"]              = df[col_x].astype(float)
    out["y_fix"]              = df[col_y].astype(float)

    if col_dur is not None:
        out["duration_ms"] = df[col_dur].astype(float)
    elif col_start is not None and col_end is not None:
        out["duration_ms"] = df[col_end].astype(float) - df[col_start].astype(float)
    else:
        raise ValueError("Cannot find duration_ms or start_time/end_time columns.")

    out["start_time"]         = df[col_start].astype(float) if col_start else np.nan
    out["end_time"]           = df[col_end].astype(float)   if col_end   else np.nan
    out["sigma_base_existing"] = df[col_sigma].astype(float) if col_sigma else np.nan

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["x_fix", "y_fix", "duration_ms"])
    out = out[out["duration_ms"] > 0].copy()

    if config is not None and config.get("DROP_SHORT_FIXATIONS", False):
        min_dur = float(config.get("MIN_DURATION_MS", 80.0))
        out = out[out["duration_ms"] >= min_dur].copy()

    return out.reset_index(drop=True)


# ===========================================================================
# 4.  RAW GAZE FILE UTILITIES
# ===========================================================================

def _find_raw_columns(raw_df: pd.DataFrame) -> dict[str, str | None]:
    lower_map = {c.lower().strip(): c for c in raw_df.columns}

    def find_col(candidates: list[str]) -> str | None:
        for c in candidates:
            if c.lower().strip() in lower_map:
                return lower_map[c.lower().strip()]
        return None

    return {
        "time": find_col(["time", "t", "timestamp", "time_us",
                           "time_ms", "timestamp_ms", "timestamp_us"]),
        "x":    find_col(["x", "gaze_x", "gaze_x_mean", "mean_x"]),
        "y":    find_col(["y", "gaze_y", "gaze_y_mean", "mean_y"]),
        "xl":   find_col(["gaze_x_left",  "x_left",  "left_x",  "gaze_left_x"]),
        "yl":   find_col(["gaze_y_left",  "y_left",  "left_y",  "gaze_left_y"]),
        "xr":   find_col(["gaze_x_right", "x_right", "right_x", "gaze_right_x"]),
        "yr":   find_col(["gaze_y_right", "y_right", "right_y", "gaze_right_y"]),
    }


def _normalize_time_to_ms(t_raw: np.ndarray) -> tuple[np.ndarray, str]:
    t_raw   = np.asarray(t_raw, dtype=float)
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


def read_raw_gaze(
    raw_path: str | Path,
    config: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, str]:
    """
    Read a raw gaze file and return
    ``(t_ms, gaze_x, gaze_y, sr_hz, median_dt_ms, sr_source)``.
    """
    raw_df = pd.read_csv(raw_path)
    cols   = _find_raw_columns(raw_df)

    if cols["time"] is None:
        raise ValueError(f"No time column found in {raw_path}")

    t_raw = raw_df[cols["time"]].astype(float).values

    if cols["x"] is not None and cols["y"] is not None:
        gaze_x = raw_df[cols["x"]].astype(float).values
        gaze_y = raw_df[cols["y"]].astype(float).values
    elif all(cols[k] is not None for k in ("xl", "yl", "xr", "yr")):
        gaze_x = np.nanmean(np.vstack([
            raw_df[cols["xl"]].values.astype(float),
            raw_df[cols["xr"]].values.astype(float),
        ]), axis=0)
        gaze_y = np.nanmean(np.vstack([
            raw_df[cols["yl"]].values.astype(float),
            raw_df[cols["yr"]].values.astype(float),
        ]), axis=0)
    elif cols["xr"] is not None and cols["yr"] is not None:
        gaze_x = raw_df[cols["xr"]].astype(float).values
        gaze_y = raw_df[cols["yr"]].astype(float).values
    elif cols["xl"] is not None and cols["yl"] is not None:
        gaze_x = raw_df[cols["xl"]].astype(float).values
        gaze_y = raw_df[cols["yl"]].astype(float).values
    else:
        raise ValueError(f"No gaze x/y columns found in {raw_path}")

    t_ms, _ = _normalize_time_to_ms(t_raw)

    t_valid = t_ms[np.isfinite(t_ms)]
    sr_hz = median_dt_ms = float("nan")
    sr_source = "no_valid_timestamps"
    if len(t_valid) >= 3:
        dt = np.diff(t_valid)
        dt = dt[(np.isfinite(dt)) & (dt > 0)]
        if len(dt) > 0:
            median_dt_ms = float(np.median(dt))
            if median_dt_ms > 0:
                sr_hz     = 1000.0 / median_dt_ms
                min_hz    = float(config.get("SAMPLING_RATE_MIN_HZ",  10.0)) if config else  10.0
                max_hz    = float(config.get("SAMPLING_RATE_MAX_HZ", 2000.0)) if config else 2000.0
                sr_source = ("raw_timestamp"
                             if min_hz <= sr_hz <= max_hz
                             else f"out_of_range_{sr_hz:.1f}Hz")

    return t_ms, gaze_x, gaze_y, float(sr_hz), float(median_dt_ms), sr_source


# ===========================================================================
# 5.  SIGMA ESTIMATION
# ===========================================================================

def compute_sigma_from_raw(
    fix_df: pd.DataFrame,
    raw_path: str | Path,
    config: dict | None = None,
) -> dict:
    """
    Estimate sigma_base = mean(std_x, std_y) of gaze residuals inside
    fixation windows.  Does NOT use dyslexia labels — no data leakage.
    """
    try:
        t_ms, gaze_x, gaze_y, sr_hz, median_dt_ms, sr_source = read_raw_gaze(
            raw_path, config=config
        )
    except Exception as exc:
        return {
            "sigma_base": float("nan"),
            "sigma_source": f"raw_read_error: {exc}",
            "n_raw_points_used": 0,
            "raw_sampling_rate_hz": float("nan"),
            "raw_sampling_rate_source": "error",
            "raw_median_dt_ms": float("nan"),
        }

    dx_all: list[float] = []
    dy_all: list[float] = []

    for _, row in fix_df.iterrows():
        st    = safe_float(row.get("start_time", float("nan")))
        et    = safe_float(row.get("end_time",   float("nan")))
        x_fix = safe_float(row.get("x_fix",      float("nan")))
        y_fix = safe_float(row.get("y_fix",      float("nan")))

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

    arr_dx = np.asarray(dx_all, dtype=float)
    arr_dy = np.asarray(dy_all, dtype=float)
    valid  = np.isfinite(arr_dx) & np.isfinite(arr_dy)
    arr_dx, arr_dy = arr_dx[valid], arr_dy[valid]

    if len(arr_dx) < 2:
        return {
            "sigma_base": float("nan"),
            "sigma_source": "not_enough_raw_points_in_fixations",
            "n_raw_points_used": int(len(arr_dx)),
            "raw_sampling_rate_hz": float(sr_hz),
            "raw_sampling_rate_source": sr_source,
            "raw_median_dt_ms": float(median_dt_ms),
        }

    sigma_x    = float(np.std(arr_dx, ddof=1))
    sigma_y    = float(np.std(arr_dy, ddof=1))
    sigma_base = (sigma_x + sigma_y) / 2.0

    return {
        "sigma_base": sigma_base,
        "sigma_x":    sigma_x,
        "sigma_y":    sigma_y,
        "sigma_source": "raw_residuals_in_fixation_windows",
        "n_raw_points_used": int(len(arr_dx)),
        "raw_sampling_rate_hz": float(sr_hz),
        "raw_sampling_rate_source": sr_source,
        "raw_median_dt_ms": float(median_dt_ms),
    }


def compute_sigma_from_existing_column(fix_df: pd.DataFrame) -> float:
    """Fallback: median of a pre-computed sigma column in the fixation file."""
    if "sigma_base_existing" not in fix_df.columns:
        return float("nan")
    vals = fix_df["sigma_base_existing"].dropna().astype(float).values
    vals = vals[(np.isfinite(vals)) & (vals > 0)]
    return float(np.median(vals)) if len(vals) > 0 else float("nan")


def clip_sigma(sigma: float, config: dict) -> float:
    """Clip sigma to [SIGMA_MIN, SIGMA_MAX] and hard cap at SIGMA_HARD_CAP_PX."""
    s_min     = config.get("SIGMA_MIN")
    s_max     = config.get("SIGMA_MAX")
    hard_cap  = float(config.get("SIGMA_HARD_CAP_PX", 7.5))

    if s_min is not None and np.isfinite(s_min):
        sigma = max(sigma, float(s_min))
    if s_max is not None and np.isfinite(s_max):
        sigma = min(sigma, float(s_max))
    return min(sigma, hard_cap)


# ===========================================================================
# 6.  ETDD70-COMPATIBLE METRICS
# ===========================================================================

#: Column order matching the ETDD70 *_metrics.csv schema
_METRICS_COLUMNS = [
    "sid", "stimfile", "eye_used", "trialid",
    "n_fix_trial", "sum_fix_dur_trial", "dwell_time_trial", "mean_fix_dur_trial",
    "n_sacc_trial", "sum_sacc_dur_trial", "mean_sacc_dur_trial", "mean_sacc_ampl_trial",
    "ratio_progress_regress_trial", "n_between_line_regress_trial",
    "n_within_line_regress_trial", "n_regress_trial", "n_progress_trial",
    "n_transit_trial",
    "aoi", "aoi_kind", "content",
    "dwell_time_aoi", "n_fix_aoi", "sum_fix_dur_aoi", "mean_fix_dur_aoi",
    "skipped_aoi", "n_fix_first_visit_aoi", "first_fix_dur_aoi",
    "first_fix_land_pos_aoi", "dwell_time_first_visit_aoi",
    "sum_fix_dur_first_visit_aoi", "sum_fix_dur_after_first_visit_aoi",
    "dwell_time_rereading_aoi", "n_revisits_aoi", "task",
]


def compute_metrics(
    fix_df:  pd.DataFrame,
    sacc_df: pd.DataFrame,
    roi_df:  pd.DataFrame,
    sid:     str,
    task:    str,
    trial_id: int = 12,
) -> pd.DataFrame:
    """
    Compute ETDD70-compatible metrics from fixation and saccade DataFrames.

    Returns a DataFrame with columns matching ``_METRICS_COLUMNS``.
    """
    stimfile  = str(roi_df.iloc[0].get("stimfile", "")) if len(roi_df) > 0 else ""
    n_fix     = len(fix_df)
    sum_fd    = float(fix_df["duration_ms"].sum()) if n_fix   > 0 else 0.0
    mean_fd   = sum_fd / n_fix                     if n_fix   > 0 else 0.0
    n_sacc    = len(sacc_df)
    sum_sd    = float(sacc_df["duration_ms"].sum()) if n_sacc > 0 else 0.0
    mean_sd   = sum_sd / n_sacc                     if n_sacc > 0 else 0.0
    mean_ampl = float(sacc_df["ampl"].mean())        if n_sacc > 0 else 0.0
    n_prog    = int((sacc_df["end_x"] > sacc_df["start_x"]).sum()) if n_sacc > 0 else 0
    n_reg     = n_sacc - n_prog
    ratio     = n_prog / n_reg if n_reg > 0 else float(n_prog)
    dwell     = sum_fd + sum_sd

    whole = {
        "sid": sid, "stimfile": stimfile, "eye_used": "b", "trialid": trial_id,
        "n_fix_trial": n_fix, "sum_fix_dur_trial": sum_fd,
        "dwell_time_trial": dwell, "mean_fix_dur_trial": mean_fd,
        "n_sacc_trial": n_sacc, "sum_sacc_dur_trial": sum_sd,
        "mean_sacc_dur_trial": mean_sd, "mean_sacc_ampl_trial": mean_ampl,
        "ratio_progress_regress_trial": ratio,
        "n_between_line_regress_trial": 0,
        "n_within_line_regress_trial":  n_reg,
        "n_regress_trial": n_reg, "n_progress_trial": n_prog,
        "n_transit_trial": n_sacc,
        "task": task,
    }

    line_rois = (roi_df[roi_df["kind"] == "line"]
                 .sort_values("id").reset_index(drop=True)
                 if "kind" in roi_df.columns else pd.DataFrame())
    sub_rois  = (roi_df[roi_df["kind"] == "sub-line"]
                 .sort_values("id").reset_index(drop=True)
                 if "kind" in roi_df.columns else pd.DataFrame())

    rows = []

    for _, lroi in line_rois.iterrows():
        ln  = str(lroi["name"])
        fln = (fix_df[fix_df["aoi_line"] == ln]
               if "aoi_line" in fix_df.columns else pd.DataFrame())
        nl  = len(fln)
        rows.append({
            **whole,
            "aoi":      ln.replace(" ", "_").replace("-", "_"),
            "aoi_kind": "line",
            "content":  np.nan,
            "dwell_time_aoi":                    float(fln["duration_ms"].sum())  if nl > 0 else 0.0,
            "n_fix_aoi":                         nl,
            "sum_fix_dur_aoi":                   float(fln["duration_ms"].sum())  if nl > 0 else 0.0,
            "mean_fix_dur_aoi":                  float(fln["duration_ms"].mean()) if nl > 0 else 0.0,
            "skipped_aoi":                       int(nl == 0),
            "n_fix_first_visit_aoi":             nl,
            "first_fix_dur_aoi":                 float(fln["duration_ms"].iloc[0]) if nl > 0 else 0.0,
            "first_fix_land_pos_aoi":            np.nan,
            "dwell_time_first_visit_aoi":        float(fln["duration_ms"].sum())  if nl > 0 else 0.0,
            "sum_fix_dur_first_visit_aoi":       float(fln["duration_ms"].sum())  if nl > 0 else 0.0,
            "sum_fix_dur_after_first_visit_aoi": 0.0,
            "dwell_time_rereading_aoi":          0.0,
            "n_revisits_aoi":                    max(nl - 1, 0),
        })

    for _, sroi in sub_rois.iterrows():
        an   = str(sroi["name"])
        fin  = (fix_df[fix_df["aoi_subline"] == an]
                if "aoi_subline" in fix_df.columns else pd.DataFrame())
        ns   = len(fin)
        rx   = float(sroi.get("x",     0))
        rw   = float(sroi.get("width", 1))
        if ns > 0:
            first = (fin.sort_values("start_ms").iloc[0]
                     if "start_ms" in fin.columns else fin.iloc[0])
            land = (float(first["fix_x"]) - rx) / rw if rw > 0 else 0.5
            sd   = float(fin["duration_ms"].sum())
            md   = sd / ns
            fd   = float(first["duration_ms"])
        else:
            land = sd = md = fd = 0.0
        rows.append({
            **whole,
            "aoi": an, "aoi_kind": "subline",
            "content": sroi.get("content", np.nan),
            "dwell_time_aoi": sd, "n_fix_aoi": ns,
            "sum_fix_dur_aoi": sd, "mean_fix_dur_aoi": md,
            "skipped_aoi": int(ns == 0), "n_fix_first_visit_aoi": min(ns, 1),
            "first_fix_dur_aoi": fd,
            "first_fix_land_pos_aoi": land if ns > 0 else 0.0,
            "dwell_time_first_visit_aoi":        fd,
            "sum_fix_dur_first_visit_aoi":       fd,
            "sum_fix_dur_after_first_visit_aoi": sd - fd,
            "dwell_time_rereading_aoi":          sd - fd,
            "n_revisits_aoi": max(ns - 1, 0),
        })

    if not rows:
        rows.append({
            **whole,
            "aoi": "trial", "aoi_kind": "trial", "content": np.nan,
            "dwell_time_aoi": sum_fd, "n_fix_aoi": n_fix,
            "sum_fix_dur_aoi": sum_fd, "mean_fix_dur_aoi": mean_fd,
            "skipped_aoi": 0, "n_fix_first_visit_aoi": n_fix,
            "first_fix_dur_aoi": float(fix_df["duration_ms"].iloc[0]) if n_fix > 0 else 0.0,
            "first_fix_land_pos_aoi": np.nan,
            "dwell_time_first_visit_aoi":        sum_fd,
            "sum_fix_dur_first_visit_aoi":       sum_fd,
            "sum_fix_dur_after_first_visit_aoi": 0.0,
            "dwell_time_rereading_aoi":          0.0,
            "n_revisits_aoi": max(n_fix - 1, 0),
        })

    df_out = pd.DataFrame(rows)
    for col in _METRICS_COLUMNS:
        if col not in df_out.columns:
            df_out[col] = np.nan
    return df_out[_METRICS_COLUMNS]


def compute_tstr_metrics_from_gaze(
    syn_df:      pd.DataFrame,
    orig_fix_df: pd.DataFrame,
    sid:         str,
    task:        str,
    trial_id:    int = 12,
    roi_df:      pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Convert gaze-level *syn_df* → fixation-level → saccades → ETDD70 metrics.

    Pipeline
    --------
    1. For each fixation_id in *orig_fix_df*, extract the gaze cluster from *syn_df*.
    2. Keep ``duration_ms`` from *orig_fix_df* (preserves original timing).
    3. Update ``fix_x/fix_y`` = mean(x/y) of the cluster.
    4. Infer saccades from consecutive fixation centroids.
    5. Call :func:`compute_metrics`.
    """
    if roi_df is None or len(roi_df) == 0:
        roi_df = pd.DataFrame(
            columns=["kind", "id", "name", "stimfile", "x", "width", "content"]
        )

    syn_fixes = []
    for _, f in orig_fix_df.iterrows():
        fid     = f.get("fixation_id", f.name)
        cluster = syn_df[syn_df["fixation_id"] == fid]
        if len(cluster) == 0:
            continue
        row           = f.to_dict()
        row["fix_x"]  = float(cluster["x"].mean())
        row["fix_y"]  = float(cluster["y"].mean())
        if "start_time" in f and not pd.isna(f.get("start_time", float("nan"))):
            row["start_ms"] = float(f["start_time"])
            row["end_ms"]   = float(f["start_time"]) + float(f["duration_ms"])
        elif "start_ms" not in row:
            row["start_ms"] = 0.0
            row["end_ms"]   = float(f["duration_ms"])
        syn_fixes.append(row)

    if not syn_fixes:
        return pd.DataFrame()

    syn_fix_df = pd.DataFrame(syn_fixes).reset_index(drop=True)

    saccades = []
    for k in range(len(syn_fix_df) - 1):
        r0 = syn_fix_df.iloc[k]
        r1 = syn_fix_df.iloc[k + 1]
        s  = float(r0.get("end_ms",   r0.get("start_ms", 0.0) + r0.get("duration_ms", 0.0)))
        e  = float(r1.get("start_ms", s))
        if e - s <= 0:
            continue
        x1, y1 = float(r0["fix_x"]), float(r0["fix_y"])
        x2, y2 = float(r1["fix_x"]), float(r1["fix_y"])
        saccades.append({
            "start_ms": s, "end_ms": e, "duration_ms": e - s,
            "ampl":     float(np.hypot(x2 - x1, y2 - y1)),
            "start_x":  x1, "start_y": y1,
            "end_x":    x2, "end_y":   y2,
        })

    syn_sacc_df = (
        pd.DataFrame(saccades) if saccades
        else pd.DataFrame(columns=[
            "start_ms", "end_ms", "duration_ms", "ampl",
            "start_x", "start_y", "end_x", "end_y",
        ])
    )
    return compute_metrics(syn_fix_df, syn_sacc_df, roi_df, sid, task, trial_id)

# ===========================================================================
# 7.  GLOBAL SIGMA STATISTICS
# ===========================================================================

def compute_global_sigma_stats(fixation_files, config: dict) -> dict:
    """
    Compute empirical sigma distribution across the full dataset.

    Returns global_sigma (median), sigma_min (P05), sigma_max (P95).
    Used by SGG to set dataset-wide sigma bounds without label leakage.
    """
    from tqdm import tqdm

    sigma_values: list[float] = []
    details: list[dict]       = []

    for path in tqdm(fixation_files, desc="Computing empirical sigma"):
        sid, task = infer_subject_task_from_path(path)
        try:
            fix_df   = normalize_fixation_dataframe(pd.read_csv(path), config=config)
            raw_path = find_raw_file(path)

            if raw_path is not None:
                info   = compute_sigma_from_raw(fix_df, raw_path, config=config)
                sigma  = safe_float(info.get("sigma_base", float("nan")))
                source = info.get("sigma_source", "unknown")
            else:
                sigma  = compute_sigma_from_existing_column(fix_df)
                source = ("fixation_column_median_no_raw"
                          if np.isfinite(sigma) else "no_raw_or_sigma")
                info   = {}

            used = bool(np.isfinite(sigma) and sigma > 0)
            if used:
                sigma_values.append(float(sigma))

            details.append({
                "fixation_path": str(path), "subject_id": sid, "task": task,
                "sigma_base": float(sigma) if np.isfinite(sigma) else None,
                "sigma_source": source, "used_for_empirical": used,
                "raw_path": str(raw_path) if raw_path else None,
                "raw_sampling_rate_hz": (
                    float(info["raw_sampling_rate_hz"])
                    if info and np.isfinite(safe_float(info.get("raw_sampling_rate_hz")))
                    else None
                ),
            })
        except Exception as exc:
            details.append({
                "fixation_path": str(path), "subject_id": sid, "task": task,
                "error": str(exc), "used_for_empirical": False,
            })

    if not sigma_values:
        if bool(config.get("REQUIRE_DATASET_SIGMA_PARAMS", True)):
            raise ValueError(
                "Could not compute sigma from dataset. "
                "Check raw files, time/x/y columns, and start_time/end_time."
            )
        if not bool(config.get("ALLOW_MANUAL_SIGMA_FALLBACK", False)):
            raise ValueError("No empirical sigma and manual fallback is disabled.")
        sigma_fixed = safe_float(config.get("SIGMA_FIXED"), float("nan"))
        if not np.isfinite(sigma_fixed) or sigma_fixed <= 0:
            raise ValueError("SIGMA_FIXED is invalid for fallback.")
        return {
            "global_sigma": float(sigma_fixed),
            "sigma_min":    float(sigma_fixed),
            "sigma_max":    float(sigma_fixed),
            "source": "manual_fallback", "n": 0, "details": details,
        }

    vals      = np.array(sigma_values)
    lower_p   = float(config.get("SIGMA_MIN_PERCENTILE",  5.0))
    upper_p   = float(config.get("SIGMA_MAX_PERCENTILE", 95.0))
    glob_sig  = float(np.median(vals))
    sig_min   = float(np.percentile(vals, lower_p))
    sig_max   = min(float(np.percentile(vals, upper_p)), 7.5)

    if not np.isfinite(sig_min) or sig_min <= 0:
        sig_min = float(np.min(vals))
    if not np.isfinite(sig_max) or sig_max <= sig_min:
        sig_max = float(np.max(vals))

    return {
        "global_sigma": glob_sig,
        "sigma_min":    sig_min,
        "sigma_max":    sig_max,
        "source": "dataset_empirical",
        "n": len(sigma_values),
        "stats": {
            "min":    float(np.min(vals)),
            "p05":    float(np.percentile(vals,  5)),
            "p25":    float(np.percentile(vals, 25)),
            "median": float(np.median(vals)),
            "p75":    float(np.percentile(vals, 75)),
            "p95":    float(np.percentile(vals, 95)),
            "max":    float(np.max(vals)),
            "mean":   float(np.mean(vals)),
            "std":    float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        },
        "details": details,
    }


# ===========================================================================
# 8.  SAMPLING RATE SELECTION
# ===========================================================================

def select_sampling_rate(
    fixation_path,
    config: dict,
    raw_info: dict | None = None,
) -> tuple[float, float, str]:
    """
    Select the sampling rate for a fixation file.

    Priority: config fixed → raw_info cache → read raw file → fallback.

    Returns ``(sr_hz, median_dt_ms, source_tag)``.
    """
    mode = config.get("SAMPLING_RATE_MODE", "subject_task")

    if mode == "fixed":
        hz = safe_float(config.get("SAMPLING_RATE_HZ"), float("nan"))
        if not np.isfinite(hz) or hz <= 0:
            raise ValueError(
                "SAMPLING_RATE_MODE=fixed but SAMPLING_RATE_HZ is invalid."
            )
        return hz, 1000.0 / hz, "fixed"

    # Use cached raw_info if available (avoids reading raw file twice)
    if raw_info is not None:
        sr_hz    = safe_float(raw_info.get("raw_sampling_rate_hz",  float("nan")))
        sr_src   = raw_info.get("raw_sampling_rate_source", "unknown")
        med_dt   = safe_float(raw_info.get("raw_median_dt_ms",      float("nan")))
        min_hz   = float(config.get("SAMPLING_RATE_MIN_HZ",  10.0))
        max_hz   = float(config.get("SAMPLING_RATE_MAX_HZ", 2000.0))
        if (np.isfinite(sr_hz) and sr_hz > 0
                and min_hz <= sr_hz <= max_hz
                and sr_src == "raw_timestamp"):
            return float(sr_hz), float(med_dt), sr_src

    # Read raw file directly
    raw_path = find_raw_file(fixation_path)
    if raw_path is None:
        if config.get("ALLOW_SAMPLING_RATE_FALLBACK", False):
            hz = safe_float(config.get("SAMPLING_RATE_HZ"), float("nan"))
            if np.isfinite(hz) and hz > 0:
                return hz, 1000.0 / hz, "fallback_no_raw_file"
        raise ValueError(f"Raw file not found for: {fixation_path}")

    sr_source = "raw_read_error"
    try:
        _, _, _, sr_hz, median_dt_ms, sr_source = read_raw_gaze(raw_path, config)
        min_hz = float(config.get("SAMPLING_RATE_MIN_HZ",  10.0))
        max_hz = float(config.get("SAMPLING_RATE_MAX_HZ", 2000.0))
        if (np.isfinite(sr_hz) and sr_hz > 0
                and min_hz <= sr_hz <= max_hz
                and sr_source == "raw_timestamp"):
            return float(sr_hz), float(median_dt_ms), sr_source
    except Exception as exc:
        sr_source = f"raw_read_error: {exc}"

    if config.get("ALLOW_SAMPLING_RATE_FALLBACK", False):
        hz = safe_float(config.get("SAMPLING_RATE_HZ"), float("nan"))
        if np.isfinite(hz) and hz > 0:
            return hz, 1000.0 / hz, f"fallback_after_{sr_source}"

    raise ValueError(
        f"Cannot estimate sampling rate: {sr_source}. "
        "Set ALLOW_SAMPLING_RATE_FALLBACK=True to enable fallback."
    )
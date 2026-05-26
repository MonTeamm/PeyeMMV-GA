"""
sggx20.py — Stochastic Gaussian Generation (SGG), extreme noisy baseline.

Method (Section 4.7):
    Gaze points sampled IID from a 2D Gaussian centred at each fixation:
        x_i ~ N(x_fix, sigma_base^2)
        y_i ~ N(y_fix, sigma_base^2)
    Pure white noise — no drift, no fBm, no 1/f structure.

Sigma source (no label leakage):
    sigma_base estimated from raw gaze residuals inside fixation windows.
    Dyslexia/control labels are never used during sigma estimation.

Pipeline per subject-task:
    1. Read fixation file  → x_fix, y_fix, duration_ms
    2. Locate raw file     → estimate sampling_rate_hz
    3. Estimate sigma_base from raw residuals in fixation windows
    4. Generate IID Gaussian points per fixation
    5. Save synthetic CSV
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from Utils import (
    ensure_dir, safe_float, stable_seed_from_string,
    normalize_task_name, infer_subject_task_from_path,
    list_fixation_files, find_raw_file,
    normalize_fixation_dataframe, read_raw_gaze,
    compute_sigma_from_raw, compute_sigma_from_existing_column,
    clip_sigma, compute_global_sigma_stats, select_sampling_rate,
    compute_metrics, compute_tstr_metrics_from_gaze,
    ROI_MAP,
)

# ============================================================
# 1. CONFIG
# ============================================================

CONFIG = {
    "DATA_DIR": "./data",
    "OUTPUT_DIR": "./sgg_output",

    # Number of synthetic sets to generate per subject-task
    "N_SYN_SETS": 20,
    "FIXATION_PATTERN": "*_fixations.csv",
    "EXCLUDE_TASKS": ["T1"],

    # --------------------------------------------------------
    # Sampling rate: estimated from raw file per subject-task
    # --------------------------------------------------------
    "SAMPLING_RATE_MODE": "subject_task",
    "SAMPLING_RATE_HZ": None,
    "ALLOW_SAMPLING_RATE_FALLBACK": False,
    "SAMPLING_RATE_MIN_HZ": 10.0,
    "SAMPLING_RATE_MAX_HZ": 2000.0,

    # --------------------------------------------------------
    # Sigma mode:
    # "subject_task": estimated from raw residuals inside fixation windows
    # → sigma reflects sensor noise characteristics per subject
    # "global": median sigma across the full dataset
    # → less precise but avoids inconsistency
    # "fixed": constant value (use only when raw files are unavailable)
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
    "SUMMARY_FILENAME": "sggx20_summary.json",
    "CSV_ENCODING": "utf-8-sig",

    # TSTR metrics output
    "TSTR_OUTPUT_DIR": "./sgg_tstr_output",
    "ROIS_DIR": "./rois",
    "SAVE_TSTR_METRICS": True,
}

# ---------------------------------------------------------------------------
# Shared utilities imported from Utils.py
# ---------------------------------------------------------------------------


# ============================================================
# 8. SGG GENERATION — CORE
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

    Per Section 4.7 of the paper:
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
# 9. PER-FILE PIPELINE
# ============================================================

def generate_sgg_for_file(fixation_path, output_dir, config, global_sigma=None,
                          global_sigma_stats=None, set_idx=0,
                          tstr_output_dir=None, roi_df=None):
    fixation_path = Path(fixation_path)
    subject_id, task = infer_subject_task_from_path(fixation_path)

    if config["USE_DETERMINISTIC_SEED"]:
        seed = stable_seed_from_string(
            f"{config['RANDOM_SEED']}_{subject_id}_{task}_{fixation_path.stem}_SGG_{set_idx}"
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
        output_path = Path(output_dir) / f"SGG_{subject_id}_{task}_{fixation_path.stem}_syn_{set_idx}.csv"
        syn_df.to_csv(output_path, index=False, encoding=config.get("CSV_ENCODING", "utf-8-sig"))

    raw_sr_hz = safe_float(raw_info.get("raw_sampling_rate_hz") if raw_info else np.nan)

    result = {
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
        "set_idx": int(set_idx),
    }

    # ---- Save TSTR metrics (evaluate_tstr.py-compatible) ----
    if (tstr_output_dir is not None
            and config.get("SAVE_TSTR_METRICS", True)
            and not syn_df.empty):
        try:
            mdf = compute_tstr_metrics_from_gaze(
                syn_df=syn_df, orig_fix_df=fix_df,
                sid=subject_id, task=task, trial_id=12, roi_df=roi_df,
            )
            if mdf is not None and not mdf.empty:
                tstr_subj_dir = Path(tstr_output_dir) / f"Subject_{subject_id}"
                tstr_subj_dir.mkdir(parents=True, exist_ok=True)
                # Write gaze-level data to raw/ subdirectory
                if not syn_df.empty:
                    raw_dir = tstr_subj_dir / "raw"
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    syn_df.to_csv(
                        raw_dir / f"Subject_{subject_id}_{task}_gaze_syn_{set_idx}.csv",
                        index=False, encoding=config.get("CSV_ENCODING", "utf-8-sig")
                    )
                mdf.to_csv(
                    tstr_subj_dir / f"Subject_{subject_id}_{task}_metrics_syn_{set_idx}.csv",
                    index=False, encoding=config.get("CSV_ENCODING", "utf-8-sig")
                )
        except Exception as e_tstr:
            import warnings
            warnings.warn(f"[SGGx20 TSTR] {subject_id} {task} set {set_idx}: {e_tstr}")

    return result


# ============================================================
# 10. PIPELINE
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
    n_sets = int(config.get("N_SYN_SETS", 1))
    print(f"Data dir             : {data_dir}")
    print(f"Output dir           : {output_dir}")
    print(f"Files found          : {len(fixation_files)}")
    print(f"Exclude tasks        : {config.get('EXCLUDE_TASKS', [])}")
    print(f"Sigma mode           : {config['SIGMA_MODE']}")
    print(f"Sampling rate mode   : {config['SAMPLING_RATE_MODE']}")
    print(f"N synthetic sets     : {n_sets}")
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
    total_jobs = len(fixation_files) * max(n_sets, 1)
    pbar = tqdm(total=total_jobs, desc="Generating SGGx20")

    for path in fixation_files:
        for i_set in range(max(n_sets, 1)):
            try:
                _sid, _task = infer_subject_task_from_path(path)
                _tstr_dir = config.get("TSTR_OUTPUT_DIR") \
                            if config.get("SAVE_TSTR_METRICS", True) else None

                # --- CHECKPOINT: skip if metrics file already exists ---
                if _tstr_dir is not None:
                    _cp = (Path(_tstr_dir) / f"Subject_{_sid}"
                           / f"Subject_{_sid}_{_task}_metrics_syn_{i_set}.csv")
                    if _cp.exists() and _cp.stat().st_size > 0:
                        pbar.update(1)
                        continue
                _roi_df = None
                if _tstr_dir is not None:
                    _rois_dir = config.get("ROIS_DIR", "./rois")
                    _roi_file = ROI_MAP.get(_task)
                    if _roi_file:
                        _roi_path = Path(_rois_dir) / _roi_file
                        if _roi_path.exists():
                            _roi_df = pd.read_csv(_roi_path)
                summary = generate_sgg_for_file(
                    path, output_dir, config,
                    global_sigma=global_sigma,
                    global_sigma_stats=global_sigma_stats,
                    set_idx=i_set, tstr_output_dir=_tstr_dir, roi_df=_roi_df,
                )
                summaries.append(summary)
            except Exception as e:
                sid, task = infer_subject_task_from_path(path)
                summaries.append({
                    "method": "SGG", "subject_id": sid, "task": task,
                    "set_idx": int(i_set), "fixation_path": str(path),
                    "ok": False, "error": str(e),
                })
            finally:
                pbar.update(1)

    pbar.close()

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
        "n_sets": int(max(n_sets, 1)),
        "n_jobs": int(len(fixation_files) * max(n_sets, 1)),
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
    parser.add_argument("--n_sets", type=int, default=CONFIG["N_SYN_SETS"],
                        help="Number of synthetic sets to generate (default: 20)")
    parser.add_argument("--tstr_output_dir", type=str, default=None)
    parser.add_argument("--rois_dir", type=str, default=None)
    parser.add_argument("--no_tstr_metrics", action="store_true")

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

    config["N_SYN_SETS"] = max(1, int(args.n_sets))
    if args.no_tstr_metrics:
        config["SAVE_TSTR_METRICS"] = False
    if args.tstr_output_dir is not None:
        config["TSTR_OUTPUT_DIR"] = args.tstr_output_dir
    if args.rois_dir is not None:
        config["ROIS_DIR"] = args.rois_dir

    run_sgg_pipeline(config)


if __name__ == "__main__":
    main()
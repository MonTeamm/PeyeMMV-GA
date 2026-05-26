"""
dcmx20.py — Deterministic Centroid Minimization (DCM), extreme noiseless baseline.

Method (Section 4.7):
    Near-static gaze with variance ≈ 0:
        x(t) = x_fix + drift_x(t) + noise_x(t)
        y(t) = y_fix + drift_y(t) + noise_y(t)
    NOISE_SIGMA = 0.1 px (IID), DRIFT_STEP ≈ 0 px/step.
    Fixed constants — not data-derived.

Three-way comparison:
    DCM : near-static (variance ≈ 0)  → high detection, non-physiological
    SGG : IID Gaussian (empirical σ)  → white noise, no correlation
    GA2 : fBm (optimised H)           → pink 1/f noise, physiological

No data leakage:
    x_fix/y_fix from fixation file, sampling_rate from raw timestamps,
    noise/drift sigma are hard-coded constants.
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
    normalize_fixation_dataframe,
    compute_metrics, compute_tstr_metrics_from_gaze,
    ROI_MAP,
)

# ============================================================
# 1. CONFIG
# ============================================================

CONFIG = {
    # Directory containing fixation and raw files
    "DATA_DIR": "./data",

    # Output directory
    "OUTPUT_DIR": "./dcm_output",

    # Number of synthetic sets to generate per subject-task
    "N_SYN_SETS": 20,

    # Pattern file fixation
    "FIXATION_PATTERN": "*_fixations.csv",

    # Excluded tasks (T1 not used in this study)
    "EXCLUDE_TASKS": ["T1"],

    # --------------------------------------------------------
    # Sampling rate: always estimated from raw file per subject-task
    # --------------------------------------------------------
    "SAMPLING_RATE_MODE": "subject_task",
    "SAMPLING_RATE_HZ": None,           # Used only when mode="fixed" or as fallback
    "ALLOW_SAMPLING_RATE_FALLBACK": False,
    "SAMPLING_RATE_MIN_HZ": 10.0,
    "SAMPLING_RATE_MAX_HZ": 2000.0,

    # --------------------------------------------------------
    # DCM Core Parameters — FIXED CONSTANTS, not data-derived
    # --------------------------------------------------------
    #
    # NOISE_SIGMA_PX: IID Gaussian noise, < 1 pixel
    "NOISE_SIGMA_PX": 0,

    # Random walk drift
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
    "SUMMARY_FILENAME": "dcmx20_summary.json",
    "CSV_ENCODING": "utf-8-sig",

    # --------------------------------------------------------
    # TSTR metrics output (evaluate_tstr.py-compatible)
    # --------------------------------------------------------
    "TSTR_OUTPUT_DIR": "./dcm_tstr_output",
    "ROIS_DIR": "./rois",
    "SAVE_TSTR_METRICS": True,
}

# ---------------------------------------------------------------------------
# Shared utilities imported from Utils.py
# ---------------------------------------------------------------------------

# ============================================================
# 6. DCM GENERATION — CORE
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

    # Noise: near-zero IID Gaussian
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
# 7. PER-FILE PIPELINE
# ============================================================

def generate_dcm_for_file(fixation_path, output_dir, config,
                          set_idx=0, tstr_output_dir=None, roi_df=None):
    fixation_path = Path(fixation_path)
    subject_id, task = infer_subject_task_from_path(fixation_path)

    # Deterministic seed per subject-task
    if config["USE_DETERMINISTIC_SEED"]:
        seed = stable_seed_from_string(
            f"{config['RANDOM_SEED']}_{subject_id}_{task}_{fixation_path.stem}_DCM_{set_idx}"
        )
    else:
        seed = None
    rng = np.random.default_rng(seed)

    # Load fixations
    fix_df = normalize_fixation_dataframe(pd.read_csv(fixation_path), config=config)
    if len(fix_df) == 0:
        raise ValueError("No valid fixations remaining after filtering.")

    # Sampling rate from raw file (not empirical distribution)
    sr_hz, median_dt_ms, sr_source = select_sampling_rate(fixation_path, config)

    # DCM parameters: FIXED CONSTANTS (not empirically derived)
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
        output_path = Path(output_dir) / f"DCM_{subject_id}_{task}_{fixation_path.stem}_syn_{set_idx}.csv"
        syn_df.to_csv(output_path, index=False, encoding=config.get("CSV_ENCODING", "utf-8-sig"))

    result = {
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
        "sigma_source": "fixed_constant_not_from_dataset",
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
            warnings.warn(f"[DCMx20 TSTR] {subject_id} {task} set {set_idx}: {e_tstr}")

    return result


# ============================================================
# 8. PIPELINE
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
    n_sets = int(config.get("N_SYN_SETS", 1))
    print(f"Data dir             : {data_dir}")
    print(f"Output dir           : {output_dir}")
    print(f"Files found          : {len(fixation_files)}")
    print(f"Exclude tasks        : {config.get('EXCLUDE_TASKS', [])}")
    print(f"Noise sigma          : {config['NOISE_SIGMA_PX']} px (IID, fixed constant)")
    print(f"Drift sigma          : {config['DRIFT_SIGMA_PX_PER_SEC']} px/s (fixed constant)")
    print(f"Sampling rate mode   : {config['SAMPLING_RATE_MODE']}")
    print(f"N synthetic sets     : {n_sets}")
    print(f"NOTE: DCM sigma is FIXED CONSTANT (no data-driven estimation)")
    print(f"      → variance ≈ 0, gaze < 1px from centroid, non-physiological by design")
    print("=" * 80)

    if not fixation_files:
        raise FileNotFoundError(f"No fixation files found in: {data_dir}")

    summaries = []
    total_jobs = len(fixation_files) * max(n_sets, 1)
    pbar = tqdm(total=total_jobs, desc="Generating DCMx20")

    for path in fixation_files:
        for i_set in range(max(n_sets, 1)):
            try:
                _sid, _task = infer_subject_task_from_path(path)

                # --- CHECKPOINT ---
                if config.get("TSTR_OUTPUT_DIR"):
                    _cp = (Path(config["TSTR_OUTPUT_DIR"]) / f"Subject_{_sid}"
                           / f"Subject_{_sid}_{_task}_metrics_syn_{i_set}.csv")
                    if _cp.exists() and _cp.stat().st_size > 0:
                        pbar.update(1)
                        continue
                _tstr_dir = config.get("TSTR_OUTPUT_DIR") \
                            if config.get("SAVE_TSTR_METRICS", True) else None
                _roi_df = None
                if _tstr_dir is not None:
                    _rois_dir = config.get("ROIS_DIR", "./rois")
                    _roi_file = ROI_MAP.get(_task)
                    if _roi_file:
                        _roi_path = Path(_rois_dir) / _roi_file
                        if _roi_path.exists():
                            _roi_df = pd.read_csv(_roi_path)
                summary = generate_dcm_for_file(
                    path, output_dir, config,
                    set_idx=i_set, tstr_output_dir=_tstr_dir, roi_df=_roi_df,
                )
                summaries.append(summary)
            except Exception as e:
                sid, task = infer_subject_task_from_path(path)
                summaries.append({
                    "method": "DCM", "subject_id": sid, "task": task,
                    "set_idx": int(i_set), "fixation_path": str(path),
                    "ok": False, "error": str(e),
                })
            finally:
                pbar.update(1)

    pbar.close()

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
        "n_sets": int(max(n_sets, 1)),
        "n_jobs": int(len(fixation_files) * max(n_sets, 1)),
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
# 9. CLI
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
    parser.add_argument("--n_sets", type=int, default=CONFIG["N_SYN_SETS"],
                        help="Number of synthetic sets to generate (default: 20)")
    parser.add_argument("--tstr_output_dir", type=str, default=None,
                        help="Output directory for TSTR metrics")
    parser.add_argument("--rois_dir", type=str, default=None,
                        help="Directory containing ROI CSV files")
    parser.add_argument("--no_tstr_metrics", action="store_true",
                        help="Do not save TSTR metrics")

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

    config["N_SYN_SETS"] = max(1, int(args.n_sets))
    if args.no_tstr_metrics:
        config["SAVE_TSTR_METRICS"] = False
    if args.tstr_output_dir is not None:
        config["TSTR_OUTPUT_DIR"] = args.tstr_output_dir
    if args.rois_dir is not None:
        config["ROIS_DIR"] = args.rois_dir

    run_dcm_pipeline(config)


if __name__ == "__main__":
    main()
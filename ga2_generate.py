"""
ga2_generate.py — Generate synthetic gaze using GA2 (engine: engine.py)

So sanh voi ga_generate.py (dung co_vt.py):
  - Fitness: F = w1*D - w2*O - w3*(E_psd + E_disp)
      E_psd : L1-distance PSD slope velocity syn vs real
      E_disp: relative dispersion error syn vs real fixation
  - H estimation: velocity PSD slope  H = (1 - slope) / 2  (fGn)
    thay vi position PSD cua co_vt.py
    - Co the dung theta CSV rieng cho GA2 (xu ly giong GA)
        hoac chay toi uu truc tiep per subject-task
  - Drift: velocity lay mau dong deu [DRIFT_VEL_MIN_PX_S, DRIFT_VEL_MAX_PX_S]
    thay vi drift_strength_ratio co dinh

Input:
  data/Subject_{sid}_{task}_raw.csv
  data/Subject_{sid}_{task}_fixations.csv
  data/dyslexia_class_label.csv   <- lay danh sach subject

Output:
  ga2_output/GA2_subject_{sid}_task_{task}_Subject_{sid}_{task}_fixations.csv
  ga2_output/ga2_generate_summary.json

Schema CSV (tuong thich compare_generators.py):
  subject_id, task, method="GA",
  fixation_id, point_index, t_ms, t_abs,
    x, y, x_fix, y_fix, peyemmv_passed,
  H_used, sigma_used, phi_used, H_base_used,
  sampling_rate_hz, dt_ms, duration_ms

Cach dung:
  python ga2_generate.py
  python ga2_generate.py --data_dir ./data --output_dir ./ga2_output
    python ga2_generate.py --theta_csv ./syn_output/ga2_theta_star.csv
  python ga2_generate.py --pop 100 --gens 50 --w1 10 --w2 2 --w3 5
  python ga2_generate.py --max_workers 4
"""
import os
import sys
import json
import argparse
import importlib.util
import concurrent.futures
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# IMPORT tu engine.py
# ============================================================
_HERE    = os.path.dirname(os.path.abspath(__file__))
_P2_FILE = os.path.join(_HERE, "engine.py")
_spec = importlib.util.spec_from_file_location("_p2", _P2_FILE)
_p2   = importlib.util.module_from_spec(_spec)
sys.modules["_p2"] = _p2   # required for Python 3.13 dataclass resolution
_spec.loader.exec_module(_p2)

load_raw_gaze        = _p2.load_raw_gaze
load_fixations       = _p2.load_fixations
detect_sampling_rate = _p2.detect_sampling_rate
estimate_baseline    = _p2.estimate_baseline
run_ga               = _p2.run_ga
peyemmv_check        = _p2.peyemmv_check
generate_displacement = _p2.generate_displacement
combine_seed         = _p2.combine_seed
stable_hash_int      = _p2.stable_hash_int
CFG = _p2.CFG   # singleton dict — patch to adjust GA settings

# ============================================================
# HANG SO
# ============================================================
TASKS = ["T4_Meaningful_Text", "T5_Pseudo_Text"]

CONFIG = {
    "DATA_DIR":         "./data",
    "OUTPUT_DIR":       "./ga2_output",
    "THETA_CSV":        "./syn_output/ga2_theta_star.csv",
    "SAVE_FILES":       True,
    "CSV_ENCODING":     "utf-8-sig",
    "SUMMARY_FILENAME": "ga2_generate_summary.json",
}

# Default GA values — mapped to CFG keys in engine.py
DEFAULT_GA = {
    "POPULATION_SIZE": 100,
    "GENERATIONS":     50,
    "CROSSOVER_RATE":  0.80,
    "MUTATION_RATE":   0.05,
    "W1_D":            8.0,
    "W2_O":            2.0,
    "W3_E":            10.0,
}


def load_ga2_theta_csv(theta_csv_path):
    """Load GA2 theta CSV and build lookup by (sid, task)."""
    df = pd.read_csv(theta_csv_path)
    if df.empty:
        raise ValueError(f"Theta CSV rong: {theta_csv_path}")

    sid_col = None
    for c in ("sid", "subject_id"):
        if c in df.columns:
            sid_col = c
            break
    if sid_col is None:
        raise ValueError("Theta CSV thieu cot sid/subject_id")

    required = ["task", "H", "sigma", "phi"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Theta CSV thieu cot: {missing}")

    lookup = {}
    for _, row in df.iterrows():
        sid = str(row[sid_col])
        task = str(row["task"])
        seed_val = row["seed_best"] if "seed_best" in df.columns else np.nan
        fit_val = row["best_fitness"] if "best_fitness" in df.columns else np.nan
        d_val = row["best_D"] if "best_D" in df.columns else np.nan
        o_val = row["best_O"] if "best_O" in df.columns else np.nan
        e_val = row["best_E"] if "best_E" in df.columns else np.nan

        lookup[(sid, task)] = {
            "H": float(row["H"]),
            "sigma": float(row["sigma"]),
            "phi": float(row["phi"]),
            "seed_best": int(seed_val) if np.isfinite(seed_val) else None,
            "best_fitness": float(fit_val) if np.isfinite(fit_val) else np.nan,
            "best_D": float(d_val) if np.isfinite(d_val) else np.nan,
            "best_O": float(o_val) if np.isfinite(o_val) else np.nan,
            "best_E": float(e_val) if np.isfinite(e_val) else np.nan,
        }
    return lookup


def resolve_theta_csv_path(user_theta_csv, output_dir):
    """Resolve theta CSV path with auto-discovery when user does not pass --theta_csv."""
    if user_theta_csv is not None:
        return user_theta_csv

    candidates = [
        CONFIG["THETA_CSV"],
        "./ga2_theta_star.csv",
        "./test_mini/ga2_theta_star.csv",
        os.path.join(output_dir, "ga2_theta_star.csv"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


# ============================================================
# WORKER (subprocess — must patch CFG locally)
# ============================================================
def worker_ga2(args_tuple):
    """
    Generate GA2 synthetic gaze for one subject-task.

        args_tuple: (sid, task, data_dir, output_dir, save_files, csv_enc, ga_cfg, theta_lookup)
      ga_cfg: dict cac key CFG can patch (POPULATION_SIZE, GENERATIONS, ...)

    Returns: result dict (ok=True/False, + metadata)
    """
    sid, task, data_dir, output_dir, save_files, csv_enc, ga_cfg, theta_lookup = args_tuple

    # Patch CFG — each subprocess re-imports _p2 from scratch
    for k, v in ga_cfg.items():
        CFG[k] = v

    try:
        raw_path = os.path.join(data_dir, f"Subject_{sid}_{task}_raw.csv")
        fix_path = os.path.join(data_dir, f"Subject_{sid}_{task}_fixations.csv")

        if not os.path.exists(raw_path) or not os.path.exists(fix_path):
            return {
                "ok": False, "sid": sid, "task": task,
                "error": f"file_not_found: {raw_path}",
            }

        # --- Load ---
        raw_df = load_raw_gaze(raw_path)
        fix_df = load_fixations(fix_path)
        fix_df = (fix_df
                  .loc[fix_df["duration_ms"] >= CFG["INPUT_MIN_FIX_DUR_MS"]]
                  .reset_index(drop=True))

        if len(fix_df) == 0:
            return {
                "ok": False, "sid": sid, "task": task,
                "error": "no_valid_fixations_after_filter",
            }

        # --- Sampling rate ---
        sr = detect_sampling_rate(raw_df)

        # --- Baseline (engine: velocity-PSD Hurst, local-window sigma) ---
        baseline = estimate_baseline(raw_df, fix_df, sr)

        # --- Seed deterministc per subject-task ---
        rng_seed = stable_hash_int(
            f"{sid}_{task}_{CFG['GLOBAL_SEED']}", mod=10_000_000
        )

        # Subsample fixations cho fitness (neu nhieu qua)
        if (CFG.get("MAX_FIXATIONS_FOR_FITNESS") is not None
                and len(fix_df) > CFG["MAX_FIXATIONS_FOR_FITNESS"]):
            fitness_fix = fix_df.sample(
                n=int(CFG["MAX_FIXATIONS_FOR_FITNESS"]),
                random_state=rng_seed,
            ).reset_index(drop=True)
        else:
            fitness_fix = fix_df.copy()

        # --- Lay theta tu CSV (neu co) hoac toi uu online ---
        theta_key = (str(sid), task)
        if theta_lookup is not None:
            if theta_key not in theta_lookup:
                return {
                    "ok": False,
                    "sid": sid,
                    "task": task,
                    "error": f"missing_theta_for_subject_task: {sid}-{task}",
                }
            th = theta_lookup[theta_key]
            best_H = float(th["H"])
            best_sigma = float(th["sigma"])
            best_phi = float(th["phi"])
            best_seed = int(th["seed_best"]) if th.get("seed_best") is not None else int(rng_seed)
            best_fitness = float(th.get("best_fitness", np.nan))
            best_D = float(th.get("best_D", np.nan))
            best_O = float(th.get("best_O", np.nan))
            best_E = float(th.get("best_E", np.nan))
            n_generations_run = 0
            theta_source = "theta_csv"
        else:
            # --- GA (engine: F = w1*D - w2*O - w3*(E_psd+E_disp)) ---
            best, history = run_ga(fitness_fix, baseline, rng_seed)
            best_H = float(best.H)
            best_sigma = float(best.sigma)
            best_phi = float(best.phi)
            best_seed = int(best.seed)
            best_fitness = float(best.fitness)
            best_D = float(best.D)
            best_O = float(best.O)
            best_E = float(best.E)
            n_generations_run = int(len(history))
            theta_source = "optimized_online"

        # --- Generate synthetic over the FULL fixation set ---
        dt_ms      = 1000.0 / sr
        all_chunks = []
        n_fix_passed = 0
        n_fix_total = 0

        for _, f in fix_df.iterrows():
            fix_id      = int(f["id"])
            fix_x       = float(f["fix_x"])
            fix_y       = float(f["fix_y"])
            duration_ms = float(f["duration_ms"])
            start_ms    = float(f["start_ms"])

            n_pts = max(2, int(round(duration_ms * sr / 1000.0)))
            dx, dy = generate_displacement(
                n_points=n_pts,
                H=best_H,
                sigma=best_sigma,
                phi=best_phi,
                sampling_rate=sr,
                seed=combine_seed(best_seed, fix_id),
                H_drift=baseline.H_base,
            )
            x_syn = fix_x + dx
            y_syn = fix_y + dy
            t_ms_arr = np.arange(n_pts, dtype=float) * dt_ms

            chk = peyemmv_check(x_syn, y_syn, duration_ms)
            passed = bool(chk["detected"])
            n_fix_total += 1
            if passed:
                n_fix_passed += 1

            chunk = pd.DataFrame({
                "fixation_id":      fix_id,
                "point_index":      np.arange(n_pts, dtype=np.int32),
                "t_ms":             t_ms_arr,
                "t_abs":            start_ms + t_ms_arr,
                "x":                x_syn,
                "y":                y_syn,
                "x_fix":            fix_x,
                "y_fix":            fix_y,
                "peyemmv_passed":   passed,
                "H_used":           best_H,
                "sigma_used":       best_sigma,
                "phi_used":         best_phi,
                "H_base_used":      baseline.H_base,
                "sampling_rate_hz": sr,
                "dt_ms":            dt_ms,
                "duration_ms":      duration_ms,
            })
            all_chunks.append(chunk)

        if not all_chunks:
            return {
                "ok": False, "sid": sid, "task": task,
                "error": "no_fixation_chunks_generated",
            }

        syn_df = pd.concat(all_chunks, ignore_index=True)
        syn_df.insert(0, "subject_id", sid)
        syn_df.insert(1, "task", task)
        syn_df.insert(2, "method", "GA")

        output_path = None
        if save_files:
            os.makedirs(output_dir, exist_ok=True)
            out_name    = f"GA2_subject_{sid}_task_{task}_Subject_{sid}_{task}_fixations.csv"
            output_path = os.path.join(output_dir, out_name)
            syn_df.to_csv(output_path, index=False, encoding=csv_enc)

        acr_pct = (100.0 * n_fix_passed / n_fix_total) if n_fix_total > 0 else 0.0

        print(
            f"  [{sid}-{task}] OK | source={theta_source} | "
            f"H={best_H:.3f}  sigma={best_sigma:.3f}  "
            f"phi={best_phi:.3f}  F={best_fitness:.4f}  "
            f"D={best_D:.3f}  O={best_O:.3f}  E={best_E:.3f}  "
            f"ACR={acr_pct:.1f}%  n_fix={len(fix_df)}"
        )

        return {
            "ok":                 True,
            "sid":                sid,
            "task":               task,
            "theta_source":       theta_source,
            "output_path":        output_path,
            "best_H":             best_H,
            "best_sigma":         best_sigma,
            "best_phi":           best_phi,
            "best_seed":          best_seed,
            "best_fitness":       best_fitness,
            "best_D":             best_D,
            "best_O":             best_O,
            "best_E":             best_E,
            "H_base":             float(baseline.H_base),
            "sigma_base":         float(baseline.sigma_base),
            "target_psd_slope":   float(baseline.target_psd_slope),
            "sampling_rate_hz":   float(sr),
            "n_fixations":        int(len(fix_df)),
            "n_fixations_peyemmv_passed": int(n_fix_passed),
            "ACR_pct":            float(acr_pct),
            "n_fix_for_fitness":  int(len(fitness_fix)),
            "n_synthetic_samples": int(len(syn_df)),
            "n_generations_run":  n_generations_run,
        }

    except Exception as e:
        import traceback
        msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"  [{sid}-{task}] ERROR: {msg[:200]}")
        return {"ok": False, "sid": sid, "task": task, "error": msg}


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="ga2_generate.py — GA2 (engine engine) synthetic gaze",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--data_dir",    default=CONFIG["DATA_DIR"],
        help="ETDD70 data directory with raw+fixation files (default: ./data)")
    parser.add_argument("--output_dir",  default=CONFIG["OUTPUT_DIR"],
        help="Output directory for GA2_*.csv files (default: ./ga2_output)")
    parser.add_argument("--theta_csv", default=None,
        help=(
            "(Tuy chon) Duong dan ga2_theta_star.csv de generate theo kieu GA.\n"
            "Neu bo trong, script tu dong tim file theta o cac vi tri pho bien."
        ))
    parser.add_argument("--max_workers", type=int, default=1,
        help=(
            "Number of parallel worker processes (default: 1).\n"
            "GA is heavier than baseline due to E_disp; recommended <=4."
        ))
    # GA hyperparameters
    parser.add_argument("--pop",  type=int,   default=DEFAULT_GA["POPULATION_SIZE"],
        help=f"POPULATION_SIZE (default: {DEFAULT_GA['POPULATION_SIZE']})")
    parser.add_argument("--gens", type=int,   default=DEFAULT_GA["GENERATIONS"],
        help=f"GENERATIONS     (default: {DEFAULT_GA['GENERATIONS']})")
    parser.add_argument("--cr",   type=float, default=DEFAULT_GA["CROSSOVER_RATE"],
        help=f"CROSSOVER_RATE  (default: {DEFAULT_GA['CROSSOVER_RATE']})")
    parser.add_argument("--mr",   type=float, default=DEFAULT_GA["MUTATION_RATE"],
        help=f"MUTATION_RATE   (default: {DEFAULT_GA['MUTATION_RATE']})")
    parser.add_argument("--w1",   type=float, default=DEFAULT_GA["W1_D"],
        help=f"W1_D  weight Detection     (default: {DEFAULT_GA['W1_D']})")
    parser.add_argument("--w2",   type=float, default=DEFAULT_GA["W2_O"],
        help=f"W2_O  weight Outlier       (default: {DEFAULT_GA['W2_O']})")
    parser.add_argument("--w3",   type=float, default=DEFAULT_GA["W3_E"],
        help=f"W3_E  weight PSD+Disp      (default: {DEFAULT_GA['W3_E']})")
    parser.add_argument("--no_save", action="store_true",
        help="Skip CSV output, save summary JSON only")
    args = parser.parse_args()

    # Xay dung ga_cfg (se truyen cho moi worker)
    ga_cfg = {
        "ETDD70_DIR":      args.data_dir,
        "OUTPUT_DIR":      args.output_dir,
        "POPULATION_SIZE": args.pop,
        "GENERATIONS":     args.gens,
        "CROSSOVER_RATE":  args.cr,
        "MUTATION_RATE":   args.mr,
        "W1_D":            args.w1,
        "W2_O":            args.w2,
        "W3_E":            args.w3,
    }

    # Patch CFG trong process chinh (dung cho max_workers=1)
    for k, v in ga_cfg.items():
        CFG[k] = v

    # Load danh sach subjects
    label_path = os.path.join(args.data_dir, "dyslexia_class_label.csv")
    label_df   = pd.read_csv(label_path)
    subjects   = [str(s) for s in label_df["subject_id"].tolist()]

    tasks_list = [(sid, task) for sid in subjects for task in TASKS]
    save_files = not args.no_save
    csv_enc    = CONFIG["CSV_ENCODING"]
    theta_lookup = None
    theta_csv_path = resolve_theta_csv_path(args.theta_csv, args.output_dir)
    if theta_csv_path is not None:
        theta_lookup = load_ga2_theta_csv(theta_csv_path)

    print("=" * 80)
    print("GA2 Generate — engine.py engine")
    print("=" * 80)
    print(f"Data dir     : {args.data_dir}")
    print(f"Output dir   : {args.output_dir}")
    print(f"Subjects     : {len(subjects)}")
    print(f"Tasks        : {TASKS}")
    print(f"Total jobs   : {len(tasks_list)}")
    print(f"Pop / Gens   : {args.pop} / {args.gens}")
    print(f"CR / MR      : {args.cr} / {args.mr}")
    print(f"W1/W2/W3     : {args.w1} / {args.w2} / {args.w3}")
    print(f"max_workers  : {args.max_workers}")
    print(f"Save files   : {save_files}")
    print(f"Theta mode   : {'theta_csv' if theta_lookup is not None else 'optimize_online'}")
    if theta_lookup is not None:
        print(f"Theta CSV    : {theta_csv_path}")
    print("-" * 80)
    print("Khac biet GA2 vs GA:")
    print("  Fitness : F = w1*D - w2*O - w3*(E_psd + E_disp)")
    print("  H estim : velocity PSD slope  H = (1-slope)/2  (fGn)")
    print("  Drift   : speed ~ Uniform[4.35, 86.92] px/s per fixation")
    print("=" * 80)

    args_list = [
        (sid, task, args.data_dir, args.output_dir, save_files, csv_enc, ga_cfg, theta_lookup)
        for sid, task in tasks_list
    ]

    results = []
    if args.max_workers > 1:
        ctx = concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers)
        with ctx as ex:
            for r in tqdm(
                ex.map(worker_ga2, args_list),
                total=len(args_list),
                desc="GA2",
            ):
                results.append(r)
    else:
        for a in tqdm(args_list, desc="GA2"):
            results.append(worker_ga2(a))

    n_ok    = sum(1 for r in results if r.get("ok"))
    n_error = sum(1 for r in results if not r.get("ok"))

    # --- Summary ---
    os.makedirs(args.output_dir, exist_ok=True)
    _desc_mode = (
        f"theta_csv mode (no GA re-run): {theta_csv_path}"
        if theta_lookup is not None
        else "optimize_online: GA runs per subject-task"
    )
    summary = {
        "method":      "GA",
        "engine":      "engine.py",
        "description": (
            "fBm displacement (velocity-PSD Hurst, drift+tremor). "
            "Fitness F = w1*D - w2*O - w3*(E_psd+E_disp). "
            f"{_desc_mode}."
        ),
        "ga_config": {
            "POPULATION_SIZE": args.pop,
            "GENERATIONS":     args.gens,
            "CROSSOVER_RATE":  args.cr,
            "MUTATION_RATE":   args.mr,
            "W1_D": args.w1, "W2_O": args.w2, "W3_E": args.w3,
        },
        "n_tasks":   len(tasks_list),
        "n_ok":      n_ok,
        "n_error":   n_error,
        "results":   results,
    }
    summary_path = os.path.join(args.output_dir, CONFIG["SUMMARY_FILENAME"])
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print("=" * 80)
    print(f"DONE  OK={n_ok}  Error={n_error}")
    print(f"Summary: {summary_path}")
    if n_error > 0:
        errors = [(r["sid"], r["task"], r.get("error", "")[:80])
                  for r in results if not r.get("ok")]
        print(f"Errors ({n_error}):")
        for sid, task, err in errors[:10]:
            print(f"  [{sid}-{task}] {err}")
    print("=" * 80)
    print("Next steps:")
    print(f"  python compare_generators.py --ga2_dir {args.output_dir}")


if __name__ == "__main__":
    main()
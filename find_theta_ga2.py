"""
find_theta_ga2.py — Find optimal parameters θ* for GA2 (engine: engine.py).

Objective:
    Run GA2 optimisation per subject-task to obtain optimal parameters
    (H, sigma, phi, seed, fitness, D, O, E).
    Does not generate synthetic gaze; only saves the parameter table.

Input:
    data/Subject_{sid}_{task}_raw.csv
    data/Subject_{sid}_{task}_fixations.csv
    data/dyslexia_class_label.csv

Output:
    <output_root>/ga2_theta_star.csv
    <output_root>/ga2_theta_summary.json

Usage:
    python find_theta_ga2.py --output_root ./syn_output
    python find_theta_ga2.py --pop 100 --gens 50 --w1 8 --w2 2 --w3 10 --max_workers 4
"""
import os
import sys
import json
import argparse
import warnings
import importlib.util
import concurrent.futures
import multiprocessing
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")


# ============================================================
# IMPORT from engine.py
# ============================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
_P2_FILE = os.path.join(_HERE, "engine.py")
_spec = importlib.util.spec_from_file_location("_p2", _P2_FILE)
_p2 = importlib.util.module_from_spec(_spec)
sys.modules["_p2"] = _p2
_spec.loader.exec_module(_p2)

load_raw_gaze = _p2.load_raw_gaze
load_fixations = _p2.load_fixations
detect_sampling_rate = _p2.detect_sampling_rate
estimate_baseline = _p2.estimate_baseline
run_ga = _p2.run_ga
stable_hash_int = _p2.stable_hash_int
CFG = _p2.CFG


# ============================================================
# CONSTANTS
# ============================================================
DATA_ROOT = "./data"
TASKS = ["T4_Meaningful_Text", "T5_Pseudo_Text"]

DEFAULT_GA = {
    "POPULATION_SIZE": 100,
    "GENERATIONS": 50,
    "CROSSOVER_RATE": 0.80,
    "MUTATION_RATE": 0.05,
    "W1_D": 8.0,
    "W2_O": 2.0,
    "W3_E": 10.0,
}


# ============================================================
# WORKER
# ============================================================
def worker_theta_ga2(args_tuple):
    sid, task, data_dir, ga_cfg, label_map = args_tuple

    for k, v in ga_cfg.items():
        CFG[k] = v
    # Override CFG per group: dyslexic vs non-dyslexic
    is_dyslexic = label_map.get(str(sid), 0) == 1
    if is_dyslexic:
        CFG["H_DELTA"]  = 0.12          # wider search range
        CFG["W1_D"]     = 7.0           # lighter detection penalty
        CFG["W2_O"]     = 1.5           # lighter outlier penalty
        # sigma upper bound multiplier, read by initialize_population
        CFG["_SIGMA_HIGH_MULT"] = 4.0
        CFG["_H_CENTER_OFFSET"] = -0.05 # shift H search centre downward
    else:
        CFG["H_DELTA"]  = 0.08          # snarrower search range, stable
        CFG["W1_D"]     = 8.0
        CFG["W2_O"]     = 3.0
        CFG["_SIGMA_HIGH_MULT"] = 2.0
        CFG["_H_CENTER_OFFSET"] = 0.0
    # W3_E unchanged for both groups

    try:
        raw_path = os.path.join(data_dir, f"Subject_{sid}_{task}_raw.csv")
        fix_path = os.path.join(data_dir, f"Subject_{sid}_{task}_fixations.csv")

        if not os.path.exists(raw_path) or not os.path.exists(fix_path):
            return sid, task, None, f"file_not_found: {raw_path}"

        raw_df = load_raw_gaze(raw_path)
        fix_df = load_fixations(fix_path)
        fix_df = (
            fix_df.loc[fix_df["duration_ms"] >= CFG["INPUT_MIN_FIX_DUR_MS"]]
            .reset_index(drop=True)
        )

        if len(fix_df) == 0:
            return sid, task, None, "no_valid_fixations_after_filter"

        sr = detect_sampling_rate(raw_df)
        baseline = estimate_baseline(raw_df, fix_df, sr)

        rng_seed = stable_hash_int(
            f"{sid}_{task}_{CFG['GLOBAL_SEED']}", mod=10_000_000
        )

        if (
            CFG.get("MAX_FIXATIONS_FOR_FITNESS") is not None
            and len(fix_df) > CFG["MAX_FIXATIONS_FOR_FITNESS"]
        ):
            fitness_fix = fix_df.sample(
                n=int(CFG["MAX_FIXATIONS_FOR_FITNESS"]),
                random_state=rng_seed,
            ).reset_index(drop=True)
        else:
            fitness_fix = fix_df.copy()

        best, history = run_ga(fitness_fix, baseline, rng_seed)

        print(
            f"  [{sid}-{task}] theta*: H={best.H:.3f}  sigma={best.sigma:.3f}  "
            f"phi={best.phi:.3f}  seed={best.seed}  F={best.fitness:.4f}  "
            f"D={best.D:.3f}  O={best.O:.3f}  E={best.E:.3f}"
        )

        rec = {
            "sid": sid,
            "task": task,
            "H": float(best.H),
            "sigma": float(best.sigma),
            "phi": float(best.phi),
            "seed_best": int(best.seed),
            "best_fitness": float(best.fitness),
            "best_D": float(best.D),
            "best_O": float(best.O),
            "best_E": float(best.E),
            "H_base": float(baseline.H_base),
            "sigma_base": float(baseline.sigma_base),
            "target_psd_slope": float(baseline.target_psd_slope),
            "sampling_rate_hz": float(sr),
            "n_fixations": int(len(fix_df)),
            "n_fix_for_fitness": int(len(fitness_fix)),
            "n_generations_run": int(len(history)),
        }
        return sid, task, rec, "OK"

    except Exception as e:
        import traceback
        return sid, task, None, f"ERROR: {e}\n{traceback.format_exc()}"


# ============================================================
# CHECKPOINT HELPERS
# ============================================================
def load_checkpoint(cp_file):
    if os.path.exists(cp_file):
        with open(cp_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_checkpoint(cp_file, records_dict):
    with open(cp_file, "w", encoding="utf-8") as f:
        json.dump(records_dict, f, indent=2, ensure_ascii=False)


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="find_theta_ga2.py - Find optimal theta* parameters for GA2",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--data_dir", default=DATA_ROOT, help="Data directory (default: ./data)")
    parser.add_argument(
        "--output_root",
        default="./syn_output",
        help="Output directory for ga2_theta_star.csv (default: ./syn_output)",
    )
    parser.add_argument(
        "--output_name",
        default="ga2_theta_star.csv",
        help="Output CSV filename for theta (default: ga2_theta_star.csv)",
    )
    parser.add_argument("--max_workers", type=int, default=4, help="Number of parallel worker processes")

    parser.add_argument("--pop", type=int, default=DEFAULT_GA["POPULATION_SIZE"])
    parser.add_argument("--gens", type=int, default=DEFAULT_GA["GENERATIONS"])
    parser.add_argument("--cr", type=float, default=DEFAULT_GA["CROSSOVER_RATE"])
    parser.add_argument("--mr", type=float, default=DEFAULT_GA["MUTATION_RATE"])
    parser.add_argument("--w1", type=float, default=DEFAULT_GA["W1_D"])
    parser.add_argument("--w2", type=float, default=DEFAULT_GA["W2_O"])
    parser.add_argument("--w3", type=float, default=DEFAULT_GA["W3_E"])

    args = parser.parse_args()

    ga_cfg = {
        "ETDD70_DIR": args.data_dir,
        "POPULATION_SIZE": args.pop,
        "GENERATIONS": args.gens,
        "CROSSOVER_RATE": args.cr,
        "MUTATION_RATE": args.mr,
        "W1_D": args.w1,
        "W2_O": args.w2,
        "W3_E": args.w3,
    }

    for k, v in ga_cfg.items():
        CFG[k] = v

    min_pop = CFG.get("TOURNAMENT_SIZE", 3)
    if ga_cfg["POPULATION_SIZE"] < min_pop:
        print(
            f"[WARN] POPULATION_SIZE={ga_cfg['POPULATION_SIZE']} < TOURNAMENT_SIZE={min_pop},"
            f" tang len {min_pop}"
        )
        ga_cfg["POPULATION_SIZE"] = min_pop

    label_path = os.path.join(args.data_dir, "dyslexia_class_label.csv")
    label_df = pd.read_csv(label_path)
    subject_ids = [str(s) for s in label_df["subject_id"].tolist()]

    # Build label map: sid -> class_id (0=non-dyslexic, 1=dyslexic)
    label_map = {str(row["subject_id"]): int(row["class_id"]) for _, row in label_df.iterrows()}

    jobs = [(sid, task, args.data_dir, ga_cfg, label_map) for sid in subject_ids for task in TASKS]
    total = len(jobs)

    print("=" * 80)
    print("Find theta GA2")
    print("=" * 80)
    print(f"Data dir     : {args.data_dir}")
    print(f"Output root  : {args.output_root}")
    print(f"Subjects     : {len(subject_ids)}")
    print(f"Tasks        : {TASKS}")
    print(f"Total jobs   : {total}")
    print(f"Pop / Gens   : {args.pop} / {args.gens}")
    print(f"CR / MR      : {args.cr} / {args.mr}")
    print(f"W1/W2/W3     : {args.w1} / {args.w2} / {args.w3}")
    print(f"max_workers  : {args.max_workers}")
    print("=" * 80)

    checkpoint_file = os.path.join(args.output_root, "ga2_theta_checkpoint.json")
    os.makedirs(args.output_root, exist_ok=True)

    # --- Load checkpoint ---
    checkpoint = load_checkpoint(checkpoint_file)
    records = list(checkpoint.values())
    done = len(records)

    # --- Filter incomplete jobs ---
    jobs_to_run = [j for j in jobs if f"{j[0]}-{j[1]}" not in checkpoint]
    total_to_run = len(jobs_to_run)
    print(f"[INFO] Jobs remaining: {total_to_run}/{total}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(worker_theta_ga2, j) for j in jobs_to_run]
        for fut in concurrent.futures.as_completed(futures):
            sid, task, rec, status = fut.result()
            done += 1
            if status == "OK":
                records.append(rec)
                checkpoint[f"{sid}-{task}"] = rec
                # --- Save checkpoint ---
                save_checkpoint(checkpoint_file, checkpoint)
                print(f"  [{done}/{total}] OK   {sid}-{task}")
            else:
                print(f"  [{done}/{total}] FAIL {sid}-{task}: {status[:200]}")

    # --- Export final CSV + JSON ---
    out_csv = os.path.join(args.output_root, args.output_name)
    out_json = os.path.join(args.output_root, "ga2_theta_summary.json")

    if records:
        df = pd.DataFrame(records)
        df.to_csv(out_csv, index=False)

        summary = {
            "method": "GA2",
            "engine": "engine.py",
            "description": "Theta finder only (no synthetic generation).",
            "ga_config": ga_cfg,
            "n_tasks": total,
            "n_ok": int(len(records)),
            "n_error": int(total - len(records)),
            "theta_csv": out_csv,
            "results": records,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        print("=" * 80)
        print(f"DONE  OK={len(records)}  Error={total - len(records)}")
        print(f"Theta CSV : {out_csv}")
        print(f"Summary   : {out_json}")
        print("=" * 80)
    else:
        print("[WARN] No theta records were saved.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
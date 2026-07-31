"""
Example: Find the optimal theta* for GA2 on a sample subject-task.

Runs in seconds - only illustrates how to call the module, not the paper's result.
Run: python examples/example_01_find_theta.py
"""
from peyemmv_ga.engine import (
    load_raw_gaze, load_fixations, detect_sampling_rate,
    estimate_baseline, run_ga, stable_hash_int, CFG,
)

DATA_DIR = "examples/sample_data"
SID, TASK = "1003", "T4_Meaningful_Text"

CFG["POPULATION_SIZE"] = 20
CFG["GENERATIONS"] = 5

raw_df = load_raw_gaze(f"{DATA_DIR}/Subject_{SID}_{TASK}_raw.csv")
fix_df = load_fixations(f"{DATA_DIR}/Subject_{SID}_{TASK}_fixations.csv")
sr = detect_sampling_rate(raw_df)
baseline = estimate_baseline(raw_df, fix_df, sr)
seed = stable_hash_int(f"{SID}_{TASK}_{CFG['GLOBAL_SEED']}", mod=10_000_000)
best, history = run_ga(fix_df, baseline, seed)

print(f"theta* Sample: H={best.H:.3f} sigma={best.sigma:.3f} phi={best.phi:.3f} fitness={best.fitness:.4f}")
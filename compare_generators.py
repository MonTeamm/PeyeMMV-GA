"""
compare_generators.py — Compare GA2 vs DCM vs SGG across the full ETDD70 dataset.

Reads SYNTHETIC data from pre-generated files:
    ga2_dir/  GA2_subject_{sid}_task_{task}_*.csv
    dcm_dir/  DCM_subject_{sid}_task_{task}_*.csv
    sgg_dir/  SGG_subject_{sid}_task_{task}_*.csv

Reads REAL data from:
    data_dir/ Subject_{sid}_{task}_raw.csv        -> omega_real (trong cua so fixation thuc)
    data_dir/ Subject_{sid}_{task}_fixations.csv  -> fixation tham chieu

Metrics computed per subject-task-method:
    ACR   : % synthetic fixations passing PeyeMMV (T1=21.73px, T2=6.52px, min_dur=100ms)
    CLE   : centroid error (px) — distance from synthetic centroid to real fixation centroid
    VPC   : mean step length (px/frame) — proxy velocity
    JSD   : Jensen-Shannon Divergence omega_syn vs omega_real (from _p1, [0,1])
    alpha : PSD slope of omega_syn (−slope, pink noise ~ 1–2)

Output:
    <output_root>/comparison_results.csv   (420 rows: 70 subs x 2 tasks x 3 methods)
    <output_root>/summary_by_method.csv    (mean+-std per method)
    <output_root>/anova_summary.csv        (F, p, eta2 per metric)
    <output_root>/tukey_hsd.csv            (post-hoc pairs)
    <output_root>/boxplot_<metric>.png     (5 metrics x 2 tasks — consistent colour scheme)
    <output_root>/fig1_spatial_scatter.png (Figure 1 — 2D spatial distribution)
    <output_root>/fig2a_loglog_psd.png     (Figure 2a — Log-Log PSD: Real vs GA vs SGG)
    <output_root>/fig2b_velocity_kde.png   (Figure 2b — Velocity KDE: Real vs GA)

Usage:
    python compare_generators.py \
        --ga2_dir ./ga2_output \
        --dcm_dir ./dcm_output \
        --sgg_dir ./sgg_output \
        --data_dir ./data \
        --output_root ./phase1_results \
        --max_workers 4
"""
import os
import sys
import argparse
import warnings
import glob
import re
import importlib.util
import concurrent.futures

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import f_oneway, entropy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

# ============================================================================
# IMPORT FUNCTIONS FROM engine.py (identical logic)
# ============================================================================
_HERE    = os.path.dirname(os.path.abspath(__file__))
_P1_FILE = os.path.join(_HERE, "engine.py")
spec = importlib.util.spec_from_file_location("_p1", _P1_FILE)
_p1  = importlib.util.module_from_spec(spec)
sys.modules["_p1"] = _p1   # required for Python 3.13 dataclass resolution
spec.loader.exec_module(_p1)

load_raw_gaze  = _p1.load_raw_gaze
load_fixations = _p1.load_fixations
detect_sampling_rate = _p1.detect_sampling_rate
peyemmv_check = _p1.peyemmv_check
_angular_velocity_from_xy_time = _p1._angular_velocity_from_xy_time
_histogram_distribution = _p1._histogram_distribution
_jensen_shannon_divergence_normalized = _p1._jensen_shannon_divergence_normalized
CFG = _p1.CFG

TASKS   = ["T4_Meaningful_Text", "T5_Pseudo_Text"]
METRICS = ["ACR", "CLE", "VPC", "JSD", "alpha"]


# ============================================================================
# BUILD OMEGA_REAL DISTRIBUTION (identical to _build_target in test_tham_so)
# ============================================================================
def build_omega_real(raw_df: pd.DataFrame, fix_df: pd.DataFrame):
    """
    Compute omega_real distribution from raw gaze INSIDE each fixation window.
    Logic identical to _build_target_angular_velocity_distribution().
    Returns: (hist, bin_edges, omega_max)
    """
    times = raw_df["time_ms"].values
    xs    = raw_df["x"].values
    ys    = raw_df["y"].values
    omega_values = []

    for _, f in fix_df.iterrows():
        left  = np.searchsorted(times, float(f["start_ms"]), side="left")
        right = np.searchsorted(times, float(f["end_ms"]),   side="right")
        if right - left < 4:
            continue
        omega = _angular_velocity_from_xy_time(
            xs[left:right], ys[left:right], t_ms=times[left:right]
        )
        if len(omega):
            omega_values.extend(omega.tolist())

    omega_arr = np.asarray(omega_values, dtype=float)
    omega_arr = omega_arr[np.isfinite(omega_arr) & (omega_arr >= 0)]

    if len(omega_arr) == 0:
        omega_max = 1.0
        bin_edges = np.linspace(0.0, omega_max, CFG["OMEGA_N_BINS"] + 1)
        hist      = np.ones(CFG["OMEGA_N_BINS"], dtype=float) / CFG["OMEGA_N_BINS"]
        return hist, bin_edges, omega_max

    omega_max = float(np.percentile(omega_arr, CFG["OMEGA_MAX_PERCENTILE"]))
    omega_max = max(omega_max, float(np.max(omega_arr)), CFG["OMEGA_EPS"])
    bin_edges = np.linspace(0.0, omega_max, CFG["OMEGA_N_BINS"] + 1)
    hist      = _histogram_distribution(np.clip(omega_arr, 0.0, omega_max), bin_edges)
    return hist, bin_edges, omega_max


# ============================================================================
# WORKER: EVALUATE ONE SYNTHETIC FILE
# ============================================================================
def _eval_synthetic_file(args_tuple):
    """
    Worker process: read one synthetic file and compute ACR/CLE/VPC/JSD/alpha.

    Input:
        syn_path  : path to synthetic file (GA2/DCM/SGG)
        data_dir  : directory containing real raw + fixation files
        label_map : {int(sid): label}

    Returns: dict row or None on error.
    """
    syn_path, data_dir, label_map = args_tuple
    try:
        # --- Read synthetic file ---
        syn_df = pd.read_csv(syn_path)
        if syn_df.empty:
            return None, f"empty file: {syn_path}"

        sid    = str(syn_df["subject_id"].iloc[0])
        task   = str(syn_df["task"].iloc[0])
        method = str(syn_df["method"].iloc[0])

        # --- Load real data ---
        raw_path = os.path.join(data_dir, f"Subject_{sid}_{task}_raw.csv")
        fix_path = os.path.join(data_dir, f"Subject_{sid}_{task}_fixations.csv")
        if not os.path.exists(raw_path) or not os.path.exists(fix_path):
            return None, f"real data missing for {sid}-{task}"

        raw_df = load_raw_gaze(raw_path)
        fix_df = load_fixations(fix_path)
        sr     = detect_sampling_rate(raw_df)

        # --- Omega_real (from raw INSIDE real fixation windows) ---
        omega_real_hist, bin_edges, omega_max = build_omega_real(raw_df, fix_df)

        # --- Iterate over fixation clusters in synthetic file ---
        if "fixation_id" not in syn_df.columns:
            return None, f"no fixation_id col in {syn_path}"

        acr_list, cle_list, vpc_list, omega_syn_all = [], [], [], []

        for fix_id, cluster in syn_df.groupby("fixation_id"):
            x     = cluster["x"].values.astype(float)
            y     = cluster["y"].values.astype(float)
            x_fix = float(cluster["x_fix"].iloc[0])
            y_fix = float(cluster["y_fix"].iloc[0])
            dur   = float(cluster["duration_ms"].iloc[0])

            # ACR: peyemmv_check returns dict (identical to test_tham_so)
            chk = peyemmv_check(x, y, dur)
            acr_list.append(float(chk["detected"]))

            # CLE: distance from synthetic centroid to real fixation centroid
            cle_list.append(float(np.sqrt((np.mean(x) - x_fix)**2 +
                                          (np.mean(y) - y_fix)**2)))

            # VPC: mean step length (proxy velocity)
            dists = np.hypot(np.diff(x), np.diff(y))
            vpc_list.append(float(dists.mean()) if len(dists) > 0 else 0.0)

            # Omega_syn: use sampling_rate (no t_ms in synthetic)
            om = _angular_velocity_from_xy_time(x, y, sampling_rate=sr)
            if len(om):
                omega_syn_all.extend(om.tolist())

        if len(acr_list) == 0:
            return None, f"no fixation clusters in {syn_path}"

        ACR = float(np.mean(acr_list)) * 100.0
        CLE = float(np.mean(cle_list))
        VPC = float(np.mean(vpc_list))

        # JSD: compare omega_syn vs omega_real (same bin_edges)
        omega_syn_arr = np.asarray(omega_syn_all, dtype=float)
        omega_syn_arr = omega_syn_arr[np.isfinite(omega_syn_arr) & (omega_syn_arr >= 0)]
        if len(omega_syn_arr) >= 5:
            q_syn = _histogram_distribution(
                np.clip(omega_syn_arr, 0.0, omega_max), bin_edges
            )
            JSD = float(_jensen_shannon_divergence_normalized(omega_real_hist, q_syn))
        else:
            JSD = 1.0

        # Alpha: PSD slope of omega_syn (−slope)
        alpha = float("nan")
        if len(omega_syn_all) >= 16:
            nperseg = min(len(omega_syn_all), 128)
            freqs, psd = welch(np.asarray(omega_syn_all), fs=sr, nperseg=nperseg)
            nz = freqs > 0
            if nz.sum() >= 4:
                slope, _ = np.polyfit(
                    np.log(freqs[nz]), np.log(psd[nz] + 1e-20), 1
                )
                alpha = float(-slope)

        label_val = label_map.get(int(sid), "unknown")
        return {
            "sid":    sid,
            "task":   task,
            "method": method,
            "ACR":    ACR,
            "CLE":    CLE,
            "VPC":    VPC,
            "JSD":    JSD,
            "alpha":  alpha,
            "label":  label_val,
        }, "OK"

    except Exception as e:
        import traceback
        return None, f"ERROR {syn_path}: {e}\n{traceback.format_exc()}"


# ============================================================================
# STATISTICS: ANOVA + TUKEY + ETA2
# ============================================================================
def run_statistics(df: pd.DataFrame, output_root: str):
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        HAS_TUKEY = True
    except ImportError:
        print("[WARN] statsmodels not installed -> skipping Tukey HSD. pip install statsmodels")
        HAS_TUKEY = False

    # Methods taken from actual data (GA2, DCM, SGG, ...)
    methods = sorted(df["method"].dropna().unique().tolist())
    anova_rows = []
    tukey_rows = []

    for task in TASKS:
        sub = df[df["task"] == task]
        for metric in METRICS:
            groups = [sub[sub["method"] == m][metric].dropna().values for m in methods]
            if any(len(g) < 3 for g in groups):
                continue
            F, p = f_oneway(*groups)
            all_vals = np.concatenate(groups)
            gm   = all_vals.mean()
            ss_b = sum(len(g) * (g.mean() - gm)**2 for g in groups)
            ss_t = np.sum((all_vals - gm)**2)
            eta2 = float(ss_b / ss_t) if ss_t > 0 else 0.0
            sig  = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            print(f"  ANOVA [{task[:4]}][{metric:5s}]: F={F:7.2f}  p={p:.4f}  eta2={eta2:.3f}  {sig}")
            anova_rows.append({
                "task": task, "metric": metric,
                "F": round(F, 4), "p": round(p, 6), "eta2": round(eta2, 4),
                "significant": p < 0.05,
            })
            if HAS_TUKEY:
                vals   = np.concatenate(groups)
                labels = sum([[m]*len(g) for m, g in zip(methods, groups)], [])
                try:
                    tukey = pairwise_tukeyhsd(vals, labels)
                    for row in tukey.summary().data[1:]:
                        g1, g2, mn_diff, _, lo, hi, reject = row
                        tukey_rows.append({
                            "task": task, "metric": metric,
                            "group1": g1, "group2": g2,
                            "mean_diff": round(float(mn_diff), 4),
                            "ci_low":    round(float(lo), 4),
                            "ci_high":   round(float(hi), 4),
                            "reject_H0": bool(reject),
                        })
                except Exception as ex:
                    print(f"    [WARN] Tukey failed: {ex}")

    os.makedirs(output_root, exist_ok=True)
    pd.DataFrame(anova_rows).to_csv(
        os.path.join(output_root, "anova_summary.csv"), index=False)
    if tukey_rows:
        pd.DataFrame(tukey_rows).to_csv(
            os.path.join(output_root, "tukey_hsd.csv"), index=False)
    print(f"  -> anova_summary.csv{'  tukey_hsd.csv' if tukey_rows else ''} saved to {output_root}")


# ============================================================================
# COLOR PALETTE — consistent throughout the paper (GA=green, SGG=red, DCM=grey)
# ============================================================================
METHOD_COLORS = {
    "GA2": "#2ca02c",   # green
    "GA":  "#2ca02c",
    "SGG": "#d62728",   # red
    "DCM": "#7f7f7f",   # grey
}
METHOD_ORDER = ["SGG", "DCM", "GA2"]   # thu tu subplot / grouped-bar


def _color(method: str) -> str:
    return METHOD_COLORS.get(method.upper(), "#1f77b4")


# ============================================================================
# FIGURE 1 — Spatial Scatter 2D (3 subplots: SGG | DCM | GA2)
# ============================================================================
def make_spatial_scatter(syn_files: list, output_root: str,
                         n_sample: int = 3000, seed: int = 42):
    """
    Read x/y from synthetic files and plot 3 subplots on one Figure.
    Only samples n_sample random points for speed.
    """
    rng = np.random.default_rng(seed)
    coords = {m: {"x": [], "y": []} for m in METHOD_ORDER}

    for fp in syn_files:
        try:
            df = pd.read_csv(fp, usecols=["x", "y", "method"])
            m = str(df["method"].iloc[0]).upper()
            if m not in coords:
                m = "GA2" if "GA" in m else m
            if m in coords:
                coords[m]["x"].extend(df["x"].tolist())
                coords[m]["y"].extend(df["y"].tolist())
        except Exception as exc:
            warnings.warn(f"[compare_generators] Skipped: {exc}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles_label = {"SGG": "SGG (baseline)", "DCM": "DCM (baseline)", "GA2": "GA (proposed)"}

    for ax, method in zip(axes, METHOD_ORDER):
        xs = np.asarray(coords[method]["x"], dtype=float)
        ys = np.asarray(coords[method]["y"], dtype=float)
        if len(xs) > n_sample:
            idx = rng.choice(len(xs), n_sample, replace=False)
            xs, ys = xs[idx], ys[idx]
        ax.scatter(xs, ys, c=_color(method), s=4, alpha=0.35, linewidths=0)
        ax.set_title(titles_label.get(method, method), fontsize=13, fontweight="bold")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.invert_yaxis()
        ax.grid(alpha=0.2)
        for sp in ax.spines.values():
            sp.set_edgecolor("#cccccc")

    plt.suptitle("Figure 1 — 2D spatial distribution of synthetic gaze", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_root, "fig1_spatial_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> fig1_spatial_scatter.png saved to {output_root}")


# ============================================================================
# FIGURE 2a — Log-Log PSD (Real vs GA2 vs SGG)
# FIGURE 2b — Velocity KDE (Real vs GA2)
# ============================================================================
def make_psd_and_velocity(df: pd.DataFrame, syn_files: list,
                          data_dir: str, output_root: str,
                          sr: float = 1000.0, n_subjects: int = 10):
    """
    Use the first n_subjects to compute average PSD.
    df: comparison_results.csv (columns: method, sid, task)
    """
    # --- Select subject subset for omega values ---
    sids = df["sid"].unique()[:n_subjects]

    omega_real_all: list = []
    omega_ga2_all: list  = []
    omega_sgg_all: list  = []

    file_map: dict = {}
    for fp in syn_files:
        fname = os.path.basename(fp)
        file_map[fname.upper()] = fp

    for sid in sids:
        for task in TASKS:
            raw_path = os.path.join(data_dir, f"Subject_{sid}_{task}_raw.csv")
            fix_path = os.path.join(data_dir, f"Subject_{sid}_{task}_fixations.csv")
            if not (os.path.exists(raw_path) and os.path.exists(fix_path)):
                continue

            try:
                raw_df = load_raw_gaze(raw_path)
                fix_df = load_fixations(fix_path)
                sr_det = detect_sampling_rate(raw_df)

                # omega_real
                _, _, omega_max = build_omega_real(raw_df, fix_df)
                times = raw_df["time_ms"].values
                xs = raw_df["x"].values
                ys = raw_df["y"].values
                for _, f in fix_df.iterrows():
                    l = np.searchsorted(times, float(f["start_ms"]), side="left")
                    r = np.searchsorted(times, float(f["end_ms"]),   side="right")
                    if r - l < 4:
                        continue
                    om = _angular_velocity_from_xy_time(
                        xs[l:r], ys[l:r], t_ms=times[l:r])
                    if len(om):
                        omega_real_all.extend(om.tolist())

                # omega_syn from files matching this sid+task
                for fp in syn_files:
                    fname_upper = os.path.basename(fp).upper()
                    sid_str = str(sid).upper()
                    task_str = task.upper()
                    if sid_str not in fname_upper or task_str[:2] not in fname_upper:
                        continue
                    try:
                        syn_df = pd.read_csv(fp)
                        m = str(syn_df["method"].iloc[0]).upper()
                        for _, cluster in syn_df.groupby("fixation_id"):
                            xi = cluster["x"].values.astype(float)
                            yi = cluster["y"].values.astype(float)
                            om = _angular_velocity_from_xy_time(xi, yi, sampling_rate=sr_det)
                            if len(om):
                                if "GA" in m:
                                    omega_ga2_all.extend(om.tolist())
                                elif "SGG" in m:
                                    omega_sgg_all.extend(om.tolist())
                    except Exception as exc:
                        warnings.warn(f"[compare_generators] Skipped: {exc}")

            except Exception as exc:
                warnings.warn(f"[compare_generators] Skipped: {exc}")

    # ── Figure 2a: Log-Log PSD ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ref_added = False
    for label, vals, color, ls in [
        ("Real data",    omega_real_all, "#333333", "-"),
        ("GA (proposed)", omega_ga2_all, _color("GA2"), "-"),
        ("SGG (baseline)", omega_sgg_all, _color("SGG"), "--"),
    ]:
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr) & (arr >= 0)]
        if len(arr) < 32:
            continue
        nperseg = min(len(arr), 512)
        freqs, psd = welch(arr, fs=sr, nperseg=nperseg)
        nz = freqs > 0
        ax.loglog(freqs[nz], psd[nz], color=color, linestyle=ls,
                  linewidth=1.8, label=label, alpha=0.85)
        if not ref_added and len(freqs[nz]) > 4:
            f_ref = freqs[nz]
            psd_mid = np.exp(np.mean(np.log(psd[nz])))
            f_mid   = np.exp(np.mean(np.log(f_ref)))
            scale   = psd_mid / f_mid**(-1)
            ax.loglog(f_ref, scale * f_ref**(-1), "k:", linewidth=1.2,
                      alpha=0.5, label="1/f reference")
            ref_added = True

    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Power Spectral Density", fontsize=12)
    ax.set_title("Figure 2a — Power Spectral Density (Log-Log PSD)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(which="both", alpha=0.2)
    plt.tight_layout()
    path2a = os.path.join(output_root, "fig2a_loglog_psd.png")
    plt.savefig(path2a, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> fig2a_loglog_psd.png saved to {output_root}")

    # ── Figure 2b: Velocity KDE ──────────────────────────────────────────────
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        print("  [WARN] scipy.stats.gaussian_kde unavailable — skipping fig2b")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, vals, color, ls in [
        ("Real data",    omega_real_all, "#333333", "-"),
        ("GA (proposed)",  omega_ga2_all,  _color("GA2"), "-"),
    ]:
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr) & (arr >= 0)]
        if len(arr) < 10:
            continue
        clip = float(np.percentile(arr, 99))
        arr_clip = arr[arr <= clip]
        kde = gaussian_kde(arr_clip, bw_method="scott")
        xs_plot = np.linspace(0, clip, 400)
        ax.plot(xs_plot, kde(xs_plot), color=color, linestyle=ls,
                linewidth=2.0, label=label, alpha=0.9)
        ax.fill_between(xs_plot, kde(xs_plot), alpha=0.12, color=color)

    ax.set_xlabel("Angular velocity (°/s)", fontsize=12)
    ax.set_ylabel("Probability density", fontsize=12)
    ax.set_title("Figure 2b — Velocity distribution (KDE)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    path2b = os.path.join(output_root, "fig2b_velocity_kde.png")
    plt.savefig(path2b, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> fig2b_velocity_kde.png saved to {output_root}")


# ============================================================================
# BOXPLOTS
# ============================================================================
def make_boxplots(df: pd.DataFrame, output_root: str):
    os.makedirs(output_root, exist_ok=True)
    # Use consistent method order and colour scheme
    existing = set(df["method"].dropna().str.upper().unique())
    methods  = [m for m in METHOD_ORDER if m in existing]
    if not methods:
        methods = sorted(df["method"].dropna().unique().tolist())
    colors    = [_color(m) for m in methods]
    title_str = " vs ".join(methods)

    for metric in METRICS:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, task in zip(axes, TASKS):
            sub  = df[df["task"] == task]
            data = [sub[sub["method"].str.upper() == m][metric].dropna().values
                    for m in methods]
            bp = ax.boxplot(data, labels=methods, patch_artist=True,
                            medianprops=dict(color="black", linewidth=1.5))
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)
            ax.set_title(f"{metric} — {task[:4]}")
            ax.set_ylabel(metric)
            ax.grid(axis="y", alpha=0.3)
        plt.suptitle(f"{metric} distribution ({title_str})", fontsize=12)
        plt.tight_layout()
        path = os.path.join(output_root, f"boxplot_{metric}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
    print(f"  -> Boxplots ({len(METRICS)}) saved to {output_root}")


# ============================================================================
# SUMMARY
# ============================================================================
def print_and_save_summary(df: pd.DataFrame, output_root: str):
    methods = sorted(df["method"].dropna().unique().tolist())
    rows = []
    for method in methods:
        for task in TASKS:
            sub = df[(df["method"] == method) & (df["task"] == task)]
            row = {"method": method, "task": task}
            for m in METRICS:
                vals = sub[m].dropna()
                row[f"{m}_mean"] = round(float(vals.mean()), 4) if len(vals) > 0 else float("nan")
                row[f"{m}_std"]  = round(float(vals.std()),  4) if len(vals) > 1 else float("nan")
            rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(output_root, "summary_by_method.csv"), index=False)

    print("\n=== SUMMARY (mean) ===")
    pivot = summary_df.pivot_table(
        values=[f"{m}_mean" for m in METRICS],
        index="method", columns="task",
    )
    print(pivot.round(4).to_string())
    print(f"\n  -> summary_by_method.csv saved to {output_root}")
    return summary_df


# ============================================================================
# COLLECT SYNTHETIC FILES
# ============================================================================
def collect_synthetic_files(dcm_dir, sgg_dir, ga2_dir=None):
    """Return list of paths for all synthetic files.

    ga2_dir: None (skip) or path to directory containing GA2_*.csv files.
    """
    dirs_prefixes = [
        (dcm_dir, "DCM_"),
        (sgg_dir, "SGG_"),
    ]
    if ga2_dir is not None:
        dirs_prefixes.append((ga2_dir, "GA2_"))

    files = []
    for d, prefix in dirs_prefixes:
        if not os.path.isdir(d):
            print(f"[WARN] Directory not found: {d}")
            continue
        found = glob.glob(os.path.join(d, f"{prefix}*.csv"))
        files.extend(found)
        print(f"  {prefix[:-1]:4s}: {len(found)} files trong {d}")
    return files


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="compare_generators.py — Compare GA2/DCM/SGG from pre-generated files",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--dcm_dir",     default="./dcm_output",
        help="Directory containing DCM_*.csv files (default: ./dcm_output)")
    parser.add_argument("--sgg_dir",     default="./sgg_output",
        help="Directory containing SGG_*.csv files (default: ./sgg_output)")
    parser.add_argument("--ga2_dir",     default=None,
        help="Directory containing GA2_*.csv from ga2_generate.py (auto-detect ./ga2_output)")
    parser.add_argument("--data_dir",    default="./data",
        help="ETDD70 data directory with raw+fixation files (default: ./data)")
    parser.add_argument("--output_root", default="./phase1_results",
        help="Output directory for results (default: ./phase1_results)")
    parser.add_argument("--max_workers", type=int, default=4,
        help="Number of parallel worker processes (default: 4)")
    args = parser.parse_args()

    # --- Auto-detect GA2 directory if not specified ---
    if args.ga2_dir is None and os.path.isdir("./ga2_output"):
        args.ga2_dir = "./ga2_output"

    # --- Load label map ---
    label_csv = os.path.join(args.data_dir, "dyslexia_class_label.csv")
    label_df  = pd.read_csv(label_csv)
    label_map = dict(zip(label_df["subject_id"].astype(int), label_df["label"]))

    # --- Collect all synthetic files ---
    print("=== COLLECTING FILES ===")
    syn_files = collect_synthetic_files(args.dcm_dir, args.sgg_dir, args.ga2_dir)
    total = len(syn_files)
    print(f"Total: {total} files")
    if total == 0:
        print("[ERROR] No files found. Run ga2_generate.py / dcm.py / sgg.py first.")
        return

    # --- Compute metrics in parallel ---
    print(f"\n=== EVALUATING METRICS (max_workers={args.max_workers}) ===")
    task_args = [(fp, args.data_dir, label_map) for fp in syn_files]
    all_rows  = []
    done = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(_eval_synthetic_file, a): a for a in task_args}
        for fut in concurrent.futures.as_completed(futs):
            done += 1
            row, status = fut.result()
            if status == "OK" and row is not None:
                all_rows.append(row)
                if done % 20 == 0 or done == total:
                    print(f"  [{done}/{total}] OK {row['method']}-{row['sid']}-{row['task'][:4]}")
            else:
                print(f"  [{done}/{total}] SKIP: {status[:120]}")

    if not all_rows:
        print("[ERROR] No rows succeeded.")
        return

    df = pd.DataFrame(all_rows)
    os.makedirs(args.output_root, exist_ok=True)
    out_csv = os.path.join(args.output_root, "comparison_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n-> comparison_results.csv: {len(df)} rows -> {out_csv}")
    print(f"   Distribution: {df.groupby('method').size().to_dict()}")

    # --- Summary ---
    print_and_save_summary(df, args.output_root)

    # --- Statistics ---
    print("\n=== STATISTICS (ANOVA + Tukey) ===")
    run_statistics(df, args.output_root)

    # --- Boxplots ---
    print("\n=== BOXPLOTS ===")
    make_boxplots(df, args.output_root)

    # --- Figure 1: Spatial Scatter ---
    print("\n=== FIGURE 1 — Spatial Scatter 2D ===")
    make_spatial_scatter(syn_files, args.output_root)

    # --- Figure 2a + 2b: PSD & Velocity KDE ---
    print("\n=== FIGURE 2a/2b — PSD & Velocity KDE ===")
    make_psd_and_velocity(df, syn_files, args.data_dir, args.output_root)

    print(f"\nDone! Results saved to: {args.output_root}")
    print("Next steps:")
    print("  python ga2x20.py --theta_csv ./syn_output/ga2_theta_star.csv")
    print("  python evaluate_tstr.py --syn_root ./ga2_tstr_output")


if __name__ == "__main__":
    main()
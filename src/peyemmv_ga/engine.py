from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy
from tqdm.auto import tqdm


CFG: Dict = {
    "ETDD70_DIR": "./data",
    "OUTPUT_DIR": "./frame_output",
    "TASKS": ["T4_Meaningful_Text", "T5_Pseudo_Text"],

    "POPULATION_SIZE": 100,
    "GENERATIONS": 50,
    "ELITISM_RATE": 0.10,
    "TOURNAMENT_SIZE": 3,
    "MUTATION_RATE": 0.15,
    "CROSSOVER_RATE": 0.80,
    "EARLY_STOP_WINDOW": 20,
    "EARLY_STOP_TOL": 1e-4,

    "H_MIN": 0.60,
    "H_MAX": 0.99,
    "H_DELTA": 0.10,

    "H_ESTIMATION_MIN_DUR_MS": 200.0,
    "H_MUT_STD": 0.05,
    "SIGMA_MIN": 1.0,
    "SIGMA_RATIO": 1.0,
    "SIGMA_MUT_RATIO": 0.15,
    "PHI_MIN": 0.0,                   
    "PHI_MAX": 2.0 * np.pi,           
    "PHI_MUT_STD": 30.0 * np.pi / 180,
    "SEED_MIN": 0,
    "SEED_MAX": 99_999,
    "SEED_MUT_RATE": 0.30,

    "W1_D": 8.0,    
    "W2_O": 2.0,    
    "W3_E": 10.0,

    # --- PeyeMMV parameters: Set 1 thresholds 
    # Set 1 angular values are converted to pixels using
    # PIXELS_PER_DEGREE = 43.46 (ETDD70 viewing geometry).
    "PEYEMMV_T1_DEG": 0.50,          # spatial dispersion threshold (deg)
    "PEYEMMV_T1_PX": 21.73,          # = 0.50 deg * 43.46 px/deg
    "PEYEMMV_T2_DEG": 0.15,          # cluster validation threshold (deg)
    "PEYEMMV_T2_PX": 6.52,           # = 0.15 deg * 43.46 px/deg
    "PEYEMMV_MIN_DUR_MS": 100.0,     # minimum fixation duration (ms)
    "PEYEMMV_LOCAL_WINDOW": 5,       # local cluster window size (samples)

    # --- Visual angle conversion for ETDD70 ---
    # Derived self-consistently from the Set 1 thresholds and reported
    # drift velocity bounds (all four values yield ~43.46 px/deg).
    "PIXELS_PER_DEGREE": 43.46,
    "OMEGA_N_BINS": 64,
    "OMEGA_MAX_PERCENTILE": 99.0,
    "OMEGA_EPS": 1e-8,

    # --- Physiological drift velocity bounds (Cherici et al., 2012;
    #     Martinez-Conde et al., 2004; Rolfs, 2009) ---
    # Drift during fixation has typical velocity in the range
    # 0.1 - 2.0 deg/s. We constrain the GA so that the synthesized
    # drift speed (drift_strength_ratio * sigma * sampling_rate /
    # n_points, projected onto the deg/s scale) remains within these
    "DRIFT_VEL_MIN_DEG_S": 0.10,     # = 4.35 px/s
    "DRIFT_VEL_MIN_PX_S": 4.35,
    "DRIFT_VEL_MAX_DEG_S": 2.00,     # = 86.92 px/s
    "DRIFT_VEL_MAX_PX_S": 86.92,
    # Fixation filtering (looser than PeyeMMV detection threshold,
    #     since short fixations 100-200ms are valid in reading;
    #     Rayner, 1998) ---
    "INPUT_MIN_FIX_DUR_MS": 80.0,
    "MAX_FIXATIONS_FOR_FITNESS": None,

    # --- Reproducibility ---
    "GLOBAL_SEED": 42,
    "RESUME": True,

    # --- Logging ---
    "LOG_LEVEL": logging.INFO,
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Individual:
    """A single GA individual encoding the genotype.

    Genotype: theta = (H, sigma, phi, seed)

    Attributes
    ----------
    H : float
        Hurst exponent in (0, 1).
    sigma : float
        Spatial scatter standard deviation (pixels).
    phi : float
        Rotation angle (radians, [0, 2π]) applied to the 2-D slow-drift
        trajectory, biasing net drift direction without fixing a
        straight-line path.
    seed : int
        PRNG seed.
    fitness : float, optional
        Cached fitness value. None until evaluated.
    D, O, E : float, optional
        Cached fitness components.
    """
    H: float
    sigma: float
    phi: float
    seed: int
    fitness: Optional[float] = None
    D: Optional[float] = None
    O: Optional[float] = None
    E: Optional[float] = None

    def copy(self) -> "Individual":
        """Return a deep copy preserving cached fitness."""
        return Individual(
            H=self.H, sigma=self.sigma, phi=self.phi,
            seed=self.seed,
            fitness=self.fitness, D=self.D, O=self.O, E=self.E,
        )

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SubjectTaskBaseline:
    """Empirical baseline parameters estimated from a single
    (subject, task) pair, used to seed the GA population.

    Attributes
    ----------
    H_base : float
        Hurst exponent estimated via PSD slope on position residuals
        within fixation intervals (H = (-slope - 1) / 2).
    sigma_base : float
        Median per-fixation local-window std of position residuals
        (pixels); window = PEYEMMV_LOCAL_WINDOW samples (fast tremor only).
    target_psd_slope : float
        PSD slope of real gaze velocity, used as target for spectral
        error.
    sampling_rate : float
        Detected sampling rate (Hz).
    n_fixations_used : int
        Number of fixations contributing to estimates.
    """
    H_base: float
    sigma_base: float
    target_psd_slope: float
    sampling_rate: float
    n_fixations_used: int
    H_source: str
    sigma_source: str


# ============================================================================
# UTILITIES
# ============================================================================

def setup_logger(name: str = "ga_synthetic") -> logging.Logger:
    """Configure a module-level logger writing to stdout."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(CFG["LOG_LEVEL"])
    handler = logging.StreamHandler()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


LOG = setup_logger()


def stable_hash_int(text: str, mod: int = 10_000_000) -> int:
    """Return a deterministic integer hash of `text`."""
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % mod


def combine_seed(base_seed: int, fixation_id: int) -> int:
    """Combine an individual seed with a fixation id without collision.

    Two distinct (seed, fixation_id) pairs yield distinct combined
    seeds, avoiding the additive-collision pathology of `seed + id`.
    """
    return stable_hash_int(f"{base_seed}_{fixation_id}", mod=2**31 - 1)


def _angular_velocity_from_xy_time(
    x: np.ndarray,
    y: np.ndarray,
    t_ms: Optional[np.ndarray] = None,
    sampling_rate: Optional[float] = None,
) -> np.ndarray:
    """Compute angular velocity magnitude from x/y positions.

    Parameters
    ----------
    x : np.ndarray
        Horizontal gaze positions.
    y : np.ndarray
        Vertical gaze positions.
    t_ms : np.ndarray, optional
        Timestamps in milliseconds for each sample.
    sampling_rate : float, optional
        Sampling rate in Hz to use when t_ms is unavailable.

    Returns
    -------
    np.ndarray
        Angular velocity magnitudes for each interval.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if t_ms is not None:
        t_ms = np.asarray(t_ms, dtype=float)
        if len(t_ms) != len(x):
            raise ValueError("t_ms length must match x/y length")
        dt = np.diff(t_ms) / 1000.0
    elif sampling_rate is not None and sampling_rate > 0:
        dt = np.full(max(0, len(x) - 1), 1.0 / float(sampling_rate), dtype=float)
    else:
        raise ValueError("Require t_ms or sampling_rate for angular velocity")

    if len(x) < 2 or len(dt) < 1:
        return np.array([], dtype=float)

    dx = np.diff(x)
    dy = np.diff(y)
    omega = np.hypot(dx, dy) / dt
    return omega


def _histogram_distribution(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bin_edges)
    hist = hist.astype(float)
    total = hist.sum()
    if total <= 0:
        return np.ones_like(hist, dtype=float) / max(1, len(hist))
    return hist / total


def _jensen_shannon_divergence_normalized(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    if p.sum() <= 0:
        p = np.ones_like(p, dtype=float) / len(p)
    else:
        p = p / p.sum()
    if q.sum() <= 0:
        q = np.ones_like(q, dtype=float) / len(q)
    else:
        q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (entropy(p, m, base=2) + entropy(q, m, base=2)))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def reflect_into(value: float, low: float, high: float) -> float:
    """Reflect a value into [low, high] instead of clipping.

    Clipping causes mutation pressure to collapse mass at boundaries;
    reflection preserves the search density near the bounds.
    """
    if high <= low:
        return low
    span = high - low
    while value < low or value > high:
        if value < low:
            value = 2 * low - value
        elif value > high:
            value = 2 * high - value
        # safety net for pathological inputs
        if not np.isfinite(value):
            return (low + high) / 2.0
    return value


def find_file(subject_id: str, task: str, file_type: str) -> Optional[str]:
    """Locate the canonical ETDD70 CSV path for a (subject, task, type)."""
    path = os.path.join(
        CFG["ETDD70_DIR"],
        f"Subject_{subject_id}_{task}_{file_type}.csv",
    )
    return path if os.path.exists(path) else None


def discover_subjects() -> List[str]:
    """Return sorted list of subject IDs found in ETDD70_DIR."""
    pattern = os.path.join(CFG["ETDD70_DIR"], "Subject_*_*_raw.csv")
    files = glob.glob(pattern)
    subjects = set()
    for path in files:
        m = re.match(r"Subject_(\d+)_T\d+_.+?_raw\.csv", os.path.basename(path))
        if m:
            subjects.add(m.group(1))
    return sorted(subjects)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_raw_gaze(raw_path: str) -> pd.DataFrame:
    """Load raw gaze CSV and compute binocular average gaze.

    Returns
    -------
    pd.DataFrame with columns: time_ms (float), x (float), y (float).

    Notes
    -----
    ETDD70 timestamps are in microseconds; converted to milliseconds.
    Binocular average follows Holmqvist et al. (2011) recommendation
    when both eyes are tracked.
    """
    raw = pd.read_csv(raw_path)
    required = ["time", "gaze_x_left", "gaze_y_left",
                "gaze_x_right", "gaze_y_right"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Raw file missing columns {missing}: {raw_path}")

    out = pd.DataFrame()
    out["time_ms"] = pd.to_numeric(raw["time"], errors="coerce") / 1000.0
    gx_left = pd.to_numeric(raw["gaze_x_left"], errors="coerce")
    gx_right = pd.to_numeric(raw["gaze_x_right"], errors="coerce")
    gy_left = pd.to_numeric(raw["gaze_y_left"], errors="coerce")
    gy_right = pd.to_numeric(raw["gaze_y_right"], errors="coerce")

    # Treat zero-coded blink samples as missing (ETDD70 convention)
    gx_left = gx_left.replace(0.0, np.nan)
    gx_right = gx_right.replace(0.0, np.nan)
    gy_left = gy_left.replace(0.0, np.nan)
    gy_right = gy_right.replace(0.0, np.nan)

    out["x"] = pd.concat([gx_left, gx_right], axis=1).mean(axis=1)
    out["y"] = pd.concat([gy_left, gy_right], axis=1).mean(axis=1)

    out = (out.dropna(subset=["time_ms", "x", "y"])
              .sort_values("time_ms")
              .drop_duplicates("time_ms")
              .reset_index(drop=True))
    return out


def load_fixations(fix_path: str) -> pd.DataFrame:
    """Load fixations CSV, filtering invalid rows."""
    fix = pd.read_csv(fix_path)
    required = ["start_ms", "end_ms", "duration_ms", "fix_x", "fix_y"]
    missing = [c for c in required if c not in fix.columns]
    if missing:
        raise ValueError(f"Fixations file missing columns {missing}: {fix_path}")

    for col in required:
        fix[col] = pd.to_numeric(fix[col], errors="coerce")
    if "id" not in fix.columns:
        fix["id"] = np.arange(len(fix))

    fix = (fix.dropna(subset=required)
              .loc[fix["duration_ms"] > 0]
              .sort_values(["start_ms", "end_ms"])
              .reset_index(drop=True))
    return fix


def detect_sampling_rate(raw_df: pd.DataFrame) -> float:
    """Detect sampling rate from inter-sample timing.

    Robust to blink gaps: takes median of dt values in the lower
    quartile range, which represent contiguous sampling intervals.
    """
    dt = np.diff(raw_df["time_ms"].values)
    dt = dt[(dt > 0) & (dt < 50)]  # discard gaps > 50 ms (blinks)
    if len(dt) == 0:
        raise ValueError("Cannot detect sampling rate: no valid intervals")
    median_dt_ms = float(np.median(dt))
    return 1000.0 / median_dt_ms


# ============================================================================
# BASELINE PARAMETER ESTIMATION
# ============================================================================

def hurst_higuchi(series: np.ndarray, k_max: Optional[int] = None) -> float:
    """Estimate Hurst exponent via Higuchi's fractal dimension.

    For a fractional Brownian motion with Hurst exponent H, Higuchi's
    fractal dimension FD satisfies FD = 2 - H (Higuchi, 1988). This
    estimator is preferred over R/S for short series (n < 1000) because
    it has lower bias and variance (Esteller et al., 2001).

    Parameters
    ----------
    series : np.ndarray
        One-dimensional time series (e.g. position residuals).
    k_max : int, optional
        Maximum lag; defaults to min(len(series)//4, 30).

    Returns
    -------
    float
        H estimate in [H_MIN, H_MAX]; returns NaN if estimation fails.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    N = len(x)
    if N < 16:
        return np.nan

    if k_max is None:
        k_max = min(N // 4, 30)
    k_max = max(k_max, 4)

    Lk = []
    ks = list(range(1, k_max + 1))
    for k in ks:
        Lmk = []
        for m in range(1, k + 1):
            indices = np.arange(m - 1, N, k)
            seg = x[indices]
            if len(seg) < 2:
                continue
            L = np.sum(np.abs(np.diff(seg))) * (N - 1) \
                / (k * (len(seg) - 1) ** 2)
            Lmk.append(L)
        if Lmk:
            Lk.append(np.mean(Lmk))
        else:
            Lk.append(np.nan)

    valid = [(k, L) for k, L in zip(ks, Lk)
             if (L is not None) and np.isfinite(L) and (L > 0)]
    if len(valid) < 3:
        return np.nan
    ks_v, Ls_v = zip(*valid)
    slope, _ = np.polyfit(np.log10(ks_v), np.log10(Ls_v), 1)
    FD = -slope
    H = 2.0 - FD
    # Return raw estimate (caller is responsible for any bound enforcement);
    # Higuchi can occasionally exceed 1.0 on highly correlated short series,
    # which is a known finite-sample bias (Esteller et al., 2001).
    return float(H)


def hurst_from_position_psd(
    series: np.ndarray, sampling_rate: float,
) -> float:
    """Estimate Hurst exponent from log-log PSD slope of position residuals.

    For fractional Brownian motion with Hurst exponent H, the PSD of the
    position signal satisfies PSD(f) ∝ f^{-(2H+1)}, giving the identity::

        H = (-slope - 1) / 2

    where slope is the log-log slope of the one-sided power spectrum
    (Mandelbrot & Van Ness, 1968). This estimator has lower finite-sample
    bias than Higuchi's fractal dimension on within-fixation segments at
    250 Hz (Eke et al., 2002).

    Parameters
    ----------
    series : np.ndarray
        De-meaned position residuals within one fixation interval.
    sampling_rate : float
        Sampling rate (Hz).

    Returns
    -------
    float
        H estimate; NaN if estimation cannot be performed.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 16:
        return np.nan
    x = x - np.mean(x)
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sampling_rate)
    valid = (freqs > 0.5) & (freqs < 30.0) & (power > 0)
    if valid.sum() < 8:
        return np.nan
    slope, _ = np.polyfit(np.log(freqs[valid]), np.log(power[valid]), 1)
    H = (-slope - 1.0) / 2.0
    return float(H)


def hurst_from_velocity_psd(
    series: np.ndarray, sampling_rate: float,
) -> float:
    """Estimate Hurst exponent from log-log PSD slope of velocity increments.

    For fractional Gaussian noise (fGn = first differences of fBm) with
    Hurst exponent H, the PSD satisfies PSD(f) ∝ f^{-(2H-1)}, giving::

        H = (1 - slope) / 2

    Using velocity (increments) instead of position makes H_base directly
    comparable to the H optimized by the GA, which targets the angular-
    velocity distribution.  Both estimators characterize the same underlying
    fBm process; the difference lies only in which PSD formula applies
    (Mandelbrot & Van Ness, 1968; Eke et al., 2002).

    Parameters
    ----------
    series : np.ndarray
        Position residuals (de-meaned); velocity increments are computed
        internally via np.diff.
    sampling_rate : float
        Sampling rate (Hz).

    Returns
    -------
    float
        H estimate in (0, 1); NaN if estimation cannot be performed.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 17:  # need >=16 increments
        return np.nan
    v = np.diff(x)  # velocity increments (fGn)
    v = v - np.mean(v)
    power = np.abs(np.fft.rfft(v)) ** 2
    freqs = np.fft.rfftfreq(len(v), d=1.0 / sampling_rate)
    valid = (freqs > 0.5) & (freqs < 30.0) & (power > 0)
    if valid.sum() < 8:
        return np.nan
    slope, _ = np.polyfit(np.log(freqs[valid]), np.log(power[valid]), 1)
    H = (1.0 - slope) / 2.0  # fGn: PSD(f) ∝ f^{-(2H-1)}
    return float(H)


def estimate_baseline(
    raw_df: pd.DataFrame,
    fix_df: pd.DataFrame,
    sampling_rate: float,
) -> SubjectTaskBaseline:
    """Estimate (H_base, sigma_base, target_psd_slope) from real data.

    H is estimated as the median of per-fixation PSD-slope H values
    computed on position residuals (de-meaned x, y) within each
    fixation interval. For fBm, PSD(f) ∝ f^{-(2H+1)}, so
    H = (-slope - 1) / 2. This has lower finite-sample bias than
    Higuchi's fractal dimension on 250 Hz within-fixation data
    (Eke et al., 2002).

    sigma is estimated as the median of per-fixation mean local-window
    std, where the window matches PeyeMMV's local detection window
    (PEYEMMV_LOCAL_WINDOW samples). This measures the fast tremor/
    jitter amplitude only, excluding slow drift, and is directly
    consistent with the PeyeMMV detection constraint.

    target_psd_slope is fit on log-log power spectral density of the
    full-task gaze velocity, providing a subject-task-specific
    spectral target rather than a hard-coded constant.
    """
    times = raw_df["time_ms"].values
    xs = raw_df["x"].values
    ys = raw_df["y"].values

    H_per_fix: List[float] = []
    sigma_per_fix: List[float] = []

    # Minimum duration for H estimation only (not for sigma / PeyeMMV).
    # Longer fixations → more samples per PSD window → lower slope bias.
    h_min_dur = CFG["H_ESTIMATION_MIN_DUR_MS"]

    for _, f in fix_df.iterrows():
        left = np.searchsorted(times, f["start_ms"], side="left")
        right = np.searchsorted(times, f["end_ms"], side="right")
        fx, fy = xs[left:right], ys[left:right]
        if len(fx) < 10:
            continue

        cx, cy = float(np.mean(fx)), float(np.mean(fy))
        dx, dy = fx - cx, fy - cy

        # sigma: use ALL valid fixations (same threshold as INPUT_MIN_FIX_DUR_MS).
        # Independent of H_ESTIMATION_MIN_DUR_MS and PEYEMMV_MIN_DUR_MS.
        win = CFG["PEYEMMV_LOCAL_WINDOW"]
        if len(dx) < win:
            continue
        swx = np.lib.stride_tricks.sliding_window_view(dx, win)
        swy = np.lib.stride_tricks.sliding_window_view(dy, win)
        s = 0.5 * (float(np.mean(np.std(swx, axis=1, ddof=1)))
                   + float(np.mean(np.std(swy, axis=1, ddof=1))))
        if not np.isfinite(s) or s > 50:  # discard blink-contaminated
            continue
        sigma_per_fix.append(s)

        # H: only use fixations >= H_ESTIMATION_MIN_DUR_MS (longer window
        # reduces finite-sample bias in PSD slope fit).
        # Velocity-based estimator (hurst_from_velocity_psd) is used so that
        # H_base and the GA-optimised H operate on the same fGn perspective,
        # making the H_DELTA search range directly meaningful.
        if float(f["duration_ms"]) < h_min_dur:
            continue
        Hx = hurst_from_velocity_psd(dx, sampling_rate)
        Hy = hurst_from_velocity_psd(dy, sampling_rate)
        if np.isfinite(Hx) and np.isfinite(Hy):
            H_per_fix.append(0.5 * (Hx + Hy))

    if len(sigma_per_fix) == 0:
        sigma_base = CFG["SIGMA_MIN"]
        sigma_source = "fallback_no_valid_fixations"
    else:
        sigma_base = float(max(np.median(sigma_per_fix), CFG["SIGMA_MIN"]))
        sigma_source = "median_local_window_std_within_fixation"

    if len(H_per_fix) < 5:
        H_base = 0.85  # high-correlation prior typical for reading gaze
        H_source = "fallback_prior"
    else:
        H_raw = float(np.median(H_per_fix))
        # Higuchi can exceed 1.0 due to finite-sample bias on short
        # highly-correlated series; clip into the GA search range.
        H_base = float(np.clip(H_raw, CFG["H_MIN"], CFG["H_MAX"]))
        H_source = (f"median_psd_slope_per_fixation_position "
                    f"(raw={H_raw:.3f})")

    target_slope = _compute_psd_slope_within_fixations(
        raw_df, fix_df, sampling_rate=sampling_rate,
    )

    return SubjectTaskBaseline(
        H_base=H_base,
        sigma_base=sigma_base,
        target_psd_slope=target_slope,
        sampling_rate=sampling_rate,
        n_fixations_used=len(sigma_per_fix),
        H_source=H_source,
        sigma_source=sigma_source,
    )


def _compute_psd_slope(
    x: np.ndarray, y: np.ndarray, sampling_rate: float,
) -> float:
    """Fit log-log PSD slope on the gaze speed signal.

    Uses Welch-style averaging via direct FFT on the de-meaned signal,
    excluding DC and the upper half of the Nyquist range (to avoid
    aliasing artefacts; Press et al., 2007).
    """
    vx = np.diff(x)
    vy = np.diff(y)
    v = np.sqrt(vx ** 2 + vy ** 2)
    v = v[np.isfinite(v)]
    if len(v) < 64:
        return -2.0  # neutral fallback
    v = v - np.mean(v)
    power = np.abs(np.fft.rfft(v)) ** 2
    freqs = np.fft.rfftfreq(len(v), d=1.0 / sampling_rate)
    valid = (freqs > 0.5) & (freqs < sampling_rate / 4) & (power > 0)
    if valid.sum() < 8:
        return -2.0
    slope, _ = np.polyfit(np.log(freqs[valid]), np.log(power[valid]), 1)
    return float(slope)


def _compute_psd_slope_within_fixations(
    raw_df: pd.DataFrame, fix_df: pd.DataFrame, sampling_rate: float,
) -> float:
    """Fit PSD slope on gaze velocity computed within fixation intervals.

    This excludes saccade-driven high-velocity transients between
    fixations, providing a fair spectral target for synthetic data
    that contains only intra-fixation samples.
    """
    times = raw_df["time_ms"].values
    xs = raw_df["x"].values
    ys = raw_df["y"].values
    velocities: List[float] = []
    for _, f in fix_df.iterrows():
        left = np.searchsorted(times, f["start_ms"], side="left")
        right = np.searchsorted(times, f["end_ms"], side="right")
        if right - left < 8:
            continue
        fx, fy = xs[left:right], ys[left:right]
        vx, vy = np.diff(fx), np.diff(fy)
        v = np.sqrt(vx ** 2 + vy ** 2)
        velocities.extend(v[np.isfinite(v)].tolist())
    if len(velocities) < 64:
        return -2.0
    v = np.asarray(velocities) - np.mean(velocities)
    power = np.abs(np.fft.rfft(v)) ** 2
    freqs = np.fft.rfftfreq(len(v), d=1.0 / sampling_rate)
    valid = (freqs > 0.5) & (freqs < sampling_rate / 4) & (power > 0)
    if valid.sum() < 8:
        return -2.0
    slope, _ = np.polyfit(np.log(freqs[valid]), np.log(power[valid]), 1)
    return float(slope)


# ============================================================================
# FRACTIONAL BROWNIAN MOTION GENERATION
# ============================================================================

def fbm_spectral(n: int, H: float, rng: np.random.Generator) -> np.ndarray:
    n = int(max(n, 2))
    if n < 16:
        # too short for spectral synthesis: fall back to BM
        white = rng.normal(0.0, 1.0, size=n)
        s = np.cumsum(white)
        return _standardize(s)

    # Use even length for clean FFT
    n_eff = n if n % 2 == 0 else n + 1

    # Frequency grid (avoid f=0 singularity by setting amplitude=0 for DC)
    freqs = np.fft.fftfreq(n_eff)
    amp = np.zeros(n_eff)
    nz = freqs != 0
    amp[nz] = np.abs(freqs[nz]) ** (-(2.0 * H + 1.0) / 2.0)

    # Random phases, with conjugate symmetry to ensure real output
    phases = rng.uniform(0.0, 2 * np.pi, size=n_eff)
    spec = amp * np.exp(1j * phases)
    # Enforce conjugate symmetry: spec[-k] = conj(spec[k])
    half = n_eff // 2
    spec[n_eff - 1 : half : -1] = np.conj(spec[1 : half])
    spec[0] = 0.0  # zero DC
    if n_eff % 2 == 0:
        spec[half] = np.real(spec[half])

    series = np.real(np.fft.ifft(spec))[:n]
    return _standardize(series)


def _standardize(x: np.ndarray) -> np.ndarray:
    """Subtract mean and divide by std, with epsilon guard."""
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-12:
        return x - mu
    return (x - mu) / sd


def generate_displacement(
    n_points: int, H: float, sigma: float, phi: float,
    sampling_rate: float, seed: int, H_drift: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (dx, dy) displacement series from genotype.

    Two-component model:

    1. Fast component (tremor/jitter):
       ``sigma * fBm(H)`` — fractional Brownian motion with Hurst
       exponent H modelling correlated high-frequency jitter and
       tremor (Engbert & Mergenthaler, 2006; Rolfs, 2009).

    2. Slow drift (wandering path):
       Drift velocity increments are drawn from fBm with H_drift
       (= H_base from the subject's real data), so drift uses the
       subject-specific correlation structure while fast tremor uses
       the GA-optimised H. This produces a non-linear, wandering drift
       trajectory consistent with real fixation drift
       (Cherici et al., 2012; Martinez-Conde et al., 2004).
       Mean drift speed is sampled uniformly from the physiological
       range [DRIFT_VEL_MIN_PX_S, DRIFT_VEL_MAX_PX_S] = [4.35, 86.92]
       px/s. phi acts as a 2-D rotation of the drift plane, biasing
       the net drift direction without constraining a straight line.

    Parameters
    ----------
    n_points : int
        Number of samples to generate.
    H : float
        Hurst exponent for the fast tremor/jitter component (GA-optimised).
    sigma : float
        Amplitude of the fast tremor component (pixels).
    phi : float
        Rotation angle (radians, [0, 2π]) of the slow-drift trajectory.
    sampling_rate : float
        Sampling rate (Hz).
    seed : int
        PRNG seed.
    H_drift : float, optional
        Hurst exponent for the slow-drift component. If None, falls back
        to H. Should be set to H_base (subject-specific baseline) so that
        each subject's drift preserves their real correlation structure.

    Returns
    -------
    (dx, dy) : Tuple[np.ndarray, np.ndarray]
        Displacement arrays of length n_points (in pixels).
    """
    rng = np.random.default_rng(int(seed))
    n_points = int(max(n_points, 2))

    # --- Fast component: correlated tremor/jitter (high-H fBm) ---
    fbm_x = fbm_spectral(n_points, H, rng)
    fbm_y = fbm_spectral(n_points, H, rng)
    noise_x = sigma * fbm_x
    noise_y = sigma * fbm_y

    # --- Slow drift: Brownian random-walk velocity → wandering path ---
    # Velocity increments: diff of fBm with H_DRIFT = H_base (subject-specific
    # baseline Hurst), so each subject's drift preserves their own real
    # correlation structure regardless of the GA-optimised H.
    H_DRIFT = H_drift if H_drift is not None else H
    vx_raw = np.diff(fbm_spectral(n_points + 1, H_DRIFT, rng))
    vy_raw = np.diff(fbm_spectral(n_points + 1, H_DRIFT, rng))

    # Scale so that mean |velocity per sample| matches target drift speed.
    drift_speed_px_s = rng.uniform(
        CFG["DRIFT_VEL_MIN_PX_S"], CFG["DRIFT_VEL_MAX_PX_S"]
    )
    drift_speed_px_sample = drift_speed_px_s / max(sampling_rate, 1e-6)
    mean_v_mag = float(np.mean(np.sqrt(vx_raw ** 2 + vy_raw ** 2)))
    if mean_v_mag < 1e-12:
        mean_v_mag = 1.0
    scale = drift_speed_px_sample / mean_v_mag
    vx = vx_raw * scale
    vy = vy_raw * scale

    # Rotate drift trajectory by phi (biases net direction, non-straight).
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    vx_rot = cos_phi * vx - sin_phi * vy
    vy_rot = sin_phi * vx + cos_phi * vy

    # Integrate velocity → drift position (starts at fixation centroid).
    drift_x = np.concatenate([[0.0], np.cumsum(vx_rot)[:-1]])
    drift_y = np.concatenate([[0.0], np.cumsum(vy_rot)[:-1]])

    return noise_x + drift_x, noise_y + drift_y


# ============================================================================
# FIXATION DETECTION (PeyeMMV-style)
# ============================================================================

def peyemmv_check(
    x: np.ndarray, y: np.ndarray, duration_ms: float,
) -> Dict[str, float]:
    """Two-phase spatial fixation validation (Krassanakis et al., 2014).

    Phase 1 — Cluster initialisation (tol1):
        Uses a sliding window: starts with the first point,
        adds each subsequent point to the cluster if its Euclidean distance
        from the dynamic centroid (moving average of accepted points)
        does not exceed tol1.
        Points that fail are rejected from the cluster (n_removed_p1).
        If fewer than 2 points remain after Phase 1 -> reject entirely.

    Phase 2 — Outlier removal (tol2):
        Compute the final centroid of the Phase 1 cluster.
        Iterate to convergence: permanently remove any point whose Euclidean
        distance from the current centroid exceeds tol2, then recompute centroid.
        n_removed_p2 = number of points removed in Phase 2 (used in fitness O).
        If remaining cluster duration < PEYEMMV_MIN_DUR_MS -> reject entirely.

    Parameters
    ----------
    x, y : np.ndarray
        Gaze point coordinates in a fixation cluster.
    duration_ms : float
        Cluster duration (ms), used for post-filtering check.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_original = len(x)

    if n_original < 2:
        return {
            "detected": False,
            "spatial_range": float("nan"),
            "local_range_max": float("nan"),
            "n_points_remaining": int(n_original),
            "n_points_removed": 0,
            "n_removed_p1": 0,
            "n_removed_p2": 0,
            "pass_duration": False,
            "pass_t1": False,
            "pass_t2": False,
        }

    # ------------------------------------------------------------------ #
    # Phase 1 — Cluster initialisation: sliding window + moving centroid
    # Each point is accepted when dist(point, moving_centroid) <= tol1.
    # moving_centroid is updated after each accepted point.
    # ------------------------------------------------------------------ #
    tol1 = CFG["PEYEMMV_T1_PX"]
    mask_p1 = np.zeros(n_original, dtype=bool)  # True = accepted into cluster

    # First point is always accepted (no centroid to compare against yet)
    mask_p1[0] = True
    sum_x = x[0]
    sum_y = y[0]
    n_accepted = 1

    for i in range(1, n_original):
        cx = sum_x / n_accepted
        cy = sum_y / n_accepted
        dist = float(np.sqrt((x[i] - cx) ** 2 + (y[i] - cy) ** 2))
        if dist <= tol1:
            mask_p1[i] = True
            sum_x += x[i]
            sum_y += y[i]
            n_accepted += 1

    n_after_p1 = int(mask_p1.sum())
    n_removed_p1 = n_original - n_after_p1
    pass_t1 = n_after_p1 >= 2

    # Compute spatial_range on Phase 1 cluster (for backward-compat / logging)
    if n_after_p1 >= 2:
        xp1 = x[mask_p1]
        yp1 = y[mask_p1]
        spatial_range = float(
            (np.max(xp1) - np.min(xp1)) + (np.max(yp1) - np.min(yp1))
        )
    else:
        spatial_range = float(
            (np.max(x) - np.min(x)) + (np.max(y) - np.min(y))
        )

    if not pass_t1:
        return {
            "detected": False,
            "spatial_range": spatial_range,
            "local_range_max": float("nan"),
            "n_points_remaining": 0,
            "n_points_removed": n_original,
            "n_removed_p1": n_removed_p1,
            "n_removed_p2": 0,
            "pass_duration": False,
            "pass_t1": False,
            "pass_t2": False,
        }

    # ------------------------------------------------------------------ #
    # Phase 2 — Outlier removal: iteratively remove points exceeding tol2 from dynamic centroid
    # Operates only on points that passed Phase 1 (mask_p1).
    # ------------------------------------------------------------------ #
    tol2 = CFG["PEYEMMV_T2_PX"]
    mask = mask_p1.copy()   # True = still in Phase 2 cluster

    for _ in range(n_original):              # at most n_original iterations
        xm, ym = x[mask], y[mask]
        if len(xm) < 2:
            break
        cx = float(np.mean(xm))
        cy = float(np.mean(ym))
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        # Only consider points still in mask
        outlier_in_cluster = mask & (dist > tol2)
        if not outlier_in_cluster.any():
            break                            # converged, no more points removed
        mask &= ~outlier_in_cluster          # permanently remove

    n_remaining  = int(mask.sum())
    n_removed_p2 = n_after_p1 - n_remaining   # points removed by tol2 (Phase 2)
    n_removed    = n_original - n_remaining    # total points removed (both phases)
    pass_t2      = n_remaining >= 2

    # Compute local_range_max on filtered cluster (for logging / fitness)
    if n_remaining >= 2:
        xm, ym = x[mask], y[mask]
        cx = float(np.mean(xm))
        cy = float(np.mean(ym))
        local_range_max = float(np.max(np.sqrt((xm - cx) ** 2 + (ym - cy) ** 2)))
    else:
        local_range_max = float("nan")

    # ------------------------------------------------------------------ #
    # Duration check — on remaining duration after point removal
    # Fraction of remaining points x original duration (proxy for valid cluster time)
    # ------------------------------------------------------------------ #
    duration_remaining = duration_ms * (n_remaining / n_original) if n_original > 0 else 0.0
    pass_dur = duration_remaining >= CFG["PEYEMMV_MIN_DUR_MS"]

    detected = bool(pass_t1 and pass_t2 and pass_dur)

    return {
        "detected": detected,
        "spatial_range": spatial_range,
        "local_range_max": local_range_max,
        "n_points_remaining": n_remaining,
        "n_points_removed": n_removed,
        "n_removed_p1": n_removed_p1,
        "n_removed_p2": n_removed_p2,
        "pass_duration": bool(pass_dur),
        "pass_t1": bool(pass_t1),
        "pass_t2": bool(pass_t2),
    }
# ============================================================================
# FITNESS EVALUATION
# ============================================================================

def evaluate_individual(
    individual: Individual,
    fixations: pd.DataFrame,
    baseline: SubjectTaskBaseline,
) -> Individual:
    """Evaluate fitness F = w1 * D - w2 * O - w3 * E for an individual.

    Parameters
    ----------
    individual : Individual
        Genotype to evaluate. Cached fitness is returned if present.
    fixations : pd.DataFrame
        Fixed sample of real fixations used for evaluation. The same
        sample must be used across all individuals in a generation
        for fitness comparability.
    baseline : SubjectTaskBaseline
        Baseline parameters; provides target PSD slope and sampling
        rate for spectral error computation.
    """
    if individual.fitness is not None:
        return individual

    sampling_rate = baseline.sampling_rate
    deg_to_px = CFG["PIXELS_PER_DEGREE"]

    detected = 0
    n_total = len(fixations)
    outlier_ratios: List[float] = []
    disp_errors: List[float] = []
    velocities: List[float] = []

    for _, f in fixations.iterrows():
        fix_x = float(f["fix_x"])
        fix_y = float(f["fix_y"])
        duration_ms = float(f["duration_ms"])
        fixation_id = int(f["id"])

        n_points = max(2, int(round(duration_ms * sampling_rate / 1000.0)))

        dx, dy = generate_displacement(
            n_points=n_points,
            H=individual.H,
            sigma=individual.sigma,
            phi=individual.phi,
            sampling_rate=sampling_rate,
            seed=combine_seed(individual.seed, fixation_id),
            H_drift=baseline.H_base,
        )
        x = fix_x + dx
        y = fix_y + dy

        # 1) Detection + retrieve n_removed_p2 to compute O
        chk = peyemmv_check(x, y, duration_ms)
        if chk["detected"]:
            detected += 1

        # 2) Outlier penalty O = fraction of points removed by tol2 filter (Phase 2).
        #    Original definition from paper: O = number of points removed by tol2,
        #    normalised to n_points so that O in [0, 1].
        n_removed_p2 = int(chk.get("n_removed_p2", chk["n_points_removed"]))
        outlier_ratios.append(float(n_removed_p2) / max(n_points, 1))

        # 3) Dispersion similarity (in pixels, comparable units)
        if "disp_x" in f.index and "disp_y" in f.index \
                and np.isfinite(f["disp_x"]) and np.isfinite(f["disp_y"]):
            target_disp_px = (float(f["disp_x"]) + float(f["disp_y"])) * deg_to_px
            target_disp_px = max(target_disp_px, 1e-6)
            syn_disp_px = max(chk["spatial_range"], 1e-6)
            disp_errors.append(abs(syn_disp_px - target_disp_px) / target_disp_px)

        # 4) Velocity for spectral error
        vx = np.diff(x)
        vy = np.diff(y)
        velocities.extend(np.sqrt(vx ** 2 + vy ** 2).tolist())

    D = detected / max(n_total, 1)
    O = float(np.mean(outlier_ratios)) if outlier_ratios else 1.0
    E_psd = _spectral_error(
        np.asarray(velocities, dtype=float),
        target_slope=baseline.target_psd_slope,
        sampling_rate=sampling_rate,
    )
    E_disp = float(np.mean(disp_errors)) if disp_errors else 0.0
    E = float(np.clip(E_psd + E_disp, 0.0, 1.0))

    F = CFG["W1_D"] * D - CFG["W2_O"] * O - CFG["W3_E"] * E

    individual.D = float(D)
    individual.O = float(O)
    individual.E = float(E)
    individual.fitness = float(F)
    return individual


def _allowed_radius_px(fixation_row: pd.Series, deg_to_px: float) -> float:
    """Allowed scatter radius for a fixation, in pixels.

    Uses the larger of (i) 3 * sigma rule and (ii) ETDD70's reported
    dispersion converted to pixels. The 3-sigma rule covers ~99% of a
    Gaussian scatter; for fBm this is a conservative outer bound.
    """
    if "disp_x" in fixation_row.index and "disp_y" in fixation_row.index \
            and np.isfinite(fixation_row["disp_x"]) \
            and np.isfinite(fixation_row["disp_y"]):
        disp_max_deg = max(float(fixation_row["disp_x"]),
                           float(fixation_row["disp_y"]))
        return max(disp_max_deg * deg_to_px, 1e-6)
    return CFG["PEYEMMV_T2_PX"] / 2.0


def _spectral_error(
    velocity: np.ndarray, target_slope: float, sampling_rate: float,
) -> float:
    """L1 distance between fitted log-log PSD slope and target slope."""
    velocity = velocity[np.isfinite(velocity)]
    if len(velocity) < 64:
        return 1.0
    velocity = velocity - np.mean(velocity)
    power = np.abs(np.fft.rfft(velocity)) ** 2
    freqs = np.fft.rfftfreq(len(velocity), d=1.0 / sampling_rate)
    valid = (freqs > 0.5) & (freqs < sampling_rate / 4) & (power > 0)
    if valid.sum() < 8:
        return 1.0
    slope, _ = np.polyfit(np.log(freqs[valid]), np.log(power[valid]), 1)
    return float(abs(slope - target_slope))


# ============================================================================
# GENETIC OPERATORS
# ============================================================================

def initialize_population(
    baseline: SubjectTaskBaseline, rng: np.random.Generator,
) -> List[Individual]:
    """Initialize population uniformly within bounds around baseline."""
    _h_offset  = CFG.get("_H_CENTER_OFFSET", 0.0)
    _sig_mult  = CFG.get("_SIGMA_HIGH_MULT", 3.0)
    H_center   = baseline.H_base + _h_offset
    H_low  = max(CFG["H_MIN"], H_center - CFG["H_DELTA"])
    H_high = min(CFG["H_MAX"], H_center + CFG["H_DELTA"])
    sigma_low  = max(CFG["SIGMA_MIN"],
                     baseline.sigma_base * 0.5)
    sigma_high = max(sigma_low + 1e-6,
                     baseline.sigma_base * _sig_mult)

    pop: List[Individual] = []
    for _ in range(CFG["POPULATION_SIZE"]):
        pop.append(Individual(
            H=float(rng.uniform(H_low, H_high)),
            sigma=float(rng.uniform(sigma_low, sigma_high)),
            phi=float(rng.uniform(CFG["PHI_MIN"], CFG["PHI_MAX"])),
            seed=int(rng.integers(CFG["SEED_MIN"], CFG["SEED_MAX"] + 1)),
        ))
    return pop


def select_elites(population: List[Individual]) -> List[Individual]:
    """Return the top fraction by fitness, preserving cached fitness."""
    sorted_pop = sorted(population, key=lambda i: i.fitness, reverse=True)
    n_elite = max(1, int(round(CFG["ELITISM_RATE"] * CFG["POPULATION_SIZE"])))
    return [ind.copy() for ind in sorted_pop[:n_elite]]


def tournament_select(
    population: List[Individual], rng: np.random.Generator,
) -> Individual:
    """k-tournament selection."""
    idxs = rng.choice(len(population),
                      size=CFG["TOURNAMENT_SIZE"], replace=False)
    return max((population[int(i)] for i in idxs), key=lambda i: i.fitness)


def crossover(
    p1: Individual, p2: Individual, rng: np.random.Generator,
) -> Individual:
    """Arithmetic crossover for continuous genes; circular mean for phi."""
    alpha = float(rng.random())
    H = alpha * p1.H + (1 - alpha) * p2.H
    sigma = alpha * p1.sigma + (1 - alpha) * p2.sigma
    sin_m = alpha * np.sin(p1.phi) + (1 - alpha) * np.sin(p2.phi)
    cos_m = alpha * np.cos(p1.phi) + (1 - alpha) * np.cos(p2.phi)
    phi = float(np.arctan2(sin_m, cos_m)) % (2 * np.pi)
    seed = p1.seed if rng.random() < 0.5 else p2.seed
    return Individual(
        H=float(H), sigma=float(sigma), phi=phi,
        seed=int(seed),
    )


def mutate(
    child: Individual, baseline: SubjectTaskBaseline,
    rng: np.random.Generator,
) -> Individual:
    """Gaussian mutation on continuous genes; uniform resample on seed.

    Bounds are enforced by reflection (not clipping) to avoid mass
    accumulation at boundaries.
    """
    if rng.random() < CFG["MUTATION_RATE"]:
        child.H += float(rng.normal(0.0, CFG["H_MUT_STD"]))
    if rng.random() < CFG["MUTATION_RATE"]:
        child.sigma += float(rng.normal(
            0.0,
            CFG["SIGMA_MUT_RATIO"] * max(baseline.sigma_base, CFG["SIGMA_MIN"]),
        ))
    if rng.random() < CFG["MUTATION_RATE"]:
        child.phi += float(rng.normal(0.0, CFG["PHI_MUT_STD"]))
    if rng.random() < CFG["SEED_MUT_RATE"]:
        child.seed = int(rng.integers(CFG["SEED_MIN"], CFG["SEED_MAX"] + 1))

    _h_offset  = CFG.get("_H_CENTER_OFFSET", 0.0)
    _sig_mult  = CFG.get("_SIGMA_HIGH_MULT", 3.0)
    H_center   = baseline.H_base + _h_offset
    H_low  = max(CFG["H_MIN"], H_center - CFG["H_DELTA"])
    H_high = min(CFG["H_MAX"], H_center + CFG["H_DELTA"])
    sigma_low  = max(CFG["SIGMA_MIN"],
                     baseline.sigma_base * 0.5)
    sigma_high = max(sigma_low + 1e-6,
                     baseline.sigma_base * _sig_mult)

    child.H = reflect_into(child.H, H_low, H_high)
    child.sigma = reflect_into(child.sigma, sigma_low, sigma_high)
    child.phi = child.phi % (2 * np.pi)  # wrap-around for full-circle angle
    child.seed = int(np.clip(child.seed, CFG["SEED_MIN"], CFG["SEED_MAX"]))

    # Mutation invalidates cached fitness
    child.fitness = None
    child.D = None
    child.O = None
    child.E = None
    return child


# ============================================================================
# GA MAIN LOOP
# ============================================================================

def run_ga(
    fixations_for_fitness: pd.DataFrame,
    baseline: SubjectTaskBaseline,
    rng_seed: int,
) -> Tuple[Individual, List[Dict]]:
    """Run the GA for one subject-task pair.

    Returns
    -------
    (best, history)
        best : Individual with highest fitness encountered.
        history : list of per-generation summary dicts.
    """
    rng = np.random.default_rng(rng_seed)
    population = initialize_population(baseline, rng)

    best: Optional[Individual] = None
    history: List[Dict] = []

    for gen in range(1, CFG["GENERATIONS"] + 1):
        # Evaluate (cached individuals skipped inside evaluate_individual)
        for ind in population:
            evaluate_individual(ind, fixations_for_fitness, baseline)

        gen_best = max(population, key=lambda i: i.fitness)
        if best is None or gen_best.fitness > best.fitness:
            best = gen_best.copy()

        history.append({
            "generation": gen,
            "best_fitness": best.fitness,
            "D": best.D, "O": best.O, "E": best.E,
            "H": best.H, "sigma": best.sigma,
            "phi": best.phi,
            "seed": best.seed,
            "gen_mean_fitness": float(np.mean([i.fitness for i in population])),
        })

        LOG.info(
            "Gen %03d | F=%.4f | D=%.3f | O=%.3f | E=%.3f | "
            "H=%.3f | sigma=%.4f | phi=%.3f",
            gen, best.fitness, best.D, best.O, best.E,
            best.H, best.sigma, best.phi,
        )

        # Early stopping
        if len(history) >= CFG["EARLY_STOP_WINDOW"]:
            recent = [h["best_fitness"]
                      for h in history[-CFG["EARLY_STOP_WINDOW"]:]]
            if max(recent) - min(recent) < CFG["EARLY_STOP_TOL"]:
                LOG.info("Converged at generation %d (early stop).", gen)
                break

        if gen >= CFG["GENERATIONS"]:
            break

        # Reproduction
        elites = select_elites(population)
        children: List[Individual] = []
        n_children = CFG["POPULATION_SIZE"] - len(elites)
        while len(children) < n_children:
            p1 = tournament_select(population, rng)
            p2 = tournament_select(population, rng)

            # Crossover is controlled by CROSSOVER_RATE.
            # If crossover is not applied, copy one parent and then mutate.
            if rng.random() < CFG["CROSSOVER_RATE"]:
                child = crossover(p1, p2, rng)
            else:
                child = p1.copy() if rng.random() < 0.5 else p2.copy()
                child.fitness = None
                child.D = None
                child.O = None
                child.E = None

            child = mutate(child, baseline, rng)
            children.append(child)
        population = elites + children

    assert best is not None
    return best, history


# ============================================================================
# FINAL SYNTHETIC GENERATION
# ============================================================================

def generate_synthetic_dataset(
    fixations: pd.DataFrame,
    theta: Individual,
    baseline: SubjectTaskBaseline,
    subject_id: str,
    task: str,
) -> pd.DataFrame:

    all_chunks: List[pd.DataFrame] = []
    sampling_rate = baseline.sampling_rate

    for _, f in fixations.iterrows():
        fixation_id = int(f["id"])
        fix_x = float(f["fix_x"])
        fix_y = float(f["fix_y"])
        duration_ms = float(f["duration_ms"])
        start_ms = float(f["start_ms"])
        end_ms = float(f["end_ms"])

        n_points = max(2, int(round(duration_ms * sampling_rate / 1000.0)))
        dx, dy = generate_displacement(
            n_points=n_points,
            H=theta.H, sigma=theta.sigma, phi=theta.phi,
            sampling_rate=sampling_rate,
            seed=combine_seed(theta.seed, fixation_id),
            H_drift=baseline.H_base,
        )
        t = np.linspace(start_ms, end_ms, n_points)

        chunk = pd.DataFrame({
            "subject_id": subject_id,
            "task": task,
            "fixation_id": fixation_id,
            "point_index": np.arange(n_points, dtype=np.int32),
            "t_ms": t,
            "x": fix_x + dx,
            "y": fix_y + dy,
            "target_fix_x": fix_x,
            "target_fix_y": fix_y,
            "duration_ms": duration_ms,
            "H": theta.H,
            "sigma": theta.sigma,
            "phi": theta.phi,
            "seed": theta.seed,
        })

        # Pass-through metadata
        for col in ("trialid", "stimfile", "aoi_line", "aoi_subline"):
            if col in f.index:
                chunk[col] = f[col]

        all_chunks.append(chunk)

    return pd.concat(all_chunks, ignore_index=True)


# ============================================================================
# PER-SUBJECT-TASK PIPELINE
# ============================================================================

def process_subject_task(
    subject_id: str, task: str,
) -> Optional[Dict]:

    raw_path = find_file(subject_id, task, "raw")
    fix_path = find_file(subject_id, task, "fixations")
    sac_path = find_file(subject_id, task, "saccades")
    met_path = find_file(subject_id, task, "metrics")

    out_dir = os.path.join(CFG["OUTPUT_DIR"], f"Subject_{subject_id}", task)
    ensure_dir(out_dir)
    paths = {
        "synthetic": os.path.join(out_dir,
            f"synthetic_GA_Subject_{subject_id}_{task}.csv"),
        "best_params": os.path.join(out_dir,
            f"best_params_GA_Subject_{subject_id}_{task}.csv"),
        "history": os.path.join(out_dir,
            f"ga_history_Subject_{subject_id}_{task}.csv"),
        "summary": os.path.join(out_dir,
            f"summary_Subject_{subject_id}_{task}.json"),
    }

    # Resume
    if (CFG["RESUME"] and os.path.exists(paths["synthetic"])
            and os.path.exists(paths["best_params"])):
        LOG.info("Skipping %s-%s: outputs already exist.", subject_id, task)
        return {"subject_id": subject_id, "task": task, "skipped": True,
                **paths}

    if raw_path is None or fix_path is None:
        LOG.warning("Skipping %s-%s: missing raw or fixations file.",
                    subject_id, task)
        return None

    LOG.info("=" * 78)
    LOG.info("Processing Subject %s | Task %s", subject_id, task)
    LOG.info("=" * 78)

    # Load
    raw_df = load_raw_gaze(raw_path)
    fix_df = load_fixations(fix_path)
    fix_df = fix_df.loc[
        fix_df["duration_ms"] >= CFG["INPUT_MIN_FIX_DUR_MS"]
    ].reset_index(drop=True)

    if len(fix_df) == 0:
        LOG.warning("No valid fixations for %s-%s.", subject_id, task)
        return None

    # Sampling rate (auto-detected, NOT hardcoded)
    sampling_rate = detect_sampling_rate(raw_df)
    LOG.info("Detected sampling rate: %.1f Hz", sampling_rate)

    # Baseline estimation
    baseline = estimate_baseline(raw_df, fix_df, sampling_rate)
    LOG.info("H_base=%.4f (%s) | sigma_base=%.4f px (%s) | "
             "PSD slope target=%.3f | n_fix=%d",
             baseline.H_base, baseline.H_source,
             baseline.sigma_base, baseline.sigma_source,
             baseline.target_psd_slope, len(fix_df))

    # Fitness fixation sample (FIXED across population for comparability)
    rng_seed = stable_hash_int(
        f"{subject_id}_{task}_{CFG['GLOBAL_SEED']}", mod=10_000_000,
    )
    if (CFG["MAX_FIXATIONS_FOR_FITNESS"] is not None
            and len(fix_df) > CFG["MAX_FIXATIONS_FOR_FITNESS"]):
        fitness_fix = fix_df.sample(
            n=CFG["MAX_FIXATIONS_FOR_FITNESS"],
            random_state=rng_seed,
        ).reset_index(drop=True)
    else:
        fitness_fix = fix_df.copy()

    # GA
    best, history = run_ga(fitness_fix, baseline, rng_seed)

    # Final synthetic on the FULL fixation set
    synth_df = generate_synthetic_dataset(
        fix_df, best, baseline, subject_id, task,
    )

    # Persist
    synth_df.to_csv(paths["synthetic"], index=False, encoding="utf-8-sig")
    pd.DataFrame([{
        "subject_id": subject_id, "task": task,
        "H_base": baseline.H_base, "sigma_base": baseline.sigma_base,
        "target_psd_slope": baseline.target_psd_slope,
        "sampling_rate_hz": baseline.sampling_rate,
        "best_H": best.H, "best_sigma": best.sigma,
        "best_phi": best.phi,
        "best_seed": best.seed,
        "best_fitness": best.fitness,
        "best_D": best.D, "best_O": best.O, "best_E": best.E,
        "n_fixations_total": len(fix_df),
        "n_fixations_for_fitness": len(fitness_fix),
        "n_synthetic_samples": len(synth_df),
        "H_source": baseline.H_source,
        "sigma_source": baseline.sigma_source,
        "peyemmv_t1_px": CFG["PEYEMMV_T1_PX"],
        "peyemmv_t2_px": CFG["PEYEMMV_T2_PX"],
        "raw_path": raw_path, "fixations_path": fix_path,
        "saccades_path": sac_path, "metrics_path": met_path,
    }]).to_csv(paths["best_params"], index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(paths["history"], index=False,
                                  encoding="utf-8-sig")
    with open(paths["summary"], "w", encoding="utf-8") as f:
        json.dump({
            "subject_id": subject_id, "task": task,
            "input_files": {"raw": raw_path, "fixations": fix_path,
                            "saccades": sac_path, "metrics": met_path},
            "baseline": asdict(baseline),
            "best_theta": best.to_dict(),
            "n_generations_run": len(history),
            "outputs": paths,
            "config": {k: v for k, v in CFG.items()
                       if not isinstance(v, (np.ndarray,))},
        }, f, indent=2, ensure_ascii=False, default=float)

    LOG.info("Saved: %s", paths["synthetic"])
    LOG.info("Saved: %s", paths["best_params"])
    return {"subject_id": subject_id, "task": task, "skipped": False,
            "best_fitness": best.fitness, "best_H": best.H,
            "best_sigma": best.sigma, "best_phi": best.phi,
            "best_seed": best.seed, "n_fixations_total": len(fix_df),
            **paths}



# ============================================================================
# PARAMETER TEST ONLY — NO SYNTHETIC DATA SAVED
# ============================================================================

def atomic_write_json(obj: Dict, path: str) -> None:
    """Safely write JSON using a temporary file then rename."""
    ensure_dir(os.path.dirname(path) or ".")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    return str(obj)


def parse_number_list(text: str, cast=float) -> List:
    """Parse comma-separated candidate values."""
    if text is None or str(text).strip() == "":
        return []
    return [cast(x.strip()) for x in str(text).split(",") if x.strip() != ""]


def parse_weight_candidates(text: str) -> List[Dict[str, float]]:
    """Parse weight candidates like '10:1:3;15:1:3;10:2:5'."""
    out = []
    for i, part in enumerate(str(text).split(";")):
        part = part.strip()
        if not part:
            continue
        vals = [float(x.strip()) for x in part.split(":")]
        if len(vals) != 3:
            raise ValueError(f"Weight candidate must be w1:w2:w3, got: {part}")
        out.append({
            "name": f"W{i+1}_{vals[0]:g}_{vals[1]:g}_{vals[2]:g}",
            "W1_D": vals[0],
            "W2_O": vals[1],
            "W3_E": vals[2],
        })
    return out


def config_snapshot(keys: Optional[List[str]] = None) -> Dict:
    """Return JSON-safe snapshot of key configuration."""
    if keys is None:
        keys = [
            "POPULATION_SIZE", "GENERATIONS", "CROSSOVER_RATE",
            "MUTATION_RATE", "W1_D", "W2_O", "W3_E",
            "ELITISM_RATE", "TOURNAMENT_SIZE",
            "PEYEMMV_T1_PX", "PEYEMMV_T2_PX", "PEYEMMV_MIN_DUR_MS",
            "MAX_FIXATIONS_FOR_FITNESS", "INPUT_MIN_FIX_DUR_MS",
        ]
    return {k: CFG.get(k) for k in keys}


def set_cfg_values(values: Dict) -> Dict:
    """Update CFG and return old values for restoration if needed."""
    old = {}
    for k, v in values.items():
        old[k] = CFG.get(k)
        CFG[k] = v
    return old


def restore_cfg_values(old: Dict) -> None:
    for k, v in old.items():
        CFG[k] = v


def make_subject_split_for_parameter_test(
    subjects: List[str],
    calib_subjects: int,
    heldout_subjects: int,
    seed: int,
) -> Dict:
    """Create subject-level calibration / held-out split."""
    subjects = sorted([str(s) for s in subjects])
    rng = np.random.default_rng(seed)
    shuffled = np.array(subjects, dtype=object)
    rng.shuffle(shuffled)

    if len(shuffled) < calib_subjects + heldout_subjects:
        raise ValueError(
            f"Not enough subjects. Have {len(shuffled)}, "
            f"need {calib_subjects + heldout_subjects}."
        )

    calib = shuffled[:calib_subjects].tolist()
    heldout = shuffled[calib_subjects:calib_subjects + heldout_subjects].tolist()
    return {
        "split_level": "subject",
        "random_seed": int(seed),
        "total_subjects_found": int(len(subjects)),
        "calib_subjects": calib,
        "heldout_subjects": heldout,
        "n_calib_subjects": int(len(calib)),
        "n_heldout_subjects": int(len(heldout)),
    }


def build_subject_task_pairs(subjects: List[str], tasks: List[str]) -> List[Tuple[str, str]]:
    """Return only available subject-task pairs with both raw and fixation files."""
    pairs = []
    for s in subjects:
        for t in tasks:
            if find_file(str(s), t, "raw") is not None and find_file(str(s), t, "fixations") is not None:
                pairs.append((str(s), t))
    return pairs


def evaluate_subject_task_no_save(
    subject_id: str,
    task: str,
    config_id: str,
    synth_out_dir: Optional[str] = None,
) -> Dict:
    """Run baseline estimation + GA for one subject-task pair.

    If synth_out_dir is provided, generate and save raw synthetic data to that directory.
    """
    t0 = time.time()
    raw_path = find_file(subject_id, task, "raw")
    fix_path = find_file(subject_id, task, "fixations")

    result = {
        "config_id": config_id,
        "subject_id": str(subject_id),
        "task": str(task),
        "raw_path": raw_path,
        "fixations_path": fix_path,
        "ok": False,
        "error": None,
    }

    try:
        if raw_path is None or fix_path is None:
            result["error"] = "missing_raw_or_fixation_file"
            return result

        raw_df = load_raw_gaze(raw_path)
        fix_df = load_fixations(fix_path)
        fix_df = fix_df.loc[
            fix_df["duration_ms"] >= CFG["INPUT_MIN_FIX_DUR_MS"]
        ].reset_index(drop=True)

        if len(fix_df) == 0:
            result["error"] = "no_valid_fixations_after_filter"
            return result

        sampling_rate = detect_sampling_rate(raw_df)
        baseline = estimate_baseline(raw_df, fix_df, sampling_rate)

        rng_seed = stable_hash_int(
            f"{config_id}_{subject_id}_{task}_{CFG['GLOBAL_SEED']}",
            mod=10_000_000,
        )

        if (CFG["MAX_FIXATIONS_FOR_FITNESS"] is not None
                and len(fix_df) > CFG["MAX_FIXATIONS_FOR_FITNESS"]):
            fitness_fix = fix_df.sample(
                n=int(CFG["MAX_FIXATIONS_FOR_FITNESS"]),
                random_state=rng_seed,
            ).reset_index(drop=True)
        else:
            fitness_fix = fix_df.copy()

        best, history = run_ga(fitness_fix, baseline, rng_seed)

        # Save raw synthetic data if requested
        synthetic_path = None
        if synth_out_dir is not None:
            try:
                ensure_dir(synth_out_dir)
                synth_df = generate_synthetic_dataset(
                    fix_df, best, baseline, subject_id, task,
                )
                fname = f"synthetic_{subject_id}_{task}.csv"
                synthetic_path = os.path.join(synth_out_dir, fname)
                synth_df.to_csv(synthetic_path, index=False, encoding="utf-8-sig")

                # Save ga_history separately
                hist_path = os.path.join(synth_out_dir, f"ga_history_{subject_id}_{task}.csv")
                pd.DataFrame(history).to_csv(hist_path, index=False, encoding="utf-8-sig")
            except Exception as _e:
                LOG.warning("Failed to save synthetic for %s/%s: %s", subject_id, task, _e)
                synthetic_path = None

        result.update({
            "ok": True,
            "runtime_seconds": float(time.time() - t0),
            "n_fixations_total": int(len(fix_df)),
            "n_fixations_for_fitness": int(len(fitness_fix)),
            "sampling_rate_hz": float(baseline.sampling_rate),
            "H_base": float(baseline.H_base),
            "sigma_base": float(baseline.sigma_base),
            "target_psd_slope": float(baseline.target_psd_slope),
            "baseline_n_fixations_used": int(baseline.n_fixations_used),
            "H_source": baseline.H_source,
            "sigma_source": baseline.sigma_source,
            "best_fitness": float(best.fitness),
            "best_D": float(best.D),
            "best_O": float(best.O),
            "best_E": float(best.E),
            "best_H": float(best.H),
            "best_sigma": float(best.sigma),
            "best_phi": float(best.phi),
            "best_seed": int(best.seed),
            "n_generations_run": int(len(history)),
            "last_generation_mean_fitness": float(history[-1]["gen_mean_fitness"]) if history else None,
            "synthetic_path": synthetic_path,
            "config": config_snapshot(),
        })
        return result

    except Exception as exc:
        result["runtime_seconds"] = float(time.time() - t0)
        result["error"] = repr(exc)
        return result


def aggregate_trial_results(results: List[Dict]) -> Dict:
    """Aggregate subject-task results for one candidate configuration."""
    ok = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]

    if not ok:
        return {
            "n_pairs": int(len(results)),
            "n_ok": 0,
            "n_failed": int(len(failed)),
            "mean_fitness": None,
            "std_fitness": None,
            "mean_D": None,
            "mean_O": None,
            "mean_E": None,
            "combined_error": None,
            "mean_runtime_seconds": None,
            "total_runtime_seconds": float(sum(r.get("runtime_seconds", 0) or 0 for r in results)),
        }

    def arr(name):
        return np.asarray([r[name] for r in ok if r.get(name) is not None], dtype=float)

    fitness = arr("best_fitness")
    D = arr("best_D")
    O = arr("best_O")
    E = arr("best_E")
    runtime = np.asarray([r.get("runtime_seconds", 0.0) or 0.0 for r in ok], dtype=float)

    return {
        "n_pairs": int(len(results)),
        "n_ok": int(len(ok)),
        "n_failed": int(len(failed)),
        "mean_fitness": float(np.mean(fitness)) if len(fitness) else None,
        "std_fitness": float(np.std(fitness, ddof=1)) if len(fitness) > 1 else 0.0,
        "mean_D": float(np.mean(D)) if len(D) else None,
        "std_D": float(np.std(D, ddof=1)) if len(D) > 1 else 0.0,
        "mean_O": float(np.mean(O)) if len(O) else None,
        "mean_E": float(np.mean(E)) if len(E) else None,
        "combined_error": float(np.mean(O) + np.mean(E)) if len(O) and len(E) else None,
        "mean_runtime_seconds": float(np.mean(runtime)) if len(runtime) else None,
        "total_runtime_seconds": float(np.sum(runtime)) if len(runtime) else None,
        "failures": [
            {"subject_id": r.get("subject_id"), "task": r.get("task"), "error": r.get("error")}
            for r in failed[:20]
        ],
    }


def choose_best_trial(trials: List[Dict], min_detection: float) -> Dict:
    """Choose best candidate: prefer detection >= threshold, then low error, then high fitness."""
    valid = [t for t in trials if t.get("aggregate", {}).get("n_ok", 0) > 0]
    if not valid:
        raise RuntimeError("No valid trials to choose from.")

    eligible = [
        t for t in valid
        if (t["aggregate"].get("mean_D") is not None
            and t["aggregate"]["mean_D"] >= min_detection)
    ]
    if not eligible:
        eligible = valid

    def key(t):
        agg = t["aggregate"]
        combined = agg.get("combined_error")
        mean_fit = agg.get("mean_fitness")
        std_fit = agg.get("std_fitness")
        runtime = agg.get("total_runtime_seconds")
        # sort ascending; negative mean_fitness means higher is better
        return (
            combined if combined is not None else 1e18,
            -(mean_fit if mean_fit is not None else -1e18),
            std_fit if std_fit is not None else 1e18,
            runtime if runtime is not None else 1e18,
        )

    return sorted(eligible, key=key)[0]


def load_checkpoint(path: str) -> Dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trials": {},
        "phase_summaries": [],
        "selected_config": {},
    }


def save_checkpoint(ckpt: Dict, path: str) -> None:
    ckpt["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    atomic_write_json(ckpt, path)


def run_candidate_trial(
    phase_name: str,
    candidate_name: str,
    overrides: Dict,
    pairs: List[Tuple[str, str]],
    checkpoint: Dict,
    checkpoint_path: str,
    pbar: tqdm,
    min_detection: float,
    phase_dir: Optional[str] = None,
) -> Dict:
    """Run one candidate configuration with checkpoint/resume.

    If phase_dir is provided, raw synthetic data for each subject-task
    is saved to phase_dir/<candidate_name>/synthetic/.
    """
    config_id = f"{phase_name}::{candidate_name}"

    if config_id not in checkpoint["trials"]:
        checkpoint["trials"][config_id] = {
            "phase": phase_name,
            "candidate": candidate_name,
            "overrides": overrides,
            "config_after_override": None,
            "subject_task_results": [],
            "aggregate": None,
            "completed": False,
        }

    trial = checkpoint["trials"][config_id]
    done_keys = {
        f"{r.get('subject_id')}::{r.get('task')}"
        for r in trial.get("subject_task_results", [])
    }

    # Candidate completed in a previous run -> skip entirely, update pbar
    if trial.get("completed"):
        pbar.update(len(pairs))
        return trial

    old_cfg = set_cfg_values(overrides)
    trial["config_after_override"] = config_snapshot()

    # Directory for raw synthetic data of this candidate
    safe_name = re.sub(r'[\\/:*?"<>|]', '_', candidate_name)
    synth_out_dir: Optional[str] = None
    if phase_dir is not None:
        synth_out_dir = os.path.join(phase_dir, safe_name, "synthetic")
        ensure_dir(synth_out_dir)

    try:
        for subject_id, task in pairs:
            pair_key = f"{subject_id}::{task}"
            if pair_key in done_keys:
                pbar.update(1)
                continue

            pbar.set_postfix({
                "phase": phase_name,
                "candidate": candidate_name,
                "pair": pair_key,
            })
            res = evaluate_subject_task_no_save(subject_id, task, config_id, synth_out_dir=synth_out_dir)
            trial["subject_task_results"].append(res)
            done_keys.add(pair_key)
            trial["aggregate"] = aggregate_trial_results(trial["subject_task_results"])
            save_checkpoint(checkpoint, checkpoint_path)
            pbar.update(1)

        trial["aggregate"] = aggregate_trial_results(trial["subject_task_results"])
        trial["completed"] = True
        save_checkpoint(checkpoint, checkpoint_path)
        return trial
    finally:
        restore_cfg_values(old_cfg)


def append_phase_summary(checkpoint: Dict, phase_name: str, trials: List[Dict], selected: Dict) -> None:
    # Remove older phase summary if rerunning same phase, then append updated.
    checkpoint["phase_summaries"] = [
        s for s in checkpoint.get("phase_summaries", [])
        if s.get("phase") != phase_name
    ]
    checkpoint["phase_summaries"].append({
        "phase": phase_name,
        "selected_config_id": selected["config_id"],
        "selected_candidate": selected["candidate"],
        "selected_overrides": selected["overrides"],
        "selected_aggregate": selected["aggregate"],
        "all_candidates": [
            {
                "config_id": t["config_id"],
                "candidate": t["candidate"],
                "overrides": t["overrides"],
                "aggregate": t["aggregate"],
            }
            for t in trials
        ],
    })


def save_candidate_results_csv(
    trial: Dict,
    phase_dir: str,
    candidate_name: str,
) -> None:
    """Save results for each subject-task of one candidate as separate CSVs.

    Directory structure:
        phase_dir/<candidate_name>/subject_task_results.csv
        phase_dir/<candidate_name>/aggregate.json
    """
    results = trial.get("subject_task_results", [])
    if not results:
        return

    # Safe directory name (strip special characters)
    safe_name = re.sub(r'[\\/:*?"<>|]', '_', candidate_name)
    candidate_dir = os.path.join(phase_dir, safe_name)
    ensure_dir(candidate_dir)

    # Flatten: drop nested "config" column for flat CSV
    rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "config"}
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(candidate_dir, "subject_task_results.csv"),
            index=False, encoding="utf-8-sig",
        )

    agg = trial.get("aggregate", {})
    if agg:
        atomic_write_json(
            {"candidate": candidate_name, "aggregate": agg},
            os.path.join(candidate_dir, "aggregate.json"),
        )


def save_phase_comparison_csv(phase_trials: List[Dict], phase_dir: str) -> None:
    """Save comparison CSV for all candidates in one phase."""
    rows = []
    for t in phase_trials:
        agg = t.get("aggregate") or {}
        row = {"config_id": t["config_id"], "candidate": t["candidate"]}
        # Add each override (e.g. POPULATION_SIZE=10)
        for k, v in (t.get("overrides") or {}).items():
            row[f"override_{k}"] = v
        # Add aggregate metrics (drop failure lists)
        for k, v in agg.items():
            if k != "failures":
                row[f"agg_{k}"] = v
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(phase_dir, "phase_comparison.csv"),
            index=False, encoding="utf-8-sig",
        )


def save_top_synthetic_for_phase(
    selected_trial: Dict,
    phase_dir: str,
    checkpoint: Dict,
    n_top: int = 5,
) -> None:
    """Regenerate and save synthetic gaze CSV for the top-N best subject-task pairs
    of the selected candidate in this phase."""
    config_id = selected_trial["config_id"]
    subject_task_results = (
        checkpoint["trials"].get(config_id, {}).get("subject_task_results", [])
    )
    ok_results = [r for r in subject_task_results if r.get("ok")]
    if not ok_results:
        return

    top_results = sorted(ok_results, key=lambda r: r.get("best_fitness", -1e18), reverse=True)[:n_top]

    synth_dir = os.path.join(phase_dir, "top_synthetic")
    ensure_dir(synth_dir)

    for r in top_results:
        try:
            subject_id = r["subject_id"]
            task = r["task"]
            fix_path = r.get("fixations_path") or find_file(subject_id, task, "fixations")
            if fix_path is None:
                continue

            fix_df = load_fixations(fix_path)
            fix_df = fix_df.loc[
                fix_df["duration_ms"] >= CFG["INPUT_MIN_FIX_DUR_MS"]
            ].reset_index(drop=True)
            if len(fix_df) == 0:
                continue

            theta = Individual(
                H=float(r["best_H"]),
                sigma=float(r["best_sigma"]),
                phi=float(r["best_phi"]),
                seed=int(r["best_seed"]),
                fitness=float(r["best_fitness"]),
                D=float(r["best_D"]),
                O=float(r["best_O"]),
                E=float(r["best_E"]),
            )
            baseline = SubjectTaskBaseline(
                H_base=float(r["H_base"]),
                sigma_base=float(r["sigma_base"]),
                target_psd_slope=float(r["target_psd_slope"]),
                sampling_rate=float(r["sampling_rate_hz"]),
                n_fixations_used=int(r.get("baseline_n_fixations_used", len(fix_df))),
                H_source=str(r.get("H_source", "")),
                sigma_source=str(r.get("sigma_source", "")),
            )

            synth_df = generate_synthetic_dataset(fix_df, theta, baseline, subject_id, task)
            fname = f"synthetic_{subject_id}_{task}_fitness{theta.fitness:.4f}.csv"
            synth_df.to_csv(os.path.join(synth_dir, fname), index=False, encoding="utf-8-sig")
        except Exception as exc:
            LOG.warning("save_top_synthetic: failed for %s/%s: %s", r.get("subject_id"), r.get("task"), exc)


def run_parameter_test_only(args) -> Dict:
    """Sequential GA hyperparameter calibration; output one detailed JSON summary."""
    ensure_dir(args.output_dir)
    parameter_dir = os.path.join(args.output_dir, "parameter_test_json_only")
    ensure_dir(parameter_dir)

    checkpoint_path = args.checkpoint_json or os.path.join(parameter_dir, "checkpoint_parameter_test.json")
    final_json_path = args.output_json or os.path.join(parameter_dir, "ga_parameter_test_summary.json")

    checkpoint = load_checkpoint(checkpoint_path)

    CFG["ETDD70_DIR"] = args.data_dir
    CFG["OUTPUT_DIR"] = args.output_dir
    CFG["TASKS"] = args.tasks.split(",") if isinstance(args.tasks, str) else args.tasks
    CFG["RESUME"] = True
    CFG["LOG_LEVEL"] = logging.WARNING
    LOG.setLevel(logging.WARNING)

    if args.max_fixations_for_fitness > 0:
        CFG["MAX_FIXATIONS_FOR_FITNESS"] = int(args.max_fixations_for_fitness)
    else:
        CFG["MAX_FIXATIONS_FOR_FITNESS"] = None

    subjects = discover_subjects()
    split = make_subject_split_for_parameter_test(
        subjects=subjects,
        calib_subjects=args.calib_subjects,
        heldout_subjects=0,
        seed=args.split_seed,
    )
    calib_pairs = build_subject_task_pairs(split["calib_subjects"], CFG["TASKS"])

    if not calib_pairs:
        raise RuntimeError("No calibration subject-task pairs found.")

    pop_candidates = parse_number_list(args.population_candidates, int)
    gen_candidates = parse_number_list(args.generation_candidates, int)
    pc_candidates = parse_number_list(args.crossover_candidates, float)
    pm_candidates = parse_number_list(args.mutation_candidates, float)
    weight_candidates = parse_weight_candidates(args.weight_candidates)

    phases = [
        ("phase1_population_size", [
            (f"POPULATION_SIZE={v}", {"POPULATION_SIZE": int(v)})
            for v in pop_candidates
        ]),
        ("phase2_generations", [
            (f"GENERATIONS={v}", {"GENERATIONS": int(v)})
            for v in gen_candidates
        ]),
        ("phase3_crossover_rate", [
            (f"CROSSOVER_RATE={v:g}", {"CROSSOVER_RATE": float(v)})
            for v in pc_candidates
        ]),
        ("phase4_mutation_rate", [
            (f"MUTATION_RATE={v:g}", {"MUTATION_RATE": float(v)})
            for v in pm_candidates
        ]),
        ("phase5_fitness_weights", [
            (w["name"], {"W1_D": w["W1_D"], "W2_O": w["W2_O"], "W3_E": w["W3_E"]})
            for w in weight_candidates
        ]),
    ]

    # Total progress = all calibration candidate evaluations + final JSON save.
    total_units = sum(len(cands) * len(calib_pairs) for _, cands in phases)
    total_units += 1

    selected_config = checkpoint.get("selected_config", {})
    all_phase_trials: List[Dict] = []

    with tqdm(total=total_units, desc="TOTAL PROGRESS — GA PARAMETER TEST", unit="eval") as pbar:
        for phase_name, candidates in phases:
            phase_dir = os.path.join(parameter_dir, phase_name)
            ensure_dir(phase_dir)

            # Phase completed in a previous run -> restore from checkpoint, skip
            existing_phase_summary = next(
                (s for s in checkpoint.get("phase_summaries", []) if s["phase"] == phase_name),
                None,
            )
            if existing_phase_summary is not None:
                selected_config.update(existing_phase_summary["selected_overrides"])
                checkpoint["selected_config"] = selected_config
                pbar.update(len(candidates) * len(calib_pairs))
                print(f"[CHECKPOINT] Phase {phase_name}: already completed, skipping.")
                continue

            phase_trials = []
            for candidate_name, phase_override in candidates:
                # Sequential tuning: current candidate override + all previous selected params.
                overrides = dict(selected_config)
                overrides.update(phase_override)

                trial = run_candidate_trial(
                    phase_name=phase_name,
                    candidate_name=candidate_name,
                    overrides=overrides,
                    pairs=calib_pairs,
                    checkpoint=checkpoint,
                    checkpoint_path=checkpoint_path,
                    pbar=pbar,
                    min_detection=args.min_detection,
                    phase_dir=phase_dir,
                )
                trial_record = {
                    "config_id": f"{phase_name}::{candidate_name}",
                    "phase": phase_name,
                    "candidate": candidate_name,
                    "overrides": overrides,
                    "aggregate": trial["aggregate"],
                }
                phase_trials.append(trial_record)
                all_phase_trials.append(trial_record)

                # Save detailed results for candidate to its own subdirectory
                save_candidate_results_csv(trial, phase_dir, candidate_name)

            selected_trial = choose_best_trial(phase_trials, min_detection=args.min_detection)
            selected_config.update(selected_trial["overrides"])
            checkpoint["selected_config"] = selected_config
            append_phase_summary(checkpoint, phase_name, phase_trials, selected_trial)
            save_checkpoint(checkpoint, checkpoint_path)

            # Save per-phase detailed results to dedicated directory
            phase_detail = {
                "phase": phase_name,
                "selected_candidate": selected_trial["candidate"],
                "selected_overrides": selected_trial["overrides"],
                "selected_aggregate": selected_trial["aggregate"],
                "all_candidates": [
                    {
                        "config_id": t["config_id"],
                        "candidate": t["candidate"],
                        "overrides": t["overrides"],
                        "aggregate": t["aggregate"],
                        "subject_task_results": checkpoint["trials"].get(t["config_id"], {}).get("subject_task_results", []),
                    }
                    for t in phase_trials
                ],
            }
            atomic_write_json(phase_detail, os.path.join(phase_dir, "phase_results.json"))

            # Save comparison CSV for all candidates in this phase
            save_phase_comparison_csv(phase_trials, phase_dir)

            # Save top-5 synthetic CSVs for the winning candidate of this phase
            save_top_synthetic_for_phase(
                selected_trial=selected_trial,
                phase_dir=phase_dir,
                checkpoint=checkpoint,
                n_top=5,
            )

        # Final summary JSON
        summary = {
            "description": (
                "GA parameter test only. This run does not save any synthetic gaze dataset. "
                "It only evaluates candidate GA hyperparameters and writes detailed trial results."
            ),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input": {
                "data_dir": args.data_dir,
                "tasks": CFG["TASKS"],
                "max_fixations_for_fitness": CFG["MAX_FIXATIONS_FOR_FITNESS"],
            },
            "method": {
                "tuning_strategy": "sequential_parameter_calibration",
                "selection_rule": (
                    "Prefer candidates with mean detection D >= min_detection; "
                    "then minimize combined_error = mean_O + mean_E; "
                    "then maximize mean_fitness; then prefer lower std_fitness and runtime."
                ),
                "fitness_formula": "F(theta) = W1_D * D - W2_O * O - W3_E * E",
                "D": "PeyeMMV-style detection rate on synthetic fixation sequences.",
                "O": "Outlier penalty.",
                "E": "Spectral error plus dispersion error.",
                "peyemmv_constraints": {
                    "t1_px": CFG["PEYEMMV_T1_PX"],
                    "t2_px": CFG["PEYEMMV_T2_PX"],
                    "min_duration_ms": CFG["PEYEMMV_MIN_DUR_MS"],
                    "local_window": CFG["PEYEMMV_LOCAL_WINDOW"],
                },
            },
            "split": split,
            "n_calibration_pairs": len(calib_pairs),
            "candidate_grids": {
                "population_size": pop_candidates,
                "generations": gen_candidates,
                "crossover_rate": pc_candidates,
                "mutation_rate": pm_candidates,
                "weights": weight_candidates,
            },
            "selected_config": selected_config,
            "phase_summaries": checkpoint.get("phase_summaries", []),
            "all_trials_detailed": {
                k: v for k, v in checkpoint["trials"].items()
                if k.startswith("phase1_")
                or k.startswith("phase2_")
                or k.startswith("phase3_")
                or k.startswith("phase4_")
                or k.startswith("phase5_")
            },
            "checkpoint_file": checkpoint_path,
            "final_json_file": final_json_path,
        }
        atomic_write_json(summary, final_json_path)
        pbar.update(1)

    print("=" * 80)
    print("GA PARAMETER TEST COMPLETE — NO SYNTHETIC DATA SAVED")
    print("=" * 80)
    print(f"Final JSON : {final_json_path}")
    print(f"Checkpoint : {checkpoint_path}")
    print("Selected config:")
    print(json.dumps(selected_config, indent=2, ensure_ascii=False))
    return summary


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GA parameter test only — no synthetic gaze data saved."
    )
    parser.add_argument("--data_dir", type=str, default=CFG["ETDD70_DIR"])
    parser.add_argument("--output_dir", type=str, default=CFG["OUTPUT_DIR"])
    parser.add_argument("--tasks", type=str, default=",".join(CFG["TASKS"]))
    parser.add_argument("--calib_subjects", type=int, default=50)
    parser.add_argument("--split_seed", type=int, default=CFG["GLOBAL_SEED"])
    parser.add_argument("--max_fixations_for_fitness", type=int, default=50)

    parser.add_argument("--population_candidates", type=str, default="10,30,50,100,200")
    parser.add_argument("--generation_candidates", type=str, default="10,30,50,100")
    parser.add_argument("--crossover_candidates", type=str, default="0.5,0.7,0.8,0.9")
    parser.add_argument("--mutation_candidates", type=str, default="0.05,0.1,0.15,0.2")
    parser.add_argument(
        "--weight_candidates",
        type=str,
        default="10:1:3;7:1.5:10;10:2:5;10:5:3;8:3:10",
    )
    parser.add_argument("--min_detection", type=float, default=0.90)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--checkpoint_json", type=str, default=None)
    args = parser.parse_args()

    run_parameter_test_only(args)


if __name__ == "__main__":
    main()
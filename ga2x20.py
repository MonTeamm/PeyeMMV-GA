"""
ga2x20.py — GA2 synthetic generation x20 sets (Train on Synthetic, Test on Real).

Reads theta* from find_theta_ga2.py, generates 20 independent synthetic fixation
sets per subject-task. Each set produces:
    - *_metrics_syn_{i}.csv   : ETDD70-compatible trial-level metrics
    - *_fixations_syn_{i}.csv : fixation-level data
    - *_saccades_syn_{i}.csv  : saccade-level data
    - raw/*_raw_syn_{i}.csv   : raw gaze-level data

Input:
    ga2_theta_star.csv         (output of find_theta_ga2.py)
    data/*_fixations.csv       (real ETDD70 fixations)
    data/dyslexia_class_label.csv

Output:
    ga2_tstr_output/Subject_{sid}/{sid}_{task}_*_syn_{i}.csv

Usage:
    python ga2x20.py --theta_csv ./syn_output/ga2_theta_star.csv
    python ga2x20.py --n_sets 20 --max_workers 4

Notes:
    - Run after find_theta_ga2.py has completed
    - Use together with dcmx20.py and sggx20.py for 3-method TSTR evaluation
"""
import os
import sys
import argparse
import warnings
import importlib.util
import concurrent.futures
import multiprocessing
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# ======================== IMPORT FROM engine.py ========================
_HERE    = os.path.dirname(os.path.abspath(__file__))   # FIX: was commented out
_P1_FILE = os.path.join(_HERE, 'engine.py')
spec = importlib.util.spec_from_file_location('_p1', _P1_FILE)
_p1  = importlib.util.module_from_spec(spec)
sys.modules["_p1"] = _p1   # required for Python 3.13 dataclass resolution
spec.loader.exec_module(_p1)

generate_displacement = _p1.generate_displacement
combine_seed          = _p1.combine_seed
load_raw_gaze         = _p1.load_raw_gaze
detect_sampling_rate  = _p1.detect_sampling_rate
peyemmv_check         = _p1.peyemmv_check
CFG                   = _p1.CFG

SAMPLING_RATE_FALLBACK = 250.0    # Hz, fallback when sampling rate cannot be read from raw file
DATA_ROOT      = './data'
ROIS_ROOT      = './rois'
ROI_MAP        = {
    'T4_Meaningful_Text': 'Meaningful_Text_rois.csv',
    'T5_Pseudo_Text':     'Pseudo_Text_rois.csv',
}
TASKS      = ['T4_Meaningful_Text', 'T5_Pseudo_Text']
N_SYN_SETS = 20   # can be overridden via --n_sets

# Fixation duration scale factor per group (based on dyslexia empirical findings)
DUR_SCALE_DYSLEXIC = 1.18   # dyslexic fixations are ~18% longer on average
DUR_SCALE_CONTROL  = 0.95   # control fixations are slightly shorter on average


def resolve_sampling_rate(sid: str, task: str, data_root: str) -> tuple:
    """Resolve sampling rate from raw file per subject-task; fall back to constant if needed."""
    raw_path = os.path.join(data_root, f'Subject_{sid}_{task}_raw.csv')
    if not os.path.exists(raw_path):
        return float(SAMPLING_RATE_FALLBACK), 'fallback_no_raw'

    try:
        raw_df = load_raw_gaze(raw_path)
        sr = float(detect_sampling_rate(raw_df))
        if np.isfinite(sr) and sr > 0:
            return sr, 'raw_timestamp_subject_task'
        return float(SAMPLING_RATE_FALLBACK), 'fallback_invalid_sr'
    except Exception:
        return float(SAMPLING_RATE_FALLBACK), 'fallback_read_error'


# ======================== GENERATE ONE SYNTHETIC SET ========================
def generate_syn_set(fix_df: pd.DataFrame, H: float, sigma: float, phi: float,
                     seed_base: int, H_base: float, sr: float, i_set: int,
                     roi_df: pd.DataFrame, sid: str, task: str, trial_id: int,
                     is_dyslexic: bool = False):
    """
    Generate one synthetic set for a single subject-task:
      - Per fixation: generate_displacement(theta*) -> cluster (x, y)
      - peyemmv_check -> keep valid fixations only
      - Infer saccades from consecutive fixation sequence
      - compute_metrics -> schema matching ETDD70

    Returns: (syn_fix_df, syn_sacc_df, syn_metrics_df, syn_raw_df)
    """
    syn_fixes = []
    raw_rows  = []          # gaze-level points
    for idx, f in fix_df.iterrows():
        fix_id    = int(f.get('id', idx))
        seed_i    = combine_seed(seed_base + i_set * 100000, fix_id)
        dur_raw   = float(f['duration_ms'])
        dur_scale = DUR_SCALE_DYSLEXIC if is_dyslexic else DUR_SCALE_CONTROL
        dur       = dur_raw * dur_scale
        n_pts     = max(2, int(round(dur * sr / 1000.0)))

        dx, dy = generate_displacement(n_pts, H, sigma, phi, sr, seed_i, H_base)
        x = float(f['fix_x']) + dx
        y = float(f['fix_y']) + dy

        passed  = peyemmv_check(x, y, dur)["detected"]
        t_start = float(f.get('start_ms', f.get('start_time', 0.0)))
        dt_ms   = dur / n_pts
        t_abs   = t_start + np.arange(n_pts) * dt_ms
        for k in range(n_pts):
            raw_rows.append({
                'fixation_id':      fix_id,
                'point_index':      k,
                't_ms':             float(k * dt_ms),
                't_abs':            float(t_abs[k]),
                'x':                float(x[k]),
                'y':                float(y[k]),
                'x_fix':            float(f['fix_x']),
                'y_fix':            float(f['fix_y']),
                'peyemmv_passed':   int(passed),
                'H_used':           float(H),
                'sigma_used':       float(sigma),
                'phi_used':         float(phi),
                'H_base_used':      float(H_base),
                'sampling_rate_hz': float(sr),
                'dt_ms':            float(dt_ms),
                'duration_ms':      float(dur),
            })

        if passed:
            row = f.to_dict()
            row['fix_x'] = float(np.mean(x))
            row['fix_y'] = float(np.mean(y))
            syn_fixes.append(row)

    syn_raw_df = pd.DataFrame(raw_rows) if raw_rows else pd.DataFrame(
        columns=['fixation_id', 'point_index', 't_ms', 't_abs',
                 'x', 'y', 'x_fix', 'y_fix', 'peyemmv_passed',
                 'H_used', 'sigma_used', 'phi_used', 'H_base_used',
                 'sampling_rate_hz', 'dt_ms', 'duration_ms'])

    if not syn_fixes:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), syn_raw_df

    syn_fix_df = pd.DataFrame(syn_fixes).reset_index(drop=True)

    # Infer saccades from consecutive fixation sequence
    saccades = []
    for k in range(len(syn_fix_df) - 1):
        r0 = syn_fix_df.iloc[k]
        r1 = syn_fix_df.iloc[k + 1]
        s  = float(r0['end_ms'])
        e  = float(r1['start_ms'])
        if e - s <= 0:
            continue
        x1, y1 = float(r0['fix_x']), float(r0['fix_y'])
        x2, y2 = float(r1['fix_x']), float(r1['fix_y'])
        saccades.append({
            'start_ms':   s, 'end_ms':   e, 'duration_ms': e - s,
            'ampl':       np.hypot(x2 - x1, y2 - y1),
            'start_x':    x1, 'start_y': y1,
            'end_x':      x2, 'end_y':   y2,
        })
    syn_sacc_df = pd.DataFrame(saccades) if saccades else pd.DataFrame(
        columns=['start_ms', 'end_ms', 'duration_ms', 'ampl',
                 'start_x', 'start_y', 'end_x', 'end_y'])

    metrics_df = compute_metrics(syn_fix_df, syn_sacc_df, roi_df, sid, task, trial_id)
    return syn_fix_df, syn_sacc_df, metrics_df, syn_raw_df


# ======================== COMPUTE METRICS ========================
def compute_metrics(fix_df: pd.DataFrame, sacc_df: pd.DataFrame,
                    roi_df: pd.DataFrame, sid: str, task: str, trial_id: int = 12):
    """Compute metrics DataFrame matching the ETDD70 _metrics.csv schema."""
    stimfile   = str(roi_df.iloc[0].get('stimfile', '')) if len(roi_df) > 0 else ''
    n_fix      = len(fix_df)
    sum_fd     = float(fix_df['duration_ms'].sum()) if n_fix > 0 else 0.0
    mean_fd    = sum_fd / n_fix if n_fix > 0 else 0.0
    n_sacc     = len(sacc_df)
    sum_sd     = float(sacc_df['duration_ms'].sum()) if n_sacc > 0 else 0.0
    mean_sd    = sum_sd / n_sacc if n_sacc > 0 else 0.0
    mean_ampl  = float(sacc_df['ampl'].mean()) if n_sacc > 0 else 0.0
    n_prog     = int((sacc_df['end_x'] > sacc_df['start_x']).sum()) if n_sacc > 0 else 0
    n_reg      = n_sacc - n_prog
    ratio      = n_prog / n_reg if n_reg > 0 else float(n_prog)
    dwell      = sum_fd + sum_sd

    whole = {
        'sid': sid, 'stimfile': stimfile, 'eye_used': 'b', 'trialid': trial_id,
        'n_fix_trial': n_fix, 'sum_fix_dur_trial': sum_fd, 'dwell_time_trial': dwell,
        'mean_fix_dur_trial': mean_fd, 'n_sacc_trial': n_sacc,
        'sum_sacc_dur_trial': sum_sd, 'mean_sacc_dur_trial': mean_sd,
        'mean_sacc_ampl_trial': mean_ampl, 'ratio_progress_regress_trial': ratio,
        'n_between_line_regress_trial': 0, 'n_within_line_regress_trial': n_reg,
        'n_regress_trial': n_reg, 'n_progress_trial': n_prog, 'n_transit_trial': n_sacc,
        'task': task,
    }

    line_rois = roi_df[roi_df['kind'] == 'line'].sort_values('id').reset_index(drop=True)
    sub_rois  = roi_df[roi_df['kind'] == 'sub-line'].sort_values('id').reset_index(drop=True)

    rows = []
    for _, lroi in line_rois.iterrows():
        ln  = str(lroi['name'])
        fln = fix_df[fix_df['aoi_line'] == ln] \
              if 'aoi_line' in fix_df.columns else pd.DataFrame()
        nl  = len(fln)
        rows.append({**whole,
            'aoi': ln.replace(' ', '_').replace('-', '_'), 'aoi_kind': 'line',
            'content': np.nan,
            'dwell_time_aoi':                    float(fln['duration_ms'].sum())  if nl > 0 else 0.0,
            'n_fix_aoi':                         nl,
            'sum_fix_dur_aoi':                   float(fln['duration_ms'].sum())  if nl > 0 else 0.0,
            'mean_fix_dur_aoi':                  float(fln['duration_ms'].mean()) if nl > 0 else 0.0,
            'skipped_aoi':                       int(nl == 0),
            'n_fix_first_visit_aoi':             nl,
            'first_fix_dur_aoi':                 float(fln['duration_ms'].iloc[0]) if nl > 0 else 0.0,
            'first_fix_land_pos_aoi':            np.nan,
            'dwell_time_first_visit_aoi':        float(fln['duration_ms'].sum())  if nl > 0 else 0.0,
            'sum_fix_dur_first_visit_aoi':       float(fln['duration_ms'].sum())  if nl > 0 else 0.0,
            'sum_fix_dur_after_first_visit_aoi': 0.0,
            'dwell_time_rereading_aoi':          0.0,
            'n_revisits_aoi':                    max(nl - 1, 0),
        })

    for _, sroi in sub_rois.iterrows():
        an  = str(sroi['name'])
        fin = fix_df[fix_df['aoi_subline'] == an] \
              if 'aoi_subline' in fix_df.columns else pd.DataFrame()
        ns  = len(fin)
        rx, rw = float(sroi.get('x', 0)), float(sroi.get('width', 1))
        if ns > 0:
            first = fin.sort_values('start_ms').iloc[0]
            land  = (float(first['fix_x']) - rx) / rw if rw > 0 else 0.5
            sd    = float(fin['duration_ms'].sum())
            md    = sd / ns
            fd    = float(first['duration_ms'])
        else:
            land = sd = md = fd = 0.0
        rows.append({**whole,
            'aoi': an, 'aoi_kind': 'subline',
            'content': sroi.get('content', np.nan),
            'dwell_time_aoi': sd, 'n_fix_aoi': ns,
            'sum_fix_dur_aoi': sd, 'mean_fix_dur_aoi': md,
            'skipped_aoi': int(ns == 0), 'n_fix_first_visit_aoi': min(ns, 1),
            'first_fix_dur_aoi': fd,
            'first_fix_land_pos_aoi': land if ns > 0 else 0.0,
            'dwell_time_first_visit_aoi': fd, 'sum_fix_dur_first_visit_aoi': fd,
            'sum_fix_dur_after_first_visit_aoi': sd - fd,
            'dwell_time_rereading_aoi': sd - fd,
            'n_revisits_aoi': max(ns - 1, 0),
        })

    COL = ['sid', 'stimfile', 'eye_used', 'trialid',
           'n_fix_trial', 'sum_fix_dur_trial', 'dwell_time_trial', 'mean_fix_dur_trial',
           'n_sacc_trial', 'sum_sacc_dur_trial', 'mean_sacc_dur_trial', 'mean_sacc_ampl_trial',
           'ratio_progress_regress_trial', 'n_between_line_regress_trial',
           'n_within_line_regress_trial', 'n_regress_trial', 'n_progress_trial', 'n_transit_trial',
           'aoi', 'aoi_kind', 'content',
           'dwell_time_aoi', 'n_fix_aoi', 'sum_fix_dur_aoi', 'mean_fix_dur_aoi', 'skipped_aoi',
           'n_fix_first_visit_aoi', 'first_fix_dur_aoi', 'first_fix_land_pos_aoi',
           'dwell_time_first_visit_aoi', 'sum_fix_dur_first_visit_aoi',
           'sum_fix_dur_after_first_visit_aoi', 'dwell_time_rereading_aoi',
           'n_revisits_aoi', 'task']

    df_out = pd.DataFrame(rows)
    for c in COL:
        if c not in df_out.columns:
            df_out[c] = np.nan
    return df_out[COL]


# ======================== PARALLEL WORKER ========================
def worker_tstr(args_tuple):
    """
    Worker process for one subject-task:
    Reads theta* from theta_row and generates n_sets synthetic fixation sets.
    """
    sid, task, theta_row, output_root, n_sets, is_dyslexic = args_tuple
    try:
        fix_path  = os.path.join(DATA_ROOT, f'Subject_{sid}_{task}_fixations.csv')
        metr_path = os.path.join(DATA_ROOT, f'Subject_{sid}_{task}_metrics.csv')
        roi_path  = os.path.join(ROIS_ROOT, ROI_MAP[task])

        if not all(os.path.exists(p) for p in [fix_path, metr_path, roi_path]):
            return sid, task, 'file_not_found'

        fix_df  = pd.read_csv(fix_path)
        metr_df = pd.read_csv(metr_path)
        roi_df  = pd.read_csv(roi_path)

        trial_id  = int(metr_df['trialid'].iloc[0]) if 'trialid' in metr_df.columns else 12
        sr, sr_source = resolve_sampling_rate(sid, task, DATA_ROOT)
        H_star    = float(theta_row['H'])
        sig_star  = float(theta_row['sigma'])
        phi_star  = float(theta_row['phi'])
        H_base    = float(theta_row['H_base'])

        # Deterministic seed derived from subject id
        sid_int   = int(''.join(filter(str.isdigit, str(sid))))
        seed_base = sid_int * 1000

        out_dir   = os.path.join(output_root, f'Subject_{sid}')
        os.makedirs(out_dir, exist_ok=True)
        base_path = os.path.join(out_dir, f'Subject_{sid}_{task}')

        raw_dir = os.path.join(output_root, f'Subject_{sid}', 'raw')
        os.makedirs(raw_dir, exist_ok=True)

        for i in range(n_sets):
            # --- CHECKPOINT: skip if metrics file already exists ---
            _cp = f'{base_path}_metrics_syn_{i}.csv'
            if os.path.exists(_cp) and os.path.getsize(_cp) > 0:
                continue

            syn_fix, syn_sacc, syn_metr, syn_raw = generate_syn_set(
                fix_df, H_star, sig_star, phi_star,
                seed_base, H_base, sr, i,
                roi_df, sid, task, trial_id,
                is_dyslexic=is_dyslexic)

            if not syn_metr.empty:
                syn_metr.to_csv(f'{base_path}_metrics_syn_{i}.csv',   index=False)
            if not syn_fix.empty:
                syn_fix.to_csv( f'{base_path}_fixations_syn_{i}.csv', index=False)
            if not syn_sacc.empty:
                syn_sacc.to_csv(f'{base_path}_saccades_syn_{i}.csv',  index=False)
            if not syn_raw.empty:
                syn_raw.insert(0, 'subject_id', sid)
                syn_raw.insert(1, 'task',       task)
                syn_raw.insert(2, 'method',     'GA2')
                raw_path = os.path.join(raw_dir,
                    f'Subject_{sid}_{task}_raw_syn_{i}.csv')
                syn_raw.to_csv(raw_path, index=False)

        print(f"  [{sid}-{task}] sampling_rate={sr:.3f} Hz ({sr_source})")
        return sid, task, 'OK'

    except Exception as e:
        import traceback
        return sid, task, f'ERROR: {e}\n{traceback.format_exc()}'


# ======================== MAIN ========================
def main():
    parser = argparse.ArgumentParser(
        description='ga2x20.py — Generate GA2 TSTR x N synthetic sets from ga2_theta_star.csv',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--theta_csv', default='./syn_output/ga2_theta_star.csv',
                        help='Path to ga2_theta_star.csv\n'
                             '(output of find_theta_ga2.py)\n'
                             'default: ./syn_output/ga2_theta_star.csv')
    parser.add_argument('--output_root', default='./ga2_tstr_output',
                        help='Output directory for synthetic sets (default: ./ga2_tstr_output)')
    parser.add_argument('--n_sets', type=int, default=N_SYN_SETS,
                        help=f'Number of synthetic sets to generate (default: {N_SYN_SETS})')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Number of parallel worker processes (default: 4)')
    args = parser.parse_args()

    # ---- Load theta_star.csv ----
    if not os.path.exists(args.theta_csv):
        print(f"[ERROR] File not found: {args.theta_csv}")
        print("        Please run find_theta_ga2.py first.")
        sys.exit(1)

    theta_df  = pd.read_csv(args.theta_csv)
    theta_map = {(str(r['sid']), r['task']): r.to_dict()
                 for _, r in theta_df.iterrows()}
    print(f"[INFO] Loaded {len(theta_df)} theta records from {args.theta_csv}")
    print(f"[INFO] Generating {args.n_sets} sets/subject-task | max_workers={args.max_workers}\n")

    # ---- Build task list ----
    label_df    = pd.read_csv(os.path.join(DATA_ROOT, 'dyslexia_class_label.csv'))
    subject_ids = [str(s) for s in label_df['subject_id'].tolist()]
    label_map   = {str(r['subject_id']): int(r['class_id'])
                   for _, r in label_df.iterrows()}

    tstr_args = []
    for sid in subject_ids:
        for task in TASKS:
            key = (sid, task)
            if key not in theta_map:
                print(f"  SKIP {sid}-{task}: no theta found")
                continue
            is_dyslexic = label_map.get(sid, 0) == 1
            tstr_args.append((sid, task, theta_map[key], args.output_root, args.n_sets, is_dyslexic))

    total = len(tstr_args)
    done  = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(worker_tstr, a): a for a in tstr_args}
        for fut in concurrent.futures.as_completed(futs):
            done += 1
            sid, task, status = fut.result()
            mark = 'OK' if status == 'OK' else 'FAIL'
            print(f"[{done}/{total}] {mark} {sid}-{task}"
                  + (f": {status[:120]}" if status != 'OK' else ''))

    print(f"\nDone! {done}/{total} subject-tasks -> {args.output_root}")
    print("Each subject folder: Subject_<sid>/<sid>_<task>_*_syn_i.csv")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
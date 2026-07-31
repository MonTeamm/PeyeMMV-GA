"""
evaluate_tstr.py — TSTR (Train on Synthetic, Test on Real) evaluation pipeline.

Pipeline:
    1. Load real metrics from data/*_metrics.csv → X_real, y_real
    2. For each synthetic method:
       - Load syn metrics from <syn_root>/Subject_{sid}/*_metrics_syn_{i}.csv → X_train
       - Train classifier on X_train → predict X_real → Accuracy, F1, AUC, MCC
    3. Report mean ± std across N sets; save CSV, plot boxplot + confusion matrix

Input:
    ./data/*_metrics.csv                                (from ETDD70 dataset)
    ./<syn_root>/Subject_{sid}/*_metrics_syn_{i}.csv    (from ga2x20.py / sggx20.py / dcmx20.py)

Output:
    <output_root>/evaluation_summary.csv
    <output_root>/confusion_matrix.png        — confusion matrix of TSTR model
    <output_root>/comparison_all_classifiers.csv — multi-classifier comparison (--classifier all)
    <output_root>/fig4_tstr_all_metrics.png   — grouped bar: Acc/F1/AUC/MCC
    <output_root>/fig4_tstr_grouped_bar_<metric>.png — per-metric grouped bar

Usage:
    # 1. Default (Random Forest, 20 sets, T4+T5)
    python evaluate_tstr.py --syn_root ./ga2_tstr_output --data_dir ./data --output_root ./tstr_results/ga2

    # 2. Task T4 only, SVM
    python evaluate_tstr.py --syn_root ./ga2_tstr_output --task T4 --classifier svm

    # 3. Random Forest with more trees, include AOI-level features
    python evaluate_tstr.py --syn_root ./ga2_tstr_output --n_estimators 200 --use_aoi_features

    # 4. Quick test with 5 sets
    python evaluate_tstr.py --syn_root ./ga2_tstr_output --n_sets 5

Notes:
    - TSTR uses full synthetic set as train, full real set as test
    - Label: 1 = dyslexic, 0 = control (from dyslexia_class_label.csv)
    - Each subject × each task = 1 feature row (trial-level)
    - Run after ga2x20.py / sggx20.py / dcmx20.py have completed
"""
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             matthews_corrcoef, confusion_matrix,
                             classification_report)
# TSTR repeated evaluation — number of bootstrap repeats for std estimation
N_TSTR_REPEATS = 10
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# ======================== CONSTANTS ========================
DATA_ROOT  = './data'
TASKS      = ['T4_Meaningful_Text', 'T5_Pseudo_Text']
N_SYN_SETS = 20

# Trial-level features (present in every row of _metrics.csv)
TRIAL_FEATURES = [
    'n_fix_trial',
    'sum_fix_dur_trial',
    'mean_fix_dur_trial',
    'dwell_time_trial',
    'n_sacc_trial',
    'sum_sacc_dur_trial',
    'mean_sacc_dur_trial',
    'mean_sacc_ampl_trial',
    'ratio_progress_regress_trial',
    'n_regress_trial',
    'n_progress_trial',
    'n_transit_trial',
]

# AOI-level features (aggregated: mean/std across AOIs)
AOI_FEATURES_BASE = [
    'dwell_time_aoi',
    'n_fix_aoi',
    'mean_fix_dur_aoi',
    'skipped_aoi',
    'n_revisits_aoi',
]

def compute_mmd(X_syn: np.ndarray, X_real: np.ndarray,
                gamma: float = 1.0) -> float:
    """
    Maximum Mean Discrepancy (RBF kernel) between syn and real.
    Small MMD → distributions are close → synthetic is less diverse.
    """
    from sklearn.metrics.pairwise import rbf_kernel
    n = min(len(X_syn), 500)   # subsample to avoid memory overhead
    m = min(len(X_real), 500)
    rng = np.random.default_rng(42)
    Xs = X_syn[rng.choice(len(X_syn), n, replace=False)]
    Xr = X_real[rng.choice(len(X_real), m, replace=False)]

    K_ss = rbf_kernel(Xs, Xs, gamma=gamma).mean()
    K_rr = rbf_kernel(Xr, Xr, gamma=gamma).mean()
    K_sr = rbf_kernel(Xs, Xr, gamma=gamma).mean()
    return float(K_ss + K_rr - 2 * K_sr)


def distribution_shift_penalty(mmd: float,
                                alpha: float = 0.15,
                                beta: float = 10.0) -> float:
    """
    Penalty factor [0, alpha] applied to the TSTR score.
    Small MMD (syn ≈ real) → high penalty → score decreases.
    Large MMD (syn differs from real) → low penalty → score unchanged.
    """
    import math
    return alpha * math.exp(-beta * mmd)

# ======================== LOAD METRICS ========================
def extract_trial_features(df: pd.DataFrame) -> pd.DataFrame:
    # From metrics DataFrame (multiple AOI rows), extract trial-level features:
    # - Take 1 row per (sid, task) from trial-level columns
    # - Return DataFrame with columns: sid, task, + TRIAL_FEATURES
    avail = [c for c in ['sid', 'task'] + TRIAL_FEATURES if c in df.columns]
    dedup = df[avail].drop_duplicates(subset=['sid', 'task']).reset_index(drop=True)
    return dedup

def extract_aoi_features(df: pd.DataFrame) -> pd.DataFrame:
    # From metrics DataFrame, aggregate AOI-level features:
    # mean/std across all AOIs per (sid, task)
    # Return DataFrame merged with trial features
    trial_df = extract_trial_features(df)
    agg_rows = []
    for (sid, task), grp in df.groupby(['sid', 'task']):
        row = {'sid': str(sid), 'task': task}
        for feat in AOI_FEATURES_BASE:
            if feat in grp.columns:
                vals = pd.to_numeric(grp[feat], errors='coerce').dropna()
                row[f'{feat}_mean'] = float(vals.mean()) if len(vals) > 0 else 0.0
                row[f'{feat}_std']  = float(vals.std())  if len(vals) > 1 else 0.0
            else:
                row[f'{feat}_mean'] = 0.0
                row[f'{feat}_std']  = 0.0
        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)
    merged = pd.merge(trial_df, agg_df, on=['sid', 'task'], how='left')
    return merged

def load_real_metrics(data_dir: str, subject_ids: list, tasks: list,
                      use_aoi: bool = False) -> pd.DataFrame:
    """Load and concatenate real metrics for all subjects."""
    frames = []
    for sid in subject_ids:
        for task in tasks:
            path = os.path.join(data_dir, f'Subject_{sid}_{task}_metrics.csv')
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            df['sid']  = str(sid)
            df['task'] = task
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No metrics found in {data_dir}")

    full = pd.concat(frames, ignore_index=True)
    return extract_aoi_features(full) if use_aoi else extract_trial_features(full)

def load_syn_set(syn_root: str, subject_ids: list, tasks: list,
                 i_set: int, use_aoi: bool = False) -> pd.DataFrame:
    """Load synthetic metrics for set i_set across all subjects."""
    frames = []
    for sid in subject_ids:
        for task in tasks:
            path = os.path.join(syn_root, f'Subject_{sid}',
                                f'Subject_{sid}_{task}_metrics_syn_{i_set}.csv')
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            df['sid']  = str(sid)
            df['task'] = task
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    full = pd.concat(frames, ignore_index=True)
    return extract_aoi_features(full) if use_aoi else extract_trial_features(full)

# ======================== BUILD X, y ========================
def build_Xy(df: pd.DataFrame, label_map: dict, feat_cols: list):
    # Fill NaN with 0, map labels from label_map.
    df = df.copy()
    df['y'] = df['sid'].astype(str).map(label_map)
    df = df.dropna(subset=['y'])

    missing = [c for c in feat_cols if c not in df.columns]
    for c in missing:
        df[c] = 0.0

    X = df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
    y = df['y'].astype(int).values
    return X, y


def normalize_binary_label(value):
    """Map common binary label formats to 0/1."""
    if pd.isna(value):
        raise ValueError("missing_label")

    text = str(value).strip().lower()
    if text in {'1', 'true', 'yes', 'y', 'dyslexic', 'dyslexia', 'case', 'positive'}:
        return 1
    if text in {'0', 'false', 'no', 'n', 'non-dyslexic', 'nondyslexic', 'control', 'negative'}:
        return 0

    # Numeric fallthrough (e.g., 0.0 / 1.0)
    num = float(text)
    if num == 1.0:
        return 1
    if num == 0.0:
        return 0
    raise ValueError(f"unsupported_label_value: {value}")

# ======================== BUILD CLASSIFIER ========================
CLF_CHOICES      = ['rf', 'svm', 'mlp', 'xgb', 'catboost']
CLF_CHOICES_FULL = CLF_CHOICES + ['all']   # 'all' = run all classifiers

def make_clf(name: str, n_estimators: int = 100, seed: int = 42):
    """
    Instantiate classifier by name.
    rf=Random Forest | svm=SVM(RBF) | mlp=MLP Neural Net
    xgb=XGBoost      | catboost=CatBoost
    """
    name = name.lower()
    if name == 'rf':
        return RandomForestClassifier(
            n_estimators=n_estimators, max_depth=None,
            class_weight='balanced', random_state=seed, n_jobs=-1)
    elif name == 'svm':
        return SVC(kernel='rbf', class_weight='balanced',
                   probability=True, random_state=seed)
    elif name == 'mlp':
        return MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed)
    elif name == 'xgb':
        if not _HAS_XGB:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        return XGBClassifier(
            n_estimators=n_estimators,
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=1,
            random_state=seed,
            verbosity=0)
    elif name == 'catboost':
        if not _HAS_CAT:
            raise ImportError("CatBoost not installed. Run: pip install catboost")
        return CatBoostClassifier(
            iterations=n_estimators,
            depth=6,
            auto_class_weights='Balanced',
            random_seed=seed,
            verbose=0)
    else:
        raise ValueError(f"Invalid classifier: {name}. Choose from: {CLF_CHOICES}")

# ======================== TSTR + 5-FOLD CV ========================
def run_tstr(X_real: np.ndarray, y_real: np.ndarray,
             syn_root: str, subject_ids: list, tasks: list,
             feat_cols: list, label_map: dict,
             n_sets: int, clf_name: str, n_estimators: int,
             use_aoi: bool, n_folds: int = 5) -> tuple:
    """
    TSTR with Stratified K-Fold CV on the real set:

    For each syn set i:
        Train = all synthetic set i   (unchanged)
        Test  = real data, chia k-fold stratified
        -> k test folds -> average k scores = 1 point per set i

    Why k-fold on real?
    - Synthetic is always train (this is the nature of TSTR)
    - Real is small (70 subjects), k-fold gives more stable estimates
      compared to testing on all real at once
    - Avoids evaluation overfitting due to high variance on small test sets

    Returns: (scores_list, best_clf, best_y_pred_concat)
    """
    scores      = []
    best_clf    = None
    best_auc    = -1.0
    best_y_pred = np.zeros_like(y_real)
    best_y_real = y_real.copy()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for i in range(n_sets):
        syn_df = load_syn_set(syn_root, subject_ids, tasks, i, use_aoi)
        if syn_df.empty:
            print(f"  Set {i:2d}: SKIP (no file found)")
            continue

        X_train, y_train = build_Xy(syn_df, label_map, feat_cols)

        if len(np.unique(y_train)) < 2:
            print(f"  Set {i:2d}: SKIP (y_train has only 1 class)")
            continue

        # ---------- k-fold on real data ----------
        fold_acc, fold_f1, fold_auc, fold_mcc = [], [], [], []
        y_pred_full = np.zeros_like(y_real)   # accumulated predictions across folds
        set_failed  = False

        for fold_k, (_, test_idx) in enumerate(skf.split(X_real, y_real)):
            X_te, y_te = X_real[test_idx], y_real[test_idx]

            # Scale: fit on synthetic train, transform real test
            scaler = StandardScaler()
            Xtr_s  = scaler.fit_transform(X_train)
            Xte_s  = scaler.transform(X_te)

            clf = make_clf(clf_name, n_estimators, seed=fold_k * 7 + i)
            try:
                clf.fit(Xtr_s, y_train)
            except Exception as e:
                print(f"  Set {i:2d} fold {fold_k}: SKIP fit error — {e}")
                set_failed = True
                break
            y_pred_k = clf.predict(Xte_s)
            y_prob_k = (clf.predict_proba(Xte_s)[:, 1]
                        if hasattr(clf, 'predict_proba')
                        else y_pred_k.astype(float))

            fold_acc.append(accuracy_score(y_te, y_pred_k))
            fold_f1.append(f1_score(y_te, y_pred_k, average='binary', zero_division=0))
            fold_auc.append(roc_auc_score(y_te, y_prob_k))
            fold_mcc.append(matthews_corrcoef(y_te, y_pred_k))
            y_pred_full[test_idx] = y_pred_k

        if set_failed or not fold_acc:
            print(f"  Set {i:2d}: SKIP (fit error)")
            continue

        acc = float(np.mean(fold_acc))
        f1  = float(np.mean(fold_f1))
        auc = float(np.mean(fold_auc))
        mcc = float(np.mean(fold_mcc))

        scores.append({
            'set': i,
            'accuracy': acc, 'f1': f1, 'auc': auc, 'mcc': mcc,
            # standard deviation across folds (stability measure)
            'acc_std': float(np.std(fold_acc)),
            'f1_std':  float(np.std(fold_f1)),
            'auc_std': float(np.std(fold_auc)),
            'mcc_std': float(np.std(fold_mcc)),
        })
        print(f"  Set {i:2d}: "
              f"Acc={acc:.3f}±{np.std(fold_acc):.3f}  "
              f"F1={f1:.3f}±{np.std(fold_f1):.3f}  "
              f"AUC={auc:.3f}±{np.std(fold_auc):.3f}  "
              f"MCC={mcc:.3f}±{np.std(fold_mcc):.3f}")

        if auc > best_auc:
            best_auc    = auc
            best_clf    = (clf, scaler)   # clf from last fold
            best_y_pred = y_pred_full

    return scores, best_clf, best_y_pred

# ======================== BASE% — 5-fold CV on real data ========================
def run_base(X_real: np.ndarray, y_real: np.ndarray,
             clf_name: str, n_estimators: int, n_folds: int = 5,
             seed: int = 42) -> dict:
    """
    Base%: Stratified 5-fold CV entirely on 70 real samples.
    Train on 4 folds, test on 1 fold, repeated 5 times.
    -> Upper bound: best achievable performance using real data.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    acc_l, f1_l, auc_l, mcc_l = [], [], [], []

    for k, (tr_idx, te_idx) in enumerate(skf.split(X_real, y_real)):
        X_tr, y_tr = X_real[tr_idx], y_real[tr_idx]
        X_te, y_te = X_real[te_idx], y_real[te_idx]
        scaler  = StandardScaler()
        X_tr_s  = scaler.fit_transform(X_tr)
        X_te_s  = scaler.transform(X_te)
        clf = make_clf(clf_name, n_estimators, seed=seed + k)
        clf.fit(X_tr_s, y_tr)
        y_pred  = clf.predict(X_te_s)
        y_prob  = (clf.predict_proba(X_te_s)[:, 1]
                   if hasattr(clf, 'predict_proba') else y_pred.astype(float))
        acc_l.append(accuracy_score(y_te, y_pred))
        f1_l.append(f1_score(y_te, y_pred, average='binary', zero_division=0))
        auc_l.append(roc_auc_score(y_te, y_prob))
        mcc_l.append(matthews_corrcoef(y_te, y_pred))

    return {
        'accuracy': float(np.mean(acc_l)), 'acc_std': float(np.std(acc_l)),
        'f1':       float(np.mean(f1_l)),  'f1_std':  float(np.std(f1_l)),
        'auc':      float(np.mean(auc_l)), 'auc_std': float(np.std(auc_l)),
        'mcc':      float(np.mean(mcc_l)), 'mcc_std': float(np.std(mcc_l)),
    }


# ======================== CVSyn% — 5-fold CV on synthetic data ========================
def run_cv_syn(syn_root: str, subject_ids: list, tasks: list,
               label_map: dict, feat_cols: list, use_aoi: bool,
               clf_name: str, n_estimators: int, n_sets: int,
               n_folds: int = 5, seed: int = 42) -> dict:
    """
    CVSyn%: Same as Base% but replacing 70 real samples with 70 synthetic samples.
    For each set i (0..n_sets-1): Stratified 5-fold CV on syn set i.
    CVSyn% = mean qua 20 sets.

    If CVSyn is low -> generator produces identical-looking groups -> data is uninformative.
    Retain% = CVSyn% / Base% x 100 — measures how much discriminative signal is retained.
    """
    skf      = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    set_rows = []

    for i in range(n_sets):
        syn_df = load_syn_set(syn_root, subject_ids, tasks, i, use_aoi)
        if syn_df.empty:
            print(f"  CVSyn Set {i:2d}: SKIP (empty)")
            continue
        X_syn, y_syn = build_Xy(syn_df, label_map, feat_cols)
        if len(np.unique(y_syn)) < 2:
            print(f"  CVSyn Set {i:2d}: SKIP (1 class)")
            continue

        acc_l, f1_l, auc_l, mcc_l = [], [], [], []
        failed = False
        for k, (tr_idx, te_idx) in enumerate(skf.split(X_syn, y_syn)):
            X_tr, y_tr = X_syn[tr_idx], y_syn[tr_idx]
            X_te, y_te = X_syn[te_idx], y_syn[te_idx]
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            clf = make_clf(clf_name, n_estimators, seed=seed + k + i * 100)
            try:
                clf.fit(X_tr_s, y_tr)
            except Exception as e:
                print(f"  CVSyn Set {i:2d} fold {k}: SKIP — {e}")
                failed = True
                break
            y_pred = clf.predict(X_te_s)
            y_prob = (clf.predict_proba(X_te_s)[:, 1]
                      if hasattr(clf, 'predict_proba') else y_pred.astype(float))
            acc_l.append(accuracy_score(y_te, y_pred))
            f1_l.append(f1_score(y_te, y_pred, average='binary', zero_division=0))
            auc_l.append(roc_auc_score(y_te, y_prob))
            mcc_l.append(matthews_corrcoef(y_te, y_pred))

        if not failed and acc_l:
            a, f, u, m = (float(np.mean(acc_l)), float(np.mean(f1_l)),
                          float(np.mean(auc_l)), float(np.mean(mcc_l)))
            print(f"  CVSyn Set {i:2d}: Acc={a:.3f}  F1={f:.3f}  AUC={u:.3f}  MCC={m:.3f}")
            set_rows.append({'set': i, 'accuracy': a, 'f1': f, 'auc': u, 'mcc': m})

    if not set_rows:
        return {}
    df = pd.DataFrame(set_rows)
    return {
        'accuracy': float(df['accuracy'].mean()), 'acc_std': float(df['accuracy'].std()),
        'f1':       float(df['f1'].mean()),        'f1_std':  float(df['f1'].std()),
        'auc':      float(df['auc'].mean()),       'auc_std': float(df['auc'].std()),
        'mcc':      float(df['mcc'].mean()),       'mcc_std': float(df['mcc'].std()),
        'n_sets_ok': len(set_rows),
        '_df': df,
    }


# ======================== TSTR% — train all syn, test all real ========================
def run_tstr_standard(X_real: np.ndarray, y_real: np.ndarray,
                      syn_root: str, subject_ids: list, tasks: list,
                      feat_cols: list, label_map: dict,
                      n_sets: int, clf_name: str, n_estimators: int,
                      use_aoi: bool, seed: int = 42) -> dict:
    """
    TSTR%: Train entirely on 2800 synthetic samples (20 sets x 70 subjects x 2 tasks),
    test on 70 real subjects the model has NEVER seen.

    -> The decisive criterion.
    Delta% = TSTR% - Base%: if positive -> synthetic even helps generalisation
                             better than real data.
    """
    # Load & concatenate all synthetic sets
    frames = []
    for i in range(n_sets):
        syn_df = load_syn_set(syn_root, subject_ids, tasks, i, use_aoi)
        if syn_df.empty:
            continue
        frames.append(syn_df)

    if not frames:
        return {}

    all_syn       = pd.concat(frames, ignore_index=True)
    X_train, y_train = build_Xy(all_syn, label_map, feat_cols)
    n_syn_sets    = len(frames)
    print(f"  TSTR train: {X_train.shape[0]} syn rows ({n_syn_sets} sets), "
          f"test: {X_real.shape[0]} real rows")

    if len(np.unique(y_train)) < 2:
        return {}

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_train)
    X_te_s  = scaler.transform(X_real)

    # MMD penalty — computed once on full train set
    mmd     = compute_mmd(X_tr_s, X_te_s)
    penalty = distribution_shift_penalty(mmd)
    print(f"  MMD={mmd:.4f}  penalty={penalty:.4f}")

    # Repeated bootstrap evaluation to obtain mean ± std
    rng_boot = np.random.default_rng(seed)
    rep_acc, rep_f1, rep_auc, rep_mcc = [], [], [], []
    best_clf   = None
    best_y_pred = None
    best_auc_val = -1.0

    for rep in range(N_TSTR_REPEATS):
        # Bootstrap sample from synthetic train set
        idx     = rng_boot.choice(len(X_tr_s), size=len(X_tr_s), replace=True)
        X_boot  = X_tr_s[idx]
        y_boot  = y_train[idx]

        clf = make_clf(clf_name, n_estimators,
                       seed=int(rng_boot.integers(1_000_000)))
        try:
            clf.fit(X_boot, y_boot)
        except Exception as exc:
            warnings.warn(f"[TSTR bootstrap rep {rep}] fit error: {exc}")
            continue

        y_pred = clf.predict(X_te_s)
        y_prob = (clf.predict_proba(X_te_s)[:, 1]
                  if hasattr(clf, 'predict_proba')
                  else y_pred.astype(float))

        a = float(accuracy_score(y_real, y_pred))
        f = float(f1_score(y_real, y_pred, average='binary', zero_division=0))
        u = float(roc_auc_score(y_real, y_prob))
        m = float(matthews_corrcoef(y_real, y_pred))

        rep_acc.append(a * (1 - penalty))
        rep_f1.append(f  * (1 - penalty))
        rep_auc.append(u * (1 - penalty))
        rep_mcc.append(m * (1 - penalty))

        if u > best_auc_val:
            best_auc_val = u
            best_clf     = (clf, scaler)
            best_y_pred  = y_pred

    if not rep_acc:
        return {}

    return {
        'accuracy':     float(np.mean(rep_acc)),
        'accuracy_std': float(np.std(rep_acc)),
        'f1':           float(np.mean(rep_f1)),
        'f1_std':       float(np.std(rep_f1)),
        'auc':          float(np.mean(rep_auc)),
        'auc_std':      float(np.std(rep_auc)),
        'mcc':          float(np.mean(rep_mcc)),
        'mcc_std':      float(np.std(rep_mcc)),
        'mmd':          mmd,
        'penalty':      penalty,
        'n_syn_sets':   n_syn_sets,
        'n_syn_rows':   int(X_train.shape[0]),
        'n_repeats':    len(rep_acc),
        '_clf':         best_clf,
        '_scaler':      scaler,
        '_y_pred':      best_y_pred,
    }


def plot_boxplot(scores_df: pd.DataFrame, output_root: str):
    metrics = ['accuracy', 'f1', 'auc', 'mcc']
    labels  = ['Accuracy', 'F1', 'AUC-ROC', 'MCC']
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle('TSTR Classification Scores across Synthetic Sets', fontsize=13)

    for ax, m, lab in zip(axes, metrics, labels):
        vals = scores_df[m].dropna().values
        bp = ax.boxplot(vals, patch_artist=True,
                        medianprops={'color': 'red', 'linewidth': 2})
        bp['boxes'][0].set_facecolor('steelblue')
        ax.set_title(lab)
        ax.set_ylabel(lab)
        ax.set_xticks([])
        mean_v = vals.mean()
        std_v  = vals.std()
        ax.axhline(mean_v, color='orange', linestyle='--', linewidth=1.5, label=f'Mean={mean_v:.3f}')
        ax.set_xlabel(f'μ={mean_v:.3f}  σ={std_v:.3f}')
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.05) if m != 'mcc' else ax.set_ylim(-1.05, 1.05)

    plt.tight_layout()
    path = os.path.join(output_root, 'tstr_boxplot.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → boxplot: {path}")

# ======================== PLOT CONFUSION MATRIX ========================
def plot_confusion(y_real: np.ndarray, y_pred: np.ndarray, output_root: str):
    cm = confusion_matrix(y_real, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Control (0)', 'Dyslexic (1)'])
    ax.set_yticklabels(['Control (0)', 'Dyslexic (1)'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix (best AUC set)')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_root, 'confusion_matrix.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → confusion: {path}")

# ======================== FEATURE IMPORTANCE ========================
def plot_feature_importance(clf_scaler, feat_cols: list, output_root: str):
    clf, _ = clf_scaler
    if not hasattr(clf, 'feature_importances_'):
        return
    imps = clf.feature_importances_
    idx  = np.argsort(imps)[::-1][:15]
    top_feats = [feat_cols[i] for i in idx]
    top_imps  = imps[idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(top_feats)), top_imps, color='steelblue')
    ax.set_xticks(range(len(top_feats)))
    ax.set_xticklabels(top_feats, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Importance')
    ax.set_title('Top Feature Importances (best AUC set)')
    plt.tight_layout()
    path = os.path.join(output_root, 'feature_importance.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → feature importance: {path}")

# ======================== FIGURE 4 — TSTR GROUPED BAR CHART ========================
# Colour palette consistent with compare_generators.py
_GA_GREEN  = "#2ca02c"
_REAL_BLUE = "#1f77b4"

def plot_tstr_grouped_bar(all_summaries: list, output_root: str,
                          metric: str = "accuracy", task_label: str = "both"):
    """
    Figure 4 — Grouped Bar Chart comparing Base% (Train on Real) vs TSTR% (Train on Synthetic/GA)
    per classifier.

    Params:
        all_summaries : list of dicts from the clf loop, each dict has
                        'classifier', 'base_accuracy', 'tstr_accuracy'
                        (and f1/auc/mcc similarly).
        metric        : 'accuracy' | 'f1' | 'auc' | 'mcc'
        task_label    : task label for the figure title
    """
    if not all_summaries:
        return

    m = metric.lower()
    clfs     = [r["classifier"] for r in all_summaries]
    base_vals = np.array([r.get(f"base_{m}", 0.0) for r in all_summaries]) * 100
    tstr_vals = np.array([r.get(f"tstr_{m}", 0.0) for r in all_summaries]) * 100

    n   = len(clfs)
    x   = np.arange(n)
    w   = 0.35

    fig, ax = plt.subplots(figsize=(max(8, n * 2), 6))
    bars_real = ax.bar(x - w / 2, base_vals, width=w,
                       color=_REAL_BLUE, alpha=0.85, label="Train on Real (Base%)",
                       edgecolor="white", linewidth=0.8)
    bars_syn  = ax.bar(x + w / 2, tstr_vals, width=w,
                       color=_GA_GREEN,  alpha=0.85, label="Train on Synthetic/GA (TSTR%)",
                       edgecolor="white", linewidth=0.8)

    # Annotate values above each bar
    for bar in bars_real:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.2f}%", ha="center", va="bottom", fontsize=8.5,
                color=_REAL_BLUE, fontweight="bold")
    for bar in bars_syn:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.2f}%", ha="center", va="bottom", fontsize=8.5,
                color=_GA_GREEN, fontweight="bold")

    # Delta annotation: arrow + delta value above the best column pair (TSTR > Base)
    for i, (bv, tv) in enumerate(zip(base_vals, tstr_vals)):
        delta = tv - bv
        x_mid = x[i] + w / 2
        y_top = max(bv, tv) + 3.5
        sign  = "▲" if delta >= 0 else "▼"
        color = _GA_GREEN if delta >= 0 else "#d62728"
        ax.annotate(f"{sign}{abs(delta):.2f}pp",
                    xy=(x_mid, y_top),
                    fontsize=8, ha="center", color=color, fontweight="bold")

    # Highlight 85% line if within range
    y_max_data = max(base_vals.max(), tstr_vals.max())
    if y_max_data > 80:
        ax.axhline(85, color="gray", linestyle=":", linewidth=1.0, alpha=0.6,
                   label="85% threshold")

    ax.set_xticks(x)
    ax.set_xticklabels(clfs, fontsize=11)
    ax.set_ylabel(f"{metric.upper()} (%)", fontsize=12)
    ax.set_title(
        f"Figure 4 — TSTR performance comparison: Real vs Synthetic/GA\n"
        f"({metric.upper()} — Task: {task_label})",
        fontsize=13,
    )
    ax.set_ylim(0, min(y_max_data + 15, 105))
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fname = f"fig4_tstr_grouped_bar_{m}.png"
    path  = os.path.join(output_root, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → fig4 grouped bar ({m.upper()}): {path}")


def plot_tstr_grouped_bar_all_metrics(all_summaries: list, output_root: str,
                                      task_label: str = "both"):
    """
    Plot 4 panels (Accuracy / F1 / AUC / MCC) combined into a single 2x2 figure.
    Also saves each metric individually for easy reference in the paper.
    """
    metrics = ["accuracy", "f1", "auc", "mcc"]
    labels  = ["Accuracy (%)", "F1-Score (%)", "AUC-ROC (%)", "MCC (scaled ×100)"]

    # ── Combined 2x2 figure ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes_flat = axes.flatten()

    for ax, m, lab in zip(axes_flat, metrics, labels):
        clfs      = [r["classifier"] for r in all_summaries]
        base_vals = np.array([r.get(f"base_{m}", 0.0) for r in all_summaries]) * 100
        tstr_vals = np.array([r.get(f"tstr_{m}", 0.0) for r in all_summaries]) * 100

        # MCC: preserve sign but scale *100 for consistent units
        if m == "mcc":
            base_vals = np.array([r.get(f"base_{m}", 0.0) for r in all_summaries]) * 100
            tstr_vals = np.array([r.get(f"tstr_{m}", 0.0) for r in all_summaries]) * 100

        n = len(clfs)
        x = np.arange(n)
        w = 0.35

        b1 = ax.bar(x - w / 2, base_vals, width=w, color=_REAL_BLUE, alpha=0.82,
                    label="Train on Real", edgecolor="white", linewidth=0.8)
        b2 = ax.bar(x + w / 2, tstr_vals, width=w, color=_GA_GREEN,  alpha=0.82,
                    label="Train on Synthetic/GA", edgecolor="white", linewidth=0.8)

        for bar in b1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=7.5,
                    color=_REAL_BLUE, fontweight="bold")
        for bar in b2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=7.5,
                    color=_GA_GREEN, fontweight="bold")

        for i, (bv, tv) in enumerate(zip(base_vals, tstr_vals)):
            delta = tv - bv
            sign  = "▲" if delta >= 0 else "▼"
            color = _GA_GREEN if delta >= 0 else "#d62728"
            y_ann = max(bv, tv) + 2.5
            ax.annotate(f"{sign}{abs(delta):.1f}pp",
                        xy=(x[i] + w / 2, y_ann),
                        fontsize=7, ha="center", color=color, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(clfs, fontsize=9.5)
        ax.set_ylabel(lab, fontsize=10)
        ax.set_title(m.upper(), fontsize=11, fontweight="bold")
        y_max = max(base_vals.max(), tstr_vals.max())
        ax.set_ylim(
            min(0, base_vals.min() - 10) if m == "mcc" else 0,
            min(y_max + 18, 110)
        )
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Figure 4 — TSTR performance comparison: Train on Real vs Train on Synthetic/GA\n"
        f"(Task: {task_label})",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    path_all = os.path.join(output_root, "fig4_tstr_all_metrics.png")
    plt.savefig(path_all, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → fig4 all metrics (2×2): {path_all}")

    # ── Individual metric plots ───────────────────────────────────────────────
    for m in metrics:
        plot_tstr_grouped_bar(all_summaries, output_root, metric=m,
                              task_label=task_label)


# ======================== MAIN ========================
def main():
    parser = argparse.ArgumentParser(
        description='evaluate_tstr.py — TSTR evaluation for dyslexia classification',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--syn_root', default='./ga2_tstr_output',
                        help='Directory containing synthetic metrics (output of ga2x20.py)\n'
                             'default: ./ga2_tstr_output')
    parser.add_argument('--data_dir', default='./data',
                        help='Directory containing real metrics (default: ./data)')
    parser.add_argument('--output_root', default='./tstr_results',
                        help='Output directory for results (default: ./tstr_results)')
    parser.add_argument('--n_sets', type=int, default=N_SYN_SETS,
                        help=f'Number of synthetic sets to evaluate (default: {N_SYN_SETS})')
    parser.add_argument('--classifier', default='rf',
                        choices=CLF_CHOICES_FULL,
                        help='Classifier type:\n'
                             '  rf       = Random Forest (default)\n'
                             '  svm      = SVM (RBF kernel)\n'
                             '  mlp      = MLP Neural Network (128→64→32)\n'
                             '  xgb      = XGBoost  [pip install xgboost]\n'
                             '  catboost = CatBoost  [pip install catboost]\n'
                             '  all      = Run ALL classifiers and compare results')
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of estimators for RF/GB/XGB/CatBoost (default: 100)')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds in Stratified K-Fold CV (default: 5)')
    parser.add_argument('--task', default='both',
                        choices=['T4', 'T5', 'both'],
                        help='Task to evaluate:\n'
                             '  both = T4 + T5 (default)\n'
                             '  T4   = T4_Meaningful_Text only\n'
                             '  T5   = T5_Pseudo_Text only')
    parser.add_argument('--use_aoi_features', action='store_true',
                        help='Include AOI-level features (mean/std across AOIs)\n'
                             'Default: trial-level features only')
    args = parser.parse_args()

    # ---- Select tasks ----
    if args.task == 'T4':
        tasks_used = ['T4_Meaningful_Text']
    elif args.task == 'T5':
        tasks_used = ['T5_Pseudo_Text']
    else:
        tasks_used = TASKS

    # ---- Load labels ----
    label_path = os.path.join(args.data_dir, 'dyslexia_class_label.csv')
    if not os.path.exists(label_path):
        print(f"[ERROR] File not found: {label_path}")
        sys.exit(1)
    label_df    = pd.read_csv(label_path)
    subject_ids = [str(s) for s in label_df['subject_id'].tolist()]

    # label_map: sid_str → 0/1
    # Find label column
    label_col = None
    for c in ['class_id', 'label', 'dyslexia', 'class', 'diagnosis', 'target']:
        if c in label_df.columns:
            label_col = c
            break
    if label_col is None:
        # Use last column if standard name not found
        label_col = [c for c in label_df.columns if c != 'subject_id'][-1]
        print(f"[WARN] Standard label column not found, using: '{label_col}'")

    label_map = {}
    for _, r in label_df.iterrows():
        label_map[str(r['subject_id'])] = normalize_binary_label(r[label_col])
    n_dyslexic = sum(v == 1 for v in label_map.values())
    n_control  = sum(v == 0 for v in label_map.values())
    print(f"[INFO] Labels: {n_dyslexic} dyslexic / {n_control} control | "
          f"total {len(label_map)} subjects")

    # ---- Load real metrics ----
    print(f"\n[INFO] Loading real metrics from {args.data_dir}...")
    real_df = load_real_metrics(args.data_dir, subject_ids, tasks_used, args.use_aoi_features)
    print(f"  -> {len(real_df)} rows (subjects x tasks)")

    # Identify feature columns
    excl_cols = {'sid', 'task', 'y', 'stimfile', 'eye_used', 'trialid',
             'subject_id', label_col}

    # Exclude saccade-duration features: inference in syn does not match real
    EXCLUDE_FEATURES = {
        'mean_sacc_dur_trial',
        'sum_sacc_dur_trial',
        'n_transit_trial',
    }

    feat_cols = [c for c in real_df.columns
                if c not in excl_cols
                and c not in EXCLUDE_FEATURES
                and pd.api.types.is_numeric_dtype(real_df[c])]
    print(f"  -> {len(feat_cols)} features: {feat_cols}")

    if not feat_cols:
        print("[ERROR] No features found. Check data_dir.")
        sys.exit(1)

    X_real, y_real = build_Xy(real_df, label_map, feat_cols)
    print(f"  -> X_real: {X_real.shape}  y: {np.bincount(y_real)}")

    # ---- Verify syn_root ----
    if not os.path.isdir(args.syn_root):
        print(f"[ERROR] syn_root not found: {args.syn_root}")
        print("        Please run ga2x20.py (or dcmx20.py / sggx20.py) first.")
        sys.exit(1)

    os.makedirs(args.output_root, exist_ok=True)

    # ---- Determine which classifiers to run ----
    if args.classifier == 'all':
        clfs_to_run = ['rf', 'svm', 'mlp']
        if _HAS_XGB:
            clfs_to_run.append('xgb')
        else:
            print(f"[WARN] XGBoost not installed (pip install xgboost) — skipping")
        if _HAS_CAT:
            clfs_to_run.append('catboost')
        else:
            print(f"[WARN] CatBoost not installed (pip install catboost) — skipping")
    else:
        clfs_to_run = [args.classifier]

    multi_mode = (len(clfs_to_run) > 1)
    all_summaries = []

    # ---- Loop over classifiers ----
    for clf_name in clfs_to_run:
        sep = '=' * 70
        print(f"\n{sep}")
        print(f"  [START] {clf_name.upper()} — tasks: {tasks_used} — {args.n_folds}-fold CV")
        print(f"{sep}")

        # ── 1. BASE% ──────────────────────────────────────────────────────────
        print(f"\n[1/3] BASE%  ({args.n_folds}-fold CV on {X_real.shape[0]} real samples)")
        base = run_base(X_real, y_real, clf_name, args.n_estimators,
                        n_folds=args.n_folds, seed=42)
        if not base:
            print(f"[WARN] {clf_name.upper()}: Base% failed.")
            continue
        print(f"       Acc={base['accuracy']:.4f}±{base['acc_std']:.4f}  "
              f"F1={base['f1']:.4f}±{base['f1_std']:.4f}  "
              f"AUC={base['auc']:.4f}±{base['auc_std']:.4f}  "
              f"MCC={base['mcc']:.4f}±{base['mcc_std']:.4f}")

        # ── 2. CVSyn% ─────────────────────────────────────────────────────────
        print(f"\n[2/3] CVSyn%  ({args.n_folds}-fold CV on each syn set, "
              f"avg over {args.n_sets} sets)")
        cvsyn = run_cv_syn(args.syn_root, subject_ids, tasks_used,
                           label_map, feat_cols, args.use_aoi_features,
                           clf_name, args.n_estimators, args.n_sets,
                           n_folds=args.n_folds, seed=42)
        if not cvsyn:
            print(f"[WARN] {clf_name.upper()}: CVSyn% failed — check syn_root.")
            cvsyn = {'accuracy': 0, 'f1': 0, 'auc': 0.5, 'mcc': 0,
                     'acc_std': 0, 'f1_std': 0, 'auc_std': 0, 'mcc_std': 0,
                     'n_sets_ok': 0}
        else:
            print(f"       Acc={cvsyn['accuracy']:.4f}±{cvsyn['acc_std']:.4f}  "
                  f"F1={cvsyn['f1']:.4f}±{cvsyn['f1_std']:.4f}  "
                  f"AUC={cvsyn['auc']:.4f}±{cvsyn['auc_std']:.4f}  "
                  f"MCC={cvsyn['mcc']:.4f}±{cvsyn['mcc_std']:.4f}  "
                  f"(n_sets={cvsyn.get('n_sets_ok', 0)})")

        # ── 3. TSTR% ──────────────────────────────────────────────────────────
        print(f"\n[3/3] TSTR%  (train {args.n_sets} syn sets -> test {X_real.shape[0]} real)")
        tstr = run_tstr_standard(X_real, y_real, args.syn_root, subject_ids,
                                 tasks_used, feat_cols, label_map,
                                 args.n_sets, clf_name, args.n_estimators,
                                 args.use_aoi_features, seed=42)
        if not tstr:
            print(f"[WARN] {clf_name.upper()}: TSTR% failed.")
            tstr = {'accuracy': 0, 'f1': 0, 'auc': 0.5, 'mcc': 0,
                    'n_syn_sets': 0, 'n_syn_rows': 0}
        else:
            print(f"       Acc={tstr['accuracy']:.4f}±{tstr.get('accuracy_std',0):.4f}  "
                  f"F1={tstr['f1']:.4f}±{tstr.get('f1_std',0):.4f}  "
                  f"AUC={tstr['auc']:.4f}±{tstr.get('auc_std',0):.4f}  "
                  f"MCC={tstr['mcc']:.4f}±{tstr.get('mcc_std',0):.4f}  "
                  f"(train rows={tstr['n_syn_rows']}, repeats={tstr.get('n_repeats',1)})")

        # ── Retain% & Delta% ──────────────────────────────────────────────────
        metrics_show = ['accuracy', 'f1', 'auc', 'mcc']
        retain = {m: (cvsyn[m] / base[m] * 100 if base[m] > 0 else 0.0)
                  for m in metrics_show}
        delta  = {m: (tstr[m]  - base[m]) * 100
                  for m in metrics_show}

        # ── Summary table ──
        print(f"\n{'─' * 70}")
        print(f"  SUMMARY — {clf_name.upper()}")
        print(f"{'─' * 70}")
        hdr = f"  {'Metric':<10}  {'Base%':>9}  {'CVSyn%':>9}  {'TSTR%':>9}  "
        hdr += f"{'Retain%':>9}  {'Delta%':>9}"
        print(hdr)
        print(f"  {'─'*10}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}")
        for m in metrics_show:
            _std_key = {'accuracy': 'acc_std', 'f1': 'f1_std',
                        'auc': 'auc_std', 'mcc': 'mcc_std'}[m]
            b_s = f"{base[m]*100:.2f}±{base[_std_key]*100:.2f}"
            c_s = f"{cvsyn[m]*100:.2f}±{cvsyn.get(_std_key, 0)*100:.2f}"
            t_s = f"{tstr[m]*100:.2f}"
            r_s = f"{retain[m]:.1f}%"
            d_s = f"{delta[m]:+.2f}pp"
            print(f"  {m.upper():<10}  {b_s:>9}  {c_s:>9}  {t_s:>9}  {r_s:>9}  {d_s:>9}")
        print(f"{'─' * 70}")
        print(f"  Retain%  = CVSyn% / Base% × 100  (how much classification signal is retained)")
        print(f"  Delta%   = TSTR% − Base%          (+ = synthetic helps generalisation)")
        print(f"{'─' * 70}")

        # ── Output directory ──
        clf_out = os.path.join(args.output_root, clf_name) if multi_mode \
                  else args.output_root
        os.makedirs(clf_out, exist_ok=True)

        # Save results table to CSV
        rows_out = []
        _std_map = {'accuracy': 'acc_std', 'f1': 'f1_std',
                    'auc': 'auc_std', 'mcc': 'mcc_std'}
        for m in metrics_show:
            _sk = _std_map[m]
            rows_out.append({
                'classifier':  clf_name.upper(), 'metric': m.upper(),
                'base_mean':   round(base[m], 4),
                'base_std':    round(base[_sk], 4),
                'cvsyn_mean':  round(cvsyn[m], 4),
                'cvsyn_std':   round(cvsyn.get(_sk, 0), 4),
                'tstr_mean':   round(tstr[m], 4),
                'tstr_std':    round(tstr.get(f'{m}_std' if m != 'accuracy'
                                              else 'accuracy_std', 0), 4),
                'retain_pct':  round(retain[m], 2),
                'delta_pp':    round(delta[m], 2),
                'n_repeats':   tstr.get('n_repeats', 1),
            })
        summary_df = pd.DataFrame(rows_out)
        summary_path = os.path.join(clf_out, 'evaluation_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\n  → evaluation summary: {summary_path}")

        # Confusion matrix & feature importance from TSTR (train all syn, test real)
        print("\n[PLOT] Generating plots...")
        if tstr.get('_y_pred') is not None:
            plot_confusion(y_real, tstr['_y_pred'], clf_out)
        if tstr.get('_clf') is not None:
            plot_feature_importance((tstr['_clf'], tstr['_scaler']),
                                    feat_cols, clf_out)

        # CVSyn boxplot (if per-set data available)
        if '_df' in cvsyn and not cvsyn['_df'].empty:
            plot_boxplot(cvsyn['_df'], clf_out)

        # Classification report (TSTR best model on real data)
        if tstr.get('_y_pred') is not None and not multi_mode:
            print("\nClassification Report (TSTR — best model on real data):")
            print(classification_report(y_real, tstr['_y_pred'],
                                        target_names=['Control', 'Dyslexic']))

        # Collect for multi-classifier comparison table
        row = {'classifier': clf_name.upper()}
        for m in metrics_show:
            row[f'base_{m}']    = round(base[m], 4)
            row[f'cvsyn_{m}']   = round(cvsyn[m], 4)
            row[f'tstr_{m}']    = round(tstr[m], 4)
            row[f'retain_{m}']  = round(retain[m], 2)
            row[f'delta_{m}']   = round(delta[m], 2)
        all_summaries.append(row)

    # ── Multi-classifier comparison table ──
    if multi_mode and all_summaries:
        comp_df   = pd.DataFrame(all_summaries)
        comp_path = os.path.join(args.output_root, 'comparison_all_classifiers.csv')
        comp_df.to_csv(comp_path, index=False)

        sep = '=' * 88
        print(f"\n{sep}")
        print(f"  ALL-CLASSIFIER COMPARISON (train {args.n_sets} syn sets, "
              f"test {X_real.shape[0]} real)")
        print(f"{sep}")
        _col_metrics = ['accuracy', 'f1', 'auc', 'mcc']
        _col_labels  = ['Acc', 'F1', 'AUC', 'MCC']
        for _m, _lab in zip(_col_metrics, _col_labels):
            print(f"\n  ── {_lab} ──")
            hdr = (f"  {'Classifier':<12}  {'Base%':>9}  {'CVSyn%':>9}  "
                   f"{'TSTR%':>9}  {'Retain%':>9}  {'Delta pp':>9}")
            print(hdr)
            print(f"  {'─'*12}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}")
            for r in all_summaries:
                print(f"  {r['classifier']:<12}  "
                      f"{r[f'base_{_m}']*100:>8.2f}%  "
                      f"{r[f'cvsyn_{_m}']*100:>8.2f}%  "
                      f"{r[f'tstr_{_m}']*100:>8.2f}%  "
                      f"{r[f'retain_{_m}']:>8.1f}%  "
                      f"{r[f'delta_{_m}']:>+8.2f}pp")
        print(f"\n{sep}")
        print(f"\n  → comparison table: {comp_path}")

        # ── Figure 4: TSTR Grouped Bar (draw when >= 1 classifier) ─────────
        print("\n[PLOT] Figure 4 — TSTR Grouped Bar Chart...")
        task_label = args.task if args.task != "both" else "T4 + T5"
        plot_tstr_grouped_bar_all_metrics(all_summaries, args.output_root,
                                          task_label=task_label)

    elif len(all_summaries) == 1:
        # Single-classifier mode: grouped bar for Accuracy
        print("\n[PLOT] Figure 4 — TSTR Grouped Bar (single classifier)...")
        task_label = args.task if args.task != "both" else "T4 + T5"
        plot_tstr_grouped_bar(all_summaries, args.output_root,
                              metric="accuracy", task_label=task_label)

    if not all_summaries:
        print("[ERROR] No classifier succeeded. Check syn_root.")
        sys.exit(1)

    print(f"\nDone! Results saved to: {args.output_root}/")


if __name__ == '__main__':
    main()
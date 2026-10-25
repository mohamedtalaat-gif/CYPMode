# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""CV-fold-internal, MCC-optimized decision thresholds -- per isoform, not
a single threshold shared across CYP3A4 and CYP2D6, and never touching a
validation/test fold's own labels to pick the threshold used on it.

For each outer fold: an inner K-fold runs entirely inside the training
portion to get inner-OOF probabilities there, a threshold is swept and
frozen from those inner-OOF probabilities (a property of the training
portion only), the real model is fit on the full training portion, and
that frozen threshold is applied to the untouched outer validation fold
for scoring. The validation fold's labels are used only to report the
resulting metrics, never to choose the threshold.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold

DEFAULT_THRESHOLD_GRID = np.linspace(0.05, 0.95, 19)


def find_optimal_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, thresholds: np.ndarray = DEFAULT_THRESHOLD_GRID
) -> tuple[float, float]:
    """Sweeps `thresholds`, returns (best_threshold, best_mcc) on the given
    (y_true, y_proba) -- callers are responsible for only passing data this
    threshold is allowed to see (see module docstring).
    """
    best_threshold, best_mcc = 0.5, -1.0
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        mcc = matthews_corrcoef(y_true, pred)
        if mcc > best_mcc:
            best_mcc, best_threshold = mcc, float(t)
    return best_threshold, best_mcc


def _inner_oof_threshold(
    features: np.ndarray, labels: np.ndarray, inner_splits: int, random_state: int
) -> float:
    """The training-portion-only threshold: an inner K-fold produces
    inner-OOF probabilities across the whole training portion (every
    training compound scored by a model that didn't train on it), and the
    threshold is swept on those -- still zero information from the outer
    validation fold.
    """
    inner_oof_proba = np.full(len(labels), np.nan)
    inner_kfold = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
    for inner_train_idx, inner_val_idx in inner_kfold.split(features, labels):
        model = LGBMClassifier(verbose=-1, random_state=random_state)
        model.fit(features[inner_train_idx], labels[inner_train_idx])
        inner_oof_proba[inner_val_idx] = model.predict_proba(features[inner_val_idx])[:, 1]
    threshold, _ = find_optimal_threshold(labels, inner_oof_proba)
    return threshold


def cross_validate_with_threshold_tuning(
    df: pd.DataFrame,
    label_col: str,
    feature_col: str = "descriptors",
    n_splits: int = 5,
    inner_splits: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Outer StratifiedKFold with a fold-internal tuned threshold per fold,
    for one isoform's label column. Returns one row per outer fold with
    both the naive-0.5 and tuned-threshold metrics, so the effect of
    tuning is directly visible, not just the tuned number in isolation.
    """
    cyp_df = df.dropna(subset=[label_col]).reset_index(drop=True)
    y = cyp_df[label_col].astype(int).to_numpy()
    features = np.stack(cyp_df[feature_col])

    rows = []
    outer_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(outer_kfold.split(features, y)):
        train_features, train_labels = features[train_idx], y[train_idx]
        val_features, val_labels = features[val_idx], y[val_idx]

        tuned_threshold = _inner_oof_threshold(train_features, train_labels, inner_splits, random_state)

        model = LGBMClassifier(verbose=-1, random_state=random_state)
        model.fit(train_features, train_labels)
        val_proba = model.predict_proba(val_features)[:, 1]

        naive_pred = (val_proba >= 0.5).astype(int)
        tuned_pred = (val_proba >= tuned_threshold).astype(int)

        rows.append(
            {
                "fold": fold,
                "n_val": len(val_idx),
                "tuned_threshold": tuned_threshold,
                "mcc_naive_0.5": matthews_corrcoef(val_labels, naive_pred),
                "mcc_tuned": matthews_corrcoef(val_labels, tuned_pred),
                "accuracy_tuned": accuracy_score(val_labels, tuned_pred),
                "f1_tuned": f1_score(val_labels, tuned_pred, zero_division=0),
                "roc_auc": roc_auc_score(val_labels, val_proba),
            }
        )
    return pd.DataFrame(rows)


DEFAULT_BASE_RATE_GRID = np.linspace(0.05, 0.50, 10)


def _mcc_from_confusion(tp: float, fn: float, fp: float, tn: float) -> float:
    """MCC from (possibly fractional) confusion-matrix counts -- fractional
    because find_robust_threshold uses this on expected counts under an
    assumed base rate, not on an integer sample. 0.0 by convention when any
    marginal is zero (matching sklearn's own MCC edge-case behavior),
    rather than raising or returning NaN.
    """
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return (tp * tn - fp * fn) / denom


def _roc_points(y_true: np.ndarray, y_proba: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    """TPR and FPR at each threshold -- the two base-rate-independent
    quantities a classifier's operating point is summarized by. Everything
    base-rate-specific (confusion counts, MCC) is reconstructed from these
    plus an assumed base rate, not recomputed from raw labels each time.
    """
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    rows = []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        tpr = ((pred == 1) & (y_true == 1)).sum() / n_pos if n_pos > 0 else 0.0
        fpr = ((pred == 1) & (y_true == 0)).sum() / n_neg if n_neg > 0 else 0.0
        rows.append({"threshold": float(t), "tpr": tpr, "fpr": fpr})
    return pd.DataFrame(rows)


def find_robust_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    base_rates: np.ndarray = DEFAULT_BASE_RATE_GRID,
    thresholds: np.ndarray = DEFAULT_THRESHOLD_GRID,
    criterion: str = "worst_case",
) -> tuple[float, pd.DataFrame]:
    """Picks a threshold from its TPR/FPR curve (computed on y_true/y_proba
    only -- callers are responsible for only passing training-portion data,
    same rule as find_optimal_threshold) that stays good across a whole
    *assumed range* of test base rates, instead of the single base rate
    find_optimal_threshold implicitly assumes (whatever y_true's own rate
    happens to be). A single MCC-optimal threshold is base-rate-specific,
    and a train/test base-rate mismatch can silently cost real MCC.

    For each candidate threshold, expected confusion counts are computed
    analytically at every base rate in `base_rates` (TP=pi*TPR,
    FN=pi*(1-TPR), FP=(1-pi)*FPR, TN=(1-pi)*(1-FPR)), converted to MCC, and
    aggregated across the base-rate grid: "worst_case" (this threshold's
    minimum MCC over the grid -- maximized, a minimax choice that's robust
    if the test base rate turns out to be anywhere in that range) or
    "mean" (average MCC over the grid). Returns the chosen threshold and
    the full per-threshold, per-criterion table for inspection.
    """
    roc = _roc_points(y_true, y_proba, thresholds)
    rows = []
    for _, r in roc.iterrows():
        mccs = [_mcc_from_confusion(pi * r.tpr, pi * (1 - r.tpr), (1 - pi) * r.fpr, (1 - pi) * (1 - r.fpr)) for pi in base_rates]
        rows.append({"threshold": r.threshold, "worst_case_mcc": min(mccs), "mean_mcc": float(np.mean(mccs))})
    summary = pd.DataFrame(rows)
    score_col = "worst_case_mcc" if criterion == "worst_case" else "mean_mcc"
    best_threshold = float(summary.loc[summary[score_col].idxmax(), "threshold"])
    return best_threshold, summary


def _resample_to_base_rate(y_true: np.ndarray, y_proba: np.ndarray, target_rate: float, rng: np.random.Generator):
    """Real base-rate-shift stress test, not simulated: undersamples the
    majority class of an actual (y_true, y_proba) pair (e.g. a real outer
    validation fold) so the resulting subset's positive rate matches
    target_rate as closely as the available counts allow, keeping every
    compound's real label and real predicted probability untouched.
    """
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    if n_pos == 0 or n_neg == 0:
        return y_true, y_proba
    # solve for the largest (n_pos', n_neg') <= (n_pos, n_neg) with n_pos'/(n_pos'+n_neg')==target_rate
    n_pos_keep = min(n_pos, round(target_rate / (1 - target_rate) * n_neg))
    n_neg_keep = min(n_neg, round((1 - target_rate) / target_rate * n_pos_keep)) if n_pos_keep > 0 else 0
    if n_pos_keep < 1 or n_neg_keep < 1:
        return y_true, y_proba
    keep = np.concatenate(
        [rng.choice(pos_idx, size=n_pos_keep, replace=False), rng.choice(neg_idx, size=n_neg_keep, replace=False)]
    )
    return y_true[keep], y_proba[keep]


def base_rate_shift_stress_test(
    y_val: np.ndarray,
    proba_val: np.ndarray,
    naive_threshold: float,
    robust_threshold: float,
    target_rates: np.ndarray = DEFAULT_BASE_RATE_GRID,
    random_state: int = 42,
) -> pd.DataFrame:
    """The check that find_robust_threshold actually buys something:
    resamples a REAL validation fold's (label, predicted-probability) pairs
    to a range of target base rates (see _resample_to_base_rate), and
    scores both the naive MCC-optimal threshold and the robust threshold
    -- both frozen beforehand from training-portion data only -- on each
    shifted subset. If the robust threshold is doing its job, its MCC
    should degrade less than the naive threshold's as the shifted rate
    moves away from y_val's own natural rate.
    """
    rng = np.random.default_rng(random_state)
    rows = []
    for target_rate in target_rates:
        y_shifted, proba_shifted = _resample_to_base_rate(y_val, proba_val, target_rate, rng)
        if len(y_shifted) == 0 or len(set(y_shifted)) < 2:
            continue
        naive_pred = (proba_shifted >= naive_threshold).astype(int)
        robust_pred = (proba_shifted >= robust_threshold).astype(int)
        rows.append(
            {
                "target_base_rate": float(target_rate),
                "n": len(y_shifted),
                "mcc_naive_threshold": matthews_corrcoef(y_shifted, naive_pred),
                "mcc_robust_threshold": matthews_corrcoef(y_shifted, robust_pred),
            }
        )
    return pd.DataFrame(rows)


def cross_validate_with_robust_threshold_tuning(
    df: pd.DataFrame,
    label_col: str,
    feature_col: str = "descriptors",
    n_splits: int = 5,
    inner_splits: int = 3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same fold-internal protocol as cross_validate_with_threshold_tuning,
    but tunes both a naive MCC-optimal threshold and a robust
    (find_robust_threshold) one per fold from the same inner-OOF
    probabilities, then reports (a) both thresholds' MCC on the real,
    unshifted outer validation fold, and (b) a base-rate-shift stress test
    on that same fold (see base_rate_shift_stress_test) -- so the tradeoff
    (does robust cost accuracy at the natural rate, does it actually help
    under a shift) is visible, not asserted.
    """
    cyp_df = df.dropna(subset=[label_col]).reset_index(drop=True)
    y = cyp_df[label_col].astype(int).to_numpy()
    features = np.stack(cyp_df[feature_col])

    fold_rows, stress_rows = [], []
    outer_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(outer_kfold.split(features, y)):
        train_features, train_labels = features[train_idx], y[train_idx]
        val_features, val_labels = features[val_idx], y[val_idx]

        inner_oof_proba = np.full(len(train_labels), np.nan)
        inner_kfold = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
        for inner_train_idx, inner_val_idx in inner_kfold.split(train_features, train_labels):
            model = LGBMClassifier(verbose=-1, random_state=random_state)
            model.fit(train_features[inner_train_idx], train_labels[inner_train_idx])
            inner_oof_proba[inner_val_idx] = model.predict_proba(train_features[inner_val_idx])[:, 1]
        naive_threshold, _ = find_optimal_threshold(train_labels, inner_oof_proba)
        robust_threshold, _ = find_robust_threshold(train_labels, inner_oof_proba)

        model = LGBMClassifier(verbose=-1, random_state=random_state)
        model.fit(train_features, train_labels)
        val_proba = model.predict_proba(val_features)[:, 1]

        naive_pred = (val_proba >= naive_threshold).astype(int)
        robust_pred = (val_proba >= robust_threshold).astype(int)
        fold_rows.append(
            {
                "fold": fold,
                "n_val": len(val_idx),
                "naive_threshold": naive_threshold,
                "robust_threshold": robust_threshold,
                "mcc_naive_threshold": matthews_corrcoef(val_labels, naive_pred),
                "mcc_robust_threshold": matthews_corrcoef(val_labels, robust_pred),
            }
        )

        stress = base_rate_shift_stress_test(val_labels, val_proba, naive_threshold, robust_threshold, random_state=random_state)
        stress.insert(0, "fold", fold)
        stress_rows.append(stress)

    return pd.DataFrame(fold_rows), pd.concat(stress_rows, ignore_index=True)


def _main() -> None:
    from cypmode.challenge.tdi_ablation import add_ablation_features
    from cypmode.challenge.tdi_baseline import (
        TDI_ISOFORMS,
        compute_descriptors,
        load_tdi_training_data,
    )

    df = add_ablation_features(compute_descriptors(load_tdi_training_data()))

    all_results = []
    for cyp in TDI_ISOFORMS:
        results = cross_validate_with_threshold_tuning(df, label_col=f"{cyp}_is_TDI", feature_col="m3_no_amine_features")
        results.insert(0, "isoform", cyp)
        all_results.append(results)
        print(f"{cyp}: mean MCC naive=0.5 -> {results['mcc_naive_0.5'].mean():.4f}, "
              f"tuned -> {results['mcc_tuned'].mean():.4f} "
              f"(mean tuned threshold={results['tuned_threshold'].mean():.3f})")

    combined = pd.concat(all_results, ignore_index=True)
    out_path = Path("data/tdi_track/outputs/comparisons/threshold_tuning.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")

    print("\n-- robust (base-rate-shift-aware) threshold, same protocol --")
    all_robust, all_stress = [], []
    for cyp in TDI_ISOFORMS:
        fold_results, stress_results = cross_validate_with_robust_threshold_tuning(df, label_col=f"{cyp}_is_TDI", feature_col="m3_no_amine_features")
        fold_results.insert(0, "isoform", cyp)
        stress_results.insert(0, "isoform", cyp)
        all_robust.append(fold_results)
        all_stress.append(stress_results)
        print(f"{cyp}: at natural rate -- naive-tuned MCC {fold_results['mcc_naive_threshold'].mean():.4f}, "
              f"robust-tuned MCC {fold_results['mcc_robust_threshold'].mean():.4f}")

    robust_combined = pd.concat(all_robust, ignore_index=True)
    stress_combined = pd.concat(all_stress, ignore_index=True)
    robust_out_path = Path("data/tdi_track/outputs/comparisons/robust_threshold_tuning.csv")
    stress_out_path = Path("data/tdi_track/outputs/comparisons/robust_threshold_stress_test.csv")
    robust_combined.to_csv(robust_out_path, index=False)
    stress_combined.to_csv(stress_out_path, index=False)
    print(f"wrote {robust_out_path}\nwrote {stress_out_path}")

    print("\n-- stress test: mean MCC by shifted base rate (naive vs robust) --")
    stress_summary = stress_combined.groupby(["isoform", "target_base_rate"])[["mcc_naive_threshold", "mcc_robust_threshold"]].mean()
    print(stress_summary.to_string())


if __name__ == "__main__":
    _main()

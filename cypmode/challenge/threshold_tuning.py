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


def _main() -> None:
    from cypmode.challenge.tdi_ablation import add_ablation_features
    from cypmode.challenge.tdi_baseline import TDI_ISOFORMS, compute_descriptors, load_tdi_training_data

    df = add_ablation_features(compute_descriptors(load_tdi_training_data()))

    all_results = []
    for cyp in TDI_ISOFORMS:
        results = cross_validate_with_threshold_tuning(df, label_col=f"{cyp}_is_TDI", feature_col="m3_features")
        results.insert(0, "isoform", cyp)
        all_results.append(results)
        print(f"{cyp}: mean MCC naive=0.5 -> {results['mcc_naive_0.5'].mean():.4f}, "
              f"tuned -> {results['mcc_tuned'].mean():.4f} "
              f"(mean tuned threshold={results['tuned_threshold'].mean():.3f})")

    combined = pd.concat(all_results, ignore_index=True)
    out_path = Path("data/cyp_challenge/tdi_track/threshold_tuning.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    _main()

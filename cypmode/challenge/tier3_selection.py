"""Fold-local Tier 3 selection mask: decides which TDI compounds are worth
the expense of a real Boltz-2 structural run, without ever letting a
validation fold's own labels influence that decision.

Originally specified as "motif-positive OR 2D-model-uncertain". Checked
against the real data first: motif-positive alone already covers ~42% of
the labeled CYP3A4 population (1,501/3,584), so that OR clause selects
46-72% of the dataset regardless of the uncertainty quantile chosen --
thousands of compounds, against a real observed Boltz-2 cost of
15-30+ minutes each. Not tractable, and redundant besides: the motif flag
is already a free Tier 2 feature, so spending an expensive structural run
on a compound the model already "knows about" via that flag doesn't add
information it doesn't already have.

Revised policy: pure uncertainty-based selection, no motif OR clause, and
a hard fixed budget (top-N most uncertain compounds) rather than an
open-ended quantile -- N is chosen to fit the real, observed compute
budget, not a fraction of an unknown-size dataset.

The uncertainty threshold itself is derived from the training fold only
(a quantile of *training* predictions) and then applied, frozen, to select
compounds from both the training and validation portions -- the
validation fold's own predictions are used only to see who a
train-fold-only policy would have picked, never to set that policy.
"""


import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

from cypmode.metrics.motifs import detect_motifs


def add_motif_selection_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["motif_selected"] = out["SMILES"].apply(lambda s: bool(detect_motifs(s)))
    return out


def uncertainty_selection_mask(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    apply_features: np.ndarray,
    quantile: float = 0.5,
    random_state: int = 42,
) -> tuple[np.ndarray, float]:
    """Fits a classifier on (train_features, train_labels) only, uses ITS
    OWN training predictions to pick an |p - 0.5| distance threshold at
    the given quantile (the most-uncertain `quantile` fraction of the
    training fold), then applies that frozen threshold to apply_features.

    Returns (selected_mask, threshold) -- the threshold is a training-fold
    property, returned so callers/tests can confirm it never used
    apply_features' own labels.
    """
    model = LGBMClassifier(verbose=-1, random_state=random_state)
    model.fit(train_features, train_labels)

    train_proba = model.predict_proba(train_features)[:, 1]
    train_distance_from_boundary = np.abs(train_proba - 0.5)
    threshold = np.quantile(train_distance_from_boundary, quantile)

    apply_proba = model.predict_proba(apply_features)[:, 1]
    apply_distance = np.abs(apply_proba - 0.5)
    selected = apply_distance <= threshold
    return selected, float(threshold)


def select_fixed_budget_oof(
    df: pd.DataFrame,
    label_col: str,
    feature_col: str = "descriptors",
    budget: int = 50,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Ranks every labeled compound by out-of-fold uncertainty
    (|OOF proba - 0.5|, ascending) and returns the `budget` most uncertain
    rows -- each compound's own uncertainty score comes from a model that
    never saw it (same OOF discipline as cypmode.challenge.oof_bridge), so
    this is a real, leakage-safe ranking over the whole labeled population.

    This is the practical alternative to running fold-local selection 5
    separate times: Boltz-2's real per-compound cost (15-30+ minutes,
    confirmed directly this session) makes a single, whole-dataset budget
    the only tractable option, not a per-fold repeat over mostly the same
    compounds.
    """
    out = df.dropna(subset=[label_col]).reset_index(drop=True)
    y = out[label_col].astype(int)
    features = np.stack(out[feature_col])

    oof_proba = np.full(len(out), np.nan)
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kfold.split(out, y):
        model = LGBMClassifier(verbose=-1, random_state=random_state)
        model.fit(features[train_idx], y.iloc[train_idx])
        oof_proba[test_idx] = model.predict_proba(features[test_idx])[:, 1]

    out["oof_proba"] = oof_proba
    out["oof_uncertainty"] = np.abs(out["oof_proba"] - 0.5)
    return out.nsmallest(budget, "oof_uncertainty").reset_index(drop=True)


def compute_selection(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    label_col: str,
    feature_col: str = "descriptors",
    uncertainty_quantile: float = 0.5,
    random_state: int = 42,
) -> pd.DataFrame:
    """The full Tier 3 selection for one fold: motif OR train-fold-derived
    uncertainty. `train_mask` marks which rows of `df` are this fold's
    training portion (the only rows the uncertainty model/threshold sees).
    """
    out = add_motif_selection_flag(df)

    features = np.stack(out[feature_col])
    labels = out[label_col].astype(int).to_numpy()

    uncertainty_selected, threshold = uncertainty_selection_mask(
        train_features=features[train_mask],
        train_labels=labels[train_mask],
        apply_features=features,
        quantile=uncertainty_quantile,
        random_state=random_state,
    )
    out["uncertainty_selected"] = uncertainty_selected
    out["uncertainty_threshold"] = threshold
    out["tier3_selected"] = out["motif_selected"] | out["uncertainty_selected"]
    return out

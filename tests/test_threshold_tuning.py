import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from cypmode.challenge.threshold_tuning import (
    _inner_oof_threshold,
    cross_validate_with_threshold_tuning,
    find_optimal_threshold,
)


def test_find_optimal_threshold_picks_the_best_mcc():
    # a perfectly separable score: threshold anywhere between 0.4 and 0.6
    # gives perfect MCC=1.0
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    threshold, mcc = find_optimal_threshold(y_true, y_proba)
    assert mcc == pytest.approx(1.0)
    assert 0.3 < threshold < 0.7


def test_inner_oof_threshold_is_a_training_only_property():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(80, 4))
    labels = (features.sum(axis=1) > 0).astype(int)

    threshold_a = _inner_oof_threshold(features, labels, inner_splits=3, random_state=42)
    threshold_b = _inner_oof_threshold(features, labels, inner_splits=3, random_state=42)
    assert threshold_a == threshold_b  # reproducible from training data alone


def _synthetic_df(n: int = 150, n_features: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    descriptors = [rng.normal(size=n_features) for _ in range(n)]
    labels = [bool(d.sum() > 0) for d in descriptors]
    return pd.DataFrame({"descriptors": descriptors, "label": labels})


def test_cross_validate_with_threshold_tuning_covers_every_labeled_compound_once():
    df = _synthetic_df()
    results = cross_validate_with_threshold_tuning(df, label_col="label", n_splits=5, inner_splits=3)
    assert results["n_val"].sum() == len(df)
    assert len(results) == 5


def test_outer_fold_threshold_matches_a_direct_training_only_call():
    """End-to-end wiring check: fold 0's tuned threshold from
    cross_validate_with_threshold_tuning must equal calling
    _inner_oof_threshold directly on exactly that fold's training split
    -- proving by construction that the outer loop passes it only the
    training portion, never anything from the validation fold."""
    from sklearn.model_selection import StratifiedKFold

    df = _synthetic_df()
    y = df["label"].astype(int).to_numpy()
    features = np.stack(df["descriptors"])

    outer_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, _ = next(outer_kfold.split(features, y))
    expected_threshold = _inner_oof_threshold(features[train_idx], y[train_idx], inner_splits=3, random_state=42)

    results = cross_validate_with_threshold_tuning(df, label_col="label", n_splits=5, inner_splits=3)
    assert results.loc[0, "tuned_threshold"] == expected_threshold

# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from cypmode.challenge.threshold_tuning import (
    _inner_oof_threshold,
    _mcc_from_confusion,
    _resample_to_base_rate,
    _roc_points,
    base_rate_shift_stress_test,
    cross_validate_with_robust_threshold_tuning,
    cross_validate_with_threshold_tuning,
    find_optimal_threshold,
    find_robust_threshold,
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


def test_mcc_from_confusion_is_perfect_for_a_perfect_classifier():
    # tp=0.2, fn=0, fp=0, tn=0.8 -- no false positives or negatives at all
    assert _mcc_from_confusion(tp=0.2, fn=0.0, fp=0.0, tn=0.8) == pytest.approx(1.0)


def test_mcc_from_confusion_is_zero_when_a_marginal_is_zero():
    # predicting everything positive: fn=tn=0, denominator is zero
    assert _mcc_from_confusion(tp=0.3, fn=0.0, fp=0.7, tn=0.0) == 0.0


def test_roc_points_recovers_known_tpr_fpr_at_a_clean_separator():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    roc = _roc_points(y_true, y_proba, thresholds=np.array([0.5]))
    assert roc.loc[0, "tpr"] == pytest.approx(1.0)
    assert roc.loc[0, "fpr"] == pytest.approx(0.0)


def test_find_robust_threshold_is_reproducible_and_returns_a_valid_summary():
    rng = np.random.default_rng(0)
    y_true = (rng.normal(size=200) > 0).astype(int)
    y_proba = np.clip(y_true + rng.normal(scale=0.3, size=200), 0, 1)

    threshold_a, summary_a = find_robust_threshold(y_true, y_proba)
    threshold_b, _ = find_robust_threshold(y_true, y_proba)
    assert threshold_a == threshold_b  # deterministic given the same inputs
    assert {"threshold", "worst_case_mcc", "mean_mcc"} <= set(summary_a.columns)
    assert 0.0 <= threshold_a <= 1.0


def test_resample_to_base_rate_hits_the_target_rate_closely():
    rng = np.random.default_rng(0)
    y_true = np.array([1] * 50 + [0] * 200)  # natural rate ~0.2
    y_proba = rng.uniform(size=len(y_true))

    y_shifted, _ = _resample_to_base_rate(y_true, y_proba, target_rate=0.4, rng=np.random.default_rng(1))
    actual_rate = y_shifted.mean()
    assert actual_rate == pytest.approx(0.4, abs=0.02)


def test_base_rate_shift_stress_test_covers_every_requested_rate():
    rng = np.random.default_rng(0)
    y_val = (rng.normal(size=300) > 0.3).astype(int)  # imbalanced, plenty of both classes
    proba_val = np.clip(y_val + rng.normal(scale=0.3, size=300), 0, 1)

    result = base_rate_shift_stress_test(
        y_val, proba_val, naive_threshold=0.5, robust_threshold=0.4, target_rates=np.array([0.1, 0.3, 0.5])
    )
    assert set(result["target_base_rate"]) == {0.1, 0.3, 0.5}
    assert {"mcc_naive_threshold", "mcc_robust_threshold"} <= set(result.columns)


def test_cross_validate_with_robust_threshold_tuning_covers_every_labeled_compound_once():
    df = _synthetic_df()
    fold_results, stress_results = cross_validate_with_robust_threshold_tuning(
        df, label_col="label", n_splits=5, inner_splits=3
    )
    assert fold_results["n_val"].sum() == len(df)
    assert len(fold_results) == 5
    assert not stress_results.empty


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

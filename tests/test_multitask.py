import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

# Real, reproducible segfault otherwise: torch and lightgbm loaded in the
# same process crash (OpenMP runtime conflict, confirmed directly on this
# machine) -- run this file in its own pytest invocation, never alongside
# the rest of the suite. See pyproject.toml's `torch` marker.
pytestmark = pytest.mark.torch

import torch
from sklearn.model_selection import KFold

from cypmode.challenge.multitask import (
    ISOFORMS,
    SharedTrunkModel,
    _find_optimal_threshold,
    _inner_oof_threshold_shared_trunk,
    cross_validate_shared_trunk,
    cross_validate_shared_trunk_with_threshold_tuning,
    masked_bce_loss,
    predict_proba,
    train_shared_trunk,
)


def test_shared_trunk_model_outputs_one_logit_per_isoform():
    model = SharedTrunkModel(input_dim=5)
    x = torch.randn(3, 5)
    out = model(x)
    assert set(out.keys()) == set(ISOFORMS)
    for cyp in ISOFORMS:
        assert out[cyp].shape == (3,)


def test_masked_bce_loss_ignores_unlabeled_examples():
    logits = {"CYP3A4": torch.tensor([10.0, -10.0]), "CYP2D6": torch.tensor([0.0, 0.0])}
    targets = {"CYP3A4": torch.tensor([1.0, 0.0]), "CYP2D6": torch.tensor([1.0, 0.0])}

    # CYP3A4 predictions are perfect; CYP2D6 fully masked out
    mask_a = {"CYP3A4": torch.tensor([1.0, 1.0]), "CYP2D6": torch.tensor([0.0, 0.0])}
    loss_a = masked_bce_loss(logits, targets, mask_a)

    # same but CYP2D6 unmasked too, with bad predictions -- loss should increase
    mask_b = {"CYP3A4": torch.tensor([1.0, 1.0]), "CYP2D6": torch.tensor([1.0, 1.0])}
    loss_b = masked_bce_loss(logits, targets, mask_b)

    assert loss_a.item() < 0.01  # near-perfect CYP3A4 predictions, CYP2D6 excluded
    assert loss_b.item() > loss_a.item()  # including bad CYP2D6 predictions raises it


def _synthetic_df(n: int = 200, n_features: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        desc = rng.normal(size=n_features)
        row = {"m3_features": desc}
        # sparse labels: most rows get only one isoform, matching the real data shape
        if i % 3 != 0:
            row["CYP3A4_is_TDI"] = bool(desc[0] > 0)
        else:
            row["CYP3A4_is_TDI"] = np.nan
        if i % 3 != 1:
            row["CYP2D6_is_TDI"] = bool(desc[1] > 0)
        else:
            row["CYP2D6_is_TDI"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def test_train_shared_trunk_runs_and_produces_finite_predictions():
    df = _synthetic_df()
    X = np.stack(df["m3_features"]).astype(np.float32)
    y = {cyp: df[f"{cyp}_is_TDI"].fillna(0).astype(int).to_numpy().astype(np.float32) for cyp in ISOFORMS}
    mask = {cyp: df[f"{cyp}_is_TDI"].notna().to_numpy().astype(np.float32) for cyp in ISOFORMS}

    model = train_shared_trunk(X, y, mask, epochs=20)
    proba = predict_proba(model, X)

    for cyp in ISOFORMS:
        assert np.isfinite(proba[cyp]).all()
        assert (proba[cyp] >= 0).all() and (proba[cyp] <= 1).all()


def test_cross_validate_shared_trunk_covers_both_isoforms():
    df = _synthetic_df()
    results = cross_validate_shared_trunk(df, n_splits=3, epochs=20)
    assert set(results["isoform"]) == set(ISOFORMS)
    assert results["mcc"].notna().any()


def test_cross_validate_shared_trunk_n_val_matches_labeled_counts():
    df = _synthetic_df()
    results = cross_validate_shared_trunk(df, n_splits=3, epochs=20)
    for cyp in ISOFORMS:
        expected = df[f"{cyp}_is_TDI"].notna().sum()
        actual = results[results["isoform"] == cyp]["n_val"].sum()
        assert actual == expected


def test_find_optimal_threshold_picks_the_best_mcc():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    threshold, mcc = _find_optimal_threshold(y_true, y_proba)
    assert mcc == pytest.approx(1.0)
    assert 0.3 < threshold < 0.7


def test_inner_oof_threshold_shared_trunk_is_a_training_only_property():
    df = _synthetic_df()
    X = np.stack(df["m3_features"]).astype(np.float32)
    y = {cyp: df[f"{cyp}_is_TDI"].fillna(0).astype(int).to_numpy().astype(np.float32) for cyp in ISOFORMS}
    mask = {cyp: df[f"{cyp}_is_TDI"].notna().to_numpy().astype(np.float32) for cyp in ISOFORMS}

    thresholds_a = _inner_oof_threshold_shared_trunk(X, y, mask, inner_splits=3, epochs=20, random_state=42)
    thresholds_b = _inner_oof_threshold_shared_trunk(X, y, mask, inner_splits=3, epochs=20, random_state=42)
    assert thresholds_a == thresholds_b  # reproducible from training data alone


def test_cross_validate_shared_trunk_with_threshold_tuning_covers_both_isoforms():
    df = _synthetic_df()
    results = cross_validate_shared_trunk_with_threshold_tuning(df, n_splits=3, inner_splits=3, epochs=20)
    assert set(results["isoform"]) == set(ISOFORMS)
    for cyp in ISOFORMS:
        expected = df[f"{cyp}_is_TDI"].notna().sum()
        actual = results[results["isoform"] == cyp]["n_val"].sum()
        assert actual == expected


def test_outer_fold_threshold_matches_a_direct_training_only_call():
    """Wiring check, same spirit as test_threshold_tuning's equivalent:
    fold 0's tuned threshold from cross_validate_shared_trunk_with_threshold_tuning
    must equal calling _inner_oof_threshold_shared_trunk directly on exactly
    that fold's training split -- proving the outer loop never lets the
    threshold see the validation fold's labels."""
    df = _synthetic_df()
    X = np.stack(df["m3_features"]).astype(np.float32)
    y = {cyp: df[f"{cyp}_is_TDI"].fillna(0).astype(int).to_numpy().astype(np.float32) for cyp in ISOFORMS}
    mask = {cyp: df[f"{cyp}_is_TDI"].notna().to_numpy().astype(np.float32) for cyp in ISOFORMS}

    outer_kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    train_idx, _ = next(outer_kfold.split(X))
    expected_thresholds = _inner_oof_threshold_shared_trunk(
        X[train_idx],
        {cyp: y[cyp][train_idx] for cyp in ISOFORMS},
        {cyp: mask[cyp][train_idx] for cyp in ISOFORMS},
        inner_splits=3,
        epochs=20,
        random_state=42,
    )

    results = cross_validate_shared_trunk_with_threshold_tuning(df, n_splits=3, inner_splits=3, epochs=20)
    for cyp in ISOFORMS:
        fold0 = results[(results["isoform"] == cyp) & (results["fold"] == 0)]
        assert fold0["tuned_threshold"].iloc[0] == expected_thresholds[cyp]

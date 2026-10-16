"""Regression guard for the standardization leak found in the 2026-09-26 audit.

cross_validate_shared_trunk and its siblings once computed the feature mean/std
over the FULL dataset before the fold loop, so every validation row's own
values leaked into the statistic used to scale it. That was fixed, but nothing
in the suite would have noticed it coming back. These tests record every
(rows, mean) pair handed to ``multitask._standardize`` and require the mean to
be the mean of the very rows being fitted on -- never of data that includes the
validation fold.

Run with the rest of the torch-marked tests, in their own pytest invocation
(see the `torch` marker in pyproject.toml).
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

pytestmark = pytest.mark.torch

from cypmode.challenge import multitask
from cypmode.challenge.multitask import (
    cross_validate_multi_branch_shared_trunk,
    cross_validate_shared_trunk,
    cross_validate_shared_trunk_with_threshold_tuning,
)


def _df(n: int = 180, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        desc = rng.normal(size=6) * (1 + i / 60)  # drifting scale so fold means genuinely differ
        rows.append(
            {
                "m3_no_amine_features": desc,
                "chemberta_embedding": rng.normal(size=8) + i / 90,
                "CYP3A4_is_TDI": bool(desc[0] > 0) if i % 3 else np.nan,
                "CYP2D6_is_TDI": bool(desc[1] > 0) if i % 3 != 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def recorded(monkeypatch):
    calls = []
    original = multitask._standardize

    def recorder(x, mean, std):
        calls.append((np.array(x, copy=True), np.array(mean, copy=True)))
        return original(x, mean, std)

    monkeypatch.setattr(multitask, "_standardize", recorder)
    return calls


def _groups(calls):
    """Consecutive calls sharing one mean array = (training rows, then the rows scaled with them)."""
    groups, current = [], [calls[0]]
    for call in calls[1:]:
        if np.array_equal(call[1], current[0][1]):
            current.append(call)
        else:
            groups.append(current)
            current = [call]
    groups.append(current)
    return groups


def assert_statistics_come_from_the_rows_they_scale(calls):
    assert calls, "_standardize was never called -- the test would pass vacuously"
    groups = _groups(calls)
    saw_validation_rows = False
    for group in groups:
        train_rows, mean = group[0]
        np.testing.assert_allclose(mean, train_rows.mean(axis=0), rtol=1e-5, atol=1e-6)
        for rows, _ in group[1:]:
            saw_validation_rows = True
            assert not np.allclose(mean, rows.mean(axis=0), rtol=1e-3, atol=1e-4), "mean looks computed from the rows it scales"
    assert saw_validation_rows


def test_the_detector_flags_full_dataset_statistics():
    rng = np.random.default_rng(0)
    train, validation = rng.normal(size=(60, 4)), rng.normal(size=(30, 4)) + 2.0
    full_mean = np.vstack([train, validation]).mean(axis=0)
    leaky_calls = [(train, full_mean), (validation, full_mean)]
    with pytest.raises(AssertionError):
        assert_statistics_come_from_the_rows_they_scale(leaky_calls)
    honest_calls = [(train, train.mean(axis=0)), (validation, train.mean(axis=0))]
    assert_statistics_come_from_the_rows_they_scale(honest_calls)


def test_shared_trunk_cv_scales_each_fold_with_its_own_training_statistics(recorded):
    cross_validate_shared_trunk(_df(), n_splits=3, epochs=2)
    assert_statistics_come_from_the_rows_they_scale(recorded)


def test_threshold_tuned_cv_scales_outer_and_inner_folds_with_training_statistics(recorded):
    cross_validate_shared_trunk_with_threshold_tuning(_df(), n_splits=3, inner_splits=3, epochs=2)
    assert_statistics_come_from_the_rows_they_scale(recorded)


def test_multi_branch_cv_scales_both_representations_with_training_statistics(recorded):
    cross_validate_multi_branch_shared_trunk(_df(), n_splits=3, epochs=2)
    assert_statistics_come_from_the_rows_they_scale(recorded)

# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The label-shift estimate of the positive rate: it recovers known rates on synthetic scores, it moves the rate to the scored compounds correctly, and its verdict follows the rule written in the module."""

import numpy as np
import pytest

from cypmode.challenge import label_shift as ls


def synthetic_folds(true_rate: float, n_folds: int = 6, n: int = 400, n_test: int = 600, shift: float = 1.2, seed: int = 0) -> list[ls.Fold]:
    """Scores of a classifier that separates the classes by ``shift`` standard deviations, held-out compounds at a rate of 0.3 and blinded compounds at ``true_rate``."""
    rng = np.random.default_rng(seed)
    folds = []
    for _ in range(n_folds):
        y = (rng.random(n) < 0.3).astype(int)
        score = rng.normal(0, 1, n) + shift * y
        z = np.where(rng.random(n) < 0.8, 2.0, 0.2)  # the same confident share in both classes
        group = rng.integers(0, 60, n)
        test_y = rng.random(n_test) < true_rate
        test_score = rng.normal(0, 1, n_test) + shift * test_y
        folds.append(ls.Fold(score, y, np.ones(n), z, group, test_score))
    return folds


@pytest.mark.parametrize("true_rate", [0.1, 0.3, 0.5])
def test_the_rate_of_a_shifted_test_set_is_recovered_without_its_labels(true_rate):
    folds = synthetic_folds(true_rate)
    summary = ls.summarize(folds, 1.0, 200)
    estimate, lo, hi = summary["all"]
    assert abs(estimate - true_rate) < 0.04 and lo <= true_rate <= hi
    assert abs(summary["scored"][0] - true_rate) < 0.05  # equal confident shares leave the rate unchanged


def test_the_rate_moves_to_the_scored_compounds_by_the_confident_share_of_each_class():
    assert ls.to_scored(0.3, 0.5, 0.5) == pytest.approx(0.3)
    assert ls.to_scored(0.5, 0.9, 0.3) == pytest.approx(0.75)
    assert ls.to_scored(0.0, 0.9, 0.3) == 0.0


def test_a_classifier_that_does_not_separate_the_classes_gives_no_estimate():
    assert np.isnan(ls.adjusted_rate(0.30, 0.31, 0.3))
    assert np.isnan(ls.adjusted_rate(float("nan"), 0.5, 0.3))


def test_the_confident_shares_use_the_weights():
    y = np.array([1, 1, 0, 0])
    z = np.array([2.0, 0.1, 2.0, 2.0])
    assert ls.confident_shares(y, np.array([1.0, 1.0, 1.0, 1.0]), z, 1.0) == (0.5, 1.0)
    assert ls.confident_shares(y, np.array([3.0, 1.0, 1.0, 1.0]), z, 1.0)[0] == pytest.approx(0.75)


def test_known_rates_are_recovered_by_the_split_of_the_held_out_compounds():
    result = ls.recovery(synthetic_folds(0.3), (0.1, 0.3, 0.5), 1.0, draws=10)
    assert ls.recovery_ratio(result) < ls.RECOVERY_LIMIT
    assert all(abs(routes["scored"][0]) < 0.06 for routes in result.values())


def test_the_verdict_follows_the_rule():
    assert ls.verdict(0.31, (0.28, 0.35), 0.1)[0] == "AGREES"
    assert ls.verdict(0.31, (0.10, 0.20), 0.1)[0] == "DISAGREES"
    assert ls.verdict(0.31, (0.10, 0.60), 0.1)[0] == "NOT INFORMATIVE"  # wider than half the rate
    assert ls.verdict(0.31, (0.28, 0.35), 0.4)[0] == "NOT INFORMATIVE"  # the recovery test failed
    assert ls.verdict(0.31, (float("nan"), 0.35), 0.1)[0] == "NOT INFORMATIVE"
    assert not ls.verdict(0.31, (0.28, 0.35), 0.1)[0].startswith(("MET", "NOT MET"))  # never read as a gate by the ledger

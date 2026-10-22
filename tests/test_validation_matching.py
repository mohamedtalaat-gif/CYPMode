# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

from itertools import pairwise

import numpy as np
import pytest

from cypmode.challenge.validation_matching import (
    nearest_neighbor_similarity,
    prior_weights,
    similarity_weights,
    tanimoto_matrix,
    weighted_auc,
    weighted_mcc,
)


def test_tanimoto_matrix_matches_the_definition_on_a_small_example():
    a = np.array([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=np.float32)
    s = tanimoto_matrix(a)
    assert s[0, 0] == pytest.approx(1.0)
    assert s[0, 1] == pytest.approx(1 / 3)  # one shared bit, three bits in the union
    assert np.allclose(s, s.T)


def test_nearest_neighbor_similarity_ignores_the_row_itself_when_asked():
    pool = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    with_self = nearest_neighbor_similarity(pool, pool)
    without_self = nearest_neighbor_similarity(pool, pool, exclude_self=True)
    assert np.allclose(with_self, 1.0)
    assert without_self[0] == pytest.approx(2 / 3)  # nearest *other* compound
    assert without_self[2] == pytest.approx(0.0)


def test_similarity_weights_reshape_the_validation_distribution_to_the_test_distribution():
    rng = np.random.default_rng(0)
    validation = rng.uniform(0.2, 0.7, 5000)  # mostly distant from training
    test = rng.uniform(0.4, 0.9, 5000)  # mostly close
    bins = np.array([0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01])
    w = similarity_weights(validation, test, bins)
    weighted_hist = np.array([w[(validation >= lo) & (validation < hi)].sum() for lo, hi in pairwise(bins)])
    weighted_hist /= weighted_hist.sum()
    test_hist = np.histogram(test, bins)[0] / len(test)
    covered = np.histogram(validation, bins)[0] > 0
    assert np.allclose(weighted_hist[covered] / weighted_hist[covered].sum(), test_hist[covered] / test_hist[covered].sum(), atol=0.02)
    assert (w[validation < 0.3] == 0).all()  # a bin the test set never reaches carries no weight


def test_similarity_weights_are_all_ones_when_the_distributions_already_match():
    x = np.random.default_rng(1).uniform(0.2, 0.9, 4000)
    assert np.allclose(similarity_weights(x, x), 1.0)


def test_prior_weights_move_the_weighted_positive_rate_to_the_target():
    y = np.array([1] * 30 + [0] * 70)
    w = prior_weights(y, 0.10)
    assert (w * y).sum() / w.sum() == pytest.approx(0.10)
    with pytest.raises(ValueError):
        prior_weights(np.ones(5), 0.1)


def test_weighted_mcc_reduces_to_plain_mcc_and_responds_to_weights():
    y = np.array([1, 1, 0, 0, 0, 1])
    pred = np.array([1, 0, 0, 0, 1, 1])
    plain = weighted_mcc(y, pred)
    assert plain == pytest.approx(weighted_mcc(y, pred, np.ones(6)))
    assert weighted_mcc(y, y) == pytest.approx(1.0)
    down_weighted_errors = weighted_mcc(y, pred, np.array([1, 0.01, 1, 1, 0.01, 1]))
    assert down_weighted_errors > plain
    assert weighted_mcc(np.array([0, 0, 0]), np.array([0, 0, 0])) == 0.0  # degenerate: no positives


def test_weighted_auc_uses_the_weights():
    y = np.array([1, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.2, 0.1])
    assert weighted_auc(y, score) == pytest.approx(0.75)
    assert weighted_auc(y, score, np.array([1, 1, 0.01, 1])) > 0.9  # the mis-ranked positive barely counts


def test_similarity_weights_fall_back_to_uniform_when_the_profiles_do_not_overlap():
    w = similarity_weights(np.full(50, 0.35), np.full(50, 0.85))
    assert np.array_equal(w, np.ones(50))

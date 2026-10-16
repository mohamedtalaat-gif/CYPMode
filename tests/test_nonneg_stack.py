import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from cypmode.challenge.nested import StackedLogit
from cypmode.challenge.nonneg_stack import NonNegativeStack, _NonNegativeLogit


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    good = y + rng.normal(size=n) * 1.0
    helper = y + rng.normal(size=n) * 1.5
    noise = rng.normal(size=n)
    return np.column_stack([good, helper, noise]), y


def test_every_coefficient_is_non_negative_and_the_intercept_is_free():
    x, y = _data()
    inverted = -(y + np.random.default_rng(1).normal(size=len(y)))  # an informative column pointing the wrong way: a free blend subtracts it
    x = np.column_stack([x, inverted])
    model = _NonNegativeLogit("l2", 1.0).fit(x, y)
    assert (model.coef_ >= -1e-12).all()
    free = LogisticRegression(C=1.0).fit(x, y)
    assert (free.coef_ < 0).any()  # the unconstrained fit does invert a column
    assert model.coef_[0, 3] == pytest.approx(0.0, abs=1e-6)


def test_it_matches_the_unconstrained_fit_when_that_fit_is_already_non_negative():
    x, y = _data()
    constrained = _NonNegativeLogit("l2", 1.0).fit(x[:, :2], y)
    free = LogisticRegression(C=1.0, tol=1e-10, max_iter=1000).fit(x[:, :2], y)
    assert np.allclose(constrained.coef_, free.coef_, atol=2e-3)
    assert np.allclose(constrained.intercept_, free.intercept_, atol=2e-3)


def test_a_strong_l1_penalty_switches_candidates_off():
    x, y = _data()
    weak, strong = _NonNegativeLogit("l1", 100.0).fit(x, y), _NonNegativeLogit("l1", 0.003).fit(x, y)
    assert strong.coef_.sum() < weak.coef_.sum() and strong.coef_[0, 2] == pytest.approx(0.0, abs=1e-9)


def test_sample_weights_move_the_fit():
    x, y = _data()
    heavy = np.where(y == 1, 5.0, 1.0)
    assert _NonNegativeLogit("l2", 1.0).fit(x, y, heavy).intercept_[0] > _NonNegativeLogit("l2", 1.0).fit(x, y).intercept_[0]


def test_the_selector_blends_without_negative_weights_and_ranks_like_the_logistic_blend_here():
    x, y = _data(600)
    names = ["good", "helper", "noise"]
    ones = np.ones(len(y))
    sel = NonNegativeStack().fit(names, x, y, ones, "auc")
    assert all(v >= 0 for v in sel.chosen.values())
    from sklearn.metrics import roc_auc_score

    base = StackedLogit().fit(names, x, y, ones, "auc")
    assert roc_auc_score(y, sel.predict(x)) == pytest.approx(roc_auc_score(y, base.predict(x)), abs=0.02)

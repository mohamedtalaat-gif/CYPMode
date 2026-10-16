import numpy as np
import pytest
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from cypmode.challenge.model_comparison import (
    compare_candidate_sets,
    corrected_resampled_t,
    naive_paired_t,
    plan_sizes,
)
from cypmode.challenge.nested import NestedPlan, PredictionCache


def test_corrected_variance_matches_the_nadeau_bengio_formula():
    d = np.array([0.02, 0.01, 0.03, 0.00, 0.02])
    result = corrected_resampled_t(d, n_train=800, n_test=200)
    assert result.standard_error == pytest.approx(np.sqrt((1 / 5 + 200 / 800) * d.var(ddof=1)))
    assert result.t == pytest.approx(d.mean() / result.standard_error)
    assert result.p_value == pytest.approx(2 * stats.t.sf(abs(result.t), df=4))
    assert result.n_better == 4 and result.n_folds == 5


def test_the_correction_always_makes_the_test_more_conservative_than_the_naive_one():
    d = np.array([0.0037, 0.0097, 0.0085, 0.0071, 0.004])  # per-fold AUC gains
    naive, corrected = naive_paired_t(d), corrected_resampled_t(d, 800, 200)
    assert naive.t == pytest.approx(d.mean() / (d.std(ddof=1) / np.sqrt(5)))
    assert corrected.p_value > naive.p_value and abs(corrected.t) < abs(naive.t)


def test_degenerate_inputs():
    assert corrected_resampled_t([0.0, 0.0, 0.0], 80, 20).p_value == 1.0
    assert corrected_resampled_t([0.1, 0.1, 0.1], 80, 20).significant
    with pytest.raises(ValueError):
        corrected_resampled_t([0.1], 80, 20)


def test_a_naive_t_over_repeated_cv_replicates_manufactures_significance_between_identical_models():
    """Two classifiers with identical information (each sees the signal feature plus a different set of pure-noise
    features), compared by 3x repeated 5-fold CV on fresh data many times. The 15 per-split differences are
    treated as independent replicates by the naive t-test, which then calls the models different about
    2-3x more often than its nominal 5%; the Nadeau-Bengio test does not. (With a single 5-fold CV the naive
    test is only mildly off -- the damage is done by pseudoreplication across repeats.)"""
    rng = np.random.default_rng(1)
    naive_hits = corrected_hits = 0
    trials, n = 150, 800
    for _ in range(trials):
        X = rng.normal(size=(n, 8))
        y = (X[:, 0] * 0.7 + rng.normal(size=n) > 0).astype(int)
        diffs = []
        for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=int(rng.integers(1 << 30))).split(X, y):
            a_cols, b_cols = [0, 1, 2, 3], [0, 4, 5, 6]
            a = LogisticRegression().fit(X[tr][:, a_cols], y[tr])
            b = LogisticRegression().fit(X[tr][:, b_cols], y[tr])
            diffs.append(roc_auc_score(y[te], a.decision_function(X[te][:, a_cols])) - roc_auc_score(y[te], b.decision_function(X[te][:, b_cols])))
        naive_hits += naive_paired_t(diffs).significant
        corrected_hits += corrected_resampled_t(diffs, n * 4 // 5, n // 5).significant
    assert naive_hits / trials > 0.09  # visibly above the nominal 5%
    assert corrected_hits / trials < 0.05  # the corrected test does not over-reject
    assert naive_hits > 3 * corrected_hits


def _cache(tmp_path, shift):
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.35).astype(int)
    plan = NestedPlan(np.arange(300), y, 1, 5, 3, 0)
    label = dict(zip(range(300), y.tolist(), strict=True))
    cache = PredictionCache(tmp_path, plan)
    for name, strength in (("weak", 0.4), ("strong", shift)):
        member = np.random.default_rng(len(name))
        for task in plan.tasks():
            lab = np.array([label[int(r)] for r in task.evaluate])
            cache.put(name, task, lab * strength + member.normal(size=len(lab)))
    return cache


def test_compare_candidate_sets_uses_whole_procedure_outer_scores_and_the_corrected_test(tmp_path):
    cache = _cache(tmp_path, shift=2.5)
    result = compare_candidate_sets(cache, ["strong"], ["weak"])
    assert result.mean_difference > 0.1 and result.n_folds == 5 and result.significant
    same = compare_candidate_sets(cache, ["weak"], ["weak"])
    assert same.mean_difference == 0 and same.p_value == 1.0
    assert plan_sizes(cache.plan) == (240, 60)

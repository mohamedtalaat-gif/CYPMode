# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The validation on the second data set: which candidates it runs, the files it reads, the barrier between the labels of evaluation compounds and the candidates, and the placement."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cypmode.challenge import external_dataset as ed
from cypmode.challenge.feature_store import OPT_IN_BLOCKS
from cypmode.challenge.nested import DEPLOY, PredictionCache
from cypmode.challenge.nested_runners import fit_predict


def test_the_candidates_read_neither_the_assay_arms_nor_a_block_trained_on_the_panel():
    candidates = ed.external_candidates()
    assert len(candidates) == 16
    assert not any(c.family in ("lgbm_reg_shift", "lgbm_reg_arms") for c in candidates)
    assert not any(set(c.blocks) & set(OPT_IN_BLOCKS) for c in candidates)


def test_the_benchmark_files_are_the_ones_whose_checksums_are_stored():
    ed.verify_files()
    for iso in ed.DATASETS:
        train_val, test = ed.load_benchmark(iso)
        assert {"Drug", "Y"} <= set(train_val.columns) and len(test) > 2000


def test_the_published_entries_are_19_for_each_isoform():
    board = pd.read_csv(ed.LEADERBOARD)
    assert board.groupby("dataset").size().to_dict() == {"CYP2D6_Veith": 19, "CYP3A4_Veith": 19}
    assert board.groupby("dataset")["rank"].apply(lambda r: sorted(r) == list(range(1, 20))).all()


def test_the_placement_counts_the_entries_above_the_mean_and_compares_with_the_median():
    board = pd.read_csv(ed.LEADERBOARD)
    median = float(board[board.dataset == "CYP3A4_Veith"].auprc_mean.median())
    above = ed.placement("CYP3A4", median + 0.001)
    below = ed.placement("CYP3A4", median - 0.001)
    assert above["verdict"] == "ABOVE THE MEDIAN" and below["verdict"] == "NOT ABOVE THE MEDIAN"
    assert above["rank_among_entries"] < below["rank_among_entries"]
    assert ed.placement("CYP3A4", 1.0)["rank_among_entries"] == 1


def test_the_official_split_is_the_one_of_the_benchmark():
    pytest.importorskip("tdc.utils")
    train_val, _ = ed.load_benchmark("CYP3A4")
    train, valid = ed.official_split(train_val, 1)
    assert (len(train), len(valid)) == (8628, 1233)
    assert set(train.Drug).isdisjoint(valid.Drug)
    assert len(set(train.Drug) | set(valid.Drug)) == len(train_val)


def test_a_label_of_an_evaluation_compound_never_reaches_a_candidate(tmp_path):
    pytest.importorskip("tdc.utils")
    bench = ed.Benchmark("CYP2D6", 2, store_path="", limit=200)
    plan, train_rows, valid_rows = ed.official_plan(bench, 1)
    candidates = [c for c in ed.external_candidates() if c.name in ("knn_tanimoto_k15", "lgbm_desc_shallow", "lgbm_A_deep_knnfeat")]
    ctx = bench.context(train_rows)
    assert (ctx.labels[valid_rows] == -1).all() and (ctx.labels[bench.test_rows] == -1).all()
    cache = PredictionCache(tmp_path, plan)
    task = plan.task(-1, 0, DEPLOY)
    first = {c.name: fit_predict(c, ctx, task) for c in candidates}
    corrupted = bench.y.copy()
    corrupted[valid_rows] = 1 - corrupted[valid_rows]
    corrupted[bench.test_rows] = np.random.default_rng(0).integers(0, 2, len(bench.test_rows))
    second = {c.name: fit_predict(c, bench.context(train_rows, corrupted), task) for c in candidates}
    for name, scores in first.items():
        assert np.array_equal(scores, second[name]), name
    assert cache.plan is plan


def test_the_scores_are_finite_and_the_metrics_follow_the_scikit_learn_definitions():
    y = np.array([0, 0, 1, 1, 0, 1])
    score = np.array([0.1, 0.3, 0.6, 0.8, 0.2, 0.9])
    out = ed.score_all(y, score, score >= 0.5)
    assert out["auc"] == pytest.approx(1.0) and out["auprc"] == pytest.approx(1.0)
    assert out["mcc"] == pytest.approx(1.0)
    swapped = ed.score_all(y, np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9]))
    assert swapped["auc"] == pytest.approx(8 / 9) and "mcc" not in swapped
    assert out["prevalence"] == pytest.approx(0.5)


class _EmptyCache:
    """A stand-in for the prediction cache that gives every inner task a zero score; only the rows of the inner tasks matter here."""

    def __init__(self, plan):
        self.plan = plan

    def get(self, name, task):
        return np.zeros(len(task.evaluate))


def _small_state(seed=0):
    rng = np.random.default_rng(seed)
    pool = np.arange(300)
    y = (rng.random(300) < 0.4).astype(int)
    plan = ed.NestedPlan(pool, y, 0, 3, 4, 7, external=np.arange(300, 700))
    cache = _EmptyCache(plan)
    labels = ed.inner_labels(plan, cache, ["a"])
    inner_scores = (2.0 * labels - 1.0) + rng.normal(0.0, 1.0, len(labels))
    y_test = (rng.random(400) < 0.4).astype(int)
    scores = (2.0 * y_test - 1.0) + rng.normal(0.0, 1.0, len(y_test))
    deployment = SimpleNamespace(selection=SimpleNamespace(inner_scores=inner_scores), scores=scores)
    bench = SimpleNamespace(iso="CYPX", y_test=y_test)
    return bench, {"plan": plan, "cache": cache, "names": ["a"], "deployment": deployment}


def test_the_labels_of_the_calibration_follow_the_order_of_the_inner_folds_not_the_order_of_the_pool():
    _, state = _small_state()
    plan = state["plan"]
    inner = ed.inner_labels(plan, state["cache"], state["names"])
    assert len(inner) == len(plan.pool) and inner.sum() == plan.y.sum()
    assert not np.array_equal(inner, plan.labels(plan.pool)), "the two orders must differ, or this test cannot tell them apart"


def test_the_isotonic_map_of_the_calibration_learns_from_scores_and_labels_of_the_same_rows():
    bench, state = _small_state()
    table = ed.calibration_on_test(bench, state).set_index("method")
    constant = float(np.mean(bench.y_test) * (1 - np.mean(bench.y_test)))
    assert table.loc["isotonic", "brier"] < 0.6 * constant, "an isotonic map that is almost constant has the Brier score of a constant prediction"
    assert 0.3 < table.loc["isotonic", "slope"] < 3.0

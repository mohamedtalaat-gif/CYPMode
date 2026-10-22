# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from cypmode.challenge import xiso_transfer as xt
from cypmode.challenge.feature_store import FeatureStore
from cypmode.challenge.nested import NestedPlan
from cypmode.challenge.nested_runners import (
    ISOFORM_ORDER,
    Candidate,
    Context,
    fit_predict,
    xiso_candidates,
)
from cypmode.challenge.validation_matching import tanimoto_matrix

PIECES = ["CCO", "CCN", "c1ccccc1", "c1ccncc1", "CC(=O)O", "C1CCCCC1", "c1ccoc1", "CCS", "c1ccsc1", "C1CCNCC1", "C1CCOC1"]
SMALL = {"n_estimators": 20, "num_leaves": 4, "min_child_samples": 3, "learning_rate": 0.1}
N_POOL = 70


@pytest.fixture(scope="module")
def store():
    smiles = [f"{a}C{b}" for a in PIECES for b in PIECES][:90]
    return FeatureStore.build(smiles, ["morgan_bits", "morgan_c2", "rdkit_desc"])


def _context(store, seed=0):
    rng = np.random.default_rng(seed)
    n = len(store.smiles)
    signal = store.blocks["rdkit_desc"][:, 0]
    y = (signal + rng.normal(size=n) * 0.5 > np.median(signal)).astype(int)
    labels = np.full(n, -1)
    labels[:N_POOL] = y[:N_POOL]
    arms = {}
    for j, iso in enumerate(ISOFORM_ORDER):
        for arm in ("tdi", "direct"):
            values = 5.0 + 0.3 * j + 0.5 * (signal - signal.mean()) / signal.std() + rng.normal(size=n) * 0.3
            values[rng.random(n) < 0.3] = np.nan  # unmeasured
            values[N_POOL + 10 :] = np.nan  # the blinded test structures have no measurement
            arms[f"{iso}_{arm}"] = values
    targets = {"shift": np.zeros(n), "tdi": np.zeros(n), "direct": np.zeros(n)}
    return Context(store, labels, targets, tanimoto_matrix(store.blocks["morgan_bits"]), n_jobs=2, isoform="CYP3A4", xiso_arms=arms)


def _plan(ctx):
    pool = np.arange(N_POOL)
    return NestedPlan(pool, ctx.labels[pool], 1, 3, 3, 0, external=np.arange(N_POOL, len(ctx.store.smiles)))


CANDIDATE = Candidate("xiso_test", "lgbm_xiso", ("rdkit_desc", "morgan_c2"), params=SMALL)


@pytest.fixture(autouse=True)
def _fresh_memo():
    xt._MEMO.clear()
    yield
    xt._MEMO.clear()


def test_scores_one_finite_value_per_evaluation_row_with_six_extra_features(store):
    ctx = _context(store)
    task = _plan(ctx).inner_tasks(0, 0)[0]
    scores = fit_predict(CANDIDATE, ctx, task)
    assert scores.shape == (len(task.evaluate),) and np.isfinite(scores).all()
    train_x, eval_x = xt.stage_one_features(ctx, task, 1)
    assert train_x.shape == (len(task.train), 6) == (len(task.train), len(xt.feature_names("CYP3A4"))) and eval_x.shape == (len(task.evaluate), 6)


def test_other_isoform_measurements_of_evaluation_and_held_out_rows_are_never_read(store):
    ctx = _context(store)
    plan = _plan(ctx)
    task = plan.inner_tasks(0, 0)[0]
    before = fit_predict(CANDIDATE, ctx, task)
    held_out = np.setdiff1d(np.arange(N_POOL), task.train)  # evaluation rows and the outer-test rows
    xt._MEMO.clear()
    corrupted = Context(store, ctx.labels.copy(), ctx.targets, ctx.similarity, ctx.n_jobs, isoform="CYP3A4", xiso_arms={k: v.copy() for k, v in ctx.xiso_arms.items()})
    for v in corrupted.xiso_arms.values():
        v[held_out] = 99.0
    assert np.allclose(before, fit_predict(CANDIDATE, corrupted, task))


def test_a_training_compound_never_receives_a_prediction_from_a_model_that_saw_its_own_measurement(store):
    ctx = _context(store)
    task = _plan(ctx).inner_tasks(0, 0)[0]
    key = "CYP1A2_tdi"
    row = next(int(r) for r in task.train if not np.isnan(ctx.xiso_arms[key][r]))
    position = int(np.flatnonzero(task.train == row)[0])
    column = xt.feature_names("CYP3A4").index("xiso_CYP1A2_tdi")
    before = xt.stage_one_features(ctx, task, 1)[0][position, column]
    xt._MEMO.clear()
    ctx.xiso_arms[key][row] = 12.0  # an extreme measurement of this very compound
    after = xt.stage_one_features(ctx, task, 1)[0][position, column]
    assert after == pytest.approx(before)


def test_allowed_rows_are_the_training_rows_and_everything_outside_the_pool():
    labels = np.array([1, 0, 1, 0, -1, -1])
    allowed = xt.allowed_rows(labels, np.array([0, 2]))
    assert allowed.tolist() == [True, False, True, False, True, True]


def test_target_isoform_arms_are_not_features():
    assert xt.other_isoforms("CYP2D6") == ["CYP1A2", "CYP2C9", "CYP3A4"]
    assert not any("CYP3A4" in name for name in xt.feature_names("CYP3A4"))


def test_correlation_table_reports_arm_and_shift_correlations_on_shared_compounds():
    rng = np.random.default_rng(0)
    base = rng.normal(size=200)
    frame = pd.DataFrame({
        "CYP3A4_pIC50_TDI_condition": base, "CYP3A4_pIC50_direct_inhibition": rng.normal(size=200),
        "CYP2D6_pIC50_TDI_condition": base + rng.normal(size=200) * 0.1, "CYP2D6_pIC50_direct_inhibition": rng.normal(size=200),
    })
    for iso in ("CYP1A2", "CYP2C9"):
        frame[f"{iso}_pIC50_TDI_condition"] = np.nan
        frame[f"{iso}_pIC50_direct_inhibition"] = np.nan
    table = xt.correlation_table(frame, "CYP3A4").set_index(["other", "quantity"])
    assert table.loc[("CYP2D6", "tdi"), "spearman"] > 0.9 and abs(table.loc[("CYP2D6", "direct"), "spearman"]) < 0.3
    assert table.loc[("CYP1A2", "tdi"), "n"] == 0


def test_candidates_cover_the_two_feature_sets_and_pair_with_their_plain_counterparts():
    assert [c.name for c in xiso_candidates()] == ["xiso_A_deep", "xiso_A+fps_deep"] and all(c.family == "lgbm_xiso" for c in xiso_candidates())
    assert xt.ALONE_PAIRS == [("lgbm_A_deep", "xiso_A_deep"), ("lgbm_A+fps_deep", "xiso_A+fps_deep")]

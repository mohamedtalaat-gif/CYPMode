import numpy as np
import pytest

pytest.importorskip("lightgbm")

from cypmode.challenge import aux_blocks as ab
from cypmode.challenge.feature_store import FeatureStore
from cypmode.challenge.nested import NestedPlan
from cypmode.challenge.nested_runners import Candidate, Context, fit_predict
from cypmode.challenge.official_extra import ENZYMES
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
    z = (signal - signal.mean()) / signal.std()
    labels = np.full(n, -1)
    labels[:N_POOL] = (z[:N_POOL] + rng.normal(size=N_POOL) * 0.5 > 0).astype(int)
    aux = {}
    for e in ENZYMES:
        screen = -1.0 - 0.8 * z + rng.normal(size=n) * 0.3
        screen[rng.random(n) < 0.25] = np.nan
        screen[N_POOL + 10 :] = np.nan  # blinded test compounds have no reading
        aux[f"{e}_screen"] = screen
        aux[f"{e}_screen_active"] = np.where(np.isnan(screen), np.nan, (screen < -0.5).astype(float))
        for arm in ("direct", "tdi"):
            emax = 40 + 20 * z + rng.normal(size=n) * 5
            emax[rng.random(n) < 0.25] = np.nan
            emax[N_POOL + 10 :] = np.nan
            aux[f"{e}_emax_{arm}"] = emax
    t = 4.0 + 0.8 * z + rng.normal(size=n) * 0.3
    d = 4.0 + 0.5 * z + rng.normal(size=n) * 0.3
    d[labels < 0] = np.nan  # compounds outside the pool have a TDI-arm curve but no direct one
    t[N_POOL + 10 :] = np.nan
    d[N_POOL + 10 :] = np.nan
    hit = np.zeros(n, dtype=np.int64)
    hit[:N_POOL] = (labels[:N_POOL] >= 0) & (z[:N_POOL] > -0.5)
    targets = {"shift": np.zeros(n), "tdi": t, "direct": d}
    return Context(store, labels, targets, tanimoto_matrix(store.blocks["morgan_bits"]), n_jobs=2, hit=hit, isoform="CYP2D6", aux=aux)


def _plan(ctx):
    pool = np.arange(N_POOL)
    return NestedPlan(pool, ctx.labels[pool], 1, 3, 3, 0, external=np.arange(N_POOL, len(ctx.store.smiles)))


BLOCKS = ("rdkit_desc", "morgan_c2")
CANDIDATES = [
    Candidate("aux_screen", "lgbm_aux", BLOCKS, params={**SMALL, "aux": "screen"}),
    Candidate("aux_all", "lgbm_aux", BLOCKS, params={**SMALL, "aux": "auxall"}),
    Candidate("gs_product", "lgbm_gate_shift", BLOCKS, params={**SMALL, "part": "product"}),
    Candidate("scr2", "lgbm_screen_two", BLOCKS, params={**SMALL, "enzyme": "CYP2D6"}),
    Candidate("ndneg", "lgbm_nodirect", BLOCKS, params={**SMALL, "mode": "neg", "extra_weight": 1.0}),
    Candidate("ndsoft", "lgbm_nodirect", BLOCKS, params={**SMALL, "mode": "soft", "extra_weight": 0.5}),
]


@pytest.fixture(autouse=True)
def _fresh_memo():
    ab._MEMO.clear()
    yield
    ab._MEMO.clear()


@pytest.mark.parametrize("candidate", CANDIDATES, ids=lambda c: c.name)
def test_one_finite_score_per_evaluation_row(store, candidate):
    ctx = _context(store)
    task = _plan(ctx).inner_tasks(0, 0)[0]
    scores = fit_predict(candidate, ctx, task)
    assert scores.shape == (len(task.evaluate),) and np.isfinite(scores).all()


def test_the_predicted_readings_have_one_column_per_reading(store):
    ctx = _context(store)
    task = _plan(ctx).inner_tasks(0, 0)[0]
    train_col, eval_col = ab.predicted_reading(ctx, task, "CYP2D6_screen", 1)
    assert train_col.shape == (len(task.train),) and eval_col.shape == (len(task.evaluate),)
    assert len(ab.AUX_SETS["auxall"]) == 12 and len(ab.AUX_SETS["screen"]) == 4


@pytest.mark.parametrize("candidate", CANDIDATES, ids=lambda c: c.name)
def test_readings_and_targets_of_evaluation_and_held_out_rows_are_never_read(store, candidate):
    ctx = _context(store)
    task = _plan(ctx).inner_tasks(0, 0)[0]
    before = fit_predict(candidate, ctx, task)
    held_out = np.setdiff1d(np.arange(N_POOL), task.train)  # evaluation rows and the outer-test rows
    ab._MEMO.clear()
    corrupted_aux = {k: v.copy() for k, v in ctx.aux.items()}
    for v in corrupted_aux.values():
        v[held_out] = 99.0
    targets = {k: v.copy() for k, v in ctx.targets.items()}
    for k in ("tdi", "direct"):
        targets[k][held_out] = 99.0
    corrupted = Context(store, ctx.labels.copy(), targets, ctx.similarity, ctx.n_jobs, hit=ctx.hit.copy(), isoform="CYP2D6", aux=corrupted_aux)
    assert np.allclose(before, fit_predict(candidate, corrupted, task))


def test_a_training_compound_is_never_predicted_by_a_model_that_saw_its_own_reading(store, monkeypatch):
    monkeypatch.setattr(ab, "MIN_ROWS", 5)  # the toy data has too few rows for the real minimum
    ctx = _context(store)
    task = _plan(ctx).inner_tasks(0, 0)[0]
    first, _ = ab.predicted_reading(ctx, task, "CYP3A4_screen", 1)
    measured = np.flatnonzero(~np.isnan(ctx.aux["CYP3A4_screen"][task.train]))
    row = measured[0]
    ab._MEMO.clear()
    changed = {k: v.copy() for k, v in ctx.aux.items()}
    changed["CYP3A4_screen"][task.train[row]] += 50.0  # a huge change of one reading
    other = Context(store, ctx.labels.copy(), ctx.targets, ctx.similarity, ctx.n_jobs, hit=ctx.hit.copy(), isoform="CYP2D6", aux=changed)
    second, _ = ab.predicted_reading(other, task, "CYP3A4_screen", 1)
    assert first[row] == pytest.approx(second[row])  # its own prediction comes from folds that did not contain it
    assert not np.allclose(first, second)  # but the other compounds' predictions do move


def test_the_no_direct_rows_are_outside_the_pool_and_never_blinded_compounds(store):
    ctx = _context(store)
    rows = ab.nodirect_rows(ctx)
    assert len(rows) > 0 and (ctx.labels[rows] < 0).all() and (rows < N_POOL + 10).all()
    assert np.isnan(ctx.targets["direct"][rows]).all() and not np.isnan(ctx.targets["tdi"][rows]).any()


def test_a_compound_with_a_low_tdi_arm_value_is_a_certain_negative_whatever_its_unknown_direct_value(store):
    ctx = _context(store)
    task = _plan(ctx).inner_tasks(0, 0)[0]
    rows = ab.nodirect_rows(ctx)
    low = Context(store, ctx.labels.copy(), {**ctx.targets, "tdi": ctx.targets["tdi"].copy()}, ctx.similarity, ctx.n_jobs, hit=ctx.hit.copy(), isoform="CYP2D6", aux=ctx.aux)
    low.targets["tdi"][rows] = ab.GATE_T - 1.0
    high = Context(store, ctx.labels.copy(), {**ctx.targets, "tdi": ctx.targets["tdi"].copy()}, ctx.similarity, ctx.n_jobs, hit=ctx.hit.copy(), isoform="CYP2D6", aux=ctx.aux)
    high.targets["tdi"][rows] = ab.GATE_T + 2.0
    candidate = CANDIDATES[-2]  # mode "neg": only certain negatives enter
    assert not np.allclose(fit_predict(candidate, low, task), fit_predict(candidate, high, task))  # extra negatives change the model; ambiguous rows do not enter in this mode
    plain = Candidate("plain", "lgbm_clf", BLOCKS, params=SMALL)
    assert np.allclose(fit_predict(candidate, high, task), fit_predict(plain, ctx, task))  # with no certain negative the mode reduces to the plain model

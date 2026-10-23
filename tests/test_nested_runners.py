# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from cypmode.challenge.feature_store import BLOCKS, FeatureStore
from cypmode.challenge.leaderboard_prior import estimate_from_board, implied_prevalence
from cypmode.challenge.nested import (
    BestSingle,
    NestedPlan,
    PredictionCache,
    nested_evaluate,
)
from cypmode.challenge.nested_runners import (
    Candidate,
    Context,
    default_candidates,
    fit_predict,
    run,
)
from cypmode.challenge.validation_matching import SimilarityMatcher, tanimoto_matrix

PIECES = ["CCO", "CCN", "c1ccccc1", "c1ccncc1", "CC(=O)O", "C1CCCCC1", "c1ccoc1", "CCS", "c1ccsc1", "C1CCNCC1", "C1CCOC1"]


def _smiles(n=90):
    out = []
    for a in PIECES:
        for b in PIECES:
            out.append(f"{a}C{b}")
    return out[:n]


@pytest.fixture(scope="module")
def store():
    return FeatureStore.build(_smiles(), ["morgan_bits", "morgan_c2", "rdkit_desc", "maccs"])


def _context(store, n_pool=70, seed=0, shift_signal=True):
    rng = np.random.default_rng(seed)
    n = len(store.smiles)
    y = (store.blocks["rdkit_desc"][:, 0] + rng.normal(size=n) * 0.5 > np.median(store.blocks["rdkit_desc"][:, 0])).astype(int)
    labels = np.full(n, -1)
    labels[:n_pool] = y[:n_pool]
    shift = np.where(np.arange(n) < n_pool, y * 0.8 - 0.2 + rng.normal(size=n) * 0.1, np.nan)
    targets = {"shift": shift, "tdi": shift + 4.5, "direct": np.where(np.arange(n) < n_pool, 4.5 + rng.normal(size=n) * 0.2, np.nan)}
    return Context(store, labels, targets, tanimoto_matrix(store.blocks["morgan_bits"]), n_jobs=2)


def _plan(ctx, n_pool=70):
    pool = np.arange(n_pool)
    return NestedPlan(pool, ctx.labels[pool], 1, 3, 3, 0, external=np.arange(n_pool, len(ctx.store.smiles)))


SMALL = {"n_estimators": 20, "num_leaves": 4, "min_child_samples": 3, "learning_rate": 0.1}
CANDIDATES = [
    Candidate("knn", "knn_tanimoto", params={"k": 5}),
    Candidate("svm", "svm_tanimoto", params={"C": 1.0}),
    Candidate("lgbm", "lgbm_clf", ("rdkit_desc", "morgan_c2"), params=SMALL),
    Candidate("lgbm_knnfeat", "lgbm_clf", ("rdkit_desc",), True, SMALL),
    Candidate("et", "extratrees", ("rdkit_desc",)),
    Candidate("reg_shift", "lgbm_reg_shift", ("rdkit_desc",), params=SMALL),
    Candidate("reg_arms", "lgbm_reg_arms", ("rdkit_desc",), params=SMALL),
]


@pytest.mark.parametrize("candidate", CANDIDATES, ids=lambda c: c.name)
def test_every_family_returns_one_finite_score_per_evaluation_row(store, candidate):
    ctx = _context(store)
    plan = _plan(ctx)
    task = plan.inner_tasks(0, 0)[0]
    scores = fit_predict(candidate, ctx, task)
    assert scores.shape == (len(task.evaluate),) and np.isfinite(scores).all()


@pytest.mark.parametrize("candidate", CANDIDATES, ids=lambda c: c.name)
def test_no_candidate_can_read_the_labels_of_the_rows_it_is_scoring(store, candidate):
    """Flip every evaluation label (and the continuous targets there) and the predictions must not move."""
    ctx = _context(store)
    plan = _plan(ctx)
    task = plan.task(0, 0, -1)  # outer refit: train on outer-train, score outer-validation
    before = fit_predict(candidate, ctx, task)
    corrupted = Context(
        store,
        ctx.labels.copy(),
        {k: v.copy() for k, v in ctx.targets.items()},
        ctx.similarity,
        ctx.n_jobs,
    )
    corrupted.labels[task.evaluate] = 1 - corrupted.labels[task.evaluate]
    for v in corrupted.targets.values():
        v[task.evaluate] = 99.0
    after = fit_predict(candidate, corrupted, task)
    assert np.allclose(before, after)


def test_predictions_are_deterministic_per_task(store):
    ctx = _context(store)
    task = _plan(ctx).inner_tasks(0, 1)[0]
    candidate = CANDIDATES[2]
    assert np.allclose(fit_predict(candidate, ctx, task), fit_predict(candidate, ctx, task))


def test_run_fills_the_cache_including_deployment_and_skips_finished_work(store, tmp_path):
    ctx = _context(store)
    plan = _plan(ctx)
    cache = PredictionCache(tmp_path, plan)
    run(CANDIDATES[:2], ctx, plan, cache)
    assert all(not cache.missing(c.name) for c in CANDIDATES[:2])
    deploy = plan.task(-1, 0, -2)
    assert cache.get("knn", deploy).shape == (len(deploy.evaluate),)
    first = cache.path("knn", plan.tasks()[0]).stat().st_mtime_ns
    run(CANDIDATES[:2], ctx, plan, cache)
    assert cache.path("knn", plan.tasks()[0]).stat().st_mtime_ns == first


def test_end_to_end_nested_evaluation_runs_on_real_runner_output(store, tmp_path):
    ctx = _context(store)
    plan = _plan(ctx)
    cache = PredictionCache(tmp_path, plan)
    run(CANDIDATES[:4], ctx, plan, cache)
    table = nested_evaluate(cache, [c.name for c in CANDIDATES[:4]], [BestSingle()], weight_fn=SimilarityMatcher(store.blocks["morgan_bits"], np.arange(70, 90)))
    assert len(table) == plan.outer_splits and table["outer_score"].between(0, 1).all()


def test_default_grid_has_unique_names_ordered_simple_to_complex():
    names = [c.name for c in default_candidates()]
    assert len(names) == len(set(names)) and len(names) >= 20
    assert names[0].startswith("knn") and any(n.startswith("lgbm_reg_arms") for n in names)


# -- feature store ------------------------------------------------------------------------------


def test_feature_store_blocks_have_one_row_per_compound_and_cache_round_trips(tmp_path):
    smiles = _smiles(12)
    built = FeatureStore.load_or_build(tmp_path / "f.npz", smiles, ["morgan_c2", "maccs", "motif", "amine"])
    again = FeatureStore.load_or_build(tmp_path / "f.npz", smiles, ["morgan_c2", "maccs", "motif", "amine"])
    for name in ("morgan_c2", "maccs", "motif", "amine"):
        assert built.blocks[name].shape[0] == 12 and np.array_equal(built.blocks[name], again.blocks[name])
    assert built.matrix(["maccs", "motif"], np.array([0, 3])).shape == (2, built.blocks["maccs"].shape[1] + built.blocks["motif"].shape[1])
    with pytest.raises(KeyError):
        built.matrix(["nope"])


def test_feature_store_rejects_unparseable_smiles_and_a_stale_cache_is_rebuilt(tmp_path):
    with pytest.raises(ValueError):
        FeatureStore.build(["CCO", "not a smiles ((("], ["maccs"])
    FeatureStore.load_or_build(tmp_path / "f.npz", ["CCO", "CCN"], ["maccs"])
    rebuilt = FeatureStore.load_or_build(tmp_path / "f.npz", ["CCO", "CCN", "CCS"], ["maccs"])
    assert rebuilt.blocks["maccs"].shape[0] == 3


def test_every_registered_block_computes_on_a_small_molecule_set():
    store = FeatureStore.build(_smiles(5))
    assert set(store.blocks) == set(BLOCKS) - {"ext_activity", "physics", "chemeleon", "physics_b", "basic_centre", "monroe", "ext_cont", "ext_tox", "ext_adme", "atomjepa", "pp3a4", "pp2d6", "st3a4", "st2d6"}  # opt-in blocks need external panels, the xTB cache or an embedding cache; tested separately
    assert all(np.isfinite(v).all() for v in store.blocks.values())


# -- leaderboard prior ---------------------------------------------------------------------------


def test_implied_prevalence_recovers_the_positive_rate_from_accuracy_precision_recall():
    rng = np.random.default_rng(0)
    n, p = 100_000, 0.29
    y = rng.random(n) < p
    rows = []
    for tpr, fpr in ((0.34, 0.06), (0.65, 0.35), (0.5, 0.2)):
        pred = np.where(y, rng.random(n) < tpr, rng.random(n) < fpr)
        tp = (pred & y).sum()
        rows.append({"Accuracy": (pred == y).mean(), "Precision": tp / pred.sum(), "Recall": tp / y.sum()})
    board = pd.DataFrame(rows)
    assert np.allclose(implied_prevalence(board.Accuracy, board.Precision, board.Recall), p, atol=0.01)
    estimate = estimate_from_board(board)
    assert estimate["prevalence"] == pytest.approx(p, abs=0.01) and estimate["spread"] < 0.01 and estimate["n_rows"] == 3


def test_degenerate_rows_are_dropped_and_an_empty_board_is_an_error():
    board = pd.DataFrame({"Accuracy": [0.9, 0.8], "Precision": [0.0, 0.5], "Recall": [0.0, 0.5]})
    assert estimate_from_board(board)["n_rows"] == 1
    with pytest.raises(ValueError):
        estimate_from_board(pd.DataFrame({"Accuracy": [0.9], "Precision": [0.0], "Recall": [0.0]}))


# -- similarity matcher --------------------------------------------------------------------------


def test_similarity_matcher_up_weights_validation_compounds_that_look_like_the_test_set(store):
    bits = store.blocks["morgan_bits"]
    matcher = SimilarityMatcher(bits, np.arange(70, 90))
    rows, reference = np.arange(0, 20), np.arange(20, 70)
    w = matcher(rows, reference)
    assert w.shape == (20,) and np.isfinite(w).all() and (w >= 0).all()
    assert np.array_equal(w, matcher(rows, reference))  # deterministic and cached


def test_load_or_build_extends_an_existing_cache_without_dropping_or_recomputing_blocks(tmp_path):
    smiles = _smiles(10)
    path = tmp_path / "f.npz"
    FeatureStore.load_or_build(path, smiles, ["maccs"])
    marker = np.load(path, allow_pickle=True)["maccs"].copy()
    extended = FeatureStore.load_or_build(path, smiles, ["maccs", "motif"])
    assert set(extended.blocks) == {"maccs", "motif"}
    stored = np.load(path, allow_pickle=True)
    assert {"maccs", "motif"} <= set(stored.files) and np.array_equal(stored["maccs"], marker)
    narrowed = FeatureStore.load_or_build(path, smiles, ["motif"])  # asking for fewer blocks keeps the rest cached
    assert set(narrowed.blocks) == {"motif"} and "maccs" in np.load(path, allow_pickle=True).files


def test_tdi_alert_block_counts_known_reactive_substructures():
    names = ["methylenedioxyphenyl", "furan", "thiophene", "terminal_alkyne", "hydrazine", "cyclopropylamine", "alkyl_halide"]
    from cypmode.challenge import feature_store as fs

    smiles = ["c1ccc2OCOc2c1", "c1ccoc1CC", "c1ccsc1", "CC#C", "CNNC", "NC1CC1", "CCBr", "CCCC"]
    mols = [fs.Chem.MolFromSmiles(s) for s in smiles]
    block = fs.BLOCKS["tdi_alerts"](mols)
    column = {n: i for i, n in enumerate(fs._ALERT_SMARTS)}
    for row, name in enumerate(names):
        assert block[row, column[name]] >= 1, name
    assert block[7, : len(column)].sum() == 0  # plain butane carries no alert
    macrocycle = fs.BLOCKS["tdi_alerts"]([fs.Chem.MolFromSmiles("C1CCCCCCCCCCC1")])
    assert macrocycle[0, len(column)] == 1.0 and macrocycle[0, len(column) + 1] == 12.0


def test_confidence_weighting_uses_only_training_row_confidence(store):
    ctx = _context(store)
    ctx.confidence = np.random.default_rng(5).uniform(0.0, 3.0, len(store.smiles))
    plan = _plan(ctx)
    task = plan.task(0, 0, 0)
    weighted = Candidate("w", "lgbm_clf", ("rdkit_desc", "morgan_c2"), params=SMALL, weighting="drop1.0")

    before = fit_predict(weighted, ctx, task)
    ctx.confidence[task.evaluate] = 99.0  # the evaluation rows' confidence must not matter to the model
    assert np.allclose(fit_predict(weighted, ctx, task), before)

    ctx.confidence[task.train] = 0.0  # no training row confident enough: nothing to fit on
    with pytest.raises(ValueError):
        fit_predict(weighted, ctx, task)


def test_soft_weighting_changes_the_fit_but_the_empty_weighting_is_the_default(store):
    ctx = _context(store)
    ctx.confidence = np.random.default_rng(6).uniform(0.0, 3.0, len(store.smiles))
    task = _plan(ctx).task(0, 0, 0)
    plain = Candidate("p", "lgbm_clf", ("rdkit_desc", "morgan_c2"), params=SMALL)
    soft = Candidate("s", "lgbm_clf", ("rdkit_desc", "morgan_c2"), params=SMALL, weighting="soft")
    assert not np.allclose(fit_predict(plain, ctx, task), fit_predict(soft, ctx, task))
    ctx.confidence = None  # only weighted candidates need the confidence array
    assert len(fit_predict(plain, ctx, task)) == len(task.evaluate)
    with pytest.raises(ValueError, match="needs Context.confidence"):
        fit_predict(soft, ctx, task)


def _population_context(store, n_pool=70, seed=2):
    """Pool of 70: the first 25 rows are 'hits' (labelled by the rule), the rest never measured (label 0)."""
    ctx = _context(store, n_pool=n_pool, seed=seed)
    hit = np.zeros(len(store.smiles), dtype=np.int64)
    hit[:25] = 1
    ctx.labels[25:n_pool] = 0
    ctx.hit = hit
    return ctx


def test_selection_aware_families_return_gate_hits_only_and_their_product(store):
    ctx = _population_context(store)
    task = _plan(ctx).task(0, 0, 0)
    gate, hits, both = (
        fit_predict(Candidate(name, family, ("rdkit_desc", "morgan_c2"), params=SMALL), ctx, task)
        for name, family in (("g", "lgbm_gate"), ("h", "lgbm_hits_only"), ("t", "lgbm_two_part"))
    )
    assert np.allclose(both, gate * hits)
    assert ((gate >= 0) & (gate <= 1)).all() and ((hits >= 0) & (hits <= 1)).all()


def test_selection_aware_families_never_read_evaluation_rows_and_need_the_hit_indicator(store):
    ctx = _population_context(store)
    task = _plan(ctx).task(0, 0, 0)
    two = Candidate("t", "lgbm_two_part", ("rdkit_desc", "morgan_c2"), params=SMALL)
    before = fit_predict(two, ctx, task)
    ctx.hit[task.evaluate] = 1 - ctx.hit[task.evaluate]  # corrupt the evaluation rows' hit status
    assert np.allclose(fit_predict(two, ctx, task), before)
    ctx.hit = None
    with pytest.raises(ValueError, match="needs Context.hit"):
        fit_predict(two, ctx, task)


def _profile_context(store):
    ctx = _population_context(store)
    n = len(store.smiles)
    rng = np.random.default_rng(4)
    profile = (rng.random((n, 4)) < 0.4).astype(np.int8)
    profile[:, 2] = ctx.hit  # the target isoform's column is the hit indicator itself
    ctx.hit_profile, ctx.isoform = profile, "CYP2D6"
    return ctx


def test_selection_v2_families_return_a_gate_and_gate_times_tdi_for_hits(store):
    ctx = _profile_context(store)
    task = _plan(ctx).task(0, 0, 0)
    cols = ("rdkit_desc", "morgan_c2")
    gate, two, heck = (fit_predict(Candidate(n, f, cols, params=SMALL), ctx, task) for n, f in (("g", "sel2_gate"), ("t", "sel2_two_part"), ("h", "sel2_heckman")))
    assert ((gate >= 0) & (gate <= 1)).all() and (two <= gate + 1e-9).all() and (heck <= gate + 1e-9).all()


def test_selection_v2_never_reads_the_evaluation_rows_hit_profile(store):
    ctx = _profile_context(store)
    task = _plan(ctx).task(0, 0, 0)
    cand = Candidate("h", "sel2_heckman", ("rdkit_desc", "morgan_c2"), params=SMALL)
    before = fit_predict(cand, ctx, task)
    ctx.hit_profile[task.evaluate] = 1 - ctx.hit_profile[task.evaluate]
    assert np.allclose(fit_predict(cand, ctx, task), before)
    ctx.hit_profile = None
    with pytest.raises(ValueError, match="needs Context.hit_profile"):
        fit_predict(cand, ctx, task)


def test_physics_block_is_opt_in_and_reads_only_its_cache(tmp_path, monkeypatch):
    from cypmode.challenge import physics_block as pb
    from cypmode.challenge.feature_store import OPT_IN_BLOCKS

    assert "physics" in OPT_IN_BLOCKS and "physics" not in FeatureStore.build(_smiles(2)).blocks
    smiles = _smiles(3)
    rows = [{"SMILES": pb.canonical(s), **{n: float(i) for n in pb.FEATURE_NAMES}} for i, s in enumerate(smiles)]
    cache = tmp_path / "physics_cache.csv"
    pd.DataFrame(rows).to_csv(cache, index=False)
    monkeypatch.setattr(pb, "CACHE_PATH", cache)
    monkeypatch.setattr(pb.load_physics_features, "__defaults__", (cache,))
    store = FeatureStore.build(smiles, ["physics"])
    assert store.blocks["physics"].shape == (3, len(pb.FEATURE_NAMES)) and (store.blocks["physics"][:, 0] == [0, 1, 2]).all()
    assert "tblite" not in sys.modules  # the evaluation process must never load tblite


def test_pair_family_scores_every_evaluation_row_without_reading_their_labels(store):
    ctx = _population_context(store)
    task = _plan(ctx).task(0, 0, 0)
    cand = Candidate("pair", "pair_lgbm", ("rdkit_desc",), params={"k": 3, "lgbm": SMALL})
    before = fit_predict(cand, ctx, task)
    assert before.shape == (len(task.evaluate),) and ((before >= 0) & (before <= 1)).all()
    ctx.labels[task.evaluate] = 1 - ctx.labels[task.evaluate]
    assert np.allclose(fit_predict(cand, ctx, task), before)


def test_clipped_arm_regressor_trains_on_floor_clipped_arms_and_keeps_labels(store):
    from cypmode.challenge.tdi_targets import ASSAY_FLOOR, SHIFT_THRESHOLD

    ctx = _population_context(store)
    task = _plan(ctx).task(0, 0, 0)
    rng = np.random.default_rng(3)
    n = len(ctx.labels)
    d, t = rng.uniform(1.9, 7.0, n), rng.uniform(1.9, 7.5, n)
    assert (((t - np.maximum(d, ASSAY_FLOOR)) > SHIFT_THRESHOLD) == ((np.maximum(t, ASSAY_FLOOR) - np.maximum(d, ASSAY_FLOOR)) > SHIFT_THRESHOLD)).all()  # clipping never changes a label
    ctx.targets = {"tdi": t, "direct": d, "shift": t - np.maximum(d, ASSAY_FLOOR)}
    cand = Candidate("reg", "lgbm_reg_arms_clip", ("rdkit_desc",), params=SMALL)
    out = fit_predict(cand, ctx, task)
    assert out.shape == (len(task.evaluate),) and np.isfinite(out).all()


def _arm_context(store):
    ctx = _population_context(store)
    rng = np.random.default_rng(11)
    n = len(ctx.labels)
    x = store.matrix(("rdkit_desc",))
    signal = (x[:, 0] - x[:, 0].mean()) / (x[:, 0].std() + 1e-9)
    d = 4.5 + 0.3 * signal + rng.normal(0, 0.2, n)
    t = d + 0.3 + 0.8 * signal + rng.normal(0, 0.2, n)
    ctx.targets = {"tdi": t, "direct": d, "shift": t - np.maximum(d, 4.0), "tdi_sd": np.full(n, 0.2), "direct_sd": np.full(n, 0.2)}
    return ctx


def _committee(output):
    light = {"n_estimators": 20, "learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 5}
    return Candidate(f"c_{output}", "arm_committee", ("rdkit_desc",), params={"member_sets": [("rdkit_desc",)], "n_seeds": 3, "lgbm": light, "output": output})


def test_committee_outputs_are_probability_and_nonnegative_improvement_and_rank_with_the_point_estimate(store):
    ctx = _arm_context(store)
    task = _plan(ctx).task(0, 0, 0)
    m, p, ei = (fit_predict(_committee(o), ctx, task) for o in ("m", "p", "ei"))
    assert ((p >= 0) & (p <= 1)).all() and (ei >= 0).all() and np.isfinite(m).all()
    from scipy.stats import spearmanr

    assert spearmanr(m, p).statistic > 0.9 and spearmanr(m, ei).statistic > 0.8


def test_committee_never_reads_the_evaluation_rows_measurements(store):
    ctx = _arm_context(store)
    task = _plan(ctx).task(0, 0, 0)
    before = fit_predict(_committee("p"), ctx, task)
    for key in ("tdi", "direct", "shift", "tdi_sd", "direct_sd"):
        ctx.targets[key][task.evaluate] = np.nan
    assert np.allclose(fit_predict(_committee("p"), ctx, task), before)


def test_committee_noise_weighting_needs_the_standard_deviations(store):
    ctx = _arm_context(store)
    task = _plan(ctx).task(0, 0, 0)
    del ctx.targets["tdi_sd"]
    with pytest.raises(ValueError, match="tdi_sd"):
        fit_predict(_committee("p"), ctx, task)


def test_testlike_weights_upweight_compounds_near_the_test_set_and_use_no_labels(store):
    from cypmode.challenge.nested_runners import testlike_weights

    ctx = _population_context(store)
    n = len(ctx.labels)
    ctx.test_rows = np.arange(n - 5, n)
    train_rows = np.arange(0, n - 5)
    w = testlike_weights(ctx, train_rows, 1.0)
    assert w.shape == train_rows.shape and w.mean() == pytest.approx(1.0) and (w > 0).all() and w.max() / w.min() <= 25.0 + 1e-9
    labels_before = ctx.labels.copy()
    ctx.labels[:] = 1 - ctx.labels
    assert np.allclose(testlike_weights(ctx, train_rows, 1.0), w)  # labels never enter
    ctx.labels[:] = labels_before
    ctx.test_rows = None
    with pytest.raises(ValueError, match="test_rows"):
        testlike_weights(ctx, train_rows, 1.0)


def test_testlike_candidate_trains_and_scores(store):
    ctx = _population_context(store)
    ctx.test_rows = np.arange(len(ctx.labels) - 5, len(ctx.labels))
    task = _plan(ctx).task(0, 0, 0)
    cand = Candidate("tl", "lgbm_clf", ("rdkit_desc",), params=SMALL, weighting="testlike0.5")
    p = fit_predict(cand, ctx, task)
    assert p.shape == (len(task.evaluate),) and ((p >= 0) & (p <= 1)).all()


def test_nonhit_share_estimators():
    from cypmode.challenge.test_nonhit_share import equivalent_nonhit_weight, pac_share

    assert pac_share(np.array([0.5, 0.5]), np.array([0.8, 0.8]), np.array([0.2, 0.2])) == pytest.approx(0.5)  # halfway between the classes
    assert equivalent_nonhit_weight(0.328, 0.328) == pytest.approx(1.0)
    assert equivalent_nonhit_weight(0.5, 0.25) == pytest.approx(1 / 3)


def _select_candidate(k=3):
    return Candidate("sel", "lgbm_clf_select", ("morgan_c2", "rdkit_desc"), params={"base_blocks": ("morgan_c2",), "select_blocks": ("rdkit_desc",), "select_k": k, "lgbm": SMALL})


def test_compact_selection_scores_every_evaluation_row_and_never_reads_their_labels(store):
    ctx = _population_context(store)
    task = _plan(ctx).task(0, 0, 0)
    cand = _select_candidate()
    before = fit_predict(cand, ctx, task)
    assert before.shape == (len(task.evaluate),) and ((before >= 0) & (before <= 1)).all()
    ctx.labels[task.evaluate] = 1 - ctx.labels[task.evaluate]
    assert np.allclose(fit_predict(cand, ctx, task), before)  # neither the labels nor the selection depend on evaluation rows


def test_compact_selection_uses_only_training_rows_to_choose_columns(store, monkeypatch):
    ctx = _population_context(store)
    task = _plan(ctx).task(0, 0, 0)
    seen = []
    import lightgbm

    original = lightgbm.LGBMClassifier.fit

    def spy(self, x, y, *a, **kw):
        seen.append(len(x))
        return original(self, x, y, *a, **kw)

    monkeypatch.setattr(lightgbm.LGBMClassifier, "fit", spy)
    fit_predict(_select_candidate(), ctx, task)
    assert seen and all(n == len(task.train) for n in seen)  # both the selection stage and the final fit see training rows only


def test_new_candidate_groups_are_complete_and_named_consistently():
    from cypmode.challenge.nested_runners import (
        compact_physics_candidates,
        physics_b_candidates,
        population_physics_candidates,
    )

    compact = compact_physics_candidates()
    assert [c.name for c in compact] == ["lgbm_A+physk20_deep", "lgbm_A+fps+physk20_deep"] and all("physics" in c.blocks for c in compact)
    assert [c.name for c in compact_physics_candidates(True)] == ["lgbm_A+physkb20_deep", "lgbm_A+fps+physkb20_deep"]
    assert {c.name for c in physics_b_candidates()} == {"lgbm_physB_deep", "lgbm_A+physB_deep", "lgbm_A+fps+physB_deep", "lgbm_A+phys+physB_deep"}
    pop = population_physics_candidates()
    assert len(pop) == 8 and {c.family for c in pop} == {"lgbm_gate", "lgbm_hits_only", "lgbm_two_part", "lgbm_clf"} and all("physics" in c.blocks for c in pop)
    assert "pophits_A+phys_deep" in {c.name for c in pop}


def test_pharmacophore_model_candidates_are_named_after_their_experiments_and_the_control_is_the_permuted_model():
    from cypmode.challenge.nested_runners import (
        hypothesis_candidates,
        site_cyp2d6_candidates,
        site_cyp3a4_candidates,
    )

    assert [c.name for c in site_cyp3a4_candidates()] == ["lgbm_A+st3a4_deep", "lgbm_A+fps+st3a4_deep"] and all("st3a4" in c.blocks for c in site_cyp3a4_candidates())
    assert [c.name for c in site_cyp2d6_candidates()] == ["lgbm_A+st2d6_deep", "lgbm_A+fps+st2d6_deep"] and all("st2d6" in c.blocks for c in site_cyp2d6_candidates())
    assert [c.name for c in hypothesis_candidates()] == ["lgbm_A+hyp_deep", "lgbm_A+fps+hyp_deep"] and {c.hyp for c in hypothesis_candidates()} == {"model"}
    assert [c.name for c in hypothesis_candidates(permuted=True)] == ["lgbm_A+hypperm_deep", "lgbm_A+fps+hypperm_deep"] and {c.hyp for c in hypothesis_candidates(permuted=True)} == {"permuted"}
    assert not any(c.blocks and "hyp" in " ".join(c.blocks) for c in hypothesis_candidates())  # the model adds its columns inside the task, not through a block of the store


@pytest.mark.parametrize("flag", ["model", "permuted"])
def test_hypothesis_candidates_score_every_row_and_never_read_the_labels_of_the_rows_they_score(store, monkeypatch, flag):
    from scipy.sparse import csr_matrix

    from cypmode.challenge import pharmacophore_hypotheses as hypotheses

    rng = np.random.default_rng(1)
    realised = csr_matrix((rng.random((len(store.smiles), 80)) < 0.3).astype(np.uint8))
    monkeypatch.setattr(hypotheses, "load", lambda smiles, cache=None: (realised, np.arange(80, dtype=np.uint64)))
    ctx = _context(store)
    task = _plan(ctx).task(0, 0, -1)
    candidate = Candidate(f"lgbm_hyp_{flag}", "lgbm_clf", ("rdkit_desc",), params=SMALL, hyp=flag)
    before = fit_predict(candidate, ctx, task)
    assert before.shape == (len(task.evaluate),) and np.isfinite(before).all()
    corrupted = Context(store, ctx.labels.copy(), {k: v.copy() for k, v in ctx.targets.items()}, ctx.similarity, ctx.n_jobs)
    corrupted.labels[task.evaluate] = 1 - corrupted.labels[task.evaluate]
    assert np.allclose(before, fit_predict(candidate, corrupted, task))
    plain = fit_predict(Candidate("lgbm_plain", "lgbm_clf", ("rdkit_desc",), params=SMALL), ctx, task)
    assert not np.allclose(before, plain)  # the five columns are in the design

import numpy as np
import pytest

from cypmode.challenge.nested import (
    DEPLOY,
    OUTER_REFIT,
    BestSingle,
    MeanOfTop,
    NestedPlan,
    OneStandardError,
    PredictionCache,
    StackedLogit,
    best_threshold,
    inner_oof,
    nested_evaluate,
    summarize,
)


def _plan(n=240, repeats=2, outer=4, inner=3, seed=0, external=False, prevalence=0.3):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    pool = np.arange(n) * 2 + 1  # global row ids that are not 0..n-1, to catch positional/global mix-ups
    return NestedPlan(pool, y, repeats, outer, inner, seed, external=np.arange(1000, 1020) if external else None), y


# --- plan structure: the leakage guarantees ------------------------------------------------------


def test_inner_folds_live_entirely_inside_the_outer_training_rows():
    plan, _ = _plan()
    for repeat, k in plan.outer_folds():
        refit = plan.task(repeat, k, OUTER_REFIT)
        outer_train, outer_val = set(refit.train.tolist()), set(refit.evaluate.tolist())
        assert not outer_train & outer_val
        for task in plan.inner_tasks(repeat, k):
            assert set(task.train.tolist()) <= outer_train and set(task.evaluate.tolist()) <= outer_train
            assert not set(task.train.tolist()) & set(task.evaluate.tolist())
            assert not (set(task.train.tolist()) | set(task.evaluate.tolist())) & outer_val  # never sees the outer-validation rows


def test_inner_validation_folds_partition_the_outer_training_rows_exactly_once():
    plan, _ = _plan()
    for repeat, k in plan.outer_folds():
        rows = np.concatenate([t.evaluate for t in plan.inner_tasks(repeat, k)])
        assert sorted(rows.tolist()) == sorted(plan.task(repeat, k, OUTER_REFIT).train.tolist())


def test_outer_validation_folds_partition_the_pool_once_per_repeat():
    plan, _ = _plan()
    for repeat in range(plan.repeats):
        rows = np.concatenate([plan.task(repeat, k, OUTER_REFIT).evaluate for k in range(plan.outer_splits)])
        assert sorted(rows.tolist()) == sorted(plan.pool.tolist())


def test_splits_are_stratified_and_reproducible():
    plan, y = _plan(n=400)
    again, _ = _plan(n=400)
    assert plan.plan_id == again.plan_id
    assert all(np.array_equal(a.evaluate, b.evaluate) for a, b in zip(plan.tasks(), again.tasks(), strict=True))
    for task in plan.inner_tasks(0, 0):
        assert abs(plan.labels(task.evaluate).mean() - y.mean()) < 0.06


def test_plan_identity_changes_with_any_ingredient_and_survives_a_save_load_round_trip(tmp_path):
    plan, _ = _plan(external=True)
    assert plan.plan_id != _plan(seed=1)[0].plan_id and plan.plan_id != _plan(inner=4)[0].plan_id
    plan.save(tmp_path / "plan.npz")
    loaded = NestedPlan.load(tmp_path / "plan.npz")
    assert loaded.plan_id == plan.plan_id and len(loaded.tasks()) == len(plan.tasks())


def test_deploy_tasks_use_the_whole_pool_and_never_leak_external_rows_into_training():
    plan, _ = _plan(external=True)
    deploy = plan.task(-1, 0, DEPLOY)
    assert sorted(deploy.train.tolist()) == sorted(plan.pool.tolist())
    assert sorted(deploy.evaluate.tolist()) == list(range(1000, 1020))
    assert not set(deploy.evaluate.tolist()) & set(plan.pool.tolist())


def test_plan_rejects_a_single_class():
    with pytest.raises(ValueError):
        NestedPlan(np.arange(10), np.zeros(10, dtype=int), 1, 2, 2, 0)


# --- cache ---------------------------------------------------------------------------------------


def _fill(cache, name, scorer):
    for task in cache.plan.tasks():
        cache.put(name, task, scorer(task))


def test_cache_validates_shape_and_finiteness_and_reports_missing_tasks(tmp_path):
    plan, _ = _plan()
    cache = PredictionCache(tmp_path, plan)
    task = plan.tasks()[0]
    with pytest.raises(ValueError):
        cache.put("m", task, np.zeros(len(task.evaluate) + 1))
    with pytest.raises(ValueError):
        cache.put("m", task, np.full(len(task.evaluate), np.nan))
    cache.put("m", task, np.zeros(len(task.evaluate)))
    assert cache.has("m", task) and len(cache.missing("m")) == len(plan.tasks()) - 1


def test_inner_oof_assembles_one_prediction_per_outer_training_row(tmp_path):
    plan, _ = _plan()
    cache = PredictionCache(tmp_path, plan)
    _fill(cache, "a", lambda t: t.evaluate.astype(float))
    rows, matrix = inner_oof(cache, ["a"], 0, 0)
    assert np.array_equal(matrix[:, 0], rows.astype(np.float32))
    assert sorted(rows.tolist()) == sorted(plan.task(0, 0, OUTER_REFIT).train.tolist())


# --- the point of the module: selection bias is measured, and removed ----------------------------


def _noise_cache(tmp_path, n_noise=40, signal_auc_shift=None, n=300, seed=3):
    plan, y = _plan(n=n, repeats=2, outer=5, inner=4, seed=seed)
    cache = PredictionCache(tmp_path, plan)
    label = dict(zip(plan.pool.tolist(), y.tolist(), strict=True))
    for j in range(n_noise):
        rng = np.random.default_rng(1000 + j)
        _fill(cache, f"noise{j:02d}", lambda t, rng=rng: rng.normal(size=len(t.evaluate)))
    if signal_auc_shift is not None:
        rng = np.random.default_rng(7)
        _fill(cache, "signal", lambda t: np.array([label[int(r)] for r in t.evaluate]) * signal_auc_shift + rng.normal(size=len(t.evaluate)))
    return cache


def test_picking_the_best_of_many_pure_noise_candidates_looks_good_inside_and_is_chance_outside(tmp_path):
    cache = _noise_cache(tmp_path)
    names = cache.candidates()
    table = nested_evaluate(cache, names, [BestSingle()])
    row = summarize(table).loc["best_single"]
    assert row["inner_score"] > 0.58  # the winner's curse: the contest score of the best of 40 noise models
    assert abs(row["outer_score"] - 0.5) < 0.045  # what the winner is actually worth
    assert row["optimism"] > 0.06


def test_a_genuine_signal_is_found_and_not_penalised(tmp_path):
    cache = _noise_cache(tmp_path, signal_auc_shift=1.0)
    names = cache.candidates()
    table = nested_evaluate(cache, names, [BestSingle(), StackedLogit(), MeanOfTop(3)])
    summary = summarize(table)
    chosen = table[table["selector"] == "best_single"]["chosen"]
    assert (chosen.str.contains("signal")).mean() > 0.9  # picked in nearly every outer fold
    assert summary.loc["best_single", "outer_score"] > 0.68  # N(0,1) vs N(1,1) has AUC ~0.76
    assert summary.loc["stack_logit", "outer_score"] > 0.68
    assert summary.loc["best_single", "optimism"] < 0.06  # little selection bias when the winner is real


def test_stacking_on_noise_does_not_manufacture_skill(tmp_path):
    cache = _noise_cache(tmp_path, n_noise=15)
    table = nested_evaluate(cache, cache.candidates(), [StackedLogit(), MeanOfTop(5), OneStandardError()])
    assert (summarize(table)["outer_score"].sub(0.5).abs() < 0.05).all()


def test_the_selector_sees_no_outer_validation_information(tmp_path):
    """Corrupt every outer-refit prediction with the *labels* and the outer score must jump while
    the inner score of a selector fitted before looking at them stays put: proof the inner and
    outer paths are separate."""
    plan, y = _plan(n=300, repeats=1, outer=5, inner=4, seed=5)
    label = dict(zip(plan.pool.tolist(), y.tolist(), strict=True))
    clean = PredictionCache(tmp_path / "clean", plan)
    leaky = PredictionCache(tmp_path / "leaky", plan)
    for cache, leak in ((clean, False), (leaky, True)):
        rng = np.random.default_rng(11)
        for task in plan.tasks():
            noise = rng.normal(size=len(task.evaluate))
            if leak and task.inner == OUTER_REFIT:
                noise = noise + 3 * np.array([label[int(r)] for r in task.evaluate])
            cache.put("m", task, noise)
    a = nested_evaluate(clean, ["m"], [BestSingle()])
    b = nested_evaluate(leaky, ["m"], [BestSingle()])
    assert np.allclose(a["inner_score"], b["inner_score"])
    assert b["outer_score"].mean() > a["outer_score"].mean() + 0.3


def test_weights_and_prior_shift_are_applied_to_both_inner_and_outer_scoring(tmp_path):
    cache = _noise_cache(tmp_path, n_noise=3, signal_auc_shift=1.0)
    calls = []

    def weight_fn(rows, reference):
        calls.append((len(rows), len(reference)))
        return np.ones(len(rows))

    nested_evaluate(cache, cache.candidates(), [BestSingle()], weight_fn=weight_fn, target_prevalence=0.1)
    assert calls and all(r < ref for r, ref in calls[1::2])  # outer-validation rows are scored against the larger training reference


def test_best_threshold_maximises_weighted_mcc():
    y = np.array([0] * 70 + [1] * 30)
    score = np.r_[np.linspace(0, 0.6, 70), np.linspace(0.4, 1.0, 30)]
    t = best_threshold(y, score, None)
    assert 0.3 < t < 0.7


def test_score_metric_is_uninformative_not_an_error_when_a_class_loses_all_its_weight():
    from cypmode.challenge.nested import score_metric

    y = np.array([0, 1, 0, 1])
    assert score_metric(y, np.array([0.1, 0.9, 0.2, 0.8]), np.array([1.0, 0.0, 1.0, 0.0])) == 0.5


def test_deploy_applies_the_frozen_selection_to_the_external_rows(tmp_path):
    from cypmode.challenge.nested import deploy

    plan, y = _plan(n=300, repeats=1, outer=4, inner=4, seed=9, external=True)
    cache = PredictionCache(tmp_path, plan)
    label = dict(zip(plan.pool.tolist(), y.tolist(), strict=True))
    rng = np.random.default_rng(2)
    for j in range(10):
        _fill(cache, f"noise{j}", lambda t, rng=rng: rng.normal(size=len(t.evaluate)))
    _fill(cache, "signal", lambda t: np.array([label.get(int(r), 0) for r in t.evaluate]) * 2.0 + rng.normal(size=len(t.evaluate)))
    result = deploy(cache, cache.candidates(), BestSingle())
    assert np.array_equal(result.rows, np.arange(1000, 1020)) and result.scores.shape == (20,)
    assert "signal" in result.selection.chosen and result.calls.dtype == bool


def test_rate_transfer_calls_the_requested_fraction_even_when_scores_shift():
    from cypmode.challenge.nested import call_top_fraction, positive_rate

    rng = np.random.default_rng(0)
    inner = rng.normal(size=1000)
    shifted = rng.normal(size=1000) + 2.0  # the same procedure, easier compounds: every score is higher
    threshold = 0.5
    rate = positive_rate(inner, threshold)
    assert np.mean(shifted >= threshold) > 0.9  # an absolute threshold would call nearly everything
    assert call_top_fraction(shifted, rate).mean() == pytest.approx(rate, abs=0.01)


def test_smoothed_threshold_is_reproducible_and_still_finds_a_clear_optimum():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.3).astype(int)
    score = y * 1.5 + rng.normal(size=2000)
    assert best_threshold(y, score, None) == best_threshold(y, score, None)
    assert 0.3 < best_threshold(y, score, None) < 1.3


# --- repeats can be extended; groups stay whole ---------------------------------------------------


def test_extending_the_repeats_keeps_the_identity_and_every_earlier_split():
    few, _ = _plan(repeats=1)
    many, _ = _plan(repeats=3)
    assert few.plan_id == many.plan_id  # a cache filled for one repeat is valid for three
    earlier = {t.key: t for t in few.tasks() if t.repeat == 0}
    for task in (t for t in many.tasks() if t.repeat == 0):
        assert np.array_equal(task.train, earlier[task.key].train) and np.array_equal(task.evaluate, earlier[task.key].evaluate)
    assert len(many.tasks()) > len(few.tasks())


def _grouped_plan(n=300, group_size=6, seed=0, **kwargs):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    groups = np.arange(n) // group_size
    return NestedPlan(np.arange(n), y, 2, 4, 3, seed, groups=groups, **kwargs), groups


def test_group_aware_plan_never_splits_a_group_across_training_and_validation_at_either_level():
    plan, groups = _grouped_plan()
    group = dict(zip(range(300), groups.tolist(), strict=True))
    for task in plan.tasks():
        train_groups = {group[int(r)] for r in task.train}
        eval_groups = {group[int(r)] for r in task.evaluate}
        assert not train_groups & eval_groups, task.key


def test_group_aware_folds_still_cover_every_row_once_and_keep_both_classes():
    plan, _ = _grouped_plan()
    for repeat in range(plan.repeats):
        rows = np.concatenate([plan.task(repeat, k, OUTER_REFIT).evaluate for k in range(plan.outer_splits)])
        assert sorted(rows.tolist()) == list(range(300))
    for task in plan.tasks():
        assert 0 < plan.labels(task.evaluate).mean() < 1 or len(task.evaluate) < 20


def test_groups_change_the_plan_identity_survive_a_round_trip_and_are_validated(tmp_path):
    grouped, groups = _grouped_plan()
    plain = NestedPlan(grouped.pool, grouped.y, 2, 4, 3, 0)
    assert grouped.plan_id != plain.plan_id
    grouped.save(tmp_path / "g.npz")
    loaded = NestedPlan.load(tmp_path / "g.npz")
    assert loaded.plan_id == grouped.plan_id and np.array_equal(loaded.groups, groups)
    plain.save(tmp_path / "p.npz")
    assert NestedPlan.load(tmp_path / "p.npz").groups is None
    with pytest.raises(ValueError):
        NestedPlan(grouped.pool, grouped.y, 1, 4, 3, 0, groups=groups[:-1])


def test_stacked_logit_does_not_depend_on_the_global_random_state():
    """liblinear shuffles with the global numpy state unless it is given a random_state: two processes must fit the same stack."""
    rng = np.random.default_rng(3)
    n = 400
    y = rng.integers(0, 2, n)
    oof = np.clip(0.25 * y[:, None] + rng.uniform(0.1, 0.8, (n, 6)) * 0.75, 0.01, 0.99)
    weights = rng.uniform(0.5, 2.0, n)
    scores = []
    for global_seed in (0, 1, 2):
        np.random.seed(global_seed)
        chosen = StackedLogit().fit([f"c{j}" for j in range(6)], oof, y, weights, "auc")
        scores.append(chosen.predict(oof))
    assert np.array_equal(scores[0], scores[1]) and np.array_equal(scores[1], scores[2])


def test_the_meta_learner_reaches_the_same_optimum_whatever_the_solver_seed():
    """With the loose default tolerance liblinear stops at a point that depends on its random coordinate order; the tight tolerance removes that."""
    rng = np.random.default_rng(5)
    n = 1500
    y = rng.integers(0, 2, n)
    base = rng.normal(size=(n, 1)) + y[:, None] * 0.6
    z = base + 0.3 * rng.normal(size=(n, 12))  # correlated columns, like the logits of 23 related candidates
    coefs = [StackedLogit._model("l1", 0.03, seed).fit(z, y).coef_[0] for seed in (0, 1, 2, 3)]
    assert max(np.abs(c - coefs[0]).max() for c in coefs) < 1e-6

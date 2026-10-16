import numpy as np
import pytest

from cypmode.challenge.committee_uncertainty import (
    BASELINE,
    COMMITTEE_SIGNALS,
    _verdict,
    collect_signals,
    committee_spread,
    decision_record,
    deploy_signals,
    evaluate_components,
    member_signals,
    selective_fallback,
    vote_entropy,
)
from cypmode.challenge.nested import NestedPlan, PredictionCache

N_POOL, N_EXTERNAL, N_MEMBERS = 300, 30, 8


def test_vote_entropy_is_zero_for_unanimity_and_maximal_for_a_split():
    assert np.allclose(vote_entropy(np.ones((3, 6))), 0.0, atol=1e-6)
    assert np.allclose(vote_entropy(np.array([[1, 1, 1, 0, 0, 0]])), np.log(2))
    assert vote_entropy(np.array([[1, 1, 1, 1, 0]]))[0] < np.log(2)


def test_committee_spread_is_the_per_row_standard_deviation():
    assert np.allclose(committee_spread(np.array([[0.2, 0.2, 0.2], [0.0, 0.5, 1.0]])), [0.0, np.std([0.0, 0.5, 1.0])])


@pytest.mark.parametrize(
    ("component", "auroc_lo", "gain", "gain_p", "expected"),
    [
        (BASELINE, 0.7, np.nan, np.nan, "baseline"),
        ("qbc_spread", 0.6, 0.02, 0.004, "adds_information"),
        ("qbc_spread", 0.6, 0.02, 0.08, "redundant_with_margin"),  # a gain, but not distinguishable from chance
        ("qbc_spread", 0.6, -0.002, 0.7, "redundant_with_margin"),  # flags errors, but not beyond the margin
        ("qbc_spread", 0.48, 0.01, 0.2, "no_signal"),
        ("qbc_spread", 0.6, np.nan, np.nan, "redundant_with_margin"),
    ],
)
def test_verdict_separates_new_information_from_redundancy_and_noise(component, auroc_lo, gain, gain_p, expected):
    assert _verdict(component, auroc_lo, gain, gain_p, alpha=0.0125) == expected


def _cache(tmp_path, informative: bool, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(N_POOL) < 0.35).astype(int)
    plan = NestedPlan(np.arange(N_POOL), y, 1, 4, 3, seed, external=np.arange(N_POOL, N_POOL + N_EXTERNAL))
    label = {**dict(zip(range(N_POOL), y.tolist(), strict=True)), **{i: int(rng.random() < 0.35) for i in range(N_POOL, N_POOL + N_EXTERNAL)}}
    hard = rng.random(N_POOL + N_EXTERNAL) < 0.3
    cache = PredictionCache(tmp_path, plan)
    for m in range(N_MEMBERS):
        member_rng = np.random.default_rng(100 + m)
        for task in plan.tasks():
            rows = task.evaluate
            lab = np.array([label[int(r)] for r in rows])
            if informative:
                base = np.where(hard[rows], 0.0, 2.0 * (2 * lab - 1))
                noise = member_rng.normal(size=len(rows)) * np.where(hard[rows], 3.0, 0.5)  # members disagree exactly where labels are unpredictable
            else:
                base, noise = np.zeros(len(rows)), member_rng.normal(size=len(rows))
            cache.put(f"m{m}", task, base + noise)
    return cache, [f"m{m}" for m in range(N_MEMBERS)], hard


def _bits(seed=1):
    return (np.random.default_rng(seed).random((N_POOL + N_EXTERNAL, 64)) < 0.3).astype(np.float32)


def test_member_signals_are_finite_and_on_the_expected_scales():
    rng = np.random.default_rng(0)
    oof, outer = rng.normal(size=(200, 6)), rng.normal(size=(40, 6))
    y = (rng.random(200) < 0.4).astype(int)
    signals = member_signals(oof, y, outer, [f"lgbm_{i}" for i in range(3)] + ["svm_a", "knn_b", "rf_c"], rate=0.4)
    assert set(signals) == set(COMMITTEE_SIGNALS)
    for values in signals.values():
        assert values.shape == (40,) and np.isfinite(values).all() and (values >= 0).all()
    assert signals["qbc_vote_entropy"].max() <= np.log(2) + 1e-9


def test_a_committee_that_disagrees_where_labels_are_unpredictable_flags_the_errors(tmp_path):
    cache, names, _ = _cache(tmp_path, informative=True)
    frame = collect_signals(cache, names, _bits())
    assert len(frame) == N_POOL and set(frame["wrong"]) <= {0, 1}
    table = evaluate_components(frame, n_boot=40, n_perm=60).set_index("component")
    assert table.loc["qbc_spread", "auroc"] > 0.62
    assert table.loc["qbc_spread", "auroc_lo"] < table.loc["qbc_spread", "auroc"] < table.loc["qbc_spread", "auroc_hi"]
    assert 0 < table.loc["qbc_spread", "gain_p"] <= 1
    assert table.loc[BASELINE, "verdict"] == "baseline" and np.isnan(table.loc[BASELINE, "gain_over_margin"])


def test_pure_noise_members_produce_no_signal_not_a_false_discovery(tmp_path):
    cache, names, _ = _cache(tmp_path, informative=False)
    frame = collect_signals(cache, names, _bits())
    table = evaluate_components(frame, n_boot=40, n_perm=60).set_index("component")
    for signal in COMMITTEE_SIGNALS:
        assert abs(table.loc[signal, "auroc"] - 0.5) < 0.1
        assert table.loc[signal, "verdict"] != "adds_information"


def test_selective_fallback_reports_the_baseline_and_each_fraction(tmp_path):
    cache, names, _ = _cache(tmp_path, informative=True)
    frame = collect_signals(cache, names, _bits())
    table = selective_fallback(frame, "qbc_spread", (0.1, 0.3))
    assert table["fallback_fraction"].tolist() == [0.0, 0.1, 0.3]
    fallback_all = selective_fallback(frame, "qbc_spread", (1.0,))
    assert fallback_all["mcc"].iloc[1] == 0.0  # everything negative: no skill


def test_deployment_signals_cover_every_external_compound(tmp_path):
    cache, names, _ = _cache(tmp_path, informative=True)
    frame = deploy_signals(cache, names, _bits())
    assert sorted(frame["row"]) == list(range(N_POOL, N_POOL + N_EXTERNAL))
    assert {BASELINE, "novelty", *COMMITTEE_SIGNALS} <= set(frame.columns) and np.isfinite(frame[list(COMMITTEE_SIGNALS)].to_numpy()).all()


def test_decision_record_states_the_policy_and_each_components_verdict(tmp_path):
    cache, names, _ = _cache(tmp_path, informative=False)
    frame = collect_signals(cache, names, _bits())
    evaluation = evaluate_components(frame, n_boot=30, n_perm=40)
    fallback = selective_fallback(frame, "qbc_vote_entropy")
    record = decision_record("CYP3A4", evaluation, fallback)
    assert record["isoform"] == "CYP3A4" and "never used to alter a prediction" in record["policy"]
    assert set(record["components"]) == set(evaluation["component"])
    assert record["components"]["qbc_vote_entropy"]["acting_on_it_improves_mcc"] in (True, False)
    assert not any(c["used_to_alter_predictions"] for c in record["components"].values())  # pure noise: nothing may touch a prediction


def test_a_component_needs_both_new_information_and_a_decision_gain_to_alter_predictions():
    import pandas as pd

    evaluation = pd.DataFrame(
        [
            {"component": "a", "auroc": 0.6, "gain_over_margin": 0.02, "gain_p": 0.001, "verdict": "adds_information"},
            {"component": "b", "auroc": 0.6, "gain_over_margin": 0.02, "gain_p": 0.001, "verdict": "adds_information"},
            {"component": "c", "auroc": 0.6, "gain_over_margin": 0.0, "gain_p": 0.9, "verdict": "redundant_with_margin"},
        ]
    )
    fallback = pd.DataFrame(
        [{"signal": s, "fallback_fraction": f, "mcc": m} for s, rows in {"a": [(0, 0.30), (0.1, 0.36)], "b": [(0, 0.30), (0.1, 0.299)], "c": [(0, 0.30), (0.1, 0.40)]}.items() for f, m in rows]
    )
    used = {k: v["used_to_alter_predictions"] for k, v in decision_record("X", evaluation, fallback)["components"].items()}
    assert used == {"a": True, "b": False, "c": False}  # b detects errors but cannot fix them; c fixes them but adds nothing the margin lacks


def test_fallback_flips_a_signal_that_is_high_where_the_stack_is_right(tmp_path):
    import pandas as pd

    rng = np.random.default_rng(0)
    wrong = (rng.random(400) < 0.4).astype(int)
    frame = pd.DataFrame({"y": rng.integers(0, 2, 400), "call": rng.integers(0, 2, 400), "wrong": wrong, "sig": -wrong + rng.normal(size=400) * 0.3})
    assert selective_fallback(frame, "sig")["direction"].iloc[0] == -1.0

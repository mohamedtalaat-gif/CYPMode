# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from cypmode.challenge import selection_decision as sd

A_MCC = [0.15, 0.10, 0.07, 0.05, 0.05, 0.04, 0.03]
B_MCC = [0.03, 0.09, 0.12, 0.13, 0.13, 0.13, 0.13]
GRID = (0.0, 0.1, 0.25, 0.4, 0.5, 0.68, 1.0)


def deployment_table(units: int = 6) -> pd.DataFrame:
    rows = [{"model": model, "repeat": u // 3, "outer": u % 3, "z0": z0, "weight": w, "mcc": values[i], "auc": 0.6}
            for model, values in (("hits_only", A_MCC), ("selection_aware", B_MCC)) for z0 in (1.0, 1.96) for i, w in enumerate(GRID) for u in range(units)]
    return pd.DataFrame(rows)


def test_the_mcc_agrees_with_scikit_learn_along_the_last_axis():
    rng = np.random.default_rng(0)
    y, call = rng.random((4, 200)) < 0.2, rng.random((4, 200)) < 0.25
    expected = [matthews_corrcoef(a, b) for a, b in zip(y, call, strict=True)]
    assert np.allclose(sd.mcc_of(y, call), expected)
    assert sd.mcc_of(np.zeros(10, dtype=bool), np.ones(10, dtype=bool)) == 0.0  # no positive: a zero denominator gives 0


def test_two_files_with_the_same_calls_never_differ_and_different_calls_are_counted():
    rng = np.random.default_rng(1)
    y, call = rng.random(500) < 0.1, rng.random(500) < 0.1
    diff, n_diff = sd.live_differences(y, call, call, np.ones(500), 100, 50, rng)
    assert np.all(diff == 0) and n_diff == 0
    other = ~call
    _, n_diff = sd.live_differences(y, call, other, np.ones(500), 100, 50, rng)
    assert n_diff == 100


def test_a_threshold_between_two_regimes_has_small_errors_and_the_rule_reads_it():
    rng = np.random.default_rng(2)
    low = rng.normal(-0.13, 0.085, 20000)
    high = [rng.normal(0.07, 0.085, 20000) for _ in range(5)]
    best = sd.choose_threshold(low, high, 0.13, 0.07)
    assert -0.1 < best["threshold"] < 0.0
    assert best["error_low"] < 0.2 and best["error_high"] < 0.2
    result = {"usable": True, "threshold": best["threshold"], "fallback": "A"}
    assert sd.apply_rule(0.05, result) == "B" and sd.apply_rule(-0.15, result) == "A"
    assert sd.apply_rule(best["threshold"], result) == "B"  # at the threshold itself the rule gives B
    assert sd.apply_rule(0.5, {**result, "usable": False}) == "A"  # a detector that is not usable falls back


def test_the_decision_on_the_deployment_values_is_usable_and_the_fallback_is_the_hits_only_file():
    table = deployment_table()
    rng = np.random.default_rng(3)
    distributions = {(z0, w): rng.normal(b - a, 0.085, 6000) for z0 in (1.0, 1.96) for w, a, b in zip(GRID, A_MCC, B_MCC, strict=True)}
    decision = sd.decide(table, distributions)
    assert decision["usable"] and -0.08 < decision["threshold"] < 0.0
    assert abs(decision["delta_low"] - 0.12) < 1e-9 and 0.06 < decision["delta_high"] < 0.09
    assert decision["fallback"] == "A" and decision["regret"]["hits_only"] < decision["regret"]["selection_aware"]


def test_a_detector_with_distributions_that_overlap_is_not_usable():
    table = deployment_table()
    rng = np.random.default_rng(4)
    distributions = {(z0, w): rng.normal(0.0, 0.3, 6000) for z0 in (1.0, 1.96) for w in GRID}
    assert not sd.decide(table, distributions)["usable"]


def test_the_bootstrap_interval_contains_the_mean_and_is_narrower_with_more_units():
    rng = np.random.default_rng(5)
    a, b = rng.normal(0.1, 0.05, 15), rng.normal(0.0, 0.05, 15)
    mean, lo, hi = sd.bootstrap_interval(a, b)
    assert lo < mean < hi
    _, lo_big, hi_big = sd.bootstrap_interval(rng.normal(0.1, 0.05, 150), rng.normal(0.0, 0.05, 150))
    assert hi_big - lo_big < hi - lo


def test_the_stored_evidence_is_read_from_the_log_and_the_band_table():
    log = "x\nequivalent non-hit weight on the job-012 axis: 0.51 (95%: 0.39-0.69)\ny\n"
    assert sd.pac_estimate(log) == {"w": 0.51, "low": 0.39, "high": 0.69}
    band = pd.DataFrame({"w": [0.0, 0.05, 1.5], "c": [0.4, 0.3, 3.0], "chi2": [7.0, 5.9, 9.0], "plausible": [True, True, True]})
    assert sd.board_band(band) == {"best_w": 0.05, "best_c": 0.3, "w_low": 0.0, "w_high": 1.5}

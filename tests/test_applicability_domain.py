"""Applicability-domain bins and tiers, and the diagnostics of why an idea added nothing (the arithmetic only; both modules read the nested caches in the queue)."""

import numpy as np
import pandas as pd

from cypmode.challenge import applicability_domain as ad
from cypmode.challenge import why_no_gain as wng


def test_every_similarity_lands_in_one_bin_and_the_edges_are_closed_on_the_left():
    similarity = np.array([0.0, 0.299, 0.3, 0.45, 0.6, 0.7, 1.0])
    assert ad.bin_of(similarity).tolist() == [0, 0, 1, 2, 4, 5, 5]
    assert len(ad.BIN_LABELS) == len(ad.BIN_EDGES) - 1


def test_bin_auc_and_its_interval_resample_compounds_and_reflect_the_signal():
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, n)
    frame = pd.DataFrame({"row": np.repeat(np.arange(n), 2), "y": np.repeat(y, 2), "score": np.repeat(y + rng.normal(0, 0.8, n), 2) + rng.normal(0, 0.05, 2 * n)})
    point, low, high = ad.auc_with_interval(frame, np.random.default_rng(1))
    assert 0.75 < point < 0.95 and low < point < high and high - low < 0.15
    one_class = frame.assign(y=1)
    assert np.isnan(ad.auc_with_interval(one_class, np.random.default_rng(1))[0])


def test_the_tier_follows_the_size_of_the_bin_and_the_lower_end_of_its_interval():
    held = pd.DataFrame({"row": np.arange(400), "repeat": 0, "outer": 0, "y": np.tile([0, 1], 200), "score": np.tile([0.0, 1.0], 200), "nn": np.r_[np.full(300, 0.45), np.full(100, 0.8)], "z": 2.0})
    table = ad.bin_table(held, "CYP3A4").query("z0 == 1.0").set_index("bin")
    assert table.loc["0.4 - 0.5", "tier"] == "supported"
    assert table.loc[">= 0.7", "tier"] == "supported" and table.loc["< 0.3", "tier"] == "unsupported"
    noisy = held.assign(score=np.random.default_rng(0).normal(size=400))
    assert ad.bin_table(noisy, "CYP3A4").query("z0 == 1.0").set_index("bin").loc["0.4 - 0.5", "tier"] == "weak"


def test_the_detectable_effect_grows_with_the_standard_error_and_is_about_two_and_a_half_ses():
    assert wng.detectable_effect(0.01, 15) < wng.detectable_effect(0.02, 15)
    assert 2.3 < wng.detectable_effect(1.0, 200) < 2.7


def test_reasons_are_assigned_by_the_rules_in_order_and_carry_their_numbers():
    row = pd.Series({"weight_share": 0.01, "redundancy_mean": 0.9, "redundancy_max": 0.95, "delta_auc": -0.004, "detectable_effect": 0.02})
    text = wng.reason(row)
    assert text.index("almost no weight") < text.index("redundant") < text.index("lowered the AUC") < text.index("smaller than the smallest effect")
    clear = pd.Series({"weight_share": 0.4, "redundancy_mean": 0.3, "redundancy_max": 0.5, "delta_auc": 0.03, "detectable_effect": 0.01})
    assert wng.reason(clear) == "no reason found in these four measures"


def test_the_cluster_bootstrap_gives_a_wider_interval_than_the_compound_bootstrap_when_series_share_their_fate():
    rng = np.random.default_rng(3)
    n_clusters, size = 40, 8
    cluster = np.repeat(np.arange(n_clusters), size)
    series_effect = np.repeat(rng.normal(0, 1.2, n_clusters), size)  # members of a series are scored alike
    y = (rng.random(n_clusters * size) < 0.3).astype(int)
    frame = pd.DataFrame({"row": np.arange(len(y)), "y": y, "score": y * 0.8 + series_effect + rng.normal(0, 0.3, len(y)), "cluster": cluster})
    _, low_c, high_c = ad.auc_with_interval(frame, np.random.default_rng(1))
    _, low_r, high_r = ad.auc_with_interval(frame.drop(columns="cluster"), np.random.default_rng(1))
    assert (high_c - low_c) > (high_r - low_r)


def test_equivalence_needs_an_interval_inside_the_margin_and_a_noisy_zero_is_inconclusive():
    assert wng.equivalence(0.0005, 0.002, 15)[2] == "equivalent"
    assert wng.equivalence(0.0005, 0.02, 15)[2] == "inconclusive"
    assert wng.equivalence(0.05, 0.005, 15)[2] == "better"
    assert wng.equivalence(-0.05, 0.005, 15)[2] == "worse"


def test_the_design_floor_does_not_shrink_with_more_repeats():
    standard_error_15 = np.sqrt((1 / 15 + 0.25) * 0.01**2)  # 5 folds x 3 repeats
    standard_error_150 = np.sqrt((1 / 150 + 0.25) * 0.01**2)  # ten times the repeats
    floor_15, floor_150 = wng.design_floor(standard_error_15, 15, 4000, 1000), wng.design_floor(standard_error_150, 150, 4000, 1000)
    assert abs(floor_15 - floor_150) < 1e-12
    assert wng.detectable_effect(standard_error_150, 150) < wng.detectable_effect(standard_error_15, 15)  # the detectable effect still falls, towards the floor


def test_the_partial_correlation_removes_what_the_stack_already_says():
    rng = np.random.default_rng(0)
    z = rng.normal(size=2000)
    y = (z + rng.normal(0, 1, 2000) > 0).astype(int)
    x_copy = z + rng.normal(0, 0.05, 2000)  # a candidate that only repeats the stack
    x_new = rng.normal(size=2000) + 0.8 * (y - 0.5)  # a candidate with its own information
    assert abs(wng.partial_correlation(x_copy, y, z)) < 0.08
    assert wng.partial_correlation(x_new, y, z) > 0.15

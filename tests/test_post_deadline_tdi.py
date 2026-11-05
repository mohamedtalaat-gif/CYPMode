# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import numpy as np
import pandas as pd
import pytest

from cypmode.post_deadline import tdi
from cypmode.post_deadline.released import DEST


def _table(rng, n=60, d_std=0.1, t_std=0.1):
    """An invented test table with both arms of both isoforms; the labels follow the rule of the challenge."""
    frame = pd.DataFrame({"Molecule_Name": [f"M{i}" for i in range(n)], "SMILES": ["C"] * n})
    for iso in ("CYP2D6", "CYP3A4"):
        d, t = rng.uniform(3.5, 6.5, n), rng.uniform(3.5, 6.5, n)
        frame[f"{iso}_pIC50_direct_inhibition"], frame[f"{iso}_pIC50_TDI_condition"] = d, t
        frame[f"{iso}_pIC50_direct_inhibition_std"], frame[f"{iso}_pIC50_TDI_condition_std"] = d_std, t_std
        shift = t - np.maximum(d, 4.0)
        frame[f"{iso}_is_TDI"] = pd.array(shift > np.log10(2.0), dtype="boolean")
    return frame


def _calls(truth, flips=()):
    frame = truth[["Molecule_Name"]].copy()
    for endpoint in tdi.ENDPOINTS:
        frame[endpoint] = truth[endpoint].astype(bool).to_numpy()
    for endpoint, rows in flips:
        frame.loc[rows, endpoint] = ~frame.loc[rows, endpoint]
    return frame


def test_the_gate_identifies_the_candidate_that_matches_the_board_exactly_and_keeps_the_registered_rule_beside_it():
    def summary(mean, std):
        return pd.DataFrame({"MCC_mean": [mean], "MCC_std": [std]}, index=pd.Index(["MA"], name="Endpoint"))

    board, board_std = 0.3723, 0.0378
    near = tdi.gate({"A": summary(board, board_std), "B": summary(board + 0.0002, board_std + 0.00001)}, board=board, board_std=board_std)
    assert near["passed"] and near["submitted"] == "A" and not near["registered"]["passed"] and near["registered"]["within_tolerance"] == ["A", "B"]  # the registered tolerance cannot tell them apart
    far = tdi.gate({"A": summary(board, board_std), "B": summary(board - 0.05, board_std)}, board=board, board_std=board_std)
    assert far["passed"] and far["submitted"] == "A" and far["registered"]["passed"]
    assert not tdi.gate({"A": summary(board + 0.001, board_std), "B": summary(board - 0.001, board_std)}, board=board, board_std=board_std)["passed"]  # neither matches
    assert not tdi.gate({"A": summary(board, board_std), "B": summary(board, board_std)}, board=board, board_std=board_std)["passed"]  # identical files cannot be told apart


def test_two_identical_files_show_no_difference_and_a_better_file_is_named():
    truth = _table(np.random.default_rng(1))
    perfect = _calls(truth)
    worse = _calls(truth, flips=[(endpoint, slice(0, 20)) for endpoint in tdi.ENDPOINTS])
    same = tdi.paired_difference(perfect, perfect, truth, resamples=200)
    assert set(same["verdict"]) == {"no difference shown"} and np.allclose(same["mean_difference"], 0.0)
    better = tdi.paired_difference(worse, perfect, truth, resamples=200)
    assert set(better["verdict"]) == {"second better"} and (better["mean_difference"] > 0).all()
    assert set(better["endpoint"]) == {*tdi.ENDPOINTS, "macro"}


def test_the_call_rates_are_taken_among_the_scored_compounds_only():
    truth = _table(np.random.default_rng(2), n=40)
    truth.loc[:9, "CYP2D6_is_TDI"] = pd.NA
    filled = truth.assign(**{endpoint: truth[endpoint].fillna(False) for endpoint in tdi.ENDPOINTS})
    rates = tdi.call_rates({"A": _calls(filled)}, truth).set_index("endpoint")
    assert rates.loc["CYP2D6_is_TDI", "scored"] == 30 and rates.loc["CYP3A4_is_TDI", "scored"] == 40
    assert rates.loc["CYP3A4_is_TDI", "called_A"] == pytest.approx(rates.loc["CYP3A4_is_TDI", "true_positive_rate"])


def test_the_rule_agrees_with_labels_that_follow_it_and_counts_a_changed_label_and_a_missing_value():
    truth = _table(np.random.default_rng(3), n=50)
    assert (tdi.rule_agreement(truth)["disagree"] == 0).all() and tdi.rule_disagreements(truth).empty
    truth.loc[4, "CYP3A4_is_TDI"] = not truth.loc[4, "CYP3A4_is_TDI"]
    truth.loc[7, "CYP3A4_pIC50_TDI_condition"] = np.nan
    truth.loc[8, "CYP2D6_is_TDI"] = pd.NA
    summary = tdi.rule_agreement(truth).set_index("isoform")
    assert summary.loc["CYP3A4", "disagree"] == 1 and summary.loc["CYP3A4", "not_computable"] == 1 and summary.loc["CYP2D6", "computable_unlabelled"] == 1
    listed = tdi.rule_disagreements(truth)
    assert list(listed["Molecule_Name"]) == ["M4"] and list(listed["isoform"]) == ["CYP3A4"]


def test_the_scored_set_is_compared_with_the_confidence_rule_at_each_cut_off():
    truth = _table(np.random.default_rng(4), n=80, d_std=0.5, t_std=0.5)
    truth.loc[:4, "CYP2D6_is_TDI"] = pd.NA
    table = tdi.scored_set_agreement(truth).set_index(["isoform", "z0"])
    for (_, _), row in table.iterrows():
        assert row[["both", "only_organisers", "only_project", "neither"]].sum() == 80
    assert table.loc[("CYP2D6", 1.0), "both"] >= table.loc[("CYP2D6", 1.96), "both"]  # a stricter cut-off keeps fewer compounds


def test_the_ceiling_is_one_for_scores_that_separate_the_classes_and_the_submitted_threshold_can_be_lower():
    truth = _table(np.random.default_rng(5), n=40)
    y = truth["CYP3A4_is_TDI"].astype(bool).to_numpy()
    scores = pd.DataFrame({"Molecule_Name": truth["Molecule_Name"], "score": np.where(y, 1.0, 0.0) + np.linspace(0, 0.01, len(y)), "call": True})
    result = tdi.threshold_ceiling(scores, truth, "CYP3A4_is_TDI")
    assert result["mcc_ceiling"] == pytest.approx(1.0) and result["mcc_submitted"] < 1.0 and result["scored"] == 40


@pytest.mark.skipif(not (DEST / "organisers/evaluation/evaluate_predictions.py").exists(), reason="the evaluation code of the organisers is not in this checkout (python -m cypmode.post_deadline.released)")
def test_the_organisers_scoring_gives_a_macro_value_that_is_the_mean_of_the_two_isoform_values():
    truth = _table(np.random.default_rng(6), n=80)
    perfect = _calls(truth)
    flipped = _calls(truth, flips=[("CYP2D6_is_TDI", slice(0, 25))])
    for frame in (perfect, flipped):
        summary = tdi.board_scores(frame, truth)
        assert summary.loc["MA", "MCC_mean"] == pytest.approx((summary.loc["CYP2D6_is_TDI", "MCC_mean"] + summary.loc["CYP3A4_is_TDI", "MCC_mean"]) / 2)
    assert tdi.board_scores(perfect, truth).loc["MA", "MCC_mean"] == pytest.approx(1.0)
    assert tdi.board_scores(flipped, truth).loc["MA", "MCC_mean"] < 1.0

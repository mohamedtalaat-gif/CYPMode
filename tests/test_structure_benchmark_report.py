# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import pandas as pd

from cypmode.validation.structure_benchmark_report import (
    selection_outcomes,
    summarize,
    summarize_variants,
    variant_comparison,
)

TABLE = pd.DataFrame(
    [
        # compound A: Boltz's own pick is bad, a later Boltz sample is good and the template score finds it
        ("A", "boltz", 0, 5.0, 0.2),
        ("A", "boltz", 1, 1.0, 0.9),
        ("A", "openfold3", 0, 3.0, 0.5),
        # compound B: only Boltz ran
        ("B", "boltz", 0, 1.5, 0.4),
        ("B", "boltz", 1, 4.0, 0.1),
    ],
    columns=["compound", "engine", "sample", "true_rmsd", "novel_template_score"],
)


def _by_rule(table):
    return selection_outcomes(table).pivot(index="compound", columns="rule", values="rmsd")


def test_each_rule_reports_the_true_rmsd_of_the_pose_it_would_submit():
    out = _by_rule(TABLE)
    assert out.loc["A", "boltz first pick"] == 5.0
    assert out.loc["A", "boltz best sample (oracle)"] == 1.0
    assert out.loc["A", "template score over Boltz samples"] == 1.0
    assert out.loc["A", "template score over both engines"] == 1.0
    assert out.loc["A", "either engine best sample (oracle)"] == 1.0


def test_a_rule_that_needs_an_engine_the_compound_never_ran_on_is_left_out_not_zero_filled():
    out = _by_rule(TABLE)
    assert pd.isna(out.loc["B", "openfold3 first sample"])
    assert pd.isna(out.loc["B", "template score over both engines"])
    assert out.loc["B", "boltz first pick"] == 1.5


def test_summary_counts_only_compounds_a_rule_could_score_and_orders_by_mean_rmsd():
    summary = summarize(selection_outcomes(TABLE))
    assert summary.loc["boltz first pick", "compounds"] == 2
    assert summary.loc["openfold3 first sample", "compounds"] == 1
    assert summary.loc["boltz first pick", "success_under_2A"] == 0.5  # B passes the 2 A cut, A does not
    assert list(summary["mean_rmsd"]) == sorted(summary["mean_rmsd"])


def _poses(compound, rmsds, scores):
    return pd.DataFrame(
        {"compound": compound, "engine": "boltz", "sample": range(len(rmsds)), "true_rmsd": rmsds, "novel_template_score": scores}
    )


def test_variant_comparison_lines_ligands_up_across_suffixes_and_budgets():
    unguided = pd.concat([_poses("L1", [5.0, 1.0, 4.0, 0.5], [0.1, 0.9, 0.2, 0.99]), _poses("L2", [3.0, 2.5, 2.0], [0.3, 0.2, 0.1]), _poses("L3", [1.0], [0.5])])
    guided = pd.concat([_poses("L1-febond", [1.5, 1.0, 4.0], [0.8, 0.2, 0.1]), _poses("L2-febond", [1.0, 6.0, 6.0], [0.9, 0.1, 0.1])])
    comparison = variant_comparison({"unguided": unguided, "febond": guided}, samples=3)

    assert set(comparison["compound"]) == {"L1", "L2"}  # L3 was not run restrained
    by = comparison.set_index(["variant", "compound"])
    assert by.loc[("unguided", "L1"), "best_sample"] == 1.0  # the 4th sample is outside the budget of 3
    assert by.loc[("unguided", "L1"), "template_pick"] == 1.0
    assert by.loc[("febond", "L2"), "first_pick"] == 1.0

    summary = summarize_variants(comparison)
    assert summary.loc["febond", "mean_rmsd_first_pick"] == 1.25 and summary.loc["unguided", "ligands"] == 2
    assert summary.loc["febond", "under_2A_first_pick"] == 1.0

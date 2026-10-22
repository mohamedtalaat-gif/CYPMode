# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import pandas as pd

from cypmode.validation import structure_gates as sg


def table(compound_prefix, rmsds, suffix=""):
    rows = []
    for ligand, values in rmsds.items():
        for sample, r in enumerate(values):
            rows.append({"compound": f"{ligand}{suffix}", "engine": "boltz", "sample": sample, "true_rmsd": r, "novel_template_score": 0.1 * sample})
    return pd.DataFrame(rows)


def test_restraint_gate_needs_lower_first_pick_rmsd_and_not_worse_validity():
    unguided = table("u", {"a": [5.0, 5, 5], "b": [4.0, 4, 4], "c": [6.0, 6, 6]})
    better = table("v", {"a": [2.0, 2, 2], "b": [3.0, 3, 3], "c": [5.0, 5, 5]}, "-febond")
    worse = table("v", {"a": [7.0, 7, 7], "b": [6.0, 6, 6], "c": [8.0, 8, 8]}, "-fecontact")
    always = lambda compound, variant: True
    out = sg.restraint_gates(unguided, {"febond": better, "fecontact": worse}, always)
    assert out["restraint_bond"]["met"] is True and out["restraint_contact"]["met"] is False
    invalid = lambda compound, variant: variant == "unguided"
    assert sg.restraint_gates(unguided, {"febond": better}, invalid)["restraint_bond"]["met"] is False


def test_pilot_gate_needs_both_wider_pools_and_a_closer_best_pose():
    pools = pd.DataFrame({
        "ligand": ["a", "b", "a", "b"], "lever": ["seeds", "seeds", "stepscale", "stepscale"],
        "spread": [3.0, 3.0, 1.0, 1.0], "unguided_spread": [1.0, 1.0, 1.0, 1.0],
        "best_rmsd": [1.0, 1.5, 1.0, 1.5], "unguided_best": [3.0, 3.0, 3.0, 3.0],
    })
    out = sg.pilot_gates(pools)
    assert out["pilot_seeds"]["met"] is True and out["pilot_stepscale"]["met"] is False  # closer but not wider

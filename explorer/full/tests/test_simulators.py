# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from simulators import confidence, label_rule, molecule


def test_the_label_rule_simulator_agrees_with_both_forms_of_the_rule_everywhere():
    rng = np.random.default_rng(0)
    for d, t in zip(rng.uniform(1, 8, 2000), rng.uniform(1, 8, 2000), strict=True):
        assert label_rule.evaluate(d, t)["agree"]
    assert label_rule.evaluate(5.0, 5.4)["literal rule"] and not label_rule.evaluate(5.0, 5.2)["literal rule"]
    assert label_rule.evaluate(3.0, 4.31)["literal rule"] and not label_rule.evaluate(3.0, 4.29)["literal rule"]


def test_the_label_rule_simulator_reproduces_every_label_of_the_real_compounds():
    points = pd.read_csv(HERE / "data/label_points.csv")
    assert len(points) == 3827
    got = np.array([label_rule.evaluate(d, t)["single threshold"] for d, t in zip(points["d"], points["t"], strict=True)])
    assert (got == points["label"].astype(bool).to_numpy()).all()


def test_the_confidence_simulator_gives_the_z_of_the_real_compounds_and_ignores_the_noise_of_d_below_the_floor():
    points = pd.read_csv(HERE / "data/label_points.csv")
    row = points[points["isoform"] == "CYP3A4"].iloc[7]
    result = confidence.evaluate(row["d"], row["t"], row["sd_d"], row["sd_t"])
    assert abs(result["z"] - row["z"]) < 1e-9
    below = confidence.evaluate(3.0, 5.0, 0.9, 0.2)
    assert abs(below["sd of the shift"] - 0.2) < 1e-12 and confidence.evaluate(5.0, 6.0, 0.9, 0.2)["sd of the shift"] > 0.9


def test_the_molecule_simulator_finds_the_donor_nitrogen_the_environment_of_a_bit_and_rejects_nonsense():
    assert molecule.parse("not a molecule") is None and molecule.parse("") is None
    mol = molecule.parse("c1ccc(cc1)Cn1ccnc1")  # a benzyl imidazole
    assert molecule.donor("c1ccc(cc1)Cn1ccnc1")["Boltz-2 atom name"].startswith("N") and molecule.type_ii_motifs(mol)
    bits = molecule.morgan_bits(mol)
    assert bits and all(radius <= 2 for envs in bits.values() for _, radius in envs)
    center, radius = next(iter(bits.values()))[0]
    atoms, _ = molecule.environment(mol, center, radius)
    assert center in atoms
    assert molecule.drawing(mol, atoms).startswith("<?xml") or "<svg" in molecule.drawing(mol, atoms)
    assert molecule.summary(mol)["heavy atoms"] == 12  # six ring atoms, one methylene, five atoms of the imidazole


def test_the_kinetics_simulators_have_the_properties_that_the_notebook_derives():
    from simulators import kinetics

    assert abs(kinetics.michaelis_menten(2.0, 10.0, 2.0) - 5.0) < 1e-12  # half of the maximum at [S] = Km
    t = kinetics.elimination(100.0, 6.0, 12.0, 1, 6.0)
    assert abs(t["one dose"].iloc[-1] - 50.0) < 1e-9  # the half-life
    assert abs(kinetics.accumulation(6.0, 6.0) - 2.0) < 1e-12  # dosing every half-life doubles the steady state
    v0 = kinetics.michaelis_menten(5.0, 10.0, 2.0)
    competitive = kinetics.inhibited_rate(1e9, 3.0, 10.0, 2.0, 1.0, "competitive")  # enough substrate wins the competition
    assert abs(competitive - 10.0) < 1e-6 and kinetics.inhibited_rate(5.0, 3.0, 10.0, 2.0, 1.0, "non-competitive") < v0 / 3.9
    assert abs(kinetics.inhibited_rate(5.0, 0.0, 10.0, 2.0, 1.0, "mixed") - v0) < 1e-12
    kobs, activity = kinetics.tdi_activity(2.0, 0.1, 2.0, [0.0, 10.0])
    assert abs(kobs - 0.05) < 1e-12 and abs(activity[1] - np.exp(-0.5)) < 1e-12  # half of kinact at [I] = K_I
    assert abs(kinetics.hill(3.0, 3.0, 2.5) - 0.5) < 1e-12 and abs(kinetics.pic50(1e-6) - 6.0) < 1e-12
    shift = kinetics.arm_shift(5.0, 5.30103)
    assert abs(shift["IC50 shift (fold)"] - 2.0) < 1e-4


def test_the_statistics_simulators_are_the_projects_functions_and_show_the_optimism_of_choosing():
    from simulators import statistics

    differences = [0.012, 0.004, 0.019, 0.008, 0.015, 0.001, 0.010, 0.007, 0.013, 0.006]
    out = statistics.compare(differences, 3000, 750)
    from cypmode.challenge.model_comparison import corrected_resampled_t

    reference = corrected_resampled_t(differences, 3000, 750)
    assert abs(out["corrected t"] - reference.t) < 1e-12 and out["corrected p"] > out["naive p"]
    assert abs(out["variance factor"] - (1 / 10 + 750 / 3000)) < 1e-12
    labels, scores = statistics.draw_scores(0.3, 1.5, 4000, 1)
    m = statistics.metrics_at(labels, scores, 0.75)
    assert 0.8 < m["AUC"] < 0.9 and m["MCC"] > 0.3 and abs(labels.mean() - 0.3) < 0.03
    optimism = statistics.selection_optimism(50, 60, repeats=80)
    assert optimism["best validation AUC"] > 0.62 and abs(optimism["its AUC on fresh data"] - 0.5) < 0.04
    assert statistics.spread(0.3, 1.0, 50, 0.5, repeats=60)["MCC sd"] > statistics.spread(0.3, 1.0, 2000, 0.5, repeats=60)["MCC sd"]


def test_the_fingerprint_simulators_give_the_tanimoto_of_the_project_and_the_descriptors_of_rdkit():
    from rdkit import Chem
    from simulators import fingerprints

    a, b = Chem.MolFromSmiles("c1ccc(cc1)Cn1ccnc1"), Chem.MolFromSmiles("c1ccc(cc1)Cn1cncn1")
    sim = fingerprints.tanimoto(a, b)
    assert abs(sim["similarity"] - sim["rdkit"]) < 1e-12 and 0 < sim["similarity"] < 1 and sim["shared"] == len(sim["shared bits"])
    assert fingerprints.tanimoto(a, a)["similarity"] == 1.0
    assert {r["descriptor"]: r["value"] for r in fingerprints.descriptor_rows(a)}["heavy atoms"] == 12


def test_the_gate_simulator_is_the_rule_of_the_project_and_the_false_alarms_follow_the_logic_of_6_7():
    from simulators import gates

    good = {1.0: [0.012, 0.015, 0.011, 0.014, 0.013, 0.016, 0.012, 0.010, 0.015, 0.013, 0.014, 0.012, 0.011, 0.016, 0.013], 1.96: [0.011, 0.014, 0.012, 0.013, 0.012, 0.015, 0.011, 0.010, 0.014, 0.012, 0.013, 0.011, 0.012, 0.015, 0.012]}
    bad = {1.0: good[1.0], 1.96: [0.001, -0.004, 0.003, -0.002, 0.002, -0.001, 0.004, -0.003, 0.001, 0.0, -0.002, 0.003, -0.001, 0.002, 0.0]}
    assert gates.verdict(good, 3000, 750)["met"] and not gates.verdict(bad, 3000, 750)["met"]
    f = gates.false_alarms(0.9)
    assert f["both cut-offs (the project's rule)"] <= f["one cut-off fixed in advance"] <= f["either of the two cut-offs"] and f["one cut-off fixed in advance"] < 0.04
    table = gates.holm_table([0.01, 0.04, 0.03, 0.2])
    assert [round(r["Holm-adjusted p"], 3) for r in table] == [0.04, 0.09, 0.09, 0.2]


def test_the_learning_simulators_show_overfitting_averaging_and_a_sensible_best_split():
    from simulators import learning

    table = learning.overfitting(12, 30, 0.3, 1)
    assert table["training error"].is_monotonic_decreasing or table["training error"].iloc[-1] < table["training error"].iloc[0]
    assert table["error on new data"].iloc[-1] > table["error on new data"].min() * 1.5  # a high degree is worse on new data than the best degree
    _, impurity, best = learning.tree_split(3.0, 400, 2)
    assert abs(best - 1.5) < 0.8 and impurity["impurity after the split"].min() < 0.3
    boosted = learning.boosting(200, 0.1, 2, 0.2, 3)["errors"]
    assert boosted["training error"].iloc[-1] < boosted["training error"].iloc[0]
    trees = learning.forest(100, 0.3, 4)
    assert trees["error on new data"].iloc[-1] < trees["error on new data"].iloc[0] and trees["error on new data"].iloc[-1] < trees["one deep tree"].iloc[0]


def test_the_neighbour_and_nested_simulators_count_and_vote_correctly():
    from rdkit import Chem  # noqa: F401
    from simulators import neighbours, nested

    prints = neighbours.prepare(["c1ccccc1CCN", "c1ccccc1CCO", "CCCCCCCC"])
    result = neighbours.nearest("c1ccccc1CCN", prints, 2)
    assert result[0] == (0, 1.0) and result[1][0] == 1
    assert neighbours.vote([1.0, 0.0, float("nan")], [0.9, 0.9, 0.9], False) == 0.5 and neighbours.nearest("garbage", prints, 2) == []
    p = nested.plan(3000, 5, 5, 3, 24)
    assert p["outer scores"] == 15 and p["total fits"] == 15 * (24 * 5 + 24) and abs(p["correction factor of the t-test"] - (1 / 15 + 600 / 2400)) < 1e-12


def test_the_physics_the_committee_the_weights_and_the_calls_are_the_projects_functions():
    from simulators import confidence, physics, statistics

    out = physics.run("c1ccc(cc1)Cn1ccnc1")
    assert out is not None and 50 < len(out["features"]) <= out["n_quantities"] == 133 and len(out["atoms"]) == 12  # the quantities that do not apply to a molecule are NaN and all(k in out["features"] for k in physics.HEADLINE)
    assert abs(sum(float(a["q"]) for a in out["atoms"])) < 3 and len(physics.charge_colors(out["atoms"])) == 12 and physics.run("not a molecule") is None
    c = confidence.committee(5.0, 5.6, 0.25, 0.25)
    assert abs(c["simulated share positive"] - c["analytic probability"]) < 0.02 and c["label as measured"]
    w = confidence.weights_by_mode(np.array([0.5, 1.5, 2.5, np.nan]), "drop1.0")
    assert w["rows kept"] == 2 and w["rows"] == 4 and confidence.weights_by_mode(np.array([0.5, 4.0]), "soft")["total weight"] == 0.25 + 1.0
    rows = statistics.calls_by_multiple(0.3, 1.5, 3000, [0.5, 1.0, 2.0])
    assert [round(r["share called"], 2) for r in rows] == [0.15, 0.3, 0.6] and rows[1]["MCC"] > rows[0]["MCC"] * 0.5


def test_the_out_of_fold_simulator_reads_the_stored_bundle_and_its_auc_is_a_rank_probability():
    from simulators import oof

    root = HERE / "data/public"
    tables = {e: pd.read_csv(root / f"oof_{e}.csv.gz") for e in ("CYP3A4", "CYP2D6")}
    candidates = pd.read_csv(root / "candidates.csv").fillna({"blocks": ""})
    y = np.array([0, 0, 1, 1, 1, 0])
    assert abs(oof.auc_of(y, np.array([0.1, 0.4, 0.35, 0.8, 0.9, 0.2])) - 8 / 9) < 1e-12
    ranking = oof.candidate_table(tables, candidates, "CYP3A4")
    assert len(ranking) > 10 and ranking["mean"].is_monotonic_decreasing and 0.5 < ranking["mean"].iloc[0] < 1
    best = ranking["candidate"].iloc[0]
    frame = tables["CYP3A4"]
    result = oof.confusion(frame["y"].to_numpy(), frame[f"{best}|0"].to_numpy(), 0.3)
    assert result["TP"] + result["FP"] == result["called positive"] and result["TP"] + result["FN"] == int(frame["y"].sum())
    assert len(oof.fold_table(tables, "CYP3A4", best)) == oof.folds_of(frame)


def test_the_simulators_after_the_deadline_reproduce_the_numbers_of_the_method_report():
    from simulators import part_13, store

    ligands = store.table("pd_ligands")
    mean, low, high = part_13.paired_interval((ligands["external_pli"] - ligands["submitted_pli"]).to_numpy(), 10_000, 42)
    assert (round(mean, 4), round(low, 4), round(high, 4)) == (0.0687, 0.0052, 0.1336)  # question 6, the same seed and draw as the analysis
    assert int((ligands["external_pli"] > ligands["pool_best"]).sum()) == 9 and len(ligands) == 20
    gate = store.table("pd_gate")
    candidate_a = gate[gate["entry"] == "candidate A"].iloc[0]
    candidate_b = gate[gate["entry"] == "candidate B"].iloc[0]
    assert abs(candidate_a["scored"] - candidate_a["board"]) < 1e-12 and 1e-4 < abs(candidate_b["scored"] - candidate_b["board"]) < 1e-3
    threshold = store.table("pd_threshold")
    best = threshold[(threshold["enzyme"] == "CYP3A4") & (threshold["kind"] == "best threshold")].iloc[0]
    assert round(best["mcc"], 4) == 0.5177 and len(threshold[threshold["kind"] == "curve"]) == 118  # 59 call rates for each enzyme
    contact = (ligands["truth_distance"] <= 2.6) & (ligands["submitted_distance"] <= 2.6)
    assert int(contact.sum()) == 4 and int(((ligands["truth_distance"] > 2.6) & (ligands["submitted_distance"] <= 2.6)).sum()) == 8
    assert part_13.verdict(0.005, 0.13) == "the first is better" and part_13.verdict(-0.01, 0.02) == "no difference shown"

# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("rdkit")

from cypmode.challenge import pocket_pharmacophore as pp
from cypmode.validation import pose_pharmacophore as pose


def _empty_fields():
    return {name: np.zeros(pp.GRID_SHAPE) for name in (*pp.FAMILIES, "occupancy")}


def test_the_frame_of_a_heme_from_its_nitrogens_puts_z_on_the_side_of_the_ligand_and_x_toward_the_nitrogen_a():
    nitrogens = {"NA": np.array([2.0, 0, 0]), "NB": np.array([0, 2.0, 0]), "NC": np.array([-2.0, 0, 0]), "ND": np.array([0, -2.0, 0])}
    above = pp.frame_from_nitrogens(nitrogens, np.zeros(3), np.array([[0.0, 0.0, 4.0]]))
    below = pp.frame_from_nitrogens(nitrogens, np.zeros(3), np.array([[0.0, 0.0, -4.0]]))
    assert np.allclose(above[2], [0, 0, 1]) and np.allclose(below[2], [0, 0, -1]) and np.allclose(above[0], [1, 0, 0])
    assert pp.frame_from_nitrogens({"NA": nitrogens["NA"]}, np.zeros(3), np.zeros((1, 3))) is None


def test_a_pose_whose_features_sit_on_the_sites_scores_full_recall_and_one_far_from_them_scores_none():
    fields = _empty_fields()
    fields["Acceptor"] = pp.gaussian_field(np.array([[0.0, 0.0, 6.0]]), np.array([1.0]), 1.5)
    fields["occupancy"] = pp.gaussian_field(np.array([[0.0, 0.0, 6.0], [1.5, 0.0, 6.0]]), np.array([1.0, 1.0]), 1.2)
    sites = pp.sites_of(fields)
    features = {f: np.zeros((0, 3)) for f in pp.FAMILIES}
    features["Acceptor"] = np.array([[0.0, 0.0, 6.0]])
    on = pose.score_pose(np.array([[0.0, 0.0, 6.0], [1.5, 0.0, 6.0]]), features, fields, sites)
    assert on["recall"] == pytest.approx(1.0) and on["matched"] == 1 and on["precision"] == pytest.approx(1.0) and on["outside"] == 0.0
    features["Acceptor"] = np.array([[10.0, 10.0, 12.0]])
    off = pose.score_pose(np.array([[10.0, 10.0, 12.0], [11.0, 10.0, 12.0]]), features, fields, sites)
    assert off["recall"] == 0.0 and off["matched"] == 0 and off["outside"] == 1.0 and off["field"] < on["field"]


def test_a_pose_with_no_typed_feature_matches_no_site_and_has_no_precision():
    fields = _empty_fields()
    fields["Donor"] = pp.gaussian_field(np.array([[0.0, 0.0, 6.0]]), np.array([1.0]), 1.5)
    result = pose.score_pose(np.array([[0.0, 0.0, 6.0]]), {f: np.zeros((0, 3)) for f in pp.FAMILIES}, fields, pp.sites_of(fields))
    assert result["recall"] == 0.0 and np.isnan(result["precision"])


def test_the_references_of_the_scored_ligand_are_left_out_and_in_the_novel_regime_so_are_its_analogues():
    refs = pose.references()
    ketoconazole = next(r["smiles"] for r in refs if r["comp_id"] == "KLN")
    own = pose.kept_indices(ketoconazole, "compound")
    novel = pose.kept_indices(ketoconazole, "novel")
    assert len(own) < len(refs) and len(novel) <= len(own)
    from cypmode.validation.pose_rescoring import connectivity_key

    assert all(connectivity_key(refs[i]["smiles"]) != connectivity_key(ketoconazole) for i in own)
    assert set(novel) <= set(own)


def _pool(rhos_good: bool):
    rng = np.random.default_rng(0)
    rows = []
    for compound in ("a", "b", "c", "d"):
        rmsd = rng.uniform(1, 7, 12)
        signal = -rmsd if rhos_good else rng.normal(size=12)
        rows += [{"compound": compound, "true_rmsd": r, "mbr": rng.normal(), "field": s + rng.normal(scale=0.1), "recall": s + rng.normal(scale=0.1), "precision": 0.5, "outside": 0.1} for r, s in zip(rmsd, signal, strict=True)]
    return pd.DataFrame(rows)


def test_a_score_that_tracks_the_true_rmsd_meets_the_gate_and_noise_does_not():
    _, good = pose.evaluate(_pool(True))
    _, noise = pose.evaluate(_pool(False))
    assert pose.gate(good, "rho_pharmacophore_model")["met"] and not pose.gate(noise, "rho_pharmacophore_model")["met"]
    assert (good["pharmacophore_model_pick"] < good["mean_pool"]).all()


def test_the_use_test_needs_a_positive_mean_gain_robust_to_dropping_one_ligand_and_a_small_p():
    table = pd.DataFrame({"mbr_pick": [5.0, 4.0, 6.0, 5.5, 4.5, 6.5], "pharmacophore_model_pick": [3.0, 3.5, 4.0, 4.0, 3.0, 4.5]})
    assert pose.use_test(table, "pharmacophore_model")["met"]
    table.loc[0, "pharmacophore_model_pick"] = 20.0
    assert not pose.use_test(table, "pharmacophore_model")["met"]


def test_the_log_carries_the_gate_lines_of_the_gated_regime_only_and_the_use_line_only_for_a_met_gate():
    _, good = pose.evaluate(_pool(True))
    lines = pose.report(good, "novel", True)
    assert any(l.startswith("STRUCTURE GATE pose_pharmacophore_model ") and l.endswith("MET") for l in lines)
    assert any("pose_pharmacophore_model_use" in l for l in lines)
    assert not any("STRUCTURE GATE" in l for l in pose.report(good, "compound", False))


def test_a_predicted_pose_is_put_in_the_frame_of_its_own_heme():
    from cypmode.validation.pose_pools import pool_paths
    from cypmode.validation.structures import load_structure

    paths = pool_paths("bench-MWS", ("boltz",))
    if not paths or not Path(paths[0][2]).exists():
        pytest.skip("the benchmark structures are not in this checkout")
    smiles = pose.benchmark_smiles()["bench-MWS"]
    frame = pose.pose_in_frame(load_structure(paths[0][2]), smiles)
    assert frame is not None and frame["heavy"].shape[1] == 3 and frame["heavy"][:, 2].mean() > 1.5  # the ligand lies on the distal side of the porphyrin plane
    assert abs(float(np.linalg.det(frame["frame"])) - 1.0) < 1e-6


def test_the_crystal_poses_of_the_benchmark_are_scored_with_the_fields_built_without_their_own_references(monkeypatch):
    two = dict(list(pose.benchmark_smiles().items())[:2])
    monkeypatch.setattr(pose, "benchmark_smiles", lambda: two)
    crystal = pose.crystal_scores("novel")
    if crystal.empty:
        pytest.skip("the crystal structures are not in this checkout")
    assert {"compound", "crystal_field", "crystal_recall", "crystal_outside"} <= set(crystal.columns)
    assert (crystal.crystal_outside < 0.5).all() and (crystal.crystal_field > 0).all()
    lines = pose.crystal_report(crystal, pd.DataFrame({"compound": [crystal.compound[0]] * 3, "true_rmsd": [1.0, 2.0, 3.0], "field": [0.1, 0.2, 0.3], "recall": [0.1, 0.1, 0.1], "precision": [0.5] * 3, "outside": [0.0] * 3}), "novel")
    assert lines[0].startswith("information (not a gate)") and len(lines) == 5

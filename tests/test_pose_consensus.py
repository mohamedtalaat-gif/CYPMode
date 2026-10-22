# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("Bio")
pytest.importorskip("yaml")

from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from rdkit import Chem
from rdkit.Chem import AllChem

from cypmode.validation import pose_consensus
from cypmode.validation.constants import COORDINATION_DISTANCE_CUTOFF_A
from cypmode.validation.pose_consensus import (
    AGREEMENT_RMSD_A,
    aligned_ligand_rmsd,
    build_consensus_table,
    choose_pose,
    pose_metrics,
)

LIGAND_SMILES = "c1ccccc1Cn1ccnc1"  # benzylimidazole: a symmetric phenyl + a Type II imidazole


def _metrics(passed=True, failed=(), has_motif=True, n_fe=2.1, coordinated=None) -> dict:
    return {
        "all_passed": passed,
        "failed_checks": list(failed),
        "has_motif": has_motif,
        "closest_motif_n_fe": n_fe,
        "coordinated": (n_fe is not None and n_fe <= COORDINATION_DISTANCE_CUTOFF_A) if coordinated is None else coordinated,
    }


# --- choose_pose: the full decision table --------------------------------------------------------


def test_no_openfold3_prediction_keeps_boltz():
    assert choose_pose(_metrics(), None, None) == {"engine": "boltz", "reason": "no_openfold3_prediction"}


def test_no_boltz_prediction_uses_openfold3():
    assert choose_pose(None, _metrics(), None) == {"engine": "openfold3", "reason": "no_boltz_prediction"}


def test_only_openfold3_passing_posebusters_switches_to_it():
    decision = choose_pose(_metrics(passed=False, failed=["no_clashes"]), _metrics(), 5.0)
    assert decision == {"engine": "openfold3", "reason": "only_openfold3_passes_posebusters"}


def test_only_boltz_passing_posebusters_keeps_boltz():
    decision = choose_pose(_metrics(), _metrics(passed=False, failed=["no_clashes"]), 0.5)
    assert decision == {"engine": "boltz", "reason": "only_boltz_passes_posebusters"}


def test_posebusters_gate_outranks_agreement_and_coordination():
    # engines "agree" by RMSD and Boltz coordinates the heme -- but it fails PoseBusters, which scores 0
    decision = choose_pose(_metrics(passed=False, failed=["x"], n_fe=2.0), _metrics(n_fe=9.0), 0.1)
    assert decision["engine"] == "openfold3"


def test_neither_passing_picks_the_pose_with_fewer_failed_checks_and_ties_go_to_boltz():
    fewer = choose_pose(_metrics(passed=False, failed=["a", "b"]), _metrics(passed=False, failed=["a"]), 3.0)
    assert fewer == {"engine": "openfold3", "reason": "neither_passes_posebusters_fewest_failed_checks"}
    tie = choose_pose(_metrics(passed=False, failed=["a"]), _metrics(passed=False, failed=["b"]), 3.0)
    assert tie["engine"] == "boltz"


def test_both_valid_and_agreeing_keeps_boltz_even_if_only_openfold3_coordinates():
    decision = choose_pose(_metrics(n_fe=5.0), _metrics(n_fe=2.1), AGREEMENT_RMSD_A - 0.01)
    assert decision == {"engine": "boltz", "reason": "engines_agree"}


def test_agreement_threshold_is_inclusive():
    assert choose_pose(_metrics(), _metrics(), AGREEMENT_RMSD_A)["reason"] == "engines_agree"
    assert choose_pose(_metrics(), _metrics(), AGREEMENT_RMSD_A + 0.01)["reason"] != "engines_agree"


def test_disagreement_resolved_by_the_heme_coordination_prior_in_either_direction():
    boltz_coordinates = choose_pose(_metrics(n_fe=2.1), _metrics(n_fe=6.0), 4.0)
    assert boltz_coordinates == {"engine": "boltz", "reason": "engines_disagree_heme_coordination_prior"}
    of3_coordinates = choose_pose(_metrics(n_fe=6.0), _metrics(n_fe=2.1), 4.0)
    assert of3_coordinates == {"engine": "openfold3", "reason": "engines_disagree_heme_coordination_prior"}


def test_disagreement_with_the_same_coordination_state_keeps_boltz():
    both = choose_pose(_metrics(n_fe=2.1), _metrics(n_fe=2.2), 4.0)
    neither = choose_pose(_metrics(n_fe=6.0), _metrics(n_fe=7.0), 4.0)
    no_motif = choose_pose(_metrics(has_motif=False, n_fe=None), _metrics(has_motif=False, n_fe=None), 4.0)
    for decision in (both, neither, no_motif):
        assert decision == {"engine": "boltz", "reason": "engines_disagree_unresolved_keep_boltz"}


def test_unknown_rmsd_is_treated_as_disagreement_not_agreement():
    assert choose_pose(_metrics(n_fe=6.0), _metrics(n_fe=2.1), None)["engine"] == "openfold3"
    assert choose_pose(_metrics(), _metrics(), None)["reason"] == "engines_disagree_unresolved_keep_boltz"


# --- geometry on synthetic complexes ------------------------------------------------------------


def _atom(name: str, xyz, element: str, serial: int) -> Atom:
    return Atom(name, np.asarray(xyz, dtype=float), 0.0, 1.0, " ", name.ljust(4), serial, element)


def _ligand_heavy_atoms(seed: int = 0):
    mol = Chem.AddHs(Chem.MolFromSmiles(LIGAND_SMILES))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    return [a.GetSymbol() for a in mol.GetAtoms()], coords + np.array([3.0, 1.0, 2.0])  # sit next to the heme


def _build_complex(ligand_elements, ligand_coords, ligand_order=None, seed: int = 0) -> Structure:
    """Chain A: 40 ALA with CA atoms 4-11 A from the origin; chain B: the ligand (resname LIG0, as
    OpenFold3 writes it); chain C: a heme whose FE sits at the origin."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(40, 3))
    protein_ca = directions / np.linalg.norm(directions, axis=1, keepdims=True) * rng.uniform(4, 11, size=(40, 1))

    structure = Structure("synthetic")
    model = Model(0)
    structure.add(model)
    chain_a = Chain("A")
    model.add(chain_a)
    for i, xyz in enumerate(protein_ca, start=1):
        residue = Residue((" ", i, " "), "ALA", " ")
        residue.add(_atom("CA", xyz, "C", i))
        chain_a.add(residue)

    chain_b = Chain("B")
    model.add(chain_b)
    ligand = Residue(("H_LIG0", 1, " "), "LIG0", " ")
    order = range(len(ligand_elements)) if ligand_order is None else ligand_order
    for serial, i in enumerate(order, start=100):
        ligand.add(_atom(f"{ligand_elements[i]}{i}", ligand_coords[i], ligand_elements[i], serial))
    chain_b.add(ligand)

    chain_c = Chain("C")
    model.add(chain_c)
    heme = Residue(("H_HEM", 1, " "), "HEM", " ")
    heme.add(_atom("FE", (0.0, 0.0, 0.0), "FE", 900))
    chain_c.add(heme)
    return structure


def _rigidly_moved(structure: Structure, seed: int = 1) -> Structure:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotation = q * np.sign(np.linalg.det(q))  # proper rotation
    translation = rng.normal(size=3) * 7
    moved = structure.copy()
    for atom in moved.get_atoms():
        atom.coord = (atom.coord @ rotation + translation).astype(np.float32)
    return moved


def test_ligand_rmsd_is_zero_between_a_complex_and_itself():
    elements, coords = _ligand_heavy_atoms()
    s = _build_complex(elements, coords)
    result = aligned_ligand_rmsd(s, s, LIGAND_SMILES)
    assert result["ligand_rmsd"] == pytest.approx(0.0, abs=1e-5)
    assert result["pocket_ca_rmsd"] == pytest.approx(0.0, abs=1e-5)


def test_ligand_rmsd_ignores_a_rigid_motion_of_the_whole_complex():
    elements, coords = _ligand_heavy_atoms()
    s = _build_complex(elements, coords)
    result = aligned_ligand_rmsd(s, _rigidly_moved(s), LIGAND_SMILES)
    assert result["ligand_rmsd"] == pytest.approx(0.0, abs=1e-3)


def test_ligand_rmsd_sees_a_real_ligand_shift_relative_to_the_pocket():
    elements, coords = _ligand_heavy_atoms()
    reference = _build_complex(elements, coords)
    shifted = _build_complex(elements, coords + np.array([3.0, 0.0, 0.0]))
    assert aligned_ligand_rmsd(reference, shifted, LIGAND_SMILES)["ligand_rmsd"] == pytest.approx(3.0, abs=1e-3)


def test_ligand_rmsd_is_symmetry_aware_so_atom_order_and_ring_flips_dont_count_as_a_pose_change():
    elements, coords = _ligand_heavy_atoms()
    reference = _build_complex(elements, coords)
    shuffled_order = np.random.default_rng(3).permutation(len(elements))
    shuffled = _build_complex(elements, coords, ligand_order=shuffled_order)
    assert aligned_ligand_rmsd(reference, shuffled, LIGAND_SMILES)["ligand_rmsd"] == pytest.approx(0.0, abs=1e-3)


def test_alignment_refuses_structures_that_dont_correspond():
    elements, coords = _ligand_heavy_atoms()
    reference = _build_complex(elements, coords)
    stranger = _build_complex(elements, coords)
    for residue in list(stranger[0]["A"].get_residues())[5:]:
        stranger[0]["A"].detach_child(residue.id)
    with pytest.raises(ValueError, match="don't correspond"):
        aligned_ligand_rmsd(reference, stranger, LIGAND_SMILES)


def test_pose_metrics_combines_posebusters_and_the_closest_motif_distance(monkeypatch):
    monkeypatch.setattr(pose_consensus, "check_pose_validity", lambda s, smi: {"all_passed": False, "checks": {"a": True, "b": False}})
    monkeypatch.setattr(pose_consensus, "find_heme_iron", lambda s: object())
    monkeypatch.setattr(
        pose_consensus,
        "motif_nitrogen_distances",
        lambda s, smi, fe: [{"atom": "N4", "motif": "imidazole_Nsub", "distance_angstrom": 2.14}, {"atom": "N3", "motif": "x", "distance_angstrom": 4.2}],
    )
    assert pose_metrics(object(), "C") == {
        "all_passed": False,
        "failed_checks": ["b"],
        "has_motif": True,
        "closest_motif_n_fe": 2.14,
        "coordinated": True,
    }


def test_pose_metrics_for_a_compound_without_a_type_ii_motif(monkeypatch):
    monkeypatch.setattr(pose_consensus, "check_pose_validity", lambda s, smi: {"all_passed": True, "checks": {"a": True}})
    monkeypatch.setattr(pose_consensus, "find_heme_iron", lambda s: object())
    monkeypatch.setattr(pose_consensus, "motif_nitrogen_distances", lambda s, smi, fe: [])
    m = pose_metrics(object(), "C")
    assert m["has_motif"] is False and m["closest_motif_n_fe"] is None and m["coordinated"] is False


def test_best_candidate_takes_the_first_passing_sample_else_the_first(monkeypatch):
    by_path = {Path("m0.cif"): _metrics(passed=False, failed=["x"]), Path("m1.cif"): _metrics(), Path("m2.cif"): _metrics()}
    monkeypatch.setattr(pose_consensus, "load_structure", lambda p: f"structure:{p.name}")
    monkeypatch.setattr(pose_consensus, "pose_metrics", lambda s, smi, refs=None: by_path[Path(s.split(":")[1])])

    cif, _, metrics = pose_consensus._best_candidate([Path("m0.cif"), Path("m1.cif"), Path("m2.cif")], "C")
    assert cif == Path("m1.cif") and metrics["all_passed"]

    cif, _, metrics = pose_consensus._best_candidate([Path("m0.cif")], "C")
    assert cif == Path("m0.cif") and not metrics["all_passed"]
    assert pose_consensus._best_candidate([], "C") is None


def test_best_candidate_records_a_pose_it_cannot_evaluate_as_failed_not_as_a_crash(monkeypatch):
    monkeypatch.setattr(pose_consensus, "load_structure", lambda p: "s")

    def boom(s, smi, refs=None):
        raise ValueError("no ligand atoms found")

    monkeypatch.setattr(pose_consensus, "pose_metrics", boom)
    _, _, metrics = pose_consensus._best_candidate([Path("m0.cif")], "C")
    assert metrics["all_passed"] is False and "no ligand atoms found" in metrics["failed_checks"][0]


# --- consensus table + submission wiring --------------------------------------------------------


def _patch_table_inputs(monkeypatch, boltz_by_name, of3_by_name, rmsd_by_name):
    monkeypatch.setattr(pose_consensus, "read_boltz_inputs", lambda name, target: {"ligand_smiles": "C", "protein_sequence": "M"})
    monkeypatch.setattr(pose_consensus, "boltz_cif_paths", lambda name, target: [Path(f"{name}_boltz.cif")])
    monkeypatch.setattr(pose_consensus, "model_cif_paths", lambda name, target: [Path(f"{name}_of3.cif")] if name in of3_by_name else [])

    def best(paths, smiles, references=None):
        if not paths:
            return None
        name = paths[0].name.split("_")[0]
        table = boltz_by_name if paths[0].name.endswith("_boltz.cif") else of3_by_name
        return paths[0], f"struct:{paths[0].name}", table[name]

    monkeypatch.setattr(pose_consensus, "_best_candidate", best)
    monkeypatch.setattr(
        pose_consensus,
        "aligned_ligand_rmsd",
        lambda ref, probe, smi: {"ligand_rmsd": rmsd_by_name[probe.split("_")[0].split(":")[1]], "pocket_ca_rmsd": 0.5},
    )


def test_build_consensus_table_rows_and_csv(tmp_path, monkeypatch):
    _patch_table_inputs(
        monkeypatch,
        boltz_by_name={"agree": _metrics(), "fixme": _metrics(passed=False, failed=["no_clashes"]), "solo": _metrics()},
        of3_by_name={"agree": _metrics(), "fixme": _metrics()},
        rmsd_by_name={"agree": 0.8, "fixme": 6.0},
    )
    out_csv = tmp_path / "sub" / "consensus.csv"
    table = build_consensus_table(["agree", "fixme", "solo"], out_csv=out_csv)

    rows = table.set_index("compound")
    assert rows.loc["agree", ["engine", "reason"]].tolist() == ["boltz", "engines_agree"]
    assert rows.loc["fixme", ["engine", "reason"]].tolist() == ["openfold3", "only_openfold3_passes_posebusters"]
    assert rows.loc["solo", ["engine", "reason"]].tolist() == ["boltz", "no_openfold3_prediction"]
    assert rows.loc["agree", "chosen_cif"] == "agree_boltz.cif" and rows.loc["fixme", "chosen_cif"] == "fixme_of3.cif"
    assert rows.loc["agree", "ligand_rmsd"] == pytest.approx(0.8)
    assert pd.isna(rows.loc["solo", "ligand_rmsd"]) and pd.isna(rows.loc["solo", "of3_posebusters_ok"])
    assert rows.loc["fixme", "boltz_failed_checks"] == "no_clashes"
    assert pd.read_csv(out_csv).shape[0] == 3  # written, parent directory created


def test_build_consensus_table_survives_an_rmsd_failure(tmp_path, monkeypatch):
    _patch_table_inputs(monkeypatch, {"a": _metrics(n_fe=6.0)}, {"a": _metrics(n_fe=2.1)}, {"a": 0.0})

    def boom(ref, probe, smi):
        raise ValueError("structures don't correspond")

    monkeypatch.setattr(pose_consensus, "aligned_ligand_rmsd", boom)
    row = build_consensus_table(["a"]).iloc[0]
    assert pd.isna(row["ligand_rmsd"]) and row["rmsd_error"] == "structures don't correspond"
    assert (row["engine"], row["reason"]) == ("openfold3", "engines_disagree_heme_coordination_prior")  # unknown RMSD = disagreement


def _official20(tmp_path: Path) -> tuple[list[str], Path]:
    names = [f"OCNT-{i:07d}" for i in range(20)]
    test_set = tmp_path / "test_set.csv"
    pd.DataFrame({"Molecule_Name": names, "SMILES": ["C"] * 20}).to_csv(test_set, index=False)
    return names, test_set


def _consensus_csv(tmp_path: Path, names: list[str], engines: dict[str, str], bad: set[str] = frozenset()) -> Path:
    rows = []
    for n in names:
        engine = engines.get(n, "boltz")
        cif = tmp_path / f"{n}_{engine}.cif"
        cif.write_text("data_x\n")
        rows.append(
            {
                "compound": n,
                "engine": engine,
                "reason": "x",
                "chosen_cif": str(cif),
                "boltz_posebusters_ok": n not in bad,
                "of3_posebusters_ok": n not in bad,
            }
        )
    path = tmp_path / "consensus.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _record_conversions(monkeypatch):
    from cypmode.challenge import structure_submission

    converted = {}

    def fake_convert(cif_path, out_pdb_path):
        converted[out_pdb_path.stem] = Path(cif_path)
        out_pdb_path.write_text("ATOM\n")

    monkeypatch.setattr(structure_submission, "convert_cif_to_pdb", fake_convert)
    return converted


def test_submission_in_consensus_mode_uses_each_compounds_chosen_structure(tmp_path, monkeypatch):
    from cypmode.challenge.structure_submission import build_structure_submission

    names, test_set = _official20(tmp_path)
    consensus = _consensus_csv(tmp_path, names, engines={names[3]: "openfold3", names[7]: "openfold3"})
    converted = _record_conversions(monkeypatch)

    summary = build_structure_submission(test_set_path=test_set, out_dir=tmp_path / "out", consensus_csv=consensus)

    assert summary["n_written"] == 20 and summary["n_posebusters_failures"] == 0
    assert converted[names[3]].name.endswith("_openfold3.cif") and converted[names[0]].name.endswith("_boltz.cif")
    with ZipFile(summary["zip_path"]) as zf:
        assert sorted(zf.namelist()) == sorted(f"{n}.pdb" for n in names)


def test_submission_consensus_mode_reports_failures_from_the_chosen_engines_own_posebusters_result(tmp_path, monkeypatch):
    from cypmode.challenge.structure_submission import build_structure_submission

    names, test_set = _official20(tmp_path)
    consensus = _consensus_csv(tmp_path, names, engines={}, bad={names[5]})
    _record_conversions(monkeypatch)
    summary = build_structure_submission(test_set_path=test_set, out_dir=tmp_path / "out", consensus_csv=consensus)
    assert summary["posebusters_failures"] == [names[5]]

    skipped = build_structure_submission(
        test_set_path=test_set, out_dir=tmp_path / "out2", consensus_csv=consensus, skip_posebusters_failures=True
    )
    assert skipped["n_written"] == 19


def test_submission_consensus_mode_rejects_a_table_missing_a_compound_or_a_structure(tmp_path, monkeypatch):
    from cypmode.challenge.structure_submission import build_structure_submission

    names, test_set = _official20(tmp_path)
    consensus = _consensus_csv(tmp_path, names, engines={})
    _record_conversions(monkeypatch)

    table = pd.read_csv(consensus)
    table.iloc[1:].to_csv(consensus, index=False)
    with pytest.raises(ValueError, match="no row for"):
        build_structure_submission(test_set_path=test_set, out_dir=tmp_path / "o", consensus_csv=consensus)

    table.to_csv(consensus, index=False)
    Path(table.loc[2, "chosen_cif"]).unlink()
    with pytest.raises(FileNotFoundError, match="missing on disk"):
        build_structure_submission(test_set_path=test_set, out_dir=tmp_path / "o", consensus_csv=consensus)


# --- the rescoring tie-breaker --------------------------------------------------------------------


def _rescored(template=None, support=None, occupancy=0.5, **kwargs) -> dict:
    return _metrics(**kwargs) | {"rescore": {"template_score": template, "template_support": support, "occupancy_score": occupancy}}


def _unresolved(boltz_rescore, of3_rescore, rmsd=4.0):
    # both valid, disagree, neither/both coordinate -> only the tie-breaker can decide
    return choose_pose(_rescored(**boltz_rescore, n_fe=6.0), _rescored(**of3_rescore, n_fe=6.5), rmsd)


def test_occupancy_tiebreaker_picks_the_better_occupied_pose_past_the_margin():
    margin = pose_consensus.OCCUPANCY_MARGIN
    for_of3 = _unresolved({"occupancy": 0.45}, {"occupancy": 0.45 + margin})
    assert for_of3 == {"engine": "openfold3", "reason": "engines_disagree_occupancy_rescoring"}
    for_boltz = _unresolved({"occupancy": 0.45 + margin}, {"occupancy": 0.45})
    assert for_boltz == {"engine": "boltz", "reason": "engines_disagree_occupancy_rescoring"}


def test_occupancy_difference_below_the_margin_keeps_boltz():
    decision = _unresolved({"occupancy": 0.50}, {"occupancy": 0.50 + pose_consensus.OCCUPANCY_MARGIN - 0.01})
    assert decision == {"engine": "boltz", "reason": "engines_disagree_unresolved_keep_boltz"}


def test_template_score_decides_first_when_both_poses_have_a_close_crystallized_analog():
    decision = _unresolved(
        {"template": 0.30, "support": 0.8, "occupancy": 0.60},
        {"template": 0.30 + pose_consensus.TEMPLATE_MARGIN, "support": 0.8, "occupancy": 0.40},  # occupancy would say the opposite
    )
    assert decision == {"engine": "openfold3", "reason": "engines_disagree_template_rescoring"}


def test_template_score_is_ignored_without_close_analog_support_and_occupancy_takes_over():
    weak = pose_consensus.TEMPLATE_MIN_SUPPORT - 0.1
    decision = _unresolved(
        {"template": 0.1, "support": weak, "occupancy": 0.60},
        {"template": 0.9, "support": weak, "occupancy": 0.40},
    )
    assert decision == {"engine": "boltz", "reason": "engines_disagree_occupancy_rescoring"}


def test_template_support_is_judged_on_the_weaker_of_the_two_poses():
    decision = _unresolved(
        {"template": 0.1, "support": 0.9, "occupancy": 0.50},
        {"template": 0.9, "support": 0.2, "occupancy": 0.50},
    )
    assert decision["reason"] == "engines_disagree_unresolved_keep_boltz"


def test_rescoring_never_overrides_the_posebusters_gate_agreement_or_the_coordination_prior():
    # PoseBusters gate
    assert choose_pose(_rescored(occupancy=0.2, passed=False, failed=["x"]), _rescored(occupancy=0.9), 4.0)["reason"] == "only_openfold3_passes_posebusters"
    # agreement
    assert choose_pose(_rescored(occupancy=0.9), _rescored(occupancy=0.1), 0.5)["reason"] == "engines_agree"
    # heme-coordination prior beats a contrary rescoring
    decision = choose_pose(_rescored(occupancy=0.9, n_fe=6.0), _rescored(occupancy=0.1, n_fe=2.1), 4.0)
    assert decision == {"engine": "openfold3", "reason": "engines_disagree_heme_coordination_prior"}


def test_missing_rescoring_on_either_pose_falls_through_to_keep_boltz():
    assert choose_pose(_metrics(n_fe=6.0), _rescored(occupancy=0.9, n_fe=6.5), 4.0)["reason"] == "engines_disagree_unresolved_keep_boltz"
    assert choose_pose(_rescored(occupancy=None, n_fe=6.0), _rescored(occupancy=0.9, n_fe=6.5), 4.0)["reason"] == "engines_disagree_unresolved_keep_boltz"


def test_pose_metrics_attaches_rescoring_only_when_references_are_given(monkeypatch):
    monkeypatch.setattr(pose_consensus, "check_pose_validity", lambda s, smi: {"all_passed": True, "checks": {"a": True}})
    monkeypatch.setattr(pose_consensus, "find_heme_iron", lambda s: object())
    monkeypatch.setattr(pose_consensus, "motif_nitrogen_distances", lambda s, smi, fe: [])
    seen = {}

    def fake_rescore(structure, smiles, references):
        seen["references"] = references
        return {"template_score": 0.4, "template_support": 0.6, "occupancy_score": 0.5, "top_template": "x", "n_refs_used": 3}

    monkeypatch.setattr(pose_consensus, "rescore_pose", fake_rescore)

    assert "rescore" not in pose_metrics(object(), "C")
    assert pose_metrics(object(), "C", references=[{"pdb_id": "1ABC"}])["rescore"] == {
        "template_score": 0.4,
        "template_support": 0.6,
        "occupancy_score": 0.5,
    }
    assert seen["references"] == [{"pdb_id": "1ABC"}]


def test_consensus_table_carries_both_engines_rescoring_columns_and_uses_the_tiebreaker(tmp_path, monkeypatch):
    _patch_table_inputs(
        monkeypatch,
        boltz_by_name={"a": _rescored(occupancy=0.40, n_fe=6.0)},
        of3_by_name={"a": _rescored(occupancy=0.60, n_fe=6.5)},
        rmsd_by_name={"a": 4.0},
    )
    row = build_consensus_table(["a"], references=[{"pdb_id": "1ABC"}]).iloc[0]
    assert (row["engine"], row["reason"]) == ("openfold3", "engines_disagree_occupancy_rescoring")
    assert row["boltz_occupancy_score"] == pytest.approx(0.40) and row["of3_occupancy_score"] == pytest.approx(0.60)

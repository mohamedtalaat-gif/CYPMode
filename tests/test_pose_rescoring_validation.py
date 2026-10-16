from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("Bio")
pytest.importorskip("rdkit")
pytest.importorskip("sklearn")

from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from rdkit import Chem
from rdkit.Chem import AllChem

from cypmode.validation import (
    openfold3,
    pose_consensus,
    pose_rescoring_validation,
    structures,
)
from cypmode.validation.pose_rescoring_validation import (
    MAX_SHIFT_A,
    NEAR_NATIVE_A,
    NOVEL_SIMILARITY,
    WRONG_A,
    collect_poses,
    crystal_mol,
    make_decoys,
    one_per_compound,
    rmsd,
    score_decoy_set,
    summarize,
    true_ligand_rmsd,
    validate,
    validate_engine_poses,
    win_rates,
)

LIGAND_SMILES = "c1ccccc1Cn1ccnc1"
LARGE_LIGAND_SMILES = "c1ccccc1Cn1ccnc1C(=O)Nc1ccccc1"  # 21 heavy atoms: clears the 15-atom floor of one_per_compound
FE = np.zeros(3)


def _embedded_ligand(smiles: str = LIGAND_SMILES, seed: int = 0, offset=(3.0, 1.0, 2.0)):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]) + np.asarray(offset)
    return [a.GetSymbol() for a in mol.GetAtoms()], coords


def _pocket(n: int = 40, seed: int = 0) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n, 3))
    xyz = directions / np.linalg.norm(directions, axis=1, keepdims=True) * rng.uniform(4, 11, size=(n, 1))
    return {i: xyz[i - 1] for i in range(1, n + 1)}


def _record(comp_id: str, smiles: str, elements, coords, pocket, resolution=2.0, pdb_id="1AAA") -> dict:
    return {
        "pdb_id": pdb_id,
        "comp_id": comp_id,
        "smiles": smiles,
        "resolution": resolution,
        "n_heavy": len(elements),
        "elements": list(elements),
        "coords": np.asarray(coords).tolist(),
        "fe": FE.tolist(),
        "pocket_ca": {str(k): v.tolist() for k, v in pocket.items()},
    }


def _atom(name, xyz, element, serial):
    return Atom(name, np.asarray(xyz, dtype=float), 0.0, 1.0, " ", name.ljust(4), serial, element)


def _complex(pocket, elements, coords) -> Structure:
    structure = Structure("s")
    model = Model(0)
    structure.add(model)
    chain_a = Chain("A")
    model.add(chain_a)
    for i, xyz in pocket.items():
        residue = Residue((" ", i, " "), "ALA", " ")
        residue.add(_atom("CA", xyz, "C", i))
        chain_a.add(residue)
    chain_b = Chain("B")
    model.add(chain_b)
    ligand = Residue(("H_LIG0", 1, " "), "LIG0", " ")
    for i, (el, xyz) in enumerate(zip(elements, coords, strict=True)):
        ligand.add(_atom(f"{el}{i}", xyz, el, 100 + i))
    chain_b.add(ligand)
    chain_c = Chain("C")
    model.add(chain_c)
    heme = Residue(("H_HEM", 1, " "), "HEM", " ")
    heme.add(_atom("FE", FE, "FE", 900))
    chain_c.add(heme)
    return structure


# --- decoys and bookkeeping ---------------------------------------------------------------------------


def test_rmsd_is_the_root_mean_square_of_per_atom_displacements():
    a = np.zeros((4, 3))
    assert rmsd(a, a) == 0.0
    assert rmsd(a, a + np.array([3.0, 4.0, 0.0])) == pytest.approx(5.0)


def test_decoys_are_rigid_motions_of_the_crystal_pose_within_the_stated_shift_bound():
    _, crystal = _embedded_ligand()
    decoys = make_decoys(crystal, 40, np.random.default_rng(0))
    assert len(decoys) == 40
    reference_distances = np.linalg.norm(crystal[:, None] - crystal[None], axis=-1)
    for decoy in decoys:
        np.testing.assert_allclose(np.linalg.norm(decoy[:, None] - decoy[None], axis=-1), reference_distances, atol=1e-6)  # internal geometry untouched
        assert np.linalg.norm(decoy.mean(axis=0) - crystal.mean(axis=0)) <= MAX_SHIFT_A + 1e-6  # centroid moves by the translation only


def test_decoys_cover_both_near_native_and_clearly_wrong_displacements_and_are_reproducible():
    _, crystal = _embedded_ligand()
    first = make_decoys(crystal, 200, np.random.default_rng(1))
    again = make_decoys(crystal, 200, np.random.default_rng(1))
    np.testing.assert_array_equal(np.array(first), np.array(again))
    displacements = np.array([rmsd(d, crystal) for d in first])
    assert (displacements < NEAR_NATIVE_A).any() and (displacements > WRONG_A).any()


def test_one_per_compound_keeps_the_best_resolved_copy_and_drops_tiny_or_smiles_less_ligands():
    def ref(comp, n_heavy, resolution, smiles="CCO", pdb="1AAA"):
        return {"comp_id": comp, "n_heavy": n_heavy, "resolution": resolution, "smiles": smiles, "pdb_id": pdb}

    kept = one_per_compound(
        [
            ref("AAA", 20, 2.8, pdb="1LOW"),
            ref("AAA", 20, 1.9, pdb="1HIG"),
            ref("AAA", 20, None, pdb="1UNK"),  # unknown resolution is the worst, not the best
            ref("BBB", 10, 1.5),  # too small to define an orientation
            ref("CCC", 25, 2.0, smiles=None),
            ref("DDD", 18, None, pdb="1ONLY"),
        ]
    )
    assert {r["comp_id"]: r["pdb_id"] for r in kept} == {"AAA": "1HIG", "DDD": "1ONLY"}


# --- decoy validation --------------------------------------------------------------------------------


def _crowded_references(n_others: int = 25, smiles: str = LIGAND_SMILES) -> tuple[list[dict], dict]:
    """A reference panel in one shared pocket: one target ligand plus n_others different compounds
    that all occupy the same site (so there is a real prior to score decoys against)."""
    pocket = _pocket()
    elements, site = _embedded_ligand(smiles)
    smiles_pool = ["CCO", "c1ccccc1", "CC(=O)O", "c1ccncc1", "CCN", "C1CCCCC1", "CCCC", "c1ccoc1", "CCS", "CC#N"]
    rng = np.random.default_rng(4)
    others = [
        _record(f"C{i:02d}", f"{smiles_pool[i % len(smiles_pool)]}{'C' * (i // len(smiles_pool))}", elements, site + rng.normal(size=site.shape) * 0.4, pocket, pdb_id=f"1X{i:02d}")
        for i in range(n_others)
    ]
    target = _record("TGT", smiles, elements, site, pocket, pdb_id="1TGT")
    return [target, *others], target


def test_score_decoy_set_ranks_the_crystal_pose_above_displaced_decoys_on_the_occupancy_prior():
    references, target = _crowded_references()
    row = score_decoy_set(target, references, 60, np.random.default_rng(0), similarity_cut=None)
    assert row is not None
    assert row["comp_id"] == "TGT" and row["n_refs_used"] == 25  # its own entry left out
    assert row["occupancy_score_true_pose_rank"] == 1
    assert row["occupancy_score_auc"] > 0.95 and row["occupancy_score_spearman"] > 0.5
    assert 0.0 <= row["occupancy_score_win_0.5_2"] <= 1.0
    for lo, hi in pose_rescoring_validation.RMSD_BINS_A:
        assert f"occupancy_score_win_{lo:g}_{hi:g}" in row


def test_score_decoy_set_is_skipped_when_too_few_references_remain_after_leave_out():
    references, target = _crowded_references(n_others=10)
    assert score_decoy_set(target, references, 20, np.random.default_rng(0), similarity_cut=None) is None


def test_an_aggressive_similarity_cut_can_leave_too_little_to_score():
    references, target = _crowded_references()
    assert score_decoy_set(target, references, 20, np.random.default_rng(0), similarity_cut=0.0) is None  # every analog is >= 0.0 similar


def test_validate_scores_each_compound_under_both_leave_out_regimes_and_writes_the_table(tmp_path):
    references, _ = _crowded_references(smiles=LARGE_LIGAND_SMILES)
    out = tmp_path / "validation.csv"
    table = validate(references, n_decoys=30, seed=0, out_csv=out)
    assert set(table["similarity_cut"].dropna()) == {NOVEL_SIMILARITY} and table["similarity_cut"].isna().any()
    assert set(table["comp_id"]) <= {"TGT"} | {f"C{i:02d}" for i in range(25)}
    assert pd.read_csv(out).shape == table.shape


def _fake_table() -> pd.DataFrame:
    rows = []
    for cut in (None, NOVEL_SIMILARITY):
        for comp, auc in (("A", 0.9), ("B", 0.7)):
            row = {"comp_id": comp, "similarity_cut": cut}
            for name in ("template_score", "occupancy_score"):
                row |= {f"{name}_spearman": auc - 0.3, f"{name}_auc": auc, f"{name}_true_pose_rank": 1 if comp == "A" else 5}
                for lo, hi in pose_rescoring_validation.RMSD_BINS_A:
                    row[f"{name}_win_{lo:g}_{hi:g}"] = auc
            rows.append(row)
    return pd.DataFrame(rows)


def test_summarize_reports_mean_auc_and_the_fraction_ranked_first_per_regime_and_score():
    summary = summarize(_fake_table())
    assert len(summary) == 4  # 2 regimes x 2 scores
    row = summary[summary["score"] == "template_score"].iloc[0]
    assert row["mean_auc"] == pytest.approx(0.8) and row["true_pose_ranked_first"] == pytest.approx(0.5) and row["n_ligands"] == 2
    assert set(summary["regime"]) == {"own compound only excluded", f"also excl. similarity >= {NOVEL_SIMILARITY:g}"}


def test_win_rates_average_over_ligands_in_each_displacement_bin():
    rates = win_rates(_fake_table())
    assert len(rates) == 4
    assert rates["2-3 A"].tolist() == pytest.approx([0.8] * 4)


# --- crystal vs engine poses -------------------------------------------------------------------------


def test_crystal_mol_builds_a_template_matched_molecule_with_the_crystal_coordinates():
    elements, coords = _embedded_ligand()
    mol = crystal_mol(LIGAND_SMILES, elements, coords)
    assert mol is not None and mol.GetNumAtoms() == len(elements)
    assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == Chem.MolToSmiles(Chem.MolFromSmiles(LIGAND_SMILES))
    np.testing.assert_allclose(mol.GetConformer().GetPositions(), coords, atol=1e-2)


def test_crystal_mol_is_none_when_the_crystal_ligand_does_not_match_the_template():
    elements, coords = _embedded_ligand()
    assert crystal_mol(LIGAND_SMILES, elements[:-2], coords[:-2]) is None  # atoms missing from the crystal
    assert crystal_mol("not a smiles ((", elements, coords) is None


def test_true_rmsd_is_zero_for_the_crystal_pose_and_equals_the_shift_for_a_displaced_one():
    pocket = _pocket()
    elements, crystal = _embedded_ligand()
    record = _record("TGT", LIGAND_SMILES, elements, crystal, pocket)
    assert true_ligand_rmsd(_complex(pocket, elements, crystal), LIGAND_SMILES, [record]) == pytest.approx(0.0, abs=1e-2)
    shifted = _complex(pocket, elements, crystal + np.array([3.0, 0.0, 0.0]))
    assert true_ligand_rmsd(shifted, LIGAND_SMILES, [record]) == pytest.approx(3.0, abs=1e-2)


def test_true_rmsd_takes_the_nearest_of_several_crystal_copies():
    pocket = _pocket()
    elements, crystal = _embedded_ligand()
    copies = [
        _record("TGT", LIGAND_SMILES, elements, crystal + np.array([4.0, 0.0, 0.0]), pocket, pdb_id="1FAR"),
        _record("TGT", LIGAND_SMILES, elements, crystal, pocket, pdb_id="1NEAR"),
    ]
    assert true_ligand_rmsd(_complex(pocket, elements, crystal), LIGAND_SMILES, copies) == pytest.approx(0.0, abs=1e-2)


def test_true_rmsd_is_none_when_no_crystal_copy_fits_the_pocket_or_matches_the_template():
    pocket = _pocket()
    elements, crystal = _embedded_ligand()
    pose = _complex(pocket, elements, crystal)
    unfit = _record("TGT", LIGAND_SMILES, elements, crystal, _pocket(seed=99))
    broken = _record("TGT", LIGAND_SMILES, elements[:-2], crystal[:-2], pocket)
    assert true_ligand_rmsd(pose, LIGAND_SMILES, [unfit, broken]) is None
    assert true_ligand_rmsd(pose, LIGAND_SMILES, []) is None


def test_collect_poses_lists_every_sampled_structure_from_each_engine(monkeypatch):
    monkeypatch.setattr(pose_consensus, "boltz_cif_paths", lambda name, target: [Path(f"{name}_b0.cif"), Path(f"{name}_b1.cif")])
    monkeypatch.setattr(openfold3, "model_cif_paths", lambda name, target: [Path(f"{name}_o0.cif")] if name == "a" else [])
    assert collect_poses(["a", "b"]) == {
        "a": [("boltz", 0, Path("a_b0.cif")), ("boltz", 1, Path("a_b1.cif")), ("openfold3", 0, Path("a_o0.cif"))],
        "b": [("boltz", 0, Path("b_b0.cif")), ("boltz", 1, Path("b_b1.cif"))],
    }


def test_validate_engine_poses_scores_only_compounds_with_crystal_copies_and_excludes_their_own_entry(monkeypatch, tmp_path):
    pocket = _pocket()
    elements, crystal = _embedded_ligand()
    references, _ = _crowded_references()
    smiles_by_name = {"has_crystal": LIGAND_SMILES, "no_crystal": "CC(C)(C)c1ccc(O)cc1Br"}
    monkeypatch.setattr(openfold3, "read_boltz_inputs", lambda name, target: {"ligand_smiles": smiles_by_name[name]})
    monkeypatch.setattr(structures, "load_structure", lambda path: _complex(pocket, elements, crystal + np.array([0.0, 0.0, 0.0 if "good" in str(path) else 4.0])))

    sources = {
        "has_crystal": [("boltz", 0, Path("good_boltz.cif")), ("openfold3", 0, Path("bad_of3.cif"))],
        "no_crystal": [("boltz", 0, Path("good_x.cif"))],
    }
    out = tmp_path / "engine_validation.csv"
    table = validate_engine_poses(sources, references=references, out_csv=out)

    assert set(table["compound"]) == {"has_crystal"} and len(table) == 2
    assert pd.read_csv(out).shape[0] == 2
    boltz, of3 = table.sort_values("engine").itertuples()
    assert boltz.true_rmsd == pytest.approx(0.0, abs=1e-2) and of3.true_rmsd == pytest.approx(4.0, abs=1e-2)
    assert boltz.occupancy_score > of3.occupancy_score  # the pose on the experimental site outscores the displaced one
    assert boltz.template_support < 1.0  # its own crystal entry was excluded, not read back as a perfect analog
    assert boltz.novel_occupancy_score is not None and "novel_template_support" in table.columns
    assert boltz.n_crystal_copies == 1


def test_validate_engine_poses_with_nothing_to_compare_returns_an_empty_table_and_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(openfold3, "read_boltz_inputs", lambda name, target: {"ligand_smiles": "CC(C)(C)c1ccc(O)cc1Br"})
    references, _ = _crowded_references()
    out = tmp_path / "none.csv"
    table = validate_engine_poses({"x": [("boltz", 0, Path("x.cif"))]}, references=references, out_csv=out)
    assert table.empty and not out.exists()

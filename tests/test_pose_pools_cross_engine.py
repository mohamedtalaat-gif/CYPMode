import numpy as np
import pytest

pytest.importorskip("rdkit")
pytest.importorskip("Bio")

from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from rdkit import Chem
from rdkit.Chem import AllChem

from cypmode.validation import pose_pools
from cypmode.validation.pose_pools import (
    symmetric_lddt_pli,
    template_automorphisms,
    template_ligand_coordinates,
)

SMILES = "CCO"


def _ethanol_xyz():
    mol = Chem.AddHs(Chem.MolFromSmiles(SMILES))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    conformer = mol.GetConformer()
    return np.array([list(conformer.GetAtomPosition(i)) for i in range(3)])  # heavy atoms in template order: C, C, O


def _structure(ligand_xyz, names, order, offset=(0.0, 0.0, 0.0)):
    """A protein chain of five ALA residues around the ligand and a ligand chain whose atoms carry ``names`` in ``order``."""
    model = Model(0)
    chain_a = Chain("A")
    rng = np.random.default_rng(0)
    for i in range(5):
        residue = Residue((" ", i + 1, " "), "ALA", i + 1)
        centre = np.array([3.0 * (i - 2), 4.0, 0.0]) + rng.normal(scale=0.3, size=3)
        for k, atom_name in enumerate(("N", "CA", "C")):
            residue.add(Atom(atom_name, centre + k * 0.8, 1.0, 1.0, " ", f" {atom_name} ", 3 * i + k, atom_name[0]))
        chain_a.add(residue)
    model.add(chain_a)
    chain_b = Chain("B")
    ligand = Residue(("H_LIG", 1, " "), "LIG", 1)
    elements = ("C", "C", "O")
    for serial, index in enumerate(order):
        ligand.add(Atom(names[index], ligand_xyz[index] + np.array(offset), 1.0, 1.0, " ", f" {names[index]} ", 100 + serial, elements[index]))
    chain_b.add(ligand)
    model.add(chain_b)
    structure = Structure("s")
    structure.add(model)
    return structure


def test_symmetry_mappings_of_a_tert_butyl_alcohol_are_the_six_methyl_permutations_with_identity_first():
    perms = template_automorphisms("CC(C)(C)O")
    assert perms.shape == (6, 5) and (perms[0] == np.arange(5)).all()
    assert all(perm[1] == 1 and perm[4] == 4 for perm in perms)  # the quaternary carbon and the oxygen never move


def test_a_swap_of_two_equivalent_atoms_costs_nothing_once_symmetry_is_allowed():
    ligand = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.2, 1.2, 0.0], [2.2, -1.2, 0.0], [2.2, 0.0, 1.3]])
    protein = {(i, "CA"): np.array([2.5 + 1.0 * i, 3.0, 0.5 * i]) for i in range(8)}
    swapped = ligand[[0, 1, 3, 2, 4]]  # the template atoms 2 and 3 are the same chemistry
    perms = template_automorphisms("CC(C)(C)O")
    identity_only = symmetric_lddt_pli(swapped, protein, ligand, protein, perms[:1])
    symmetric = symmetric_lddt_pli(swapped, protein, ligand, protein, perms)
    assert symmetric == pytest.approx(1.0) and identity_only < 1.0


def test_pose_atoms_are_reordered_to_the_template_whatever_their_names_and_order():
    xyz = _ethanol_xyz()
    structure = _structure(xyz, ("C42", "C48", "O32"), order=[2, 0, 1])  # oxygen first, then the two carbons
    assert np.allclose(template_ligand_coordinates(structure, SMILES), xyz, atol=1e-3)


def test_a_pool_mixing_two_atom_namings_is_compared_through_the_smiles(tmp_path, monkeypatch):
    xyz = _ethanol_xyz()
    boltz_like = _structure(xyz, ("C42", "C48", "O32"), order=[0, 1, 2])
    of3_like = _structure(xyz, ("C1", "C2", "O1"), order=[2, 1, 0], offset=(5.0, 5.0, 5.0))  # same pose, other names, other frame
    structures = {}
    paths = []
    for name, structure in (("boltz.cif", boltz_like), ("of3.cif", of3_like)):
        path = tmp_path / name
        path.write_text("x")
        structures[path] = structure
        paths.append(path)
    monkeypatch.setattr("cypmode.validation.structures.load_structure", lambda p: structures[p])
    # the protein of the second pose is shifted with its ligand, so the two complexes are the same up to a rigid motion
    for atom in of3_like[0]["A"].get_atoms():
        atom.coord = atom.coord + np.array([5.0, 5.0, 5.0])
    with pytest.raises(ValueError, match="share no ligand"):
        pose_pools.pairwise_lddt(paths, cache_dir=tmp_path / "plain")
    matrix = pose_pools.pairwise_lddt(paths, cache_dir=tmp_path / "template", smiles=SMILES)
    assert matrix[0, 1] == pytest.approx(1.0) and matrix[1, 0] == pytest.approx(1.0)
    assert any(f.name.endswith("_template.npy") for f in (tmp_path / "template").iterdir())


def test_a_pool_of_one_naming_keeps_the_name_based_matrix_even_when_the_smiles_is_given(tmp_path, monkeypatch):
    xyz = _ethanol_xyz()
    a, b = _structure(xyz, ("C42", "C48", "O32"), [0, 1, 2]), _structure(xyz, ("C42", "C48", "O32"), [0, 1, 2])
    structures, paths = {}, []
    for name, structure in (("a.cif", a), ("b.cif", b)):
        path = tmp_path / name
        path.write_text("x")
        structures[path] = structure
        paths.append(path)
    monkeypatch.setattr("cypmode.validation.structures.load_structure", lambda p: structures[p])
    matrix = pose_pools.pairwise_lddt(paths, cache_dir=tmp_path / "c", smiles=SMILES)
    assert matrix[0, 1] == pytest.approx(1.0)
    assert not any(f.name.endswith("_template.npy") for f in (tmp_path / "c").iterdir())

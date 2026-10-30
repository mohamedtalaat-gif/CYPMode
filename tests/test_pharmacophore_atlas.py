# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("rdkit")
pytest.importorskip("matplotlib")

FIGURES = Path("docs/figures")
sys.path[:0] = [str(FIGURES), str(FIGURES.parent.parent)]

from catalog import REGISTRY, caption_text
from gen import pharmacophore_atlas as atlas
from gen import structure_pharmacophore_atlas as structure_atlas
from rdkit import Chem

from cypmode.challenge import pocket_pharmacophore as pp

DATA = Path("data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv")
needs_data = pytest.mark.skipif(not DATA.exists(), reason="the training data is not in this checkout")
needs_structures = pytest.mark.skipif(not Path("data/structure_track/inputs/pdb_cache/2V0M.cif").exists(), reason="the cache of the crystal structures is not in a clean checkout")
needs_pymol = pytest.mark.skipif(shutil.which("pymol") is None and not Path("/opt/homebrew/bin/pymol").exists(), reason="PyMOL is not installed")
KETOCONAZOLE = "CC(=O)N1CCN(CC1)c1ccc(OCC2COC(Cn3ccnc3)(O2)c2ccc(Cl)cc2Cl)cc1"


def test_the_feature_drawing_names_the_families_that_the_molecule_has_and_returns_an_image():
    image, families = atlas.draw_families(Chem.MolFromSmiles(KETOCONAZOLE), "light", size=(300, 240))
    assert image.shape == (240, 300, 3) and {"Acceptor", "Aromatic", "Hydrophobe"} <= set(families) and "NegIonizable" not in families


def test_the_group_drawing_lists_each_group_found_with_its_colour():
    image, legend = atlas.draw_groups(Chem.MolFromSmiles(KETOCONAZOLE), "dark", size=(300, 240))
    names = [name for name, _ in legend]
    assert image.shape == (240, 300, 3) and "azole or pyridine (heme binder)" in names and "aryl halide" in names and len({key for _, key in legend}) == len(legend)


def test_the_feature_atoms_of_a_molecule_are_atom_indices_of_that_molecule():
    mol = Chem.MolFromSmiles("CC(=O)Nc1ccccc1")
    found = atlas.feature_atoms(mol)
    assert all(a < mol.GetNumAtoms() for groups in found.values() for atoms in groups for a in atoms) and len(found["Aromatic"]) == 1 and len(found["Aromatic"][0]) == 6


def test_the_change_in_features_between_two_compounds_is_counted_by_family():
    assert atlas.feature_change(Chem.MolFromSmiles("c1ccccc1"), Chem.MolFromSmiles("Oc1ccccc1")) != "no change in the features"
    assert atlas.feature_change(Chem.MolFromSmiles("CCC"), Chem.MolFromSmiles("CCC")) == "no change in the features"


def test_the_fragment_that_a_pair_exchanges_is_found_by_cutting_bonds_and_not_by_matching_it_anywhere():
    mol = Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1")
    assert atlas.fragment_atoms(mol, "[*:1]O") == [8]
    assert atlas.fragment_atoms(mol, "[*:1]Cl") == []


def test_the_typical_pair_of_a_group_is_the_one_whose_change_is_closest_to_the_mean():
    pairs = {"a": np.array([0, 1, 2, 3]), "b": np.array([4, 5, 6, 7]), "t": np.array([9, 8, 7, 6]), "gained": np.array([2, 2, 2, 1]), "delta": np.array([1.0, 0.0, -1.0, 1.0])}
    assert atlas.typical_pair(2, 0.1, pairs) == (1, 5, 8)  # the change of 0.0 is the closest to 0.1
    assert atlas.typical_pair(1, 1.0, pairs) == (3, 7, 6)


def test_the_colour_map_gives_an_rgb_for_each_atom_and_a_scale_for_the_bar():
    colours, mappable = atlas.colour_map({0: 1.0, 1: 3.0, 2: 2.0}, "coolwarm_r")
    assert set(colours) == {0, 1, 2} and all(len(c) == 3 for c in colours.values()) and mappable.norm.vmin == 1.0 and mappable.norm.vmax == 3.0
    _, mappable = atlas.colour_map({0: -0.4, 1: 0.2}, "coolwarm_r", symmetric=True)
    assert mappable.norm.vmin == -0.4 and mappable.norm.vmax == 0.4


@needs_data
@pytest.mark.parametrize("figure_id", ["pharm_functional_groups", "pharm_features_examples", "pharm_mmp_sar"])
def test_the_figures_of_the_functional_groups_and_the_pairs_are_drawn_from_the_training_data(figure_id):
    import matplotlib.pyplot as plt

    entry = REGISTRY[figure_id]
    fig = entry.make("light")
    assert len(fig.axes) >= 4
    plt.close(fig)
    assert len(caption_text(entry)) > 300 and entry.group == "pharmacophore"


@needs_data
def test_the_group_table_has_the_groups_that_p450s_are_known_to_bioactivate_at_the_top_for_cyp3a4():
    table = atlas.group_table("CYP3A4")
    assert table.group[0] in {"methylenedioxyphenyl", "thiophene", "terminal alkyne", "furan"} and (table.n >= 10).all()


@needs_data
@needs_structures
@needs_pymol
def test_the_pocket_figure_and_the_atlas_are_drawn_with_real_structures():
    import matplotlib.pyplot as plt

    for figure_id in ("pharm_pocket_3d", "pharm_atlas"):
        fig = REGISTRY[figure_id].make("light")
        assert len(fig.axes) >= 2
        plt.close(fig)


def test_the_chemotypes_group_ligands_by_similarity_with_the_largest_group_first():
    smiles = {"a": "c1ccccc1CCN", "b": "c1ccccc1CCNC", "c": "c1ccccc1CCNCC", "d": "CCOC(=O)C1CCCCC1C(=O)O", "e": "Cn1ccnc1"}
    groups = structure_atlas.chemotypes(smiles)
    assert groups[0] == sorted(groups[0]) or set(groups[0]) == {"a", "b", "c"}
    assert {"a", "b", "c"} <= set(groups[0]) and sum(len(g) for g in groups) == 5


def test_the_changed_atoms_of_a_pair_are_the_atoms_outside_the_common_part_and_the_ones_that_change_element():
    thioether = Chem.MolFromSmiles("c1ccccc1CSCC(=O)NC")
    amine = Chem.MolFromSmiles("c1ccccc1CNCC(=O)NC")
    changed_a, changed_b = structure_atlas.changed_atoms(thioether, amine)
    assert changed_a == [7] and changed_b == [7]  # the sulfur and the nitrogen at the same place, nothing else


@needs_pymol
def test_the_series_of_the_benchmark_has_three_pairs_of_two_ligands():
    pairs = structure_atlas.series_pairs()
    assert len(pairs) == 3 and all(a.series == b.series and a.Molecule_Name != b.Molecule_Name for a, b in pairs)


def test_the_ligands_shown_in_3d_are_the_smallest_the_median_and_the_largest_first_sample_rmsd():
    if not Path("data/structure_track/outputs/cyp3a4/benchmark_engine_pose_validation_unguided_boltz.csv").exists():
        pytest.skip("the benchmark tables are not in this checkout")
    chosen = structure_atlas.chosen_ligands()
    table = pd.read_csv("data/structure_track/outputs/cyp3a4/benchmark_engine_pose_validation_unguided_boltz.csv")
    first = table[table["sample"] == 0].set_index("compound").true_rmsd
    assert first[chosen[0]] == first.min() and first[chosen[2]] == first.max() and first.min() < first[chosen[1]] < first.max()


def test_the_site_spheres_of_a_pose_are_firmer_when_a_feature_of_the_pose_matches_them():
    frame = np.eye(3)
    in_frame = {"features": {f: np.zeros((0, 3)) for f in pp.FAMILIES}, "iron": np.array([10.0, 0.0, 0.0]), "frame": frame}
    in_frame["features"]["Acceptor"] = np.array([[0.0, 0.0, 6.0]])
    sites = {f: np.zeros((0, 4)) for f in pp.FAMILIES}
    sites["Acceptor"] = np.array([[0.2, 0.0, 6.0, 1.0], [8.0, 0.0, 6.0, 0.5]])
    spheres, table = structure_atlas.sites_in_complex(in_frame, sites)
    assert [s[5] for s in spheres] == [0.4, 0.9] and table.matched.tolist() == [True, False] and spheres[0][1] == pytest.approx(10.2)

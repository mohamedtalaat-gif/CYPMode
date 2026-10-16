import numpy as np
import pytest

pytest.importorskip("Bio")
pytest.importorskip("rdkit")

from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

from cypmode.validation.pose_rescoring import (
    MAX_ALIGN_RMSD_A,
    MIN_POCKET_CA,
    align_references,
    connectivity_key,
    gaussian_overlap_tanimoto,
    morgan_fingerprint,
    rescore_pose,
    score_against_aligned,
    score_pose_arrays,
    structure_arrays,
    superpose_reference,
)

BENZYLIMIDAZOLE = "c1ccccc1Cn1ccnc1"
TOLUENE = "Cc1ccccc1"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"

FE = np.zeros(3)


def _pocket(n: int = 40, seed: int = 0) -> dict[int, np.ndarray]:
    """n C-alpha atoms 4-11 A from the iron, residue numbers 1..n."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n, 3))
    xyz = directions / np.linalg.norm(directions, axis=1, keepdims=True) * rng.uniform(4, 11, size=(n, 1))
    return {i: xyz[i - 1] for i in range(1, n + 1)}


def _ligand(seed: int = 1, n_atoms: int = 14, center=(0.0, 0.0, 6.0)) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n_atoms, 3)) * 1.6 + np.asarray(center)


def _reference(pocket: dict[int, np.ndarray], coords: np.ndarray, smiles: str = BENZYLIMIDAZOLE, comp_id: str = "BZI", pdb_id: str = "1ABC") -> dict:
    return {
        "pdb_id": pdb_id,
        "comp_id": comp_id,
        "smiles": smiles,
        "coords": coords.tolist(),
        "fe": FE.tolist(),
        "pocket_ca": {str(k): v.tolist() for k, v in pocket.items()},
    }


def _rotation(seed: int = 5) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _moved_pocket(pocket, rotation, translation, noise: float = 0.0, seed: int = 9):
    rng = np.random.default_rng(seed)
    return {k: v @ rotation + translation + rng.normal(size=3) * noise for k, v in pocket.items()}


# --- gaussian_overlap_tanimoto ---------------------------------------------------------------------


def test_overlap_is_one_for_identical_point_sets_and_symmetric():
    a, b = _ligand(1), _ligand(2)
    assert gaussian_overlap_tanimoto(a, a) == pytest.approx(1.0)
    assert gaussian_overlap_tanimoto(a, b) == pytest.approx(gaussian_overlap_tanimoto(b, a))


def test_overlap_falls_monotonically_as_a_pose_is_translated_away():
    a = _ligand(1)
    values = [gaussian_overlap_tanimoto(a, a + np.array([shift, 0.0, 0.0])) for shift in (0.0, 0.5, 1.5, 3.0, 8.0)]
    assert values == sorted(values, reverse=True)
    assert values[0] == pytest.approx(1.0) and values[-1] < 0.05


def test_overlap_penalizes_a_size_mismatch_even_when_one_set_contains_the_other():
    big = _ligand(1, n_atoms=20)
    assert gaussian_overlap_tanimoto(big[:8], big) < 0.6


def test_overlap_is_invariant_to_atom_order():
    a, b = _ligand(1), _ligand(2)
    assert gaussian_overlap_tanimoto(a[::-1], b) == pytest.approx(gaussian_overlap_tanimoto(a, b))


# --- fingerprints --------------------------------------------------------------------------------


def test_connectivity_key_ignores_stereochemistry_and_charge_state_notation():
    assert connectivity_key("C[C@H](N)C(=O)O") == connectivity_key("C[C@@H](N)C(=O)O")
    assert connectivity_key(BENZYLIMIDAZOLE) != connectivity_key(TOLUENE)


def test_unparseable_smiles_give_none_not_an_exception():
    assert morgan_fingerprint("not a smiles ((") is None
    assert connectivity_key("not a smiles ((") is None
    assert connectivity_key("") is None


# --- superpose_reference --------------------------------------------------------------------------


def test_superposing_a_rigidly_moved_pocket_puts_the_reference_ligand_back_in_the_candidate_frame():
    pocket, ligand = _pocket(), _ligand()
    rotation, translation = _rotation(), np.array([5.0, -3.0, 2.0])
    # the reference was solved in a different frame; the candidate's pocket is the same pocket, moved
    candidate_ca = _moved_pocket(pocket, rotation, translation)
    candidate_fe = FE @ rotation + translation

    superposed = superpose_reference(_reference(pocket, ligand), candidate_fe, candidate_ca)

    assert superposed is not None
    coords, rms = superposed
    assert rms == pytest.approx(0.0, abs=1e-4)
    np.testing.assert_allclose(coords, ligand @ rotation + translation, atol=1e-3)


def test_too_few_shared_pocket_residues_drops_the_reference_rather_than_forcing_a_fit():
    pocket = _pocket()
    candidate_ca = {k: v for k, v in pocket.items() if k <= MIN_POCKET_CA - 1}
    assert superpose_reference(_reference(pocket, _ligand()), FE, candidate_ca) is None


def test_residues_outside_the_pocket_radius_of_the_candidate_iron_are_not_used_in_the_fit():
    pocket = _pocket(60)
    displaced = dict(pocket)
    for k in range(1, 21):  # 20 residues pushed well outside the 12 A shell *and* out of register with the reference
        displaced[k] = pocket[k] * 4.0
    # the 40 residues still inside the shell fit perfectly; counting the displaced ones would wreck the RMSD
    result = superpose_reference(_reference(pocket, _ligand()), FE, displaced)
    assert result is not None and result[1] == pytest.approx(0.0, abs=1e-4)

    for k in range(21, 41):  # now fewer than MIN_POCKET_CA remain inside the shell
        displaced[k] = pocket[k] * 4.0
    assert superpose_reference(_reference(pocket, _ligand()), FE, displaced) is None


def test_a_pocket_that_does_not_superpose_within_tolerance_is_dropped():
    pocket = _pocket()
    rng = np.random.default_rng(3)
    scrambled = {k: v + rng.normal(size=3) * 4.0 for k, v in pocket.items()}
    assert superpose_reference(_reference(pocket, _ligand()), FE, scrambled) is None


def test_small_pocket_noise_is_tolerated_and_reported_in_the_ca_rmsd():
    pocket = _pocket()
    noisy = _moved_pocket(pocket, np.eye(3), np.zeros(3), noise=0.3)
    result = superpose_reference(_reference(pocket, _ligand()), FE, noisy)
    assert result is not None
    assert 0.05 < result[1] < MAX_ALIGN_RMSD_A


def test_only_residues_numbered_in_both_structures_are_matched():
    pocket = _pocket(60)
    reference = _reference({k: v for k, v in pocket.items() if k <= 45}, _ligand())
    candidate = {k: v for k, v in pocket.items() if k >= 20}  # overlap 20..45 = 26 residues
    assert superpose_reference(reference, FE, candidate) is not None
    assert superpose_reference(reference, FE, {k: v for k, v in pocket.items() if k >= 30}) is None  # 16 shared


# --- align_references -------------------------------------------------------------------------------


def _reference_panel(pocket) -> list[dict]:
    return [
        _reference(pocket, _ligand(1), BENZYLIMIDAZOLE, "BZI", "1AAA"),
        _reference(pocket, _ligand(2), TOLUENE, "TOL", "1BBB"),
        _reference(pocket, _ligand(3, n_atoms=14), CAFFEINE, "CAF", "1CCC"),
    ]


def test_every_fitting_reference_is_aligned_with_its_chemical_similarity_and_label():
    pocket = _pocket()
    aligned = align_references(BENZYLIMIDAZOLE, FE, pocket, _reference_panel(pocket))
    assert [label for _, label, _ in aligned] == ["1AAA:BZI", "1BBB:TOL", "1CCC:CAF"]
    similarities = {label: s for s, label, _ in aligned}
    assert similarities["1AAA:BZI"] == pytest.approx(1.0)
    assert similarities["1AAA:BZI"] > similarities["1BBB:TOL"] > similarities["1CCC:CAF"]


def test_excluding_the_compound_itself_removes_its_own_crystal_entry_whatever_the_notation():
    pocket = _pocket()
    stereo_noise = "c1ccccc1Cn1ccnc1"  # same connectivity as the query
    aligned = align_references(stereo_noise, FE, pocket, _reference_panel(pocket), exclude_compound=BENZYLIMIDAZOLE)
    assert "1AAA:BZI" not in [label for _, label, _ in aligned]
    assert len(aligned) == 2


def test_similarity_cutoff_mimics_a_novel_scaffold_by_dropping_close_analogs():
    pocket = _pocket()
    aligned = align_references(BENZYLIMIDAZOLE, FE, pocket, _reference_panel(pocket), exclude_similarity_at_least=0.5)
    assert all(s < 0.5 for s, _, _ in aligned)
    assert "1AAA:BZI" not in [label for _, label, _ in aligned]


def test_a_reference_whose_pocket_does_not_fit_is_skipped_not_forced():
    pocket = _pocket()
    panel = _reference_panel(pocket)
    panel.append(_reference(_pocket(seed=77), _ligand(4), TOLUENE, "BAD", "1DDD"))  # a different pocket geometry
    labels = [label for _, label, _ in align_references(BENZYLIMIDAZOLE, FE, pocket, panel)]
    assert "1DDD:BAD" not in labels and len(labels) == 3


def test_a_reference_with_an_unparseable_smiles_still_counts_for_occupancy_with_no_similarity():
    pocket = _pocket()
    panel = [_reference(pocket, _ligand(1), "not a smiles ((", "XXX")]
    aligned = align_references(BENZYLIMIDAZOLE, FE, pocket, panel)
    assert len(aligned) == 1 and aligned[0][0] is None


# --- score_against_aligned --------------------------------------------------------------------------


def test_no_references_gives_empty_scores_and_zero_count():
    result = score_against_aligned(_ligand(), [])
    assert result == {"template_score": None, "template_support": None, "top_template": None, "occupancy_score": None, "n_refs_used": 0}


def test_a_pose_matching_its_close_analogs_template_beats_a_displaced_pose():
    ligand = _ligand(1)
    aligned = [(0.9, "1AAA:BZI", ligand), (0.4, "1BBB:TOL", _ligand(2))]
    on_template = score_against_aligned(ligand, aligned)
    displaced = score_against_aligned(ligand + np.array([6.0, 0.0, 0.0]), aligned)
    assert on_template["template_score"] > displaced["template_score"]
    assert on_template["top_template"] == "1AAA:BZI"
    assert on_template["template_support"] == pytest.approx(0.9)


def test_template_score_is_a_similarity_weighted_mean_so_the_closest_analog_dominates():
    ligand = _ligand(1)
    near_match, far_match = ligand, ligand + np.array([8.0, 0.0, 0.0])
    heavy_on_match = score_against_aligned(ligand, [(0.9, "a", near_match), (0.1, "b", far_match)])["template_score"]
    heavy_on_miss = score_against_aligned(ligand, [(0.1, "a", near_match), (0.9, "b", far_match)])["template_score"]
    assert heavy_on_match > 0.8 and heavy_on_miss < 0.2


def test_only_the_top_k_most_similar_references_enter_the_template_score():
    ligand = _ligand(1)
    decoys = [(0.5 + 0.01 * i, f"d{i}", ligand + np.array([9.0, 0.0, 0.0])) for i in range(5)]
    low_similarity_perfect_match = (0.2, "match", ligand)
    with_it = score_against_aligned(ligand, [*decoys, low_similarity_perfect_match], top_k=5)
    assert with_it["template_score"] < 0.05  # crowded out by five more similar, but wrong, references


def test_occupancy_is_high_inside_where_reference_ligands_sit_and_low_outside():
    reference = _ligand(1)
    aligned = [(0.3, "r", reference)]
    inside = score_against_aligned(reference, aligned)["occupancy_score"]
    outside = score_against_aligned(reference + np.array([12.0, 0.0, 0.0]), aligned)["occupancy_score"]
    assert inside > 0.9 and outside < 0.01


def test_occupancy_is_a_fraction_of_all_references_not_just_the_best_one():
    ligand = _ligand(1)
    one_hit_one_miss = score_against_aligned(ligand, [(0.3, "hit", ligand), (0.3, "miss", ligand + 15.0)])["occupancy_score"]
    two_hits = score_against_aligned(ligand, [(0.3, "h1", ligand), (0.3, "h2", ligand)])["occupancy_score"]
    assert one_hit_one_miss == pytest.approx(two_hits / 2, rel=1e-3)


def test_unparseable_reference_fingerprints_leave_the_template_score_empty_but_keep_occupancy():
    ligand = _ligand(1)
    result = score_against_aligned(ligand, [(None, "x:y", ligand)])
    assert result["template_score"] is None and result["template_support"] is None
    assert result["occupancy_score"] > 0.9 and result["n_refs_used"] == 1


# --- score_pose_arrays / structure_arrays / rescore_pose --------------------------------------------


def test_score_pose_arrays_scores_a_pose_against_the_references_superposed_into_its_frame():
    pocket, ligand = _pocket(), _ligand(1)
    rotation, translation = _rotation(), np.array([2.0, 7.0, -4.0])
    candidate_ca = _moved_pocket(pocket, rotation, translation)
    candidate_fe = FE @ rotation + translation
    panel = [_reference(pocket, ligand, BENZYLIMIDAZOLE, "BZI")]

    in_frame_pose = ligand @ rotation + translation  # exactly where the reference ligand lands
    result = score_pose_arrays(in_frame_pose, candidate_fe, candidate_ca, TOLUENE, panel)

    assert result["n_refs_used"] == 1
    assert result["template_score"] == pytest.approx(1.0, abs=1e-3)
    assert result["occupancy_score"] == pytest.approx(1.0, abs=1e-3)


def _atom(name, xyz, element, serial):
    return Atom(name, np.asarray(xyz, dtype=np.float32), 0.0, 1.0, " ", name.ljust(4), serial, element)


def _complex(pocket: dict[int, np.ndarray], ligand: np.ndarray, hydrogens: bool = False) -> Structure:
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
    lig = Residue(("H_LIG0", 1, " "), "LIG0", " ")
    for serial, xyz in enumerate(ligand, start=100):
        lig.add(_atom(f"C{serial}", xyz, "C", serial))
    if hydrogens:
        lig.add(_atom("H1", ligand[0] + 1.0, "H", 300))
    chain_b.add(lig)
    chain_c = Chain("C")
    model.add(chain_c)
    heme = Residue(("H_HEM", 1, " "), "HEM", " ")
    heme.add(_atom("FE", FE, "FE", 900))
    chain_c.add(heme)
    return structure


def test_structure_arrays_reads_ligand_heavy_atoms_iron_and_ca_by_residue_number():
    pocket, ligand = _pocket(), _ligand()
    coords, fe, ca = structure_arrays(_complex(pocket, ligand, hydrogens=True))
    assert coords.shape == (len(ligand), 3)  # the hydrogen is excluded
    np.testing.assert_allclose(coords, ligand, atol=1e-4)
    np.testing.assert_allclose(fe, FE)
    assert sorted(ca) == sorted(pocket)
    np.testing.assert_allclose(ca[7], pocket[7], atol=1e-4)


def test_rescore_pose_on_a_predicted_complex_matches_the_array_level_scores():
    pocket, ligand = _pocket(), _ligand()
    panel = [_reference(pocket, ligand, BENZYLIMIDAZOLE, "BZI")]
    structure = _complex(pocket, ligand)
    from_structure = rescore_pose(structure, TOLUENE, panel)
    from_arrays = score_pose_arrays(*structure_arrays(structure), TOLUENE, panel)
    assert from_structure["occupancy_score"] == pytest.approx(from_arrays["occupancy_score"])
    assert from_structure["template_score"] == pytest.approx(from_arrays["template_score"])


def test_a_pose_in_the_experimental_site_outscores_one_in_the_access_channel_on_both_scores():
    pocket, site = _pocket(), _ligand(1)
    panel = [_reference(pocket, site, BENZYLIMIDAZOLE, "BZI"), _reference(pocket, _ligand(2), TOLUENE, "TOL")]
    in_site = rescore_pose(_complex(pocket, site), BENZYLIMIDAZOLE, panel)
    channel = rescore_pose(_complex(pocket, site + np.array([0.0, 0.0, 14.0])), BENZYLIMIDAZOLE, panel)
    assert in_site["occupancy_score"] > channel["occupancy_score"]
    assert in_site["template_score"] > channel["template_score"]


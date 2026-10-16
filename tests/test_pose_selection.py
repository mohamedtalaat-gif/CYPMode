import numpy as np
import pytest

from cypmode.validation.pose_selection import expected_lddt, lddt_pli_arrays

RNG = np.random.default_rng(3)
PROTEIN = {(i, "CA"): RNG.normal(scale=5.0, size=3) for i in range(60)}
LIGAND = {"C1": np.array([0.0, 0.0, 0.0]), "C2": np.array([1.5, 0.0, 0.0]), "N3": np.array([3.0, 0.0, 0.0])}


def _shifted(delta):
    return {k: v + np.asarray(delta, dtype=float) for k, v in LIGAND.items()}


def test_identical_complexes_score_one_and_a_displaced_ligand_scores_lower():
    same = lddt_pli_arrays(LIGAND, PROTEIN, LIGAND, PROTEIN)
    moved = lddt_pli_arrays(_shifted([0.0, 0.0, 3.0]), PROTEIN, LIGAND, PROTEIN)
    way_off = lddt_pli_arrays(_shifted([0.0, 0.0, 12.0]), PROTEIN, LIGAND, PROTEIN)
    assert same == 1.0 and way_off < moved < same


def test_atoms_missing_from_either_complex_are_ignored_and_no_overlap_is_an_error():
    partial = {"C1": LIGAND["C1"], "C2": LIGAND["C2"]}
    assert lddt_pli_arrays(partial, PROTEIN, LIGAND, PROTEIN) == 1.0
    with pytest.raises(ValueError):
        lddt_pli_arrays({"X9": LIGAND["C1"]}, PROTEIN, LIGAND, PROTEIN)


def test_expected_lddt_prefers_the_pose_most_of_the_pool_agrees_with():
    # three poses cluster together, one is an outlier; the pick must come from the cluster
    pool = [(_shifted(d), PROTEIN) for d in ([0, 0, 0], [0, 0, 0.3], [0, 0.3, 0], [0, 0, 8.0])]
    scores = expected_lddt(pool)
    assert int(scores.argmax()) in (0, 1, 2) and scores[3] == scores.min()


def test_a_single_pose_pool_is_returned_as_is():
    assert list(expected_lddt([(LIGAND, PROTEIN)])) == [1.0]

# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import numpy as np
import pytest

from cypmode.validation.lddt_to_crystal import CrystalReference, lddt_against_references
from cypmode.validation.pose_pools import template_automorphisms

# a ligand of three atoms next to a pocket of eight protein atoms
PROTEIN = {(i, "CA"): np.array([float(i), 3.0, 0.0]) for i in range(8)}
LIGAND = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])


def reference(ligand=LIGAND) -> CrystalReference:
    return CrystalReference("TEST", ligand, PROTEIN)


def test_a_pose_identical_to_the_crystal_scores_one():
    identity = np.arange(3)[None, :]
    assert lddt_against_references(LIGAND, PROTEIN, [reference()], identity) == pytest.approx(1.0)


def test_a_pose_moved_away_from_the_pocket_scores_low():
    identity = np.arange(3)[None, :]
    moved = LIGAND + np.array([0.0, -6.0, 0.0])
    assert lddt_against_references(moved, PROTEIN, [reference()], identity) < 0.3


def test_symmetric_atoms_are_mapped_to_the_best_permutation():
    permutations = template_automorphisms("CCC")  # propane read as a graph: its two ends can be exchanged
    flipped = LIGAND[::-1]
    assert lddt_against_references(flipped, PROTEIN, [reference()], permutations) == pytest.approx(1.0)
    assert lddt_against_references(flipped, PROTEIN, [reference()], np.arange(3)[None, :]) < 1.0


def test_the_best_crystal_copy_counts():
    identity = np.arange(3)[None, :]
    far = reference(LIGAND + np.array([0.0, -6.0, 0.0]))
    assert lddt_against_references(LIGAND, PROTEIN, [far, reference()], identity) == pytest.approx(1.0)


def test_there_is_nothing_to_compare_with_without_a_reference():
    assert lddt_against_references(LIGAND, PROTEIN, [], np.arange(3)[None, :]) is None

# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import numpy as np
import pytest

pytest.importorskip("rdkit")
pytest.importorskip("gemmi")

from cypmode.challenge import pocket_pharmacophore as pp


def test_the_orientations_are_rotations_inside_the_cone_with_every_spin():
    rotations = pp.orientations()
    assert rotations.shape == (pp.N_DIRECTIONS * pp.N_SPINS, 3, 3)
    assert np.allclose(rotations @ np.transpose(rotations, (0, 2, 1)), np.eye(3), atol=1e-9)
    tilt = np.degrees(np.arccos(np.clip(rotations[:, 2, 2], -1, 1)))  # the angle between z and the image of z of the molecule's own axis
    assert tilt.max() <= pp.CONE_DEGREES + 1.0 + 1e-6


def test_rotation_between_takes_one_unit_vector_to_the_other_also_when_they_are_opposite():
    a, b = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    assert np.allclose(pp.rotation_between(a, b) @ a, b)
    assert np.allclose(pp.rotation_between(a, -a) @ a, -a)


def test_a_gaussian_field_peaks_at_one_where_its_points_are_and_is_zero_outside_the_grid():
    field = pp.gaussian_field(np.array([[1.0, 2.0, 5.0]]), np.array([1.0]), 1.5)
    assert field.max() == pytest.approx(1.0)
    assert pp.lookup(field, np.array([[1.0, 2.0, 5.0]]))[0] == pytest.approx(1.0)
    assert pp.lookup(field, np.array([[40.0, 0.0, 5.0]]))[0] == 0.0
    assert pp.lookup(field, np.array([[1.0, 2.0, 5.0], [1.0, 2.0, 20.0]])).tolist()[1] == 0.0


def test_the_frame_of_a_heme_has_z_on_the_side_of_the_ligand_and_x_toward_the_nitrogen_a():
    import gemmi

    st = gemmi.Structure()
    model, chain = gemmi.Model("1"), gemmi.Chain("A")
    residue = gemmi.Residue()
    residue.name, residue.seqid = "HEM", gemmi.SeqId(1, " ")
    for name, element, xyz in (("FE", "Fe", (0, 0, 0)), ("NA", "N", (2, 0, 0)), ("NB", "N", (0, 2, 0)), ("NC", "N", (-2, 0, 0)), ("ND", "N", (0, -2, 0))):
        atom = gemmi.Atom()
        atom.name, atom.element, atom.pos = name, gemmi.Element(element), gemmi.Position(*xyz)
        residue.add_atom(atom)
    chain.add_residue(residue)
    model.add_chain(chain)
    st.add_model(model)
    below = pp.heme_frame(st, np.zeros(3), np.array([[0.0, 0.0, -4.0]]))
    above = pp.heme_frame(st, np.zeros(3), np.array([[0.0, 0.0, 4.0]]))
    assert np.allclose(above[2], [0, 0, 1]) and np.allclose(below[2], [0, 0, -1])
    assert np.allclose(above[0], [1, 0, 0]) and np.allclose(np.cross(above[0], above[1]), above[2])
    assert np.allclose(pp.to_frame(np.array([[0.0, 0.0, 3.0]]), np.zeros(3), above), [[0, 0, 3]])


def test_a_pose_whose_feature_meets_the_field_scores_higher_than_a_field_that_is_elsewhere():
    heavy = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]])  # a line; atom 0 is the anchor
    feats = {f: np.zeros((0, 3)) for f in pp.FAMILIES}
    feats["Acceptor"] = np.array([[3.0, 0.0, 0.0]])
    empty = {name: np.zeros(pp.GRID_SHAPE) for name in (*pp.FAMILIES, "occupancy")}
    near = dict(empty, Acceptor=pp.gaussian_field(np.array([[0.0, 0.0, 7.0]]), np.array([1.0]), 1.5))  # the donor-mode anchor is at z 2.2, so an atom 3 A on is at about 5 A
    far = dict(empty, Acceptor=pp.gaussian_field(np.array([[-10.0, -10.0, 15.0]]), np.array([1.0]), 1.5))
    good = pp.score_poses(heavy, 0, feats, "donor", near)
    bad = pp.score_poses(heavy, 0, feats, "donor", far)
    assert good is not None and bad is not None and good["best"] > bad["best"] + 0.3
    assert 0 < good["accepted"] <= 1 and good["z_centroid"] > pp.MIN_HEIGHT


def test_a_compound_without_a_donor_has_no_donor_mode_columns_and_a_donor_has_both():
    fields = {name: np.zeros(pp.GRID_SHAPE) for name in (*pp.FAMILIES, "occupancy")}
    fields["occupancy"] = pp.gaussian_field(np.array([[0.0, 0.0, 6.0], [2.0, 2.0, 8.0]]), np.array([1.0, 1.0]), 3.0)
    plain = pp.compound_columns("CC(C)Cc1ccc(cc1)C(C)C(=O)O", fields, "cyp3a4")
    azole = pp.compound_columns("OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F", fields, "cyp3a4")
    assert set(plain) == set(azole) and len(plain) == 2 * len(pp.PART_COLUMNS)
    assert np.isnan(plain["pp_cyp3a4_donor_best"]) and not np.isnan(plain["pp_cyp3a4_substrate_best"])
    assert not np.isnan(azole["pp_cyp3a4_donor_best"])

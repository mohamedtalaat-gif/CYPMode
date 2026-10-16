import numpy as np
import pytest

from cypmode.validation import heme_frame_prior as hp


def planar_heme(radius=2.0):
    angles = np.radians([0, 90, 180, 270])
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros(4)])


def random_rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def test_axis_is_the_heme_normal_and_descriptor_reads_height_and_radius():
    ns = planar_heme()
    assert abs(abs(hp.heme_axis(ns)[2]) - 1.0) < 1e-9
    ligand = np.array([[0.0, 0.0, 2.1], [0.0, 0.0, 5.0], [1.0, 0.0, 7.0]])  # a ligand standing on the iron
    d = hp.descriptor(ligand, np.zeros(3), hp.heme_axis(ns))
    assert d[0] == pytest.approx(2.1) and d[1] == pytest.approx(2.1) and d[2] == pytest.approx(0.0)
    assert d[5] > 0.9  # its long axis points along the heme normal


def test_descriptor_ignores_rigid_motion_and_the_side_of_the_porphyrin():
    rng = np.random.default_rng(0)
    ns, fe = planar_heme(), np.zeros(3)
    ligand = rng.normal(size=(12, 3)) + np.array([0.5, 0.3, 6.0])
    base = hp.descriptor(ligand, fe, hp.heme_axis(ns))
    r = random_rotation(rng)
    moved = hp.descriptor(ligand @ r.T + 10, fe @ r.T + 10, hp.heme_axis(ns @ r.T))
    assert np.allclose(base, moved, atol=1e-8)
    mirrored = hp.descriptor(ligand * np.array([1, 1, -1]), fe, hp.heme_axis(ns))  # the other face of the heme
    assert np.allclose(base, mirrored, atol=1e-8)


def test_prior_prefers_typical_poses_and_excludes_the_scored_compound():
    rng = np.random.default_rng(1)
    typical = rng.normal([2.1, 2.1, 0.3, 6.0, 1.0, 0.8, 0.5], 0.1, size=(60, 7))
    prior = hp.HemePrior(typical)
    near, far = typical.mean(axis=0), typical.mean(axis=0) + np.array([4.0, 4.0, 3.0, 5.0, 4.0, 0.0, 0.0])
    assert prior.log_density(near)[0] > prior.log_density(far)[0] + 5
    poses = [(near + rng.normal(0, 0.05, 7), 1.0 + i * 0.01, 0.5) for i in range(3)] + [(far, 7.0, 0.5), (far + 0.1, 6.5, 0.5)]
    keys = ["K1"] * 30 + ["K2"] * 30
    out = hp.evaluate({"c": poses}, typical, keys, {"c": "K1"})
    assert out.loc[0, "prior_pick"] < 2.0 and out.loc[0, "rho_prior_vs_minus_rmsd"] > 0.5
    assert hp.gate(out.assign(rho_prior_vs_minus_rmsd=[0.9]))["met"] is False  # needs at least three ligands

import numpy as np
import pytest

from cypmode.validation.conformers import (
    MIN_SHARED_RESIDUES,
    entry_pockets,
    excluded_entries,
    pick_conformers,
    pocket_distance,
    template_path,
    template_spec,
)

RNG = np.random.default_rng(0)
BASE = {r: RNG.normal(scale=8.0, size=3) for r in range(100, 140)}


def _moved(shift: float, residues=None):
    return {r: BASE[r] + np.array([shift if r % 2 else -shift, 0.0, 0.0]) for r in (residues or BASE)}


def _rotated(pocket):
    c, s = np.cos(0.7), np.sin(0.7)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return {r: rot @ v + 5.0 for r, v in pocket.items()}


def test_pocket_distance_ignores_rigid_motion_and_grows_with_deformation():
    assert pocket_distance(BASE, _rotated(BASE)) == pytest.approx(0.0, abs=1e-6)
    assert pocket_distance(BASE, _moved(1.0)) < pocket_distance(BASE, _moved(3.0))


def test_pocket_distance_needs_enough_shared_residues():
    assert pocket_distance(BASE, {r: BASE[r] for r in list(BASE)[: MIN_SHARED_RESIDUES - 1]}) is None


def test_entry_pockets_keeps_one_record_per_entry_and_drops_poorly_resolved_ones():
    refs = [
        {"pdb_id": "AAAA", "resolution": 2.0, "pocket_ca": {"100": [0, 0, 0]}},
        {"pdb_id": "AAAA", "resolution": 2.0, "pocket_ca": {"100": [9, 9, 9]}},
        {"pdb_id": "BBBB", "resolution": 3.9, "pocket_ca": {"100": [1, 1, 1]}},
    ]
    pockets = entry_pockets(refs)
    assert list(pockets) == ["AAAA"] and list(pockets["AAAA"][100]) == [0, 0, 0]


def test_excluded_entries_hide_the_compound_itself_and_close_analogs_but_not_unrelated_ligands():
    refs = [
        {"pdb_id": "OWN1", "smiles": "CC(C)(c1cccnc1)C(=O)c1cccnc1"},
        {"pdb_id": "ANALOG", "smiles": "CC(C)(c1cccnc1)C(=O)c1ccc(C)nc1"},
        {"pdb_id": "OTHER", "smiles": "CCCCCCCCCCCC(=O)OCC"},
    ]
    assert excluded_entries("CC(C)(c1cccnc1)C(=O)c1cccnc1", refs) == {"OWN1", "ANALOG"}


def test_pick_conformers_starts_at_the_medoid_then_spreads_out_and_respects_exclusions(tmp_path):
    shifts = {"E0": 0.0, "E1": 0.2, "E2": 0.4, "E3": 4.0, "E4": 8.0}
    pockets = {e: _moved(s) for e, s in shifts.items()}
    for e in shifts:
        (tmp_path / f"{e}.cif").write_text("data_x\n")
    picked = pick_conformers(pockets, 3, cache_dir=tmp_path)
    assert picked[0] in {"E1", "E2"}  # the middle of the tight group
    assert "E4" in picked and "E3" not in picked[:2]  # the farthest conformer is taken before the next-farthest
    assert "E4" not in pick_conformers(pockets, 3, exclude={"E4"}, cache_dir=tmp_path)
    (tmp_path / "E4.cif").unlink()
    assert "E4" not in pick_conformers(pockets, 3, cache_dir=tmp_path)  # no structure file, no template


def test_template_path_is_absolute_and_must_exist(tmp_path):
    (tmp_path / "1ABC.cif").write_text("data_x\n")
    assert template_path("1ABC", tmp_path).is_absolute()
    with pytest.raises(FileNotFoundError):
        template_path("9ZZZ", tmp_path)


def test_template_spec_uses_chain_a_for_cryoem_entries_and_looks_there_first(tmp_path):
    crystal, cryoem = tmp_path / "crystal", tmp_path / "cryoem"
    crystal.mkdir(), cryoem.mkdir()
    (crystal / "1ABC.cif").write_text("x")
    (crystal / "38FU.cif").write_text("x")
    (cryoem / "38FU.cif").write_text("x")
    assert template_spec("1ABC", crystal, cryoem) == ((crystal / "1ABC.cif").resolve(), None)
    assert template_spec("38FU", crystal, cryoem) == ((cryoem / "38FU.cif").resolve(), "A")
    with pytest.raises(FileNotFoundError):
        template_spec("9ZZZ", crystal, cryoem)

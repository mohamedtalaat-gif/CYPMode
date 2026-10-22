# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

from pathlib import Path
from zipfile import ZipFile

import pytest

pytest.importorskip("Bio")

from cypmode.validation.structures import load_structure

RESULT_DIR = Path("data/structure_track/outputs/out/cyp3a4/boltz_results_cyp3a4_ketoconazole")
CIF_PATH = RESULT_DIR / "predictions" / "cyp3a4_ketoconazole" / "cyp3a4_ketoconazole_model_0.cif"

pytestmark = pytest.mark.skipif(
    not CIF_PATH.exists(),
    reason="requires a local Boltz-2 run; see README.md's Validation section",
)


def test_convert_cif_to_pdb_renames_the_ligand_residue(tmp_path):
    from cypmode.challenge.structure_submission import convert_cif_to_pdb
    from cypmode.validation.structures import _ligand_atoms

    structure = load_structure(CIF_PATH)
    original_resname = _ligand_atoms(structure)[0].get_parent().resname
    assert original_resname == "LIG1"  # real Boltz-2 convention, confirmed directly

    out_pdb = tmp_path / "OCNT-TEST.pdb"
    convert_cif_to_pdb(CIF_PATH, out_pdb)

    assert out_pdb.exists()
    written_text = out_pdb.read_text()
    assert "LIG1" not in written_text  # renamed, not just aliased
    assert " LIG " in written_text  # real 3-character PDB field name present


def test_convert_cif_to_pdb_does_not_mutate_the_source_cif():
    from cypmode.challenge.structure_submission import convert_cif_to_pdb

    before = CIF_PATH.read_bytes()
    convert_cif_to_pdb(CIF_PATH, Path("/tmp/cypmode_test_conversion_scratch.pdb"))
    after = CIF_PATH.read_bytes()
    assert before == after


def test_build_structure_submission_writes_exactly_20_pdb_files_and_a_zip(tmp_path):
    from cypmode.challenge.structure_submission import build_structure_submission

    summary = build_structure_submission(out_dir=tmp_path)

    assert summary["n_written"] == 20
    assert summary["zip_path"].exists()

    with ZipFile(summary["zip_path"]) as zf:
        names = zf.namelist()
        assert len(names) == 20
        assert all(n.endswith(".pdb") for n in names)
        # every entry named by the real official Molecule_Name, not e.g. "cyp3a4_<name>"
        assert not any(n.startswith("cyp3a4_") for n in names)


def test_build_structure_submission_pdb_entries_never_contain_lig1(tmp_path):
    from cypmode.challenge.structure_submission import build_structure_submission

    summary = build_structure_submission(out_dir=tmp_path)
    with ZipFile(summary["zip_path"]) as zf:
        for name in zf.namelist():
            content = zf.read(name).decode()
            assert "LIG1" not in content


def test_the_zip_has_the_same_bytes_when_rebuilt_from_the_same_structures(tmp_path):
    import time

    from cypmode.challenge.structure_submission import build_structure_submission

    first = build_structure_submission(out_dir=tmp_path / "a")
    time.sleep(2.2)  # zip entries carry a two-second clock: the files of the second build get other times, the content is the same
    second = build_structure_submission(out_dir=tmp_path / "b")
    assert first["zip_path"].read_bytes() == second["zip_path"].read_bytes()

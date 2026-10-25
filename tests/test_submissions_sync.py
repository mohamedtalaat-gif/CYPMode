# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import pandas as pd
import pytest

from cypmode.challenge import submissions_sync as sync


def entries():
    return [{"file": "tdi_track_candidate_A_hits_only.csv", "candidate": "candidate_A_hits_only", "sha256": "ab" * 32, "rows": 750, "CYP3A4_positive_rate": 0.372, "CYP2D6_positive_rate": 0.0587,
             "schema_valid": True, "portal_valid": True, "portal_message": "ok"}]


MANIFEST = {"isoforms": {iso: {"stack": "default", "n_candidates": 23, "called_rate": 0.372, "call_rate_multiplier": 1.2, "target_prevalence": 0.31} for iso in ("CYP3A4", "CYP2D6")}}


def test_the_tdi_readme_names_the_file_its_hash_and_where_the_upload_is_recorded():
    text = sync.render_tdi_readme(entries(), MANIFEST, {"CYP3A4_is_TDI": 0, "CYP2D6_is_TDI": 40})
    assert "tdi_track_candidate_A_hits_only.csv" in text and "ab" * 32 in text
    assert "the receipts of the portal, not this folder, record what was uploaded and when" in text and "has been uploaded" not in text
    assert "differ in 0 CYP3A4 calls and 40 CYP2D6 calls" in text


def test_the_tdi_readme_shows_a_failed_check_instead_of_hiding_it():
    bad = entries()
    bad[0].update(portal_valid=False, portal_message="Expected 750 rows, got 3.")
    assert "INVALID: Expected 750 rows, got 3." in sync.render_tdi_readme(bad, MANIFEST, None)


def test_the_structure_readme_lists_every_member_and_the_reasons():
    text = sync.render_structure_readme("cd" * 32, ["B.pdb", "A.pdb"], 0, {"recipe": {"reasons": ["restraint 'bond': gate MET"]}}, "ok")
    assert text.index("`A.pdb`") < text.index("`B.pdb`")
    assert "restraint 'bond': gate MET" in text and "0 of 2 fail any check" in text and "record whether and when it was uploaded" in text


def test_the_top_readme_does_not_link_to_a_track_that_is_not_part_of_the_project():
    assert "direct_inhibition" not in sync.render_top_readme()


def test_stale_files_are_moved_aside_never_deleted(tmp_path, monkeypatch):
    monkeypatch.setattr(sync, "SUBMISSIONS", tmp_path)
    folder = tmp_path / "tdi_track"
    folder.mkdir()
    for name in ("README.md", "old.csv", "keep.csv"):
        (folder / name).write_text(name)
    moved = sync.archive_stale(folder, {"keep.csv"})
    assert moved == ["old.csv"]
    assert (tmp_path / "_superseded" / "tdi_track" / "old.csv").read_text() == "old.csv"
    assert (folder / "keep.csv").exists() and (folder / "README.md").exists() and not (folder / "old.csv").exists()


def test_the_calls_that_differ_between_two_candidates_are_counted(tmp_path):
    base = pd.DataFrame({"SMILES": list("abcd"), "Molecule_Name": list("wxyz"), "CYP2D6_is_TDI": [True, False, False, False], "CYP3A4_is_TDI": [True, True, False, False]})
    other = base.assign(CYP2D6_is_TDI=[False, True, False, False])
    base.to_csv(tmp_path / "a.csv", index=False)
    other.to_csv(tmp_path / "b.csv", index=False)
    assert sync.differing_calls(tmp_path / "a.csv", tmp_path / "b.csv") == {"CYP3A4_is_TDI": 0, "CYP2D6_is_TDI": 2}


def test_the_hash_is_the_sha256_of_the_bytes(tmp_path):
    (tmp_path / "f").write_bytes(b"abc")
    assert sync.sha256(tmp_path / "f") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


@pytest.mark.skipif(not sync.FINAL_STRUCTURE.joinpath("structures_final.zip").exists(), reason="the final structure zip is not built in this checkout")
def test_the_final_files_pass_their_checks():
    assert all(e["schema_valid"] and e["portal_valid"] for e in sync.sync_tdi(check_only=True))
    assert sync.sync_structure(check_only=True)["members"] == 20

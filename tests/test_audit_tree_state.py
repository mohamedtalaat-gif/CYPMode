# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

from pathlib import Path

from cypmode.audit.sections import AUDIT_OWN_FILES, failed_test_names, tree_state


def test_the_files_of_the_audit_itself_do_not_count_in_the_state_of_the_tree():
    first_run = " M data/structure_track/outputs/cyp3a4/logs/audit_run.log\n"
    second_run = first_run + " M audit/AUDIT_REPORT.md\n?? audit/AUDIT_REPORT.run2.md\n"
    assert tree_state(first_run) == tree_state(second_run) == (0, 0, 0)  # the two runs of run_audit.sh see the same tree


def test_every_other_modified_untracked_or_deleted_file_counts():
    porcelain = " M cypmode/audit/sections.py\nM  README.md\nMM notebooks/parts/part00.py\n?? notes.txt\n?? audit/AUDIT_REPORT.run2.md\n D old.py\n"
    assert tree_state(porcelain) == (3, 1, 1)


def test_the_files_of_the_audit_are_the_ones_that_run_audit_writes():
    script = (Path(__file__).resolve().parents[1] / "scripts/run_audit.sh").read_text()
    assert "audit/AUDIT_REPORT.md" in script and "audit/AUDIT_REPORT.run2.md" in script
    assert "audit/AUDIT_REPORT.md" in AUDIT_OWN_FILES and "audit/AUDIT_REPORT.run2.md" in AUDIT_OWN_FILES


def test_the_report_names_the_tests_that_failed_or_errored_and_nothing_when_all_passed():
    red = "..F.E\n=== short test summary info ===\nFAILED tests/test_a.py::test_one - assert 1 == 2\nERROR tests/test_video.py::test_last - subprocess.TimeoutExpired\n2 failed, 5 passed\n"
    assert failed_test_names(red) == ["FAILED tests/test_a.py::test_one", "ERROR tests/test_video.py::test_last"]
    assert failed_test_names("........\n970 passed, 1 skipped in 130.72s\n") == []


def _tdi_submission_tree(root, name, flip_hash=False):
    import hashlib
    import json

    import pandas as pd

    test = pd.DataFrame({"Molecule_Name": [f"m{i}" for i in range(750)], "SMILES": ["C" * (1 + i % 5) for i in range(750)]})
    calls = test.assign(CYP2D6_is_TDI=[i % 17 == 0 for i in range(750)], CYP3A4_is_TDI=[i % 3 == 0 for i in range(750)])
    folder = root / "submissions/tdi_track"
    folder.mkdir(parents=True)
    calls.to_csv(folder / name, index=False)
    digest = hashlib.sha256((folder / name).read_bytes()).hexdigest()
    (root / "submissions/manifest.json").write_text(json.dumps({"tdi_track": [{"file": name, "sha256": ("0" * 64) if flip_hash else digest}]}))
    return test


def test_the_readiness_section_checks_the_tdi_files_whatever_they_are_called(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from cypmode.audit import sections

    test = _tdi_submission_tree(tmp_path, "any_name_at_all.csv")
    monkeypatch.setattr(sections, "ROOT", tmp_path)
    section = sections.readiness(SimpleNamespace(test=test, prevalence=lambda iso: 0.3))
    names = [c.name for c in section.checks if "any_name_at_all.csv" in c.name]
    assert len(names) == 7 and all(c.status == "PASS" for c in section.checks if "any_name_at_all.csv" in c.name)


def test_the_readiness_section_fails_a_tdi_file_whose_hash_differs_from_the_manifest(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from cypmode.audit import sections

    test = _tdi_submission_tree(tmp_path, "file.csv", flip_hash=True)
    monkeypatch.setattr(sections, "ROOT", tmp_path)
    section = sections.readiness(SimpleNamespace(test=test, prevalence=lambda iso: 0.3))
    assert [c.name.rsplit(": ", 1)[1] for c in section.checks if c.status == "FAIL"] == ["SHA-256 equals the one recorded in the manifest"]


def test_the_readiness_section_says_so_when_there_is_no_tdi_file(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from cypmode.audit import sections

    monkeypatch.setattr(sections, "ROOT", tmp_path)
    section = sections.readiness(SimpleNamespace(test=None, prevalence=lambda iso: 0.3))
    assert "No TDI submission file on disk." in section.body and not any(c.status == "FAIL" for c in section.checks)

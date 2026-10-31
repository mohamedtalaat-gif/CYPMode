# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import json
import subprocess

import pytest

from cypmode import data_record as dr

POLICY = {
    "archive_suffix": ".tar.gz",
    "record": {"doi": "", "base_url": "", "url_suffix": ""},
    "groups": [
        {"name": "outputs", "paths": ["data/out"], "exclude": ["data/out/raw"]},
        {"name": "raw", "paths": ["data/out/raw"]},
    ],
    "not_in_the_record": [{"paths": ["data/inputs"], "reason": "downloaded from their own sources"}],
}


def _repository(tmp_path):
    root = tmp_path / "repo"
    (root / "data_record").mkdir(parents=True)
    (root / "data_record/record.json").write_text(json.dumps(POLICY))
    for name, text in {"data/out/a.csv": "a", "data/out/b/c.csv": "cc", "data/out/raw/r.bin": "raw", "data/inputs/third.bin": "third"}.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / ".gitignore").write_text("data/\n")  # ignored files belong to the record all the same
    return root


def test_a_group_holds_the_untracked_files_of_its_folders_minus_the_excluded_ones(tmp_path):
    root = _repository(tmp_path)
    policy = dr.load_policy(root)
    assert dr.group_files(root, policy["groups"][0]) == ["data/out/a.csv", "data/out/b/c.csv"]
    assert dr.group_files(root, policy["groups"][1]) == ["data/out/raw/r.bin"]


def test_the_build_writes_an_archive_per_group_and_the_manifest_with_their_sizes_and_hashes(tmp_path):
    root = _repository(tmp_path)
    manifest = dr.build(root, tmp_path / "archives")
    assert [e["name"] for e in manifest["archives"]] == ["outputs", "raw"]
    for entry in manifest["archives"]:
        archive = tmp_path / "archives" / entry["archive"]
        assert archive.stat().st_size == entry["bytes"] and dr.sha256_of(archive) == entry["sha256"]
    assert manifest["archives"][0]["files"] == 2 and dr.listed_paths(root, "outputs") == ["data/out/a.csv", "data/out/b/c.csv"]
    assert json.loads((root / dr.MANIFEST).read_text()) == manifest


def test_nothing_is_dropped_without_being_named(tmp_path):
    root = _repository(tmp_path)
    assert dr.left_out(root) == []  # the third-party inputs are left out by name
    without_raw = {**POLICY, "groups": POLICY["groups"][:1]}  # the folder that the first group excludes is now covered by no group
    (root / "data_record/record.json").write_text(json.dumps(without_raw))
    assert dr.left_out(root) == ["data/out/raw/r.bin"]


def test_the_missing_files_are_named_per_group_and_fetch_restores_them_from_a_folder(tmp_path):
    root = _repository(tmp_path)
    dr.build(root, tmp_path / "archives")
    assert dr.missing(root) == {}
    (root / "data/out/a.csv").unlink()
    (root / "data/out/raw/r.bin").unlink()
    gone = dr.missing(root)
    assert gone == {"outputs": ["data/out/a.csv"], "raw": ["data/out/raw/r.bin"]}
    assert "python scripts/fetch_data.py fetch" in dr.message_for(gone)
    assert dr.fetch(root, str(tmp_path / "archives"), work=tmp_path / "work") == ["outputs", "raw"]
    assert dr.missing(root) == {} and (root / "data/out/a.csv").read_text() == "a"


def test_an_archive_whose_hash_differs_from_the_manifest_is_refused_before_anything_is_extracted(tmp_path):
    root = _repository(tmp_path)
    dr.build(root, tmp_path / "archives")
    (root / "data/out/a.csv").unlink()
    archive = tmp_path / "archives/outputs.tar.gz"
    archive.write_bytes(archive.read_bytes() + b"x")
    with pytest.raises(SystemExit, match="differs from the manifest"):
        dr.fetch(root, str(tmp_path / "archives"), work=tmp_path / "work")
    assert not (root / "data/out/a.csv").exists()


def test_fetch_needs_an_address_when_the_manifest_has_none(tmp_path):
    root = _repository(tmp_path)
    dr.build(root, tmp_path / "archives")
    with pytest.raises(SystemExit, match="names no address"):
        dr.fetch(root)


def test_there_is_no_check_before_the_record_exists(tmp_path):
    assert dr.missing(_repository(tmp_path)) == {}


def test_the_policy_of_the_repository_names_every_group_and_the_left_out_folders():
    policy = dr.load_policy()
    assert [g["name"] for g in policy["groups"]] and all(g["paths"] for g in policy["groups"])
    assert policy["not_in_the_record"] and all(e["reason"] for e in policy["not_in_the_record"])


def test_the_audit_names_the_missing_inputs_and_how_to_get_them_before_it_reads_one(tmp_path, monkeypatch):
    from cypmode.audit import report

    root = _repository(tmp_path)
    dr.build(root, tmp_path / "archives")
    (root / "data/out/a.csv").unlink()
    monkeypatch.setattr(report, "ROOT", root)
    monkeypatch.setattr("sys.argv", ["report", "--skip-tests"])
    with pytest.raises(SystemExit) as stopped:
        report.main()
    assert "data/out/a.csv" in str(stopped.value) and "fetch_data.py fetch" in str(stopped.value)

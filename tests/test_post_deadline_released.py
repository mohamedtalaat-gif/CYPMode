# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import io
import json
import tarfile

from cypmode.post_deadline import released


def _tarball(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(f"repo-abc/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _fake_downloads(monkeypatch, listing):
    tarball = _tarball({"evaluation/evaluate_predictions.py": b"x = 1\n", "inputs/boltz-2_cyp3a4_predictions/a.cif": b"data_a\n", "notebooks/skipped.ipynb": b"{}"})

    def download(url: str) -> bytes:
        if url.endswith("/tree/main/structures_unblinded"):
            return json.dumps([{"path": f"structures_unblinded/{name}"} for name in listing]).encode()
        if url.startswith("https://codeload.github.com/"):
            return tarball
        return b"content of " + url.rsplit("/", 1)[1].encode()

    monkeypatch.setattr(released, "_download", download)


def test_the_fetch_copies_the_named_files_keeps_only_what_the_plan_uses_and_records_every_checksum(tmp_path, monkeypatch):
    _fake_downloads(monkeypatch, ["OCNT-1.pdb", "OCNT-2.pdb", "readme.md"])
    manifest = released.fetch(tmp_path / "in", tmp_path / "manifest.json")
    paths = [entry["path"] for entry in manifest["files"]]
    assert paths == sorted(paths) and f"dataset/{released.TDI_FILE}" in paths
    assert "dataset/structures_unblinded/OCNT-1.pdb" in paths and "dataset/structures_unblinded/readme.md" not in paths  # only the structures are taken from the folder
    assert "organisers/evaluation/evaluate_predictions.py" in paths and "organisers/inputs/boltz-2_cyp3a4_predictions/a.cif" in paths and "organisers/notebooks/skipped.ipynb" not in paths
    assert all(len(entry["sha256"]) == 64 and entry["bytes"] > 0 and entry["source"].startswith("https://") for entry in manifest["files"])
    assert manifest["tutorial_commit"] == released.TUTORIAL_COMMIT and json.loads((tmp_path / "manifest.json").read_text()) == manifest


def test_a_second_fetch_downloads_nothing_that_is_already_there(tmp_path, monkeypatch):
    _fake_downloads(monkeypatch, ["OCNT-1.pdb"])
    first = released.fetch(tmp_path / "in", tmp_path / "m.json")
    calls = []
    monkeypatch.setattr(released, "_download", lambda url: calls.append(url) or b"")
    # the listing of the structures is read again; the files and the archive are not
    monkeypatch.setattr(released, "structure_names", lambda: ["structures_unblinded/OCNT-1.pdb"])
    assert released.fetch(tmp_path / "in", tmp_path / "m.json") == first and calls == []


def test_the_verification_names_a_file_that_changed_or_is_missing(tmp_path, monkeypatch):
    _fake_downloads(monkeypatch, ["OCNT-1.pdb"])
    manifest = released.fetch(tmp_path / "in", tmp_path / "m.json")
    assert released.verify(tmp_path / "in", manifest) == []
    (tmp_path / "in" / "dataset" / released.TDI_FILE).write_bytes(b"changed")
    (tmp_path / "in" / "dataset/structures_unblinded/OCNT-1.pdb").unlink()
    assert released.verify(tmp_path / "in", manifest) == sorted([f"dataset/{released.TDI_FILE}", "dataset/structures_unblinded/OCNT-1.pdb"])

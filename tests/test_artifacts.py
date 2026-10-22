# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import sys
from pathlib import Path

import pytest

from cypmode.audit import artifacts

sys.path[:0] = ["docs/video", "docs/video/anime"]
from chrome import CHROME

needs_chrome = pytest.mark.skipif(not Path(CHROME).exists(), reason="Google Chrome is not installed")


@pytest.fixture
def one_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "REQUIRED", ("card",))
    (tmp_path / "card.png").write_bytes(b"picture")
    manifest = tmp_path / "audit" / "artifacts.json"
    artifacts.record("card", [tmp_path / "card.png"], {"html_sha256": "x"}, fingerprint_fn=lambda: "numbers-v1", manifest=manifest, root=tmp_path)
    return tmp_path, manifest


def statuses(rows):
    return [status for _, status, _ in rows]


def test_an_artifact_built_from_the_current_numbers_and_untouched_passes(one_artifact):
    root, manifest = one_artifact
    assert statuses(artifacts.verify(fingerprint_fn=lambda: "numbers-v1", manifest=manifest, root=root, deep_checks={})) == ["PASS", "PASS"]


def test_an_artifact_built_before_the_numbers_changed_is_stale(one_artifact):
    root, manifest = one_artifact
    rows = artifacts.verify(fingerprint_fn=lambda: "numbers-v2", manifest=manifest, root=root, deep_checks={})
    assert statuses(rows) == ["PASS", "FAIL"] and "rebuild" in rows[1][2]


def test_an_artifact_that_was_edited_after_the_build_fails(one_artifact):
    root, manifest = one_artifact
    (root / "card.png").write_bytes(b"retouched")
    assert statuses(artifacts.verify(fingerprint_fn=lambda: "numbers-v1", manifest=manifest, root=root, deep_checks={}))[0] == "FAIL"


def test_a_missing_record_is_a_failure_and_a_failed_deep_check_is_reported(one_artifact, monkeypatch):
    root, manifest = one_artifact
    monkeypatch.setattr(artifacts, "REQUIRED", ("card", "film"))
    rows = artifacts.verify(deep=True, fingerprint_fn=lambda: "numbers-v1", manifest=manifest, root=root, deep_checks={"card": lambda entry: (False, "differs")})
    assert ("film: built by its builder and recorded", "FAIL") == rows[-1][:2]
    assert any(name.startswith("card: its source") and status == "FAIL" for name, status, _ in rows)


@needs_chrome
def test_the_dom_of_the_poster_is_deterministic():
    """The raster of a frame can differ in a few pixels between two renders; the DOM that produces it must not."""
    import render_video

    assert render_video.dom_digests(poster_only=True)["poster"] == render_video.dom_digests(poster_only=True)["poster"]

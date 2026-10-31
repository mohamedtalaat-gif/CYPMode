# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("bump", Path("scripts/bump_version.py"))
bump = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bump)

NAMES = ["pyproject.toml", "CITATION.cff", ".zenodo.json", "README.md", "USE_AND_CITATION.md"]


def _files():
    return {n: Path(n).read_text() for n in NAMES}


def test_every_file_that_states_the_version_is_changed_and_the_old_release_is_left_nowhere_in_them():
    before = _files()
    old_version = before["pyproject.toml"].split('version = "')[1].split('"')[0]
    after = bump.bump(before, "9.8.7", "2031-02-03", "10.5281/zenodo.99999999")
    assert all(after[n] != before[n] for n in NAMES)
    assert 'version = "9.8.7"' in after["pyproject.toml"] and "version: 9.8.7" in after["CITATION.cff"] and "date-released: 2031-02-03" in after["CITATION.cff"]
    assert '"version": "9.8.7"' in after[".zenodo.json"]
    for name in ("README.md", "USE_AND_CITATION.md"):
        assert "version = {9.8.7}" in after[name] and "doi     = {10.5281/zenodo.99999999}" in after[name] and f"version = {{{old_version}}}" not in after[name]
    assert "release 9.8.7 has its own DOI, [10.5281/zenodo.99999999](https://doi.org/10.5281/zenodo.99999999)" in after["README.md"]
    assert "(release 9.8.7: 10.5281/zenodo.99999999)" in after["USE_AND_CITATION.md"] and "A ready entry for release 9.8.7:" in after["USE_AND_CITATION.md"]


def test_the_doi_of_all_versions_is_never_touched():
    before = _files()
    concept = before["CITATION.cff"].split("\ndoi: ")[1].split("\n")[0]
    after = bump.bump(before, "9.8.7", "2031-02-03", "10.5281/zenodo.99999999")
    assert f"\ndoi: {concept}\n" in after["CITATION.cff"] and concept in after["README.md"] and concept in after["USE_AND_CITATION.md"]


def test_a_pattern_that_does_not_match_stops_the_whole_change():
    broken = _files()
    broken["CITATION.cff"] = broken["CITATION.cff"].replace("date-released:", "released:")
    with pytest.raises(SystemExit, match="CITATION.cff date-released"):
        bump.bump(broken, "9.8.7", "2031-02-03", "10.5281/zenodo.99999999")


def test_running_it_twice_with_the_same_values_changes_nothing_the_second_time():
    once = bump.bump(_files(), "9.8.7", "2031-02-03", "10.5281/zenodo.99999999")
    assert bump.bump(once, "9.8.7", "2031-02-03", "10.5281/zenodo.99999999") == once

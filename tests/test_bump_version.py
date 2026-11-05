# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("bump", Path("scripts/bump_version.py"))
bump = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bump)

CONCEPT = "10.5281/zenodo.23816450"
BIBTEX = "@software{x,\n  author  = {A},\n  year    = {2026},\n  version = {0.3.0},\n  doi     = {10.5281/zenodo.23822983},\n  url     = {https://example.org}\n}"
# the files of a release that has a DOI of its own, reduced to what the script reads
WITH_DOI = {
    "pyproject.toml": '[project]\nname = "x"\nversion = "0.3.0"\n',
    "CITATION.cff": f"cff-version: 1.2.0\nversion: 0.3.0\ndoi: {CONCEPT}\nidentifiers:\n  - type: doi\n    value: 10.5281/zenodo.23822983\n    description: \"Release 0.3.0 of the archive on Zenodo\"\n  - type: swh\n    value: swh:1:rel:abc\n    description: \"Release 0.3.0 in the Software Heritage archive\"\nabstract: \"a\"\ndate-released: 2026-11-05\nurl: \"https://example.org\"\n",
    ".zenodo.json": '{\n  "title": "x",\n  "version": "0.3.0"\n}\n',
    "README.md": f"```bibtex\n{BIBTEX}\n```\n\nThe DOI that stands for every version of the archive on Zenodo is [{CONCEPT}](https://doi.org/{CONCEPT}); release 0.3.0 has its own DOI, [10.5281/zenodo.23822983](https://doi.org/10.5281/zenodo.23822983), which the entry above uses.\n",
    "USE_AND_CITATION.md": f"Cite the archive by the DOI that stands for all its versions, {CONCEPT}, or a release by its own DOI (release 0.3.0: 10.5281/zenodo.23822983). Release 0.2.0 held less.\nA ready entry for release 0.3.0:\n\n```bibtex\n{BIBTEX}\n```\n",
}
NAMES = list(WITH_DOI)


def test_every_file_that_states_the_version_is_changed_and_the_old_release_is_left_nowhere_in_them():
    after = bump.bump(WITH_DOI, "9.8.7", "2031-02-03", "10.5281/zenodo.99999999")
    assert all(after[n] != WITH_DOI[n] for n in NAMES)
    assert 'version = "9.8.7"' in after["pyproject.toml"] and "version: 9.8.7" in after["CITATION.cff"] and "date-released: 2031-02-03" in after["CITATION.cff"]
    assert '"version": "9.8.7"' in after[".zenodo.json"]
    for name in ("README.md", "USE_AND_CITATION.md"):
        assert "version = {9.8.7}" in after[name] and "doi     = {10.5281/zenodo.99999999}" in after[name] and "version = {0.3.0}" not in after[name]
    assert "release 9.8.7 has its own DOI, [10.5281/zenodo.99999999](https://doi.org/10.5281/zenodo.99999999)" in after["README.md"]
    assert "(release 9.8.7: 10.5281/zenodo.99999999)" in after["USE_AND_CITATION.md"] and "A ready entry for release 9.8.7:" in after["USE_AND_CITATION.md"]


def test_the_doi_of_all_versions_is_never_touched():
    after = bump.bump(WITH_DOI, "9.8.7", "2031-02-03", "10.5281/zenodo.99999999")
    assert f"\ndoi: {CONCEPT}\n" in after["CITATION.cff"] and CONCEPT in after["README.md"] and CONCEPT in after["USE_AND_CITATION.md"]


def test_a_pattern_that_does_not_match_stops_the_whole_change():
    broken = dict(WITH_DOI)
    broken["CITATION.cff"] = broken["CITATION.cff"].replace("date-released:", "released:")
    with pytest.raises(SystemExit, match="CITATION.cff date-released"):
        bump.bump(broken, "9.8.7", "2031-02-03", "10.5281/zenodo.99999999")


def test_running_it_twice_with_the_same_values_changes_nothing_the_second_time():
    once = bump.bump(WITH_DOI, "9.8.7", "2031-02-03", "10.5281/zenodo.99999999")
    assert bump.bump(once, "9.8.7", "2031-02-03", "10.5281/zenodo.99999999") == once


def test_a_release_without_a_doi_of_its_own_cites_the_doi_of_all_versions_and_names_the_last_deposited_release():
    after = bump.bump_without_doi(WITH_DOI, "9.8.7", "2031-02-03")
    assert 'version = "9.8.7"' in after["pyproject.toml"] and "version: 9.8.7" in after["CITATION.cff"] and "date-released: 2031-02-03" in after["CITATION.cff"] and '"version": "9.8.7"' in after[".zenodo.json"]
    assert "identifiers:" not in after["CITATION.cff"] and "swh:1:rel" not in after["CITATION.cff"] and f"\ndoi: {CONCEPT}\n" in after["CITATION.cff"]
    for name in ("README.md", "USE_AND_CITATION.md"):
        assert "version = {9.8.7}" in after[name] and f"doi     = {{{CONCEPT}}}" in after[name]
    assert "release 9.8.7 has no DOI of its own, and release 0.3.0, the last one that is deposited on Zenodo, has [10.5281/zenodo.23822983]" in after["README.md"]
    assert "Release 9.8.7 has no DOI of its own; the last release that is deposited on Zenodo, 0.3.0, has 10.5281/zenodo.23822983." in after["USE_AND_CITATION.md"] and "A ready entry for release 9.8.7:" in after["USE_AND_CITATION.md"]


def test_a_release_without_a_doi_can_be_set_again_and_keeps_the_last_deposited_release_through_a_second_release_without_one():
    first = bump.bump_without_doi(WITH_DOI, "9.8.7", "2031-02-03")
    assert bump.bump_without_doi(first, "9.8.7", "2031-02-03") == first
    second = bump.bump_without_doi(first, "9.8.8", "2031-03-04")
    assert "release 9.8.8 has no DOI of its own, and release 0.3.0, the last one that is deposited on Zenodo" in second["README.md"]
    assert "A ready entry for release 9.8.8:" in second["USE_AND_CITATION.md"] and "version = {9.8.8}" in second["README.md"]


def test_the_files_of_the_repository_state_one_version_and_the_same_date_format():
    version = {n: Path(n).read_text() for n in ("pyproject.toml", "CITATION.cff", ".zenodo.json")}
    found = {version["pyproject.toml"].split('version = "')[1].split('"')[0], version["CITATION.cff"].split("\nversion: ")[1].split("\n")[0], version[".zenodo.json"].split('"version": "')[1].split('"')[0]}
    assert len(found) == 1, found

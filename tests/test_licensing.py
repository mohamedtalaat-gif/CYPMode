# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The licences of the repository: the texts, the mapping to files, the additional term, and the same name everywhere."""

import subprocess
from pathlib import Path

import tomllib

from cypmode import provenance

ROOT = Path(__file__).resolve().parents[1]


def test_the_texts_of_the_licences_are_in_the_repository():
    for name in ("AGPL-3.0-or-later", "CC-BY-NC-SA-4.0", "CC-BY-4.0", "LicenseRef-third-party"):
        assert (ROOT / "LICENSES" / f"{name}.txt").stat().st_size > 500
    assert (ROOT / "LICENSE").read_text()[:60].startswith("GNU AFFERO GENERAL PUBLIC LICENSE")


def test_the_citation_file_pyproject_and_the_notice_agree_on_the_licence_of_the_code():
    assert provenance.load()["license"] == "AGPL-3.0-or-later"
    assert tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["license"]["text"] == "AGPL-3.0-or-later"
    notice = (ROOT / "NOTICE").read_text()
    assert "AGPL-3.0-or-later" in notice and "CC-BY-NC-SA-4.0" in notice and "CC-BY-4.0" in notice and "section 7(b)" in notice


def test_the_additional_term_names_the_notice_the_headers_and_the_provenance_records():
    notice = (ROOT / "NOTICE").read_text()
    assert "PROVENANCE.json" in notice and "cypmode/provenance.py" in notice and "5(d)" in notice


def test_reuse_maps_every_kind_of_file_to_a_licence_that_has_a_text():
    reuse = tomllib.loads((ROOT / "REUSE.toml").read_text())
    ids = {a["SPDX-License-Identifier"] for a in reuse["annotations"]}
    assert ids == {"AGPL-3.0-or-later", "CC-BY-NC-SA-4.0", "CC-BY-4.0", "LicenseRef-third-party"}
    assert all((ROOT / "LICENSES" / f"{i}.txt").exists() for i in ids | {"GPL-3.0-or-later"})


def test_every_tracked_python_file_has_its_header():
    files = [f for f in subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.split() if not f.startswith("third_party/")]
    missing = [f for f in files if "SPDX-License-Identifier" not in (ROOT / f).read_text()]
    assert not missing, missing[:5]

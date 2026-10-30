# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Who made the data and the software, written next to everything the code generates.

``CITATION.cff`` is the single source of the author, the affiliation, the ORCID, the repository, the identifiers (the DOI once it exists) and the licence. This module reads it and writes the same facts
into the places where generated data live, so that a copy of a cache or of an output directory carries its origin with it:

  * ``announce()`` prints a short notice with the citation at the start of an entry-point script (it is called from the ``__main__`` block of the scripts that generate data, never on import);
  * ``stamp(path)`` writes ``PROVENANCE.json`` into a directory, or ``<file>.provenance.json`` next to a file, once: author, repository, identifiers, licence, the text to cite, the commit and the date of the run.

Nothing is rewritten and nothing is enforced by code: a file that already exists keeps its bytes (the SHA-256 of the submitted files and of the caches is recorded in the reports), the
programs never refuse to run, and a licence, not a program, is what asks that the notice stays. The record is what a reader needs to cite the work.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CITATION = ROOT / "CITATION.cff"
SIDECAR = "PROVENANCE.json"


def load(path: Path = CITATION) -> dict:
    """The citation file as a dictionary."""
    return yaml.safe_load(path.read_text())


def authors(cff: dict) -> str:
    return ", ".join(f"{a['given-names']} {a['family-names']}" for a in cff["authors"])


def identifiers(cff: dict) -> dict[str, str]:
    """The DOIs and other identifiers that the citation file lists (empty until the archive has one)."""
    found = {}
    if cff.get("doi"):
        found["doi"] = cff["doi"]
    for item in cff.get("identifiers", []):
        found[item["type"]] = item["value"]
    return found


def citation_text(cff: dict) -> str:
    """The sentence to cite, built from the citation file."""
    ids = identifiers(cff)
    reference = f"https://doi.org/{ids['doi']}" if "doi" in ids else cff["repository-code"]
    return f"{authors(cff)} ({str(cff['date-released'])[:4]}). {cff['title']}. Version {cff['version']}. {reference}"


def commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() or None if result.returncode == 0 else None


def _relative(argument: str) -> str:
    """An argument that points inside the repository, written relative to it, so that a record does not carry the path of the machine that made it."""
    prefix = str(ROOT) + os.sep
    if argument.startswith(prefix):
        return argument.removeprefix(prefix).replace(os.sep, "/")  # the record reads the same on every system
    return argument


def record(command: list[str] | None = None) -> dict:
    """Everything the sidecar says, for the run that is happening now."""
    cff = load()
    licence = cff.get("license")
    return {
        "work": cff["title"],
        "authors": [{"name": f"{a['given-names']} {a['family-names']}", "orcid": a.get("orcid"), "affiliation": a.get("affiliation")} for a in cff["authors"]],
        "repository": cff["repository-code"],
        "identifiers": identifiers(cff),
        "licence": licence if isinstance(licence, list) else [licence],
        "cite_as": citation_text(cff),
        "commit": commit(),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": [_relative(a) for a in (command if command is not None else sys.argv)],
    }


def notice() -> str:
    cff = load()
    return f"{cff['title']} | {authors(cff)} | {cff['repository-code']} | cite: {citation_text(cff)}"


def announce() -> None:
    """The notice, once, on the error stream (the output of a script stays clean). A missing or unreadable citation file never stops a run."""
    try:
        print(notice(), file=sys.stderr)
    except (OSError, KeyError, yaml.YAMLError):
        print("CYPMode (citation file not readable)", file=sys.stderr)


def stamp(path: str | Path) -> Path | None:
    """Write the sidecar for a directory or a file unless one is there already; returns its path (None when the target does not exist)."""
    try:
        return _stamp(Path(path))
    except (OSError, KeyError, yaml.YAMLError):  # a record that cannot be written must not stop the computation that it describes
        return None


def _stamp(target: Path) -> Path | None:
    if target.is_dir():
        sidecar = target / SIDECAR
    elif target.exists():
        sidecar = target.with_name(target.name + ".provenance.json")
    else:
        return None
    if not sidecar.exists():
        sidecar.write_text(json.dumps(record(), indent=2) + "\n")
    return sidecar

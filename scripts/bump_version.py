#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Set the version of a release in every file that states it: ``python scripts/bump_version.py --version 0.3.0 --date 2026-11-06 --doi 10.5281/zenodo.NNNNNNNN [--dry-run]``.

The files are ``pyproject.toml``, ``CITATION.cff``, ``.zenodo.json``, the BibTeX entry and the DOI sentence of ``README.md`` and the citation paragraph and entry of ``USE_AND_CITATION.md``. ``--doi`` is the DOI of this release (the
version DOI that Zenodo reserves); the DOI that stands for all versions is not touched. Each pattern must match exactly once, or nothing is written."""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZENODO = r"10\.5281/zenodo\.\d+"


def _sub(text: str, pattern: str, replacement: str, label: str, flags: int = 0) -> str:
    result, count = re.subn(pattern, lambda m: m.expand(replacement), text, flags=flags)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}; nothing was written")
    return result


def bibtex(text: str, version: str, doi: str, label: str) -> str:
    return _sub(text, r"version = \{[^}]+\},\n  doi     = \{[^}]+\}", f"version = {{{version}}},\n  doi     = {{{doi}}}", label)


def bump(files: dict[str, str], version: str, date: str, doi: str) -> dict[str, str]:
    out = dict(files)
    out["pyproject.toml"] = _sub(out["pyproject.toml"], r'^version = "[^"]+"', f'version = "{version}"', "pyproject.toml version", re.MULTILINE)
    c = out["CITATION.cff"]
    c = _sub(c, r"^version: \S+$", f"version: {version}", "CITATION.cff version", re.MULTILINE)
    c = _sub(c, r"^date-released: \S+$", f"date-released: {date}", "CITATION.cff date-released", re.MULTILINE)
    c = _sub(c, rf"(- type: doi\n    value: ){ZENODO}(\n    description: \")Release \S+( of the archive on Zenodo\")", f"\\g<1>{doi}\\g<2>Release {version}\\g<3>", "CITATION.cff identifier")
    out["CITATION.cff"] = c
    out[".zenodo.json"] = _sub(out[".zenodo.json"], r'("version": ")[^"]+(")', f"\\g<1>{version}\\g<2>", ".zenodo.json version")
    link = f"[{doi}](https://doi.org/{doi})"
    r = bibtex(out["README.md"], version, doi, "README.md BibTeX entry")
    out["README.md"] = _sub(r, rf"release \S+ has its own DOI, \[{ZENODO}\]\(https://doi\.org/{ZENODO}\)", f"release {version} has its own DOI, {link}", "README.md DOI sentence")
    u = bibtex(out["USE_AND_CITATION.md"], version, doi, "USE_AND_CITATION.md BibTeX entry")
    u = _sub(u, rf"\(release \S+: {ZENODO}\)", f"(release {version}: {doi})", "USE_AND_CITATION.md DOI of the release")
    out["USE_AND_CITATION.md"] = _sub(u, r"A ready entry for release \S+:", f"A ready entry for release {version}:", "USE_AND_CITATION.md entry heading")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--date", required=True, help="the date of the release, YYYY-MM-DD")
    parser.add_argument("--doi", required=True, help="the DOI of this release, 10.5281/zenodo.N")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"\d+\.\d+\.\d+", args.version) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.date) or not re.fullmatch(ZENODO, args.doi):
        raise SystemExit("the version must look like 0.3.0, the date like 2026-11-06 and the DOI like 10.5281/zenodo.123")
    names = ["pyproject.toml", "CITATION.cff", ".zenodo.json", "README.md", "USE_AND_CITATION.md"]
    new = bump({n: (ROOT / n).read_text() for n in names}, args.version, args.date, args.doi)
    for name in names:
        changed = new[name] != (ROOT / name).read_text()
        print(f"{name}: {'changed' if changed else 'already as asked'}")
        if changed and not args.dry_run:
            (ROOT / name).write_text(new[name])
    return 0


if __name__ == "__main__":
    sys.exit(main())

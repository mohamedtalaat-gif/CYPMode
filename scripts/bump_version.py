#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Set the version of a release in every file that states it: ``python scripts/bump_version.py --version 0.3.0 --date 2026-11-06 --doi 10.5281/zenodo.NNNNNNNN [--dry-run]``, or, for a release that is not
deposited on Zenodo and so has no DOI of its own, ``--no-doi`` instead of ``--doi``.

The files are ``pyproject.toml``, ``CITATION.cff``, ``.zenodo.json``, the BibTeX entry and the DOI sentence of ``README.md`` and the citation paragraph and entry of ``USE_AND_CITATION.md``. ``--doi`` is the DOI of this release (the
version DOI that Zenodo reserves); the DOI that stands for all versions is not touched. ``--no-doi`` removes the identifiers of the previous release from the citation file, cites the DOI of all versions in the two entries and
says in the two documents that the release has no DOI of its own and which release is the last one that is deposited. A release with a DOI cannot be set from the files of a release without one (the sentences differ), and the script
says so. Each pattern must match exactly once, or nothing is written."""

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


def _concept(files: dict[str, str]) -> str:
    match = re.search(rf"^doi: ({ZENODO})$", files["CITATION.cff"], re.MULTILINE)
    if match is None:
        raise SystemExit("CITATION.cff: the DOI of all versions is missing; nothing was written")
    return match.group(1)


def bump_without_doi(files: dict[str, str], version: str, date: str) -> dict[str, str]:
    """The files of a release that has no DOI of its own: the version and the date are set, the identifiers of the previous release leave the citation file, and the two entries cite the DOI of all versions."""
    out = dict(files)
    concept = _concept(files)
    out["pyproject.toml"] = _sub(out["pyproject.toml"], r'^version = "[^"]+"', f'version = "{version}"', "pyproject.toml version", re.MULTILINE)
    c = _sub(out["CITATION.cff"], r"^version: \S+$", f"version: {version}", "CITATION.cff version", re.MULTILINE)
    c = _sub(c, r"^date-released: \S+$", f"date-released: {date}", "CITATION.cff date-released", re.MULTILINE)
    c = re.sub(r"^identifiers:\n(?:  - .*\n(?:    .*\n)*)+", "", c, flags=re.MULTILINE)  # nothing to remove when the previous release had none either
    out["CITATION.cff"] = c
    out[".zenodo.json"] = _sub(out[".zenodo.json"], r'("version": ")[^"]+(")', f"\\g<1>{version}\\g<2>", ".zenodo.json version")
    readme = bibtex(out["README.md"], version, concept, "README.md BibTeX entry")
    have = re.search(rf"release (\S+) has its own DOI, \[({ZENODO})\]\(https://doi\.org/{ZENODO}\), which the entry above uses\.", readme)
    kept = re.search(rf"release (\S+), the last one that is deposited on Zenodo, has \[({ZENODO})\]", readme)
    last, last_doi = (have.group(1), have.group(2)) if have else (kept.group(1), kept.group(2)) if kept else (None, None)
    if last is None:
        raise SystemExit("README.md DOI sentence: neither form was found; nothing was written")
    sentence = (f"The DOI that stands for every version of the archive on Zenodo is [{concept}](https://doi.org/{concept}), which the entry above uses; release {version} has no DOI of its own, "
                f"and release {last}, the last one that is deposited on Zenodo, has [{last_doi}](https://doi.org/{last_doi}).")
    out["README.md"] = _sub(readme, rf"The DOI that stands for every version of the archive on Zenodo is \[{ZENODO}\]\(https://doi\.org/{ZENODO}\)(?:, which the entry above uses; release \S+ has no DOI of its own, and release \S+, the last one that is deposited on Zenodo, has \[{ZENODO}\]\(https://doi\.org/{ZENODO}\)\.|; release \S+ has its own DOI, \[{ZENODO}\]\(https://doi\.org/{ZENODO}\), which the entry above uses\.)", sentence, "README.md DOI sentence")
    u = bibtex(out["USE_AND_CITATION.md"], version, concept, "USE_AND_CITATION.md BibTeX entry")
    paragraph = (f"Cite the archive by the DOI that stands for all its versions, {concept}, and name the version. Release {version} has no DOI of its own; the last release that is deposited on Zenodo, {last}, has {last_doi}. ")
    u = _sub(u, rf"Cite the archive by the DOI that stands for all its versions, {ZENODO}, (?:or a release by its own DOI \(release \S+: {ZENODO}\)\. |and name the version\. Release \S+ has no DOI of its own; the last release that is deposited on Zenodo, \S+, has {ZENODO}\. )", paragraph, "USE_AND_CITATION.md citation paragraph")
    out["USE_AND_CITATION.md"] = _sub(u, r"A ready entry for release \S+:", f"A ready entry for release {version}:", "USE_AND_CITATION.md entry heading")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--date", required=True, help="the date of the release, YYYY-MM-DD")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--doi", help="the DOI of this release, 10.5281/zenodo.N")
    group.add_argument("--no-doi", action="store_true", help="the release is not deposited on Zenodo and has no DOI of its own")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"\d+\.\d+\.\d+", args.version) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.date) or (args.doi is not None and not re.fullmatch(ZENODO, args.doi)):
        raise SystemExit("the version must look like 0.3.0, the date like 2026-11-06 and the DOI like 10.5281/zenodo.123")
    names = ["pyproject.toml", "CITATION.cff", ".zenodo.json", "README.md", "USE_AND_CITATION.md"]
    current = {n: (ROOT / n).read_text() for n in names}
    new = bump_without_doi(current, args.version, args.date) if args.no_doi else bump(current, args.version, args.date, args.doi)
    for name in names:
        changed = new[name] != (ROOT / name).read_text()
        print(f"{name}: {'changed' if changed else 'already as asked'}")
        if changed and not args.dry_run:
            (ROOT / name).write_text(new[name])
    return 0


if __name__ == "__main__":
    sys.exit(main())

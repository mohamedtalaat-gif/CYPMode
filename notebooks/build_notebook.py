# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Build the notebook: ``python -m notebooks.build_notebook [--draft] [--no-export]``.

The cells of ``notebook_cells.py`` get their generated contents table and coverage appendix, run against the repository (a failing cell fails the build), pass the glossary check (every defined term
is defined in bold where it is first used) and the coverage check (every module, script, queue job and test file is named in the text), and are written to ``cypmode_notebook.ipynb``; the HTML and the
PDF follow from ``build_exports``. ``--draft`` builds the parts that exist and reports, instead of failing on, the files that are not explained yet and the terms not used yet.
"""

import argparse
import sys
from pathlib import Path

from notebooks.notebook_builder import (
    build,
    contents_cell,
    coverage,
    coverage_cell,
    failures,
    field_problems,
    glossary_problems,
    headings,
    symbol_problems,
    unwritten_parts,
)
from notebooks.notebook_cells import ALL_CELLS, TERMS

OUT = Path(__file__).resolve().parent / "cypmode_notebook.ipynb"


def assemble(cells: list[dict]) -> tuple[list[dict], list[str]]:
    """The cells with the coverage appendix appended and the contents table filled in; also the repository files no section names."""
    mapping, missing = coverage(cells)
    full = [*cells, coverage_cell(mapping)]
    contents = contents_cell(full)
    return [contents if c.get("placeholder") == "contents" else c for c in full], missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", action="store_true", help="build what exists; do not fail on unexplained files or terms not used yet")
    parser.add_argument("--no-export", action="store_true", help="stop after the .ipynb")
    parser.add_argument("--written", action="store_true", help="only report the planned parts and appendices that have no text yet; exit status 1 when any is missing")
    args = parser.parse_args()
    if args.written:
        left = unwritten_parts(ALL_CELLS, (Path(__file__).resolve().parent / "PLAN.md").read_text())
        print("every planned part and appendix is written" if not left else f"{len(left)} not written yet: " + ", ".join(left))
        return 1 if left else 0
    cells, missing = assemble(ALL_CELLS)
    part_zero = next((i for lv, text, i in headings(cells) if lv == 1 and text.startswith("Part 0")), 0)
    problems = glossary_problems(cells, TERMS, part_zero)
    problems += field_problems(cells) + symbol_problems(cells, part_zero)  # every section names its field; every symbol of an equation is named in the text before it
    if args.draft:
        problems = [p for p in problems if "never used" not in p]
    executed = build(cells, OUT)
    bad = failures(executed)
    print(f"{OUT.name}: {len(executed)} cells, {len(bad)} failed cells {bad}, {len(problems)} problems (glossary, fields, symbols), {len(missing)} repository files not yet explained")
    for line in problems:
        print("  problem:", line)
    if missing and not args.draft:
        for line in missing[:40]:
            print("  not explained:", line)
    if bad or problems or (missing and not args.draft):
        return 1
    if not args.no_export:
        from notebooks import build_exports

        page = build_exports.build_html()
        print(f"{page.name}: {page.stat().st_size / 1e6:.1f} MB")
        if Path(build_exports.CHROME).exists():
            pdf = build_exports.build_pdf(page)
            print(f"{pdf.name}: {pdf.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

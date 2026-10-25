# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Build the notebook: ``python -m notebooks.build_notebook [--draft] [--no-export] [--public]``.

The cells of ``notebook_cells.py`` get their generated contents table and coverage appendix, run against the repository (a failing cell fails the build), pass the glossary check (every defined term
is defined in bold where it is first used) and the coverage check (every module, script, queue job and test file is named in the text), and are written to ``cypmode_notebook.ipynb``; the HTML and the
PDF follow from ``build_exports``. ``--draft`` builds the parts that exist and reports, instead of failing on, the files that are not explained yet and the terms not used yet.

``--public`` builds the edition that is published before the challenge closes (``public/cypmode_notebook_public.*``): the same cells, run the same way, but the outputs and figures that cells marked
``held_back`` show (results for the blinded compounds) are replaced by a note, a banner says so, and the build fails if any cell still names an official structure ligand or five or more blinded
compounds. The complete edition replaces it after the challenge.
"""

import argparse
import fnmatch
import sys
from pathlib import Path

from notebooks.notebook_builder import (
    FIGURES,
    PUBLIC_BANNER,
    ROOT,
    build,
    contents_cell,
    coverage,
    coverage_cell,
    failures,
    field_problems,
    figure_sections,
    glossary_problems,
    headings,
    md,
    place_unshown_figures,
    public_problems,
    symbol_problems,
    unwritten_parts,
)
from notebooks.notebook_cells import ALL_CELLS, TERMS

OUT = Path(__file__).resolve().parent / "cypmode_notebook.ipynb"
PUBLIC_OUT = Path(__file__).resolve().parent / "public" / "cypmode_notebook_public.ipynb"


def assemble(cells: list[dict]) -> tuple[list[dict], list[str]]:
    """The cells with the coverage appendix appended and the contents table filled in; also the repository files no section names."""
    catalog = FIGURES / "CATALOG.md"
    if catalog.exists():  # a figure that is drawn and that no part shows is put at the end of the section that the catalog names
        from cypmode.explorer.data import held_back_patterns

        patterns = [pattern.rstrip("/") for pattern in held_back_patterns()]
        paths = {fid: next(iter(sorted(FIGURES.glob(f"*/{fid}-dark.png"))), None) for fid in figure_sections(catalog.read_text())}
        held = {fid for fid, path in paths.items() if path is not None and any(fnmatch.fnmatch(path.relative_to(ROOT).as_posix(), pattern + "*" if not pattern.endswith("*") else pattern) for pattern in patterns)}
        cells, _added = place_unshown_figures(cells, figure_sections(catalog.read_text()), held)
    mapping, missing = coverage(cells)
    full = [*cells, coverage_cell(mapping)]
    contents = contents_cell(full)
    return [contents if c.get("placeholder") == "contents" else c for c in full], missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", action="store_true", help="build what exists; do not fail on unexplained files or terms not used yet")
    parser.add_argument("--no-export", action="store_true", help="stop after the .ipynb")
    parser.add_argument("--public", action="store_true", help="the edition published before the challenge closes: the outputs marked held_back become a note (public/cypmode_notebook_public.*)")
    parser.add_argument("--written", action="store_true", help="only report the planned parts and appendices that have no text yet; exit status 1 when any is missing")
    args = parser.parse_args()
    if args.written:
        left = unwritten_parts(ALL_CELLS, (Path(__file__).resolve().parent / "PLAN.md").read_text())
        print("every planned part and appendix is written" if not left else f"{len(left)} without text: " + ", ".join(left))
        return 1 if left else 0
    cells, missing = assemble(ALL_CELLS)
    part_zero = next((i for lv, text, i in headings(cells) if lv == 1 and text.startswith("Part 0")), 0)
    problems = glossary_problems(cells, TERMS, part_zero)
    problems += field_problems(cells) + symbol_problems(cells, part_zero)  # every section names its field; every symbol of an equation is named in the text before it
    if args.draft:
        problems = [p for p in problems if "never used" not in p]
    if args.public:
        cells = [cells[0], md(PUBLIC_BANNER), *cells[1:]]
        PUBLIC_OUT.parent.mkdir(exist_ok=True)
    target = PUBLIC_OUT if args.public else OUT
    executed = build(cells, target, public=args.public)
    bad = failures(executed)
    if args.public:
        import pandas as pd

        problems += public_problems(executed, pd.read_csv("data/structure_track/inputs/structure_track_test_blinded.csv")["Molecule_Name"].tolist())
    print(f"{target.name}: {len(executed)} cells, {len(bad)} failed cells {bad}, {len(problems)} problems (glossary, fields, symbols), {len(missing)} repository files not yet explained")
    for line in problems:
        print("  problem:", line)
    if missing and not args.draft:
        for line in missing[:40]:
            print("  not explained:", line)
    if bad or problems or (missing and not args.draft):
        return 1
    if not args.no_export:
        from notebooks import build_exports

        page = build_exports.build_html(target)
        print(f"{page.name}: {page.stat().st_size / 1e6:.1f} MB")
        if Path(build_exports.CHROME).exists():
            pdf = build_exports.build_pdf(page)
            print(f"{pdf.name}: {pdf.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

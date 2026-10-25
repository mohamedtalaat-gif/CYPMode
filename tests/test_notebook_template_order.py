# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import ast
import builtins
import re

from notebooks.notebook_builder import TEMPLATE, code, md


def names_set_by(source: str) -> set[str]:
    """Every name that a code cell binds: assignments (tuples included), imports, definitions, loop targets and exception names."""
    bound = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bound |= {(alias.asname or alias.name).split(".")[0] for alias in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    return bound


def names_used_before_set(cells: list[dict]) -> list[tuple[int, str]]:
    """(index of the Markdown cell, name) for every {{=expression}} that uses a name which no earlier code cell binds."""
    bound, problems = set(), []
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            bound |= names_set_by(cell["source"])
            continue
        for match in TEMPLATE.finditer(cell["source"]):
            expression = re.sub(r":[-+.\d,a-z%<>^ ]+$", "", match.group(1).strip())  # the format specification
            tree = ast.parse(expression, mode="eval")
            local = {t.id for n in ast.walk(tree) if isinstance(n, ast.comprehension) for t in ast.walk(n.target) if isinstance(t, ast.Name)}
            for name in sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - local):
                if name not in bound and not hasattr(builtins, name):
                    problems.append((index, name))
    return problems


def test_a_number_in_the_text_that_is_computed_by_a_later_cell_is_found():
    cells = [md("The table has {{=n_rows}} rows and {{=len(kept)}} are kept."), code("n_rows, kept = 3, [1]")]
    assert names_used_before_set(cells) == [(0, "n_rows"), (0, "kept")]
    assert names_used_before_set([cells[1], cells[0]]) == []


def test_a_format_specification_a_comprehension_and_a_builtin_are_not_names_to_define():
    cells = [code("values = [1, 2]"), md("{{=sum(v for v in values):+.2f}} and {{=len(values)}} and {{=max(values):.1f}}")]
    assert names_used_before_set(cells) == []


def test_every_number_in_the_text_of_the_notebook_is_computed_before_the_cell_that_reads_it():
    from notebooks.notebook_cells import ALL_CELLS

    assert names_used_before_set(ALL_CELLS) == []  # the build fails on the first one, after the cells before it have run for as long as a quarter of an hour

# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import ast
import re

import pytest

from cypmode.audit import docs_state
from cypmode.audit.method_report import gate_rows
from notebooks.parts import part00

needs_logs = pytest.mark.skipif(not gate_rows(), reason="the logs of the experiments are not on this machine")


def stems() -> list[str]:
    found = set()
    for log, _, _ in gate_rows():
        found.add(re.sub(r"(_f10)?\.log$", "", re.sub(r"_(CYP3A4|CYP2D6)(?=(_f10)?\.log)", "", log)))
    return sorted(found)


def table(name: str) -> dict:
    """A dictionary literal of the ledger cell of Part 0, read without running the cell."""
    source = part00.LEDGER
    start = source.index(f"{name} = {{")
    return ast.literal_eval(source[source.index("{", start): source.index("\n}", start) + 2])


@needs_logs
def test_every_experiment_of_the_ledger_has_a_readable_name_and_a_section_in_the_notebook():
    described, sections = table("DESCRIPTION"), table("SECTIONS")
    without_name = [s for s in stems() if s not in described and s not in docs_state.TITLES]
    without_section = [s for s in stems() if s not in described and s not in sections]
    assert not without_name, f"shown under a file name: {without_name}"
    assert not without_section, f"shown with a question mark as the section: {without_section}"


def test_the_sections_are_written_as_numbers_that_the_table_can_sort():
    for section in table("SECTIONS").values():
        assert all(re.fullmatch(r"\d+\.\d+", part.strip()) for part in section.split(","))

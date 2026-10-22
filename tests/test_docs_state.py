# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import pytest

from cypmode.audit import docs_state as docs


def test_a_block_is_rewritten_between_its_markers_and_nothing_else_changes():
    text = "intro\n<!-- generated:a begin (x) -->\nold\n<!-- generated:a end -->\ntail\n<!-- generated:b begin (x) -->\nkeep\n<!-- generated:b end -->\n"
    out = docs.render(text, ("a",), {"a": lambda: "new\nlines"})
    assert out == "intro\n<!-- generated:a begin (x) -->\nnew\nlines\n<!-- generated:a end -->\ntail\n<!-- generated:b begin (x) -->\nkeep\n<!-- generated:b end -->\n"
    assert docs.render(out, ("a",), {"a": lambda: "new\nlines"}) == out  # rewriting is idempotent


def test_a_missing_marker_pair_is_an_error_not_a_silent_skip():
    with pytest.raises(ValueError, match="'a'"):
        docs.render("no markers here\n", ("a",), {"a": lambda: "x"})


def test_tables_and_verdict_cells():
    assert docs.table([["a", 1]], ["x", "y"]) == "| x | y |\n| --- | --- |\n| a | 1 |"
    assert docs.verdict_cell([]) == "-"
    assert docs.verdict_cell([("alone", "NOT MET"), ("stacked", "NOT MET")]) == "NOT MET"
    assert docs.verdict_cell([("alone", "MET"), ("stacked", "NOT MET")]) == "alone MET; stacked NOT MET"


def test_the_readme_and_the_specification_carry_every_marker_the_generator_writes():
    for filename, names in docs.TARGETS.items():
        text = (docs.ROOT / filename).read_text()
        for name in names:
            assert docs.pattern(name).search(text), f"{filename} lacks the block {name}"

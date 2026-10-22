# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import json

from notebooks.notebook_builder import (
    build,
    code,
    contents_cell,
    coverage,
    failures,
    glossary_problems,
    headings,
    md,
    planned_parts,
    section_of_cells,
    unwritten_parts,
)


def test_cells_run_in_a_shared_namespace_and_outputs_are_recorded(tmp_path):
    cells = [md("# title"), code("x = 21\nprint('first')"), code("print(x * 2)\nimport matplotlib.pyplot as plt\nplt.plot([1, 2, 3])")]
    done = build(cells, tmp_path / "n.ipynb")
    nb = json.loads((tmp_path / "n.ipynb").read_text())
    assert nb["nbformat"] == 4 and len(nb["cells"]) == 3 and not failures(done)
    streams = [o["text"] for c in nb["cells"] for o in c.get("outputs", []) if o["output_type"] == "stream"]
    assert "".join(streams[0]).strip() == "first" and "".join(streams[1]).strip() == "42"
    assert any(o["output_type"] == "display_data" and "image/png" in o["data"] for o in nb["cells"][2]["outputs"])


def test_a_failing_cell_is_reported_not_hidden(tmp_path):
    done = build([code("raise ValueError('boom')")], tmp_path / "n.ipynb")
    assert failures(done) == [0]


def test_glossary_rule_wants_the_definition_in_the_cell_that_first_uses_a_term():
    good = [md("An **enzyme** is a protein that speeds up a reaction."), md("The enzyme binds a molecule.")]
    assert glossary_problems(good, ["enzyme"]) == []
    late = [md("The enzyme binds a molecule."), md("An **enzyme** is a protein.")]
    assert "first used in cell 0" in glossary_problems(late, ["enzyme"])[0]
    assert "never used" in glossary_problems(good, ["ligand"])[0]


def test_a_plural_counts_as_a_use_and_the_front_matter_can_be_left_out():
    cells = [md("Enzymes are everywhere."), md("An **enzyme** is a protein.")]
    assert "first used in cell 0" in glossary_problems(cells, ["enzyme"])[0]
    assert glossary_problems(cells, ["enzyme"], start=1) == []


def test_numbers_in_markdown_are_filled_from_the_namespace_when_the_notebook_runs(tmp_path):
    done = build([code("n = 1234"), md("There are {{=n:,}} molecules and {{=n / 4:.1f}} in each quarter.")], tmp_path / "n.ipynb")
    assert done[1]["source"] == "There are 1,234 molecules and 308.5 in each quarter."


def test_the_contents_follow_the_headings_and_a_section_belongs_to_the_last_numbered_heading():
    cells = [md("# Part 0: Orientation"), md("## 0.1 The problem\nText."), code("x = 1"), md("## 0.2 The data")]
    assert [text for _, text, _ in headings(cells)] == ["Part 0: Orientation", "0.1 The problem", "0.2 The data"]
    assert section_of_cells(cells) == ["Part 0", "0.1", "0.1", "0.2"]
    assert "[0.2 The data](#0.2-The-data)" in contents_cell(cells)["source"]


def test_coverage_names_the_files_that_no_section_explains(tmp_path):
    (tmp_path / "cypmode").mkdir()
    (tmp_path / "cypmode/alpha.py").write_text("")
    (tmp_path / "cypmode/beta.py").write_text("")
    mapping, missing = coverage([md("## 1.1 Alpha\nThe module `alpha.py` does it.")], root=tmp_path)
    assert mapping["module"]["cypmode/alpha.py"] == ["1.1"] and missing == ["module: cypmode/beta.py"]


def test_every_numbered_section_must_name_its_field_under_its_heading():
    from notebooks.notebook_builder import field_problems, md

    ok = md("## 1.1 Drugs in the body\n\n> **Field:** Pharmacology · Mathematics (differential equations)\n\nText.")
    missing = md("## 1.2 Enzymes\n\nText without a field line.")
    wrong = md("## 1.3 Heme\n\n> **Field:** Alchemy\n\nText.")
    assert field_problems([ok]) == []
    assert "has no" in field_problems([missing])[0] and "Alchemy" in field_problems([wrong])[0]
    assert field_problems([md("## Abstract\n\nUnnumbered headings need no field.")]) == []


def test_a_symbol_of_an_equation_must_be_named_in_the_text_of_its_section():
    from notebooks.notebook_builder import md, symbol_problems

    md("## 1.1 Decay\n\nWrite $C$ for the concentration and $t$ for the time.\n\n$$C(t) = C_0 e^{-k t}$$\n\nwhere $k$ and $C_0$ are constants.")
    named_after = md("## 1.1 Decay\n\nThe concentration $C$ at the time $t$ falls with the rate constant $k$, starting from $C_0$.\n\n$$C(t) = C_0 e^{-k t}$$")
    unnamed = md("## 1.1 Decay\n\nThe concentration falls.\n\n$$C(t) = C_0 e^{-k t}$$")
    assert symbol_problems([named_after]) == []
    problems = symbol_problems([unnamed])
    assert problems and all(sym in problems[0] for sym in ("C", "k", "t")) and "e" not in problems[0].split("symbols")[1].split("of an equation")[0].replace(" ", "").split(",")


def test_a_symbol_named_in_one_section_does_not_count_for_the_next():
    from notebooks.notebook_builder import md, symbol_problems

    cells = [md("## 1.1 A\n\nWrite $Q$ for the flow."), md("## 1.2 B\n\n$$Q = 2$$")]
    assert "Q" in symbol_problems(cells)[0]


PLAN = "## Front matter\n\n## Part 0. Orientation\n\n## Part IV. Machine learning from zero\n\n## Part XIII. Closing\n\n## Appendices\n\n- A. Practical walkthrough\n- B. Symbols\n- C. References\n- D. Coverage map\n"


def test_the_plan_lists_every_part_and_appendix_and_the_cells_open_some_of_them():
    assert planned_parts(PLAN) == ["Part 0", "Part IV", "Part XIII", "Appendix A", "Appendix B", "Appendix C", "Appendix D"]
    cells = [md("# CYPMode"), md("# Part 0: Orientation"), md("# Part IV. Machine learning from zero"), md("## 4.1 A section")]
    assert unwritten_parts(cells, PLAN) == ["Part XIII", "Appendix A", "Appendix B", "Appendix C"]  # Appendix D is generated by the builder
    done = [*cells, md("# Part XIII. Closing")] + [md(f"# Appendix {x}. Title") for x in "ABC"]
    assert unwritten_parts(done, PLAN) == []

# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

from notebooks.notebook_builder import (
    HELD_BACK_NOTE,
    code,
    execute,
    figure,
    md,
    public_problems,
)

CELLS = [
    code("y = 41 + 1\nprint('OCNT-2395436 pose table')"),
    code("z = 7\nprint('OCNT-2395503 pose table')", held_back=True),
    md("The held back sentence says {{=z}}.", held_back=True),
    md("The open sentence says {{=z}}."),
    code("print('z is', z)"),
]


def test_a_held_back_cell_still_runs_but_shows_a_note_in_the_public_edition():
    public = execute(CELLS, public=True)
    outputs = [o["data"]["text/markdown"] for o in public[1]["outputs"]]
    assert outputs == [HELD_BACK_NOTE] and "OCNT" not in str(public[1]["outputs"])
    assert public[2]["source"] == HELD_BACK_NOTE
    assert public[3]["source"] == "The open sentence says 7."  # the variable of the held-back cell exists for the cells that follow
    assert "z is 7" in public[4]["outputs"][0]["text"]
    assert "OCNT-2395436" in str(public[0]["outputs"])  # a cell that is not marked is untouched


def test_the_complete_edition_keeps_every_output():
    full = execute(CELLS)
    assert "OCNT-2395503" in str(full[1]["outputs"]) and full[2]["source"] == "The held back sentence says 7."


def test_a_figure_can_be_marked_held_back():
    assert figure("struct_all20_gallery", held_back=True)["held_back"] is True and "held_back" not in figure("mol_tanimoto")


def test_the_scan_refuses_an_official_ligand_and_a_table_of_blinded_compounds_but_not_one_example():
    official = ["OCNT-2395436", "OCNT-2395503"]
    one = [{"cell_type": "markdown", "source": "an example is OCNT-0493927"}]
    assert public_problems(one, official) == []
    leaked = [{"cell_type": "code", "source": "x", "outputs": [{"output_type": "stream", "text": "OCNT-2395436 1.9"}]}]
    assert public_problems(leaked, official) == ["cell 0 names 1 official structure ligand(s)"]
    many = [{"cell_type": "markdown", "source": " ".join(f"OCNT-{n:07d}" for n in range(1000001, 1000006))}]
    assert "names 5 compounds of the blinded sets" in public_problems(many, official)[0]


def test_the_per_ligand_table_of_the_relaxation_is_held_back_in_the_notebook():
    from notebooks.parts import part11h

    marked = [c for c in part11h.CELLS if c.get("held_back")]
    assert len(marked) == 1 and "decision_table" in marked[0]["source"] and "show_table" in marked[0]["source"]

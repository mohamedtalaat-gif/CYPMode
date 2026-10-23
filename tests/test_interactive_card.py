# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "docs/card/build_interactive_card.py"
pytest.importorskip("gemmi")
pytest.importorskip("rdkit")
if not (ROOT / "data/structure_track/inputs/pdb_cache/6MA7.cif").exists():
    pytest.skip("the cached public structures are not in this checkout", allow_module_level=True)

spec = importlib.util.spec_from_file_location("build_interactive_card", BUILDER)
card = importlib.util.module_from_spec(spec)
spec.loader.exec_module(card)


def test_the_iron_nitrogen_distance_measured_on_the_page_coordinates_matches_the_deposited_one():
    import pandas as pd

    crystals = pd.read_csv(card.card.EXPANSION).set_index("comp_id")
    for code, pdb_id in card.card.PUBLIC_STRIP:
        entry = card.structure_entry(card.PDB_CACHE / f"{pdb_id}.cif", code)
        assert abs(entry["distance"] - float(crystals.loc[code, "min_fe_distance"])) < 0.06, (code, pdb_id, entry["distance"])
        assert "LIG" in entry["pdb"] and "HEM" in entry["pdb"] and entry["near"]


def test_the_structure_of_a_page_holds_one_chain_with_protein_heme_and_ligand_only():
    entry = card.structure_entry(card.PDB_CACHE / f"{card.card.PUBLIC_HERO[0]}.cif", card.card.PUBLIC_HERO[1])
    residues = {line[17:20].strip() for line in entry["pdb"].splitlines()}
    assert "HOH" not in residues and "DMS" not in residues and {"HEM", "LIG"} <= residues
    assert len({line[21] for line in entry["pdb"].splitlines()}) == 1


def test_a_sketch_is_an_svg_without_a_background_or_a_fixed_size():
    svg = card.sketch("Cc1ncc2n1-c3ccc(cc3C(=NC2)c4ccccc4F)Cl", "dark", None)
    assert svg.startswith("<svg") and "width='" not in svg.split(">", 1)[0] and "<?xml" not in svg


def test_the_page_makes_no_request_and_leaves_no_placeholder(monkeypatch):
    entry = {"id": "AAAA", "code": "X", "title": "x (PDB AAAA)", "caption": "X", "sketch": {"dark": "<svg></svg>", "light": "<svg></svg>"}, "pdb": "ATOM", "fe": [0, 0, 0], "donor": [1, 1, 1], "distance": 2.0, "near": [1]}
    monkeypatch.setattr(card, "entries", lambda mode="public": [entry, entry])
    monkeypatch.setattr(card.facts_module, "load", lambda: SimpleNamespace(auc={"CYP3A4": 0.8, "CYP2D6": 0.7}, poses=20, pose_failures=0, gates=10, gates_not_met=9))
    text = card.page()
    assert not re.search(r"__[A-Z]+__", text.replace("__webpack", ""))  # no placeholder of the template is left (the library has its own double underscores)
    own = text.split("<script>")[2]  # the script of the page, after the library
    assert "fetch(" not in own and "XMLHttpRequest" not in own and not re.search(r"""(src|href)=["']https?:""", text.split("<script>")[0])
    assert text == card.page()  # deterministic

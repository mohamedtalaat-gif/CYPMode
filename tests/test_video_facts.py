# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import json
import sys
from pathlib import Path

import pandas as pd

sys.path[:0] = ["docs/video", "."]

import facts as facts_module


def write_final(folder: Path, chosen: str) -> None:
    table = pd.DataFrame({"stack": ["default", "default+all_survivors"], "n_candidates": [23, 28], "auc_z1.0": [0.817, 0.834], "mcc_z1.0": [0.44, 0.476]})
    table.to_csv(folder / "CYP3A4_stack_choice.csv", index=False)
    (folder / "manifest.json").write_text(json.dumps({"isoforms": {"CYP3A4": {"stack": chosen}}}))


def test_the_results_of_the_film_are_those_of_the_stack_that_is_submitted_and_the_committee_is_the_default_one(tmp_path, monkeypatch):
    monkeypatch.setattr(facts_module, "FINAL_TDI", tmp_path)
    write_final(tmp_path, "default+all_survivors")
    assert facts_module.stack_row("CYP3A4", final=True)["n_candidates"] == 28 and facts_module.stack_row("CYP3A4", final=True)["auc_z1.0"] == 0.834
    assert facts_module.stack_row("CYP3A4")["n_candidates"] == 23  # the Stacker scene describes the default committee


def test_a_final_stack_that_is_the_default_reads_the_default_row(tmp_path, monkeypatch):
    monkeypatch.setattr(facts_module, "FINAL_TDI", tmp_path)
    write_final(tmp_path, "default")
    assert facts_module.stack_row("CYP3A4", final=True)["mcc_z1.0"] == 0.44

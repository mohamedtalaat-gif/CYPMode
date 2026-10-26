# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The architecture and exploration sections of the method report: present, numbered, and built from the ledger and the manifests."""

import re

import pandas as pd
import pytest

from cypmode.audit import method_report as report
from cypmode.audit.ledger_kinds import KINDS, TRIED, ledger_by_kind

needs_outputs = pytest.mark.skipif(not (report.FINAL_TDI / "manifest.json").exists(), reason="the final build has not been run on this machine")


def test_every_kind_of_idea_has_a_line_on_what_it_tried():
    assert set(KINDS) == set(TRIED)


def test_a_family_belongs_to_one_kind_only():
    stems = [stem for stems in KINDS.values() for stem in stems]
    assert len(stems) == len(set(stems))


@needs_outputs
def test_the_report_holds_the_architecture_and_the_exploration_before_the_results():
    text = report.build_method_report()
    order = [text.index(h) for h in ("## 2. Data and the label rule", "## 3. Architecture of the two tracks", "## 4. Exploration: what was tried and with what result", "## 5. Scoring-set-aware yardstick")]
    assert order == sorted(order)
    assert "### 3.1 TDI track" in text and "### 3.2 Structure track" in text and "### 4.2 The accounting of the number of attempts" in text
    named = set(pd.read_csv("data/structure_track/inputs/structure_track_test_blinded.csv")["Molecule_Name"]) & set(re.findall(r"OCNT-\d+", text))
    assert not named, f"the report names ligands of the structure set: {sorted(named)}"  # the recipe is summarised without an entry per ligand


@needs_outputs
def test_the_summary_holds_the_hashes_and_the_disclosures_before_section_one():
    text = report.build_method_report()
    assert text.index("## Summary and disclosures") < text.index("## 1. Scope and what is not claimed")
    manifest = report.FINAL_TDI / "manifest.json"
    import json

    for entry in json.loads(manifest.read_text())["files"]:
        assert entry["sha256"] in text
    assert "**Proprietary data.** None" in text and "**Open source.**" in text


@needs_outputs
def test_the_map_counts_every_verdict_of_the_ledger():
    frame = ledger_by_kind()
    section = report.exploration_section()
    assert f"In all {len(frame)} pre-registered verdicts in {frame['stem'].nunique()} families" in section
    assert "other" not in set(frame["kind"]), "a ledger family that no kind claims: add it to cypmode/audit/ledger_kinds.py"


def test_the_ledger_carries_the_registered_caveat_when_the_ensemble_misses_the_bound_state(tmp_path):
    log = tmp_path / "conformer_bias.log"
    log.write_text("the ensemble of the block (12 conformers within 5 kcal/mol): a member within 1.0 A of the crystal pose for 12% of the 131 copies; median nearest RMSD 2.45 A\n")
    caveat = report.ensemble_caveat(log)
    assert "12% of 131 crystal ligand copies" in caveat and "88% of the crystal ligands" in caveat and "aux_site_*" in caveat


def test_the_ledger_has_no_caveat_when_the_ensemble_covers_the_bound_state_or_the_check_is_missing(tmp_path):
    covered = tmp_path / "conformer_bias.log"
    covered.write_text("the ensemble of the block (12 conformers within 5 kcal/mol): a member within 1.0 A of the crystal pose for 83% of the 131 copies; median nearest RMSD 0.60 A\n")
    assert report.ensemble_caveat(covered) == "" and report.ensemble_caveat(tmp_path / "missing.log") == ""


def test_the_scores_of_the_uploads_are_printed_only_when_the_file_of_them_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "LEADERBOARD", tmp_path / "leaderboard_scores.csv")
    assert report.leaderboard_scores() == ""
    (tmp_path / "leaderboard_scores.csv").write_text("slot,file,uploaded_utc,cyp2d6_mcc\n1,candidate_A,2026-10-28 10:00,0.31\n")
    text = report.leaderboard_scores()
    assert "candidate_A" in text and "0.31" in text and "| slot | file |" in text


@needs_outputs
def test_the_report_says_what_each_group_that_joined_a_stack_is_and_what_it_added():
    import json

    manifest = json.loads((report.FINAL_TDI / "manifest.json").read_text())
    text = report.architecture_section()
    for iso, row in manifest["isoforms"].items():
        for group in row["surviving_groups"]:
            assert f"`{group}`" in text and report.GROUP_TEXT[group] in text, f"{iso}: the group {group} joined the stack and the report does not say what it is"
            assert "raises the AUC by" in text
    assert "Holm adjustment" in text


@needs_outputs
def test_the_ledger_of_negative_results_reads_idea_by_idea_with_the_numbers():
    text = report.negative_results()
    assert text.index("## Ideas that were added to the TDI stack and not admitted") < text.index("## Every gate that was not met")
    table = text.split("## Ideas that were added to the TDI stack and not admitted")[1].split("## Structure-track levers")[0]
    rows = [line for line in table.splitlines() if line.startswith("| ") and "---" not in line][1:]
    assert len(rows) > 40 and all(re.search(r"[+-]\d\.\d{4} \(p = \d\.\d{3}\)", row) for row in rows)
    admitted = {group for row in report.json.loads((report.FINAL_TDI / "manifest.json").read_text())["isoforms"].values() for group in row["surviving_groups"]}
    assert not any(f"`{group}`" in table for group in admitted)


@needs_outputs
def test_every_document_quotes_the_one_count_of_what_was_tried():
    from cypmode.audit import docs_state
    from cypmode.audit.ledger_kinds import totals

    verdicts, families, met = totals()
    assert f"{verdicts} pre-registered verdicts from {families} families of ideas: {met} MET" in docs_state.gate_verdicts_block()
    assert f"In all {verdicts} pre-registered verdicts in {families} families, {met} of them MET" in report.exploration_section()
    from docs.video import facts

    assert facts.gate_counts() == (verdicts, verdicts - met)


@needs_outputs
def test_the_innovation_summary_stands_between_the_disclosures_and_section_one_and_leaves_no_gap():
    text = report.build_method_report()
    assert text.index("## Summary and disclosures") < text.index("## Innovation summary") < text.index("## 1. Scope and what is not claimed")
    section = report.innovation_section()
    assert "(pending" not in section and "?" not in re.sub(r"`[^`]*`", "", section), "a claim of the innovation summary has no source"
    assert "### What is not claimed" in section
    named = set(pd.read_csv("data/structure_track/inputs/structure_track_test_blinded.csv")["Molecule_Name"]) & set(re.findall(r"OCNT-\d+", section))
    assert not named


@needs_outputs
def test_the_innovation_table_has_one_row_per_claim_with_a_source():
    rows = [line for line in report.innovation_table().splitlines()[2:]]
    assert len(rows) >= 7 and all(line.count("|") == 4 for line in rows)

# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The architecture and exploration sections of the method report: present, numbered, and built from the ledger and the manifests."""

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

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
    order = [text.index(h) for h in ("## 2. Data and the label rule", "## 3. Architecture of the two tracks", "## 4. Exploration: what was tried, and how honestly", "## 5. Scoring-set-aware yardstick")]
    assert order == sorted(order)
    assert "### 3.1 TDI track" in text and "### 3.2 Structure track" in text and "### 4.2 The accounting of the number of attempts" in text


@needs_outputs
def test_the_map_counts_every_verdict_of_the_ledger():
    frame = ledger_by_kind()
    section = report.exploration_section()
    assert f"In all {len(frame)} pre-registered verdicts in {frame['stem'].nunique()} families" in section
    assert "other" not in set(frame["kind"]), "a ledger family that no kind claims: add it to cypmode/audit/ledger_kinds.py"

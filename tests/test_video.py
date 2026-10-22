# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import json
import sys
from itertools import pairwise
from pathlib import Path

import pytest

VIDEO = Path("docs/video")
sys.path[:0] = [str(VIDEO), str(VIDEO / "anime"), "."]

import facts as facts_module
import render_video
from chrome import CHROME, Chrome

FACTS = facts_module.Facts(n_train=6145, n_test=750, n_columns=12242, n_models=23, gates=78, gates_not_met=76, auc={"CYP3A4": 0.817, "CYP2D6": 0.731}, mcc={"CYP3A4": 0.44, "CYP2D6": 0.17},
                           poses=20, pose_failures=0, audit_reproducible=True, zip_hash="05a0cfd1")
CHECKS = ["mol pred loaded", "bond lengths", "internal steric clash", "minimum distance to protein"]
STUB = {  # a small stand-in for the real data of facts.real_data(), so that the film draws in a clean checkout
    "molecule": {"name": "OCNT-0000001", "smiles": "CCO", "atoms": [[-0.4, 0.0, "C"], [0.0, 0.2, "C"], [0.4, 0.0, "O"]], "bonds": [[0, 1, 1], [1, 2, 1]], "bits": [3, 77, 500, 2000],
                 "descriptors": {"molecular weight": 46.1, "logP": -0.1, "polar surface area": 20.2, "H-bond donors": 1, "H-bond acceptors": 1, "rotatable bonds": 0, "heavy atoms": 3}, "direct": 5.5, "tdi": 5.6},
    "scatter": {"points": [[4.0, 4.1, 0], [5.0, 5.9, 1], [6.0, 6.1, 0], [5.5, 6.4, 1]], "n": 2334, "positives": 764},
    "curves": {iso: {"fpr": [0, 0.5, 1], "tpr": [0, 0.8, 1], "auc": 0.8} for iso in ("CYP3A4", "CYP2D6")},
    "gates": [{"label": "an idea", "met": 0, "not_met": 4}, {"label": "a physics idea", "met": 0, "not_met": 2}, {"label": "a structure lever", "met": 2, "not_met": 3}],
    "physics": [{"column": "sasa total", "auc": 0.626}, {"column": "ip ev", "auc": 0.365}],
    "structure": {"checks": CHECKS, "passed": [20] * len(CHECKS), "n": 20, "donors": ["OCNT-0000002"]},
    "blocks": [{"name": "morgan", "columns": 2048}, {"name": "maccs", "columns": 167}], "families": {"lgbm": 15, "knn": 3, "svm": 3, "extratrees": 1, "rf": 1},
    "repo": {"modules": 119, "test_files": 90, "test_functions": 641, "jobs": 63, "audit_hash": "abcdef123456"},
    "scout": {"sources": [{"label": "a release", "rows": 1340, "compounds": 1340, "in_training": 1250, "in_test": 1}, {"label": "a screen", "rows": 17504, "compounds": 4376, "in_training": 4375, "in_test": 0},
                          {"label": "another file", "rows": 6145, "compounds": 6145, "in_training": 6145, "in_test": 4}], "excluded": ["OCNT-0000001", "OCNT-0000002", "OCNT-0000003", "OCNT-0000004", "OCNT-0000005"]},
    "foundry": [{"label": "a pretrained model", "enzymes": {"CYP3A4": {"delta_auc": 0.0001, "p": 0.945, "verdict": "NOT MET"}, "CYP2D6": {"delta_auc": -0.0008, "p": 0.724, "verdict": "NOT MET"}}},
                {"label": "a model still running", "enzymes": {"CYP3A4": {"verdict": "pending"}, "CYP2D6": {"verdict": "pending"}}},
                {"label": "a model screened on five folds", "enzymes": {"CYP3A4": {"delta_auc": 0.0031, "p": 0.31, "verdict": "screened"}, "CYP2D6": {"delta_auc": -0.002, "p": 0.6, "verdict": "screened"}}},
                {"label": "a model that was not run", "enzymes": {"CYP3A4": {"verdict": "not run"}, "CYP2D6": {"verdict": "not run"}}}],
}
ASSET_STUB = {name: {"file": f"assets/{name}.png", "size": [1800, 1100], "anchors": {"fe": [880, 600], "ligand": [900, 500], "coordinating_n": [870, 540]}}
              for name in ("enzyme", "pocket_type_ii", "pocket_type_i", "cap_pose", "example_restrained", "example_unguided", "example_compare")}
ASSET_STUB["distances"] = {"ligand": "OCNT-0000002", "restrained": 2.4, "unguided": 5.7, "own": 2.0, "own_ligand": "OCNT-0000001"}
needs_chrome = pytest.mark.skipif(not Path(CHROME).exists(), reason="Google Chrome is not installed")

OVERFLOW = """
(() => {
  const eff = (el) => { let o = 1; for (let n = el; n && n.nodeType === 1; n = n.parentNode) { const a = n.getAttribute && n.getAttribute('opacity'); if (a) o *= parseFloat(a); } return o; };
  const boxes = [...document.querySelectorAll('#stage [data-box]')].map((r) => ({ b: r.getBoundingClientRect(), o: eff(r) })).filter((r) => r.o > 0.5);
  const out = [];
  for (const t of document.querySelectorAll('#stage text')) {
    if (!t.textContent.trim() || eff(t) < 0.5) continue;
    const b = t.getBoundingClientRect(), cx = (b.left + b.right) / 2, cy = (b.top + b.bottom) / 2;
    if (b.left < -2 || b.right > 1922 || b.top < -2 || b.bottom > 1082) out.push('outside the frame: ' + t.textContent.slice(0, 40));
    for (const r of boxes) {
      if (cx < r.b.left || cx > r.b.right || cy < r.b.top || cy > r.b.bottom) continue;
      if (Math.max(r.b.left - b.left, b.right - r.b.right, r.b.top - b.top, b.bottom - r.b.bottom) > 3) out.push('leaves its box: ' + t.textContent.slice(0, 40));
    }
  }
  return out;
})()
"""


def test_the_facts_reach_the_page_as_valid_javascript():
    text = facts_module.to_javascript(FACTS, STUB)
    assert text.startswith("window.FACTS = ") and "\nwindow.REAL = " in text and text.rstrip().endswith(";")
    assert json.loads(text.removeprefix("window.FACTS = ").split(";\nwindow.REAL")[0])["n_models"] == 23


def test_the_gates_on_screen_do_not_include_the_exploratory_rescorings(monkeypatch):
    rows = [("a.log", "mbr_selection", "NOT MET"), ("a.log", "restraint_bond", "MET"), ("b.log", "mbr_lddt_exploratory", "NOT MET")]
    monkeypatch.setattr(facts_module, "gate_rows", lambda: rows)
    assert facts_module.gate_counts() == (2, 1)


def test_the_render_refuses_to_say_the_audit_reproduced_when_it_did_not(monkeypatch):
    monkeypatch.setattr(render_video.facts_module, "load", lambda: facts_module.Facts(**{**FACTS.__dict__, "audit_reproducible": False}))
    monkeypatch.setattr(sys, "argv", ["render_video.py"])
    assert render_video.main() == 1


@pytest.fixture(scope="module")
def page():
    with Chrome(port=9411) as chrome:
        chrome.open(VIDEO / "anime" / "index.html")
        chrome.evaluate(facts_module.to_javascript(FACTS, STUB) + f"window.ASSETS = {json.dumps(ASSET_STUB)};")  # the numbers of the test, whatever the real facts.js holds
        yield chrome


@needs_chrome
def test_the_scenes_cover_the_film_without_gaps(page):
    scenes = page.evaluate("SCENES")
    assert scenes[0]["start"] == 0 and all(a["end"] == b["start"] for a, b in pairwise(scenes))
    assert page.evaluate("DURATION") == scenes[-1]["end"] > 200


@needs_chrome
def test_every_second_of_the_film_draws_without_an_error(page):
    for second in range(int(page.evaluate("DURATION"))):
        page.evaluate(f"render({second})")
    assert page.evaluate("document.getElementById('stage').children.length") > 1


@needs_chrome
def test_a_frame_is_a_full_hd_picture(page, tmp_path):
    page.evaluate("render(20)")
    page.screenshot(tmp_path / "frame.png")
    data = (tmp_path / "frame.png").read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and int.from_bytes(data[16:20], "big") == 1920 and int.from_bytes(data[20:24], "big") == 1080


@needs_chrome
def test_no_text_leaves_its_box_or_the_frame(page):
    scenes = page.evaluate("SCENES")
    starts = [s["start"] for s in scenes] + [next(s["start"] for s in scenes if s["id"] == "gate") + page.evaluate("CARD") + 16]  # the second card of the gate scene is the Quanta card
    problems = []
    for t in range(2, int(page.evaluate("DURATION")), 2):
        if any(start <= t < start + 3 for start in starts[1:]):  # the first seconds of a scene are the slide-in of its card
            continue
        page.evaluate(f"render({t})")
        problems += [f"t={t}: {p}" for p in page.evaluate(OVERFLOW)]
    assert not problems, problems[:10]


LEDGER = """
(() => {
  const title = [...document.querySelectorAll('#stage text')].find((n) => n.textContent.includes('THE LEDGER'));
  if (!title) return ['the ledger title is not on the page'];
  const tb = title.getBoundingClientRect(), cx = (tb.left + tb.right) / 2, cy = (tb.top + tb.bottom) / 2;
  const panel = [...document.querySelectorAll('#stage [data-box]')].map((r) => r.getBoundingClientRect()).filter((b) => cx > b.left && cx < b.right && cy > b.top && cy < b.bottom).sort((a, b) => (b.width * b.height) - (a.width * a.height))[0];
  if (!panel) return ['the ledger panel is not on the page'];
  const out = [];
  for (const n of document.querySelectorAll('#stage text')) {
    if (!n.textContent.trim()) continue;
    const b = n.getBoundingClientRect(), mid = (b.top + b.bottom) / 2;
    if (mid < panel.top || mid > panel.bottom) continue;
    if (b.left < panel.left - 2 || b.right > panel.right + 2) out.push('outside the ledger panel: ' + n.textContent.slice(0, 40));
  }
  return out;
})()
"""


@needs_chrome
@pytest.mark.parametrize("ideas", [26, 34])
def test_the_ledger_fits_its_panel_whatever_the_number_of_ideas(page, ideas):
    """The real ledger holds one row per idea (26 when this test was written, more as ideas are added); the layout must keep every row inside its panel."""
    rows = [{"label": f"an idea with a descriptive name {i}", "met": 0, "not_met": 1 + i % 3} for i in range(ideas)]
    page.evaluate(f"REAL.gates = {json.dumps(rows)}")
    gate = next(s for s in page.evaluate("SCENES") if s["id"] == "gate")
    page.evaluate(f"render({gate['start'] + page.evaluate('CARD') + 13.5})")
    assert page.evaluate(LEDGER) == []


@needs_chrome
def test_the_numbers_on_screen_come_from_the_facts(page):
    stacker = page.evaluate("SCENES.find((s) => s.id === 'stacker').start + CARD + 12")
    page.evaluate(f"render({stacker})")
    assert "23 MODELS" in page.evaluate("document.getElementById('stage').innerHTML")
    page.evaluate(f"FACTS.n_models = 99; render({stacker})")
    assert "99 MODELS" in page.evaluate("document.getElementById('stage').innerHTML")
    page.evaluate("FACTS.n_models = 23")
    page.evaluate("render(DURATION - 22)")
    html = page.evaluate("document.getElementById('stage').innerHTML")
    assert "0.817" in html and "0.731" in html


@needs_chrome
def test_every_member_of_the_cast_has_a_card_that_says_what_they_do(page):
    scenes = {s["id"]: s["start"] for s in page.evaluate("SCENES")}
    card = page.evaluate("CARD")
    for scene, offset, role in (("kai", 5, "ASSAY ANALYST"), ("scout", 5, "DATA SCOUT"), ("bits", 5, "CHEMOINFORMATICIAN"), ("stacker", 5, "MODELLER"), ("gate", 5, "STATISTICIAN"), ("gate", card + 16 + 5, "QUANTUM CHEMIST"),
                                ("foundry", 5, "FOUNDATION-MODEL ENGINEER"), ("dock", 5, "STRUCTURAL BIOLOGIST"), ("auditor", 5, "REPRODUCIBILITY OFFICER")):
        page.evaluate(f"render({scenes[scene] + offset})")
        assert role in page.evaluate("document.getElementById('stage').innerHTML")


TABLE = """
((title) => {
  const head = [...document.querySelectorAll('#stage text')].find((n) => n.textContent.includes(title));
  if (!head) return ['the title is not on the page: ' + title];
  const tb = head.getBoundingClientRect(), cx = (tb.left + tb.right) / 2, cy = (tb.top + tb.bottom) / 2;
  const panel = [...document.querySelectorAll('#stage [data-box]')].map((r) => r.getBoundingClientRect()).filter((b) => cx > b.left && cx < b.right && cy > b.top && cy < b.bottom).sort((a, b) => (b.width * b.height) - (a.width * a.height))[0];
  if (!panel) return ['the panel is not on the page: ' + title];
  const out = [];
  for (const n of document.querySelectorAll('#stage text')) {
    if (!n.textContent.trim()) continue;
    const b = n.getBoundingClientRect(), mid = (b.top + b.bottom) / 2;
    if (mid < panel.top || mid > panel.bottom) continue;
    if (b.left < panel.left - 2 || b.right > panel.right + 2) out.push('outside the panel: ' + n.textContent.slice(0, 40));
  }
  return out;
})
"""


@needs_chrome
@pytest.mark.parametrize("models", [9, 14])
def test_the_foundry_table_fits_its_panel_whatever_the_number_of_models(page, models):
    """The table holds one row per pretrained model (nine when it was written, more as models are added); the layout must keep every row inside its panel, judged or still running."""
    rows = [{"label": f"a pretrained model with a long descriptive name {i}", "enzymes": {iso: ({"delta_auc": -0.0123, "p": 0.456, "verdict": "NOT MET"} if i % 3 == 1 else {"verdict": "not run"} if i % 3 == 2 else {"verdict": "pending"}) for iso in ("CYP3A4", "CYP2D6")}} for i in range(models)]
    page.evaluate(f"REAL.foundry = {json.dumps(rows)}")
    foundry = next(s for s in page.evaluate("SCENES") if s["id"] == "foundry")
    page.evaluate(f"render({foundry['start'] + page.evaluate('CARD') + 14})")
    assert page.evaluate(f"({TABLE})('PRETRAINED MODELS ADDED TO THE STACK')") == []


@needs_chrome
def test_the_scout_scene_reads_the_files_it_checked_from_the_facts(page):
    scout = next(s for s in page.evaluate("SCENES") if s["id"] == "scout")
    page.evaluate(f"render({scout['start'] + page.evaluate('CARD') + 12})")
    html = page.evaluate("document.getElementById('stage').innerHTML")
    assert "a release" in html and "1,340" in html and "5 compounds of the blinded test set taken out" in html
    assert page.evaluate(f"({TABLE})('FILES RELEASED BY THE ORGANISERS, CHECKED BY NAME')") == []
    page.evaluate("REAL.scout.excluded = ['OCNT-0000001']")
    page.evaluate(f"render({scout['start'] + page.evaluate('CARD') + 12})")
    assert "1 compound of the blinded test set taken out" in page.evaluate("document.getElementById('stage').innerHTML")


@needs_chrome
def test_the_poster_shows_the_title_the_team_and_real_numbers_inside_the_frame(page):
    page.evaluate("poster()")
    html = page.evaluate("document.getElementById('stage').innerHTML")
    assert "CYPMode" in html and "0.817" in html and "0.731" in html and "20 poses" in html
    assert page.evaluate(OVERFLOW) == []

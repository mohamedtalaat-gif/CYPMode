# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import html
import json
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
MARIMO = HERE.parents[1] / ".venv-marimo/bin/marimo"

pytestmark = pytest.mark.skipif(not MARIMO.exists(), reason="the explorer environment .venv-marimo is not on this machine")


def test_every_live_section_runs_and_the_page_has_no_error(tmp_path):
    page = tmp_path / "index.html"
    run = subprocess.run([str(MARIMO), "export", "html", str(HERE / "app.py"), "-o", str(page), "--no-include-code"], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr[-600:]
    text = page.read_text(encoding="utf-8")
    for needle in ("Traceback", "MarimoExceptionRaisedError", "NameError", "KeyError", "ModuleNotFoundError"):
        assert needle not in text, needle
    live = len(__import__("re").findall(r"\bLIVE\b", html.unescape(text)))
    assert live >= 129  # every section of the notebook has a simulator


def test_the_sections_follow_the_notebook_and_every_one_has_exactly_one_simulator():
    import importlib
    import sys

    sys.path[:0] = [str(HERE), str(HERE.parents[1])]
    parts = json.loads((HERE / "data/sections.json").read_text())["parts"]
    numbers = {s["number"] for p in parts for s in p["sections"]}
    source = (HERE / "app.py").read_text()
    in_app = set(__import__("re").findall(r'"(\d+\.\d+)": sim_', source))
    registered = [set(importlib.import_module(f"simulators.{path.stem}").SIMS) for path in sorted((HERE / "simulators").glob("part_*.py"))]
    in_parts = set().union(*registered)
    assert len(numbers) == 129 and sum(len(r) for r in registered) == len(in_parts)  # no section is registered twice among the parts
    assert not (in_app & in_parts) and in_app | in_parts == numbers

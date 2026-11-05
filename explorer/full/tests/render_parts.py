# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Run by tests/test_parts.py with the interpreter of the explorer environment (the one that has marimo): every simulator of every part is built with its default controls and its page is made.
Prints one JSON object, section number -> "ok" or the error."""

import importlib
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

import marimo as mo
from simulators import framework

result = {}
for path in sorted((HERE / "simulators").glob("part_*.py")):
    module = importlib.import_module(f"simulators.{path.stem}")
    ctl = framework.controls(mo, module.SIMS)
    framework.views(mo, ctl, module.SIMS)
    for number, entry in module.SIMS.items():
        view = entry[1]
        embed = entry[2] if len(entry) > 2 else None
        try:
            page = view(mo, ctl.value[number])
            assert hasattr(page, "text") or hasattr(page, "_mime_"), type(page)
            if embed:
                assert "iframe" in embed(mo, ctl.value[number]).text
            result[number] = "ok"
        except Exception:
            result[number] = traceback.format_exc(limit=3)[-700:]
print(json.dumps(result))

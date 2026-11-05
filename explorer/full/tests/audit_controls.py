# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Run by tests/test_parts.py with the interpreter of the explorer environment: for every control of every simulator, the page is made with the default values and again with that one control set to another value, and the two pages are compared.
Prints one JSON object, "section/control" -> "changes the page" or why it does not. A control that changes nothing is a control that does nothing."""

import importlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

import marimo as mo
from simulators import framework


def other(element, value):
    """A value of the element that differs from ``value``, or None when the element cannot be varied here (a button, a file)."""
    kind = type(element).__name__
    if kind == "slider":
        return element.start if value != element.start else element.stop
    if kind == "number":
        return value + 1 if value is not None and (element.stop is None or value + 1 <= element.stop) else (value - 1 if value is not None else None)
    if kind == "checkbox":
        return not value
    if kind in ("radio", "dropdown"):
        options = list(element.options.keys())
        return next((o for o in options if o != value), None)
    if kind == "multiselect":
        options = list(element.options.keys())
        return [] if value else options
    if kind == "text":
        return "CCO" if value != "CCO" else "c1ccccc1"
    return None


def text_of(page):
    return page.text if hasattr(page, "text") else str(page)


result = {}
for path in sorted((HERE / "simulators").glob("part_*.py")):
    module = importlib.import_module(f"simulators.{path.stem}")
    ctl = framework.controls(mo, module.SIMS)
    for number, entry in module.SIMS.items():
        view = entry[1]
        embed = entry[2] if len(entry) > 2 else None
        elements = ctl[number].elements
        base = dict(ctl.value[number])
        for name, element in elements.items():
            if type(element).__name__ == "run_button":
                base[name] = True  # the page behind a button is the one that the audit looks at
        def page_of(values, view=view, embed=embed):
            return text_of(view(mo, values)) + (text_of(embed(mo, values)) if embed else "")

        reference = page_of(base)
        for name, element in elements.items():
            varied = other(element, base[name])
            key = f"{number}/{name}"
            if varied is None:
                result[key] = "not varied here"
                continue
            try:
                changed = page_of({**base, name: varied}) != reference
            except Exception as error:  # a value that the view cannot take is reported, not hidden
                result[key] = f"error: {type(error).__name__}: {str(error)[:120]}"
                continue
            result[key] = "changes the page" if changed else "CHANGES NOTHING"
print(json.dumps(result))

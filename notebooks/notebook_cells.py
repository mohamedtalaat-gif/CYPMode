# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""All the cells of the notebook in reading order, and the terms that must be defined where they are first used.

The content lives in ``notebooks/parts/``: ``front.py`` (title, abstract, how to read, contents), ``part00.py`` to ``part13.py`` (the parts) and the appendices. A part that does not exist
yet is simply not in the notebook; ``python -m notebooks.build_notebook --draft`` builds what exists, and the full build requires every file of the plan.
"""

import importlib

from notebooks.parts.glossary import GLOSSARY

MODULES = ["front", "part00", "part01", "part01b", "part02", "part02b", "part03", "part03b", "part04", "part04b", "part04c", "part05", "part05b", "part05c", "part05d", "part06", "part06b", "part06c", "part07", "part07b", "part07c", "part07d", "part07e", "part08", "part08b", "part09", "part09b", "part09c", "part09d", "part09e", "part10", "part10b", "part10c", "part11", "part11b", "part11c", "part11d", "part11e", "part11f", "part11g", "part11h", "part11i", "part11j", "part12", "part12b", "part13", "appendix_a", "appendix_b", "appendix_c"]  # the parts written so far, in reading order; the plan in notebooks/PLAN.md lists the rest


def collect() -> list[dict]:
    cells: list[dict] = []
    for name in MODULES:
        cells += importlib.import_module(f"notebooks.parts.{name}").CELLS
    return cells


ALL_CELLS = collect()
TERMS = list(GLOSSARY)

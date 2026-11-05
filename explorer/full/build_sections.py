# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The table of contents of the notebook as data: ``python explorer/full/build_sections.py`` writes ``explorer/full/data/sections.json``.

The complete explorer follows the notebook part by part and section by section, so its structure is read from the notebook itself (the level-1 headings are the parts, the level-2 headings with a
number are the sections, the line ``> **Field:**`` under a heading names the field) and cannot drift from it. It runs in the main environment, because it imports the cells of the notebook.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from notebooks.notebook_builder import headings, slug
from notebooks.notebook_cells import ALL_CELLS

FIELD = re.compile(r"^>\s*\*\*Field:\*\*\s*(.+)$", re.MULTILINE)
NUMBERED = re.compile(r"^(\d+\.\d+)\s+(.*)$")
PART = re.compile(r"^(Part (?:0|[IVX]+)|Appendix [A-D])\b[.:]?\s*(.*)$")


def build() -> dict:
    parts: list[dict] = []
    for level, text, index in headings(ALL_CELLS):
        if level == 1 and (match := PART.match(text)):
            parts.append({"label": match.group(1), "title": match.group(2).strip(" ."), "heading": text, "anchor": slug(text), "sections": []})
        elif level == 2 and parts and (match := NUMBERED.match(text)):
            field = FIELD.search(ALL_CELLS[index]["source"])
            parts[-1]["sections"].append({"number": match.group(1), "title": match.group(2), "heading": text, "anchor": slug(text), "field": field.group(1).strip() if field else ""})
    return {"parts": parts}


if __name__ == "__main__":
    data = build()
    out = Path(__file__).parent / "data" / "sections.json"
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    print(f"{len(data['parts'])} parts, {sum(len(p['sections']) for p in data['parts'])} sections -> {out.relative_to(ROOT)}")

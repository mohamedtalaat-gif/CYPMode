# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Build, execute and check the notebook without a Jupyter kernel.

``build(cells, path)`` runs every code cell in one shared namespace (standard output, matplotlib figures, tables and project figures are captured as cell outputs) and
writes a valid nbformat-4 notebook, so every number in the notebook is produced by the repository's code and files at build time.

The same module holds the checks the notebook must pass: ``glossary_problems`` (a term is defined, in bold, in the first cell that uses it), ``coverage`` (every module,
script, queue job and test file of the repository is named somewhere in the text) and the generated cells (table of contents, coverage map).

Usage: from notebooks.notebook_builder import md, code, build, glossary_problems, coverage
"""

import base64
import contextlib
import hashlib
import io
import json
import os
import re
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures"
FIGURE_WIDTH = 1500  # pixels; the stored figures are larger than a page needs


def md(text: str, **flags) -> dict:
    """A Markdown cell. ``{{=expression}}`` (or ``{{=expression:format}}``) is replaced, when the notebook runs, by the value of the expression in the notebook's namespace,
    so a number in the text always comes from the repository's files and never from the author's hand."""
    return {"cell_type": "markdown", "source": text.strip("\n"), **flags}


def code(text: str, **flags) -> dict:
    return {"cell_type": "code", "source": text.strip("\n"), **flags}


def figure(figure_id: str, **flags) -> dict:
    """A cell that shows a catalogued figure (dark variant) with its caption; the line of code that does so is not shown. ``held_back=True`` marks a figure that shows results for the blinded compounds."""
    return code(f'show_figure("{figure_id}")', hide_input=True, **flags)


HELD_BACK_NOTE = "*This output shows results for the blinded compounds of the challenge. It appears in the complete edition of the notebook, which is published after the challenge closes on 3 November 2026.*"
PUBLIC_BANNER = ("> **Public edition.** Everything of the notebook is here except the outputs and pictures that show results for the blinded compounds of the challenge: where one stood, a note stands instead. "
                 "The complete edition replaces this one when the challenge has closed (3 November 2026, 23:59 UTC). The code, the method, every gate and every aggregate number are the same in both.")


TEMPLATE = re.compile(r"\{\{=(.+?)\}\}")


def fill(source: str, namespace: dict) -> str:
    """Replace every ``{{=expression}}`` of a Markdown cell by its value, formatted like an f-string field."""
    return TEMPLATE.sub(lambda m: eval("f" + repr("{" + m.group(1).strip() + "}"), namespace), source)


# -- outputs that cells can ask for --------------------------------------------------------------------------------------------------------

def _png_b64(path: Path) -> str:
    from PIL import Image

    with Image.open(path) as image:
        if image.width > FIGURE_WIDTH:
            image = image.resize((FIGURE_WIDTH, round(image.height * FIGURE_WIDTH / image.width)), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode()


def figure_captions() -> dict[str, tuple[str, str]]:
    """Title and caption of every catalogued figure, read from docs/figures/CATALOG.md (written by the figure builder)."""
    catalog = FIGURES / "CATALOG.md"
    if not catalog.exists():
        return {}
    out: dict[str, tuple[str, str]] = {}
    for block in re.split(r"\n### ", catalog.read_text())[1:]:
        head, _, rest = block.partition("\n")
        figure_id, _, title = head.partition(": ")
        caption = rest.strip().split("\n\n")[0].strip()
        out[figure_id.strip()] = (title.strip(), caption)
    return out


def figure_sections(catalog_text: str) -> dict[str, str]:
    """The notebook section that the catalog names for every catalogued figure (``- notebook section: 6.2``), in catalog order."""
    out: dict[str, str] = {}
    for block in re.split(r"\n### ", catalog_text)[1:]:
        found = re.search(r"- notebook section: (\S+)", block)
        if found:
            out[block.split(":", 1)[0].strip()] = found.group(1)
    return out


def shown_figures(cells: list[dict]) -> set[str]:
    """The ids of the figures that a cell shows."""
    return {m.group(1) for c in cells if c.get("cell_type") == "code" for m in [re.search(r'show_figure\("([a-z0-9_]+)"', c["source"])] if m}


def place_unshown_figures(cells: list[dict], sections: dict[str, str], held_back: set[str] = frozenset()) -> tuple[list[dict], list[str]]:
    """Add a cell for every catalogued figure that no part shows, at the end of the section that the catalog names (the end of its part when the section does not exist).
    Returns the cells and the ids that were added, so that a figure that is drawn can never be missing from the notebook."""
    found = headings(cells)
    where: dict[int, list[str]] = {}
    added = []
    for figure_id, section in sections.items():
        if figure_id in shown_figures(cells):
            continue
        position = None
        starts = [(lv, text, i) for lv, text, i in found if text.split()[0] == section]
        if starts:
            level, _text, start = starts[0]
            later = [i for lv, _t, i in found if i > start and lv <= level]
            position = later[0] if later else len(cells)
            position = max(position, start + 1) if later and later[0] <= start else position
        else:
            part = section.split(".")[0]
            inside = [(lv, text, i) for lv, text, i in found if re.match(rf"{part}\.\d+ ", text)]
            if inside:
                last = inside[-1][2]
                after = [i for lv, _t, i in found if i > last and lv == 1]
                position = after[0] if after else len(cells)
        if position is None:
            continue
        where.setdefault(position, []).append(figure_id)
        added.append(figure_id)
    out = list(cells)
    for position in sorted(where, reverse=True):
        out[position:position] = [figure(i, held_back=True) if i in held_back else figure(i) for i in where[position]]
    return out, added


class Outputs:
    """The functions a code cell calls to put something other than printed text into its output.

    While the notebook is being built the outputs are collected for the cell that produced them; in a plain Jupyter session they are displayed directly, so the same
    cells run in both."""

    def __init__(self) -> None:
        self.capturing = False
        self.pending: list[dict] = []

    def emit(self, bundle: dict, metadata: dict | None = None) -> None:
        if self.capturing:
            self.pending.append({"output_type": "display_data", "data": bundle, "metadata": metadata or {}})
        else:
            from IPython.display import display

            display(bundle, raw=True, metadata=metadata or {})

    def show_figure(self, figure_id: str, theme: str = "dark") -> None:
        """Embed a catalogued figure (the dark-background variant by default) with its caption."""
        matches = sorted(FIGURES.glob(f"*/{figure_id}-{theme}.png"))
        if not matches:
            raise FileNotFoundError(f"figure {figure_id}-{theme}.png not found under docs/figures; run docs/figures/build_all.py")
        title, caption = figure_captions().get(figure_id, (figure_id, ""))
        self.emit({"image/png": _png_b64(matches[0]), "text/plain": f"<{title}>"}, {"figure": figure_id, "title": title})
        self.emit({"text/markdown": f"**Figure: {title}.** {caption}", "text/plain": caption}, {"caption": figure_id})

    def show_markdown(self, text: str) -> None:
        self.emit({"text/markdown": text, "text/plain": text})

    def show_table(self, frame, caption: str | None = None, index: bool = False, floatfmt: str = ".3f") -> None:
        """Show a pandas DataFrame as a Markdown table."""
        self.show_markdown(markdown_table(frame, index=index, floatfmt=floatfmt) + (f"\n\n*{caption}*" if caption else ""))

    def show_math(self, tex: str) -> None:
        self.show_markdown(f"$$\n{tex.strip()}\n$$")


OUTPUTS = Outputs()
show_figure, show_markdown, show_table, show_math = OUTPUTS.show_figure, OUTPUTS.show_markdown, OUTPUTS.show_table, OUTPUTS.show_math


def markdown_table(frame, index: bool = False, floatfmt: str = ".3f") -> str:
    if index:
        frame = frame.reset_index()
    header = [str(c) for c in frame.columns]

    def cell(v) -> str:
        if isinstance(v, float):
            return format(v, floatfmt)
        return str(v).replace("|", "\\|").replace("\n", " ")

    rows = [[cell(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |", *("| " + " | ".join(r) + " |" for r in rows)])


def _matplotlib_figures() -> list[dict]:
    import matplotlib.pyplot as plt

    out = []
    for number in plt.get_fignums():
        buffer = io.BytesIO()
        plt.figure(number).savefig(buffer, format="png", dpi=130, bbox_inches="tight")
        out.append({"output_type": "display_data", "data": {"image/png": base64.b64encode(buffer.getvalue()).decode(), "text/plain": "<Figure>"}, "metadata": {}})
        plt.close(number)
    return out


def execute(cells: list[dict], public: bool = False) -> list[dict]:
    """Run the code cells in order in one namespace (with the repository root as the working directory); returns the cells with nbformat outputs attached.

    With ``public`` a cell marked ``held_back`` still runs (the variables that later cells use exist) but its outputs are replaced by one note, and a Markdown cell marked ``held_back`` becomes the note."""
    import matplotlib

    matplotlib.use("Agg")
    outputs = OUTPUTS
    namespace: dict = {"__name__": "__notebook__"}
    done, count, here = [], 0, Path.cwd()
    os.chdir(ROOT)
    outputs.capturing = True
    try:
        for cell in cells:
            if cell["cell_type"] != "code":
                if public and cell.get("held_back"):
                    done.append({**cell, "source": HELD_BACK_NOTE})
                    continue
                done.append({**cell, "source": fill(cell["source"], namespace)} if TEMPLATE.search(cell["source"]) else cell)
                continue
            count += 1
            stdout, produced = io.StringIO(), []
            outputs.pending = []
            try:
                with contextlib.redirect_stdout(stdout):
                    exec(compile(cell["source"], f"<cell {count}>", "exec"), namespace)  # noqa: S102 - the notebook source is project code
            except Exception:
                produced.append({"output_type": "error", "ename": "Error", "evalue": "cell failed", "traceback": traceback.format_exc().splitlines()})
            if stdout.getvalue():
                produced.insert(0, {"output_type": "stream", "name": "stdout", "text": stdout.getvalue()})
            produced.extend(outputs.pending)
            produced.extend(_matplotlib_figures())
            if public and cell.get("held_back") and not any(o["output_type"] == "error" for o in produced):
                produced = [{"output_type": "display_data", "data": {"text/markdown": HELD_BACK_NOTE, "text/plain": HELD_BACK_NOTE}, "metadata": {"held_back": True}}]
            done.append({**cell, "execution_count": count, "outputs": produced})
    finally:
        outputs.capturing = False
        os.chdir(here)
    return done


# -- structure: headings, sections, generated cells -----------------------------------------------------------------------------------------

HEADING = re.compile(r"^(#{1,3}) (.+)$", re.MULTILINE)
FENCE = re.compile(r"```.*?```", re.DOTALL)


def slug(text: str) -> str:
    """The anchor of a heading: its text with the spaces replaced by hyphens (the scheme Jupyter uses)."""
    return re.sub(r"\s+", "-", re.sub(r"[`*]", "", text).strip())


def headings(cells: list[dict]) -> list[tuple[int, str, int]]:
    """(level, text, cell index) of every heading in the Markdown cells, outside code fences."""
    found = []
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "markdown" and (not cell.get("generated") or cell.get("in_contents")):
            for m in HEADING.finditer(FENCE.sub("", cell["source"])):
                found.append((len(m.group(1)), m.group(2).strip(), i))
    return found


def planned_parts(plan_text: str) -> list[str]:
    """The labels of the parts and appendices that the plan lists: 'Part 0', 'Part IV', ..., 'Appendix A' to 'Appendix D'."""
    found = [f"Part {m.group(1)}" for m in re.finditer(r"^## Part (0|[IVX]+)\.", plan_text, re.MULTILINE)]
    if re.search(r"^## Appendices", plan_text, re.MULTILINE):
        found += [f"Appendix {m.group(1)}" for m in re.finditer(r"^- ([A-D])\. ", plan_text, re.MULTILINE)]
    return found


GENERATED_APPENDICES = ("Appendix D",)  # the coverage map is produced by the builder itself, not written


def unwritten_parts(cells: list[dict], plan_text: str) -> list[str]:
    """The planned parts and appendices that no level-1 heading of the cells opens yet (the generated coverage map is not counted)."""
    opened = {m.group(1) for level, text, _ in headings(cells) if level == 1 for m in [re.match(r"(Part (?:0|[IVX]+)|Appendix [A-D])\b", text)] if m}
    return [label for label in planned_parts(plan_text) if label not in opened and label not in GENERATED_APPENDICES]


def section_of_cells(cells: list[dict]) -> list[str]:
    """The label of the section each cell belongs to: the number in the last '## N.M title' (or 'Appendix X') heading before it, else the part."""
    labels, current = [], "Front matter"
    heads: dict[int, list[tuple[int, str]]] = {}
    for level, text, i in headings(cells):
        heads.setdefault(i, []).append((level, text))
    for i in range(len(cells)):
        for level, text in heads.get(i, []):
            m = re.match(r"(\d+\.\d+)\b", text) or re.match(r"(Appendix [A-Z])\b", text) or re.match(r"(Part [0IVXL]+)\b", text)
            if m and level <= 2:
                current = m.group(1)
        labels.append(current)
    return labels


def contents_cell(cells: list[dict]) -> dict:
    """The table of contents, built from the headings so that it can never disagree with the text. Every entry is a link to the heading's anchor; the sections of one part are kept
    together in a block so that a printed page never starts in the middle of a part's list."""
    blocks: list[list[str]] = [[]]
    anchors: list[str] = []
    for level, text, _ in headings(cells):
        if level == 1 and not re.match(r"(Part|Appendix)\b", text):
            continue
        if level == 1:
            blocks.append([f"**[{text}](#{slug(text)})**", ""])
        elif level == 2:
            blocks[-1].append(f"- [{text}](#{slug(text)})")
        else:
            continue
        anchors.append(slug(text))
    duplicates = sorted({a for a in anchors if anchors.count(a) > 1})
    if duplicates:
        raise ValueError(f"headings with the same anchor, the contents links would point to the wrong place: {duplicates}")
    body = "\n\n".join('<div class="toc-part">\n\n' + "\n".join(block) + "\n\n</div>" for block in blocks if block)
    return md("## Table of contents\n\n" + body, generated=True)


# -- checks ---------------------------------------------------------------------------------------------------------------------------------

def failures(cells: list[dict]) -> list[int]:
    return [i for i, c in enumerate(cells) if c["cell_type"] == "code" and any(o["output_type"] == "error" for o in c.get("outputs", []))]


def glossary_problems(cells: list[dict], terms: list[str], start: int = 0) -> list[str]:
    """Terms whose first use (from cell ``start`` on, the front matter being only a summary) is in a cell that does not define them in bold, and terms never used at all. A plural counts as a use."""
    problems = []
    for term in terms:
        use = re.compile(rf"(?<![\w-]){re.escape(term)}(?:e?s)?(?![\w-])", re.IGNORECASE)
        bold = re.compile(rf"\*\*{re.escape(term)}(?:e?s)?\*\*", re.IGNORECASE)
        for i, c in enumerate(cells):
            if i >= start and c["cell_type"] == "markdown" and not c.get("generated") and use.search(FENCE.sub("", c["source"])):
                if not bold.search(c["source"]):
                    problems.append(f"'{term}' is first used in cell {i} without a definition there")
                break
        else:
            problems.append(f"'{term}' is never used")
    return problems


FIELDS = (
    "Overview", "Mathematics", "Statistics", "Biology", "Biochemistry", "Pharmacology", "Chemistry", "Cheminformatics", "Computational chemistry", "Quantum chemistry", "Physics",
    "Bioinformatics", "Structural biology", "Machine learning", "Data science", "Software engineering",
)
FIELD_LINE = re.compile(r"^> \*\*Field:\*\* (.+)$", re.MULTILINE)
SECTION_HEADING = re.compile(r"^## (?:\d+\.\d+|Appendix [A-Z]\b)", re.MULTILINE)


def field_problems(cells: list[dict]) -> list[str]:
    """Every numbered section (and every appendix) names, right under its heading, the sciences it speaks from: ``> **Field:** Pharmacology · Mathematics (differential equations)``.
    The names come from ``FIELDS``; a parenthesis may say which branch. A reader always knows from which discipline the next pages are written."""
    problems = []
    for i, c in enumerate(cells):
        if c["cell_type"] != "markdown" or c.get("generated"):
            continue
        source = FENCE.sub("", c["source"])
        for heading in SECTION_HEADING.finditer(source):
            title = source[heading.start() : source.find("\n", heading.start())].strip()
            line = FIELD_LINE.search(source[heading.end() :])
            if line is None or source[heading.end() :].find(line.group(0)) > 400:
                problems.append(f"cell {i}: '{title}' has no '> **Field:** ...' line under its heading")
                continue
            for name in (part.split("(")[0].strip() for part in line.group(1).split("·")):
                if name not in FIELDS:
                    problems.append(f"cell {i}: '{title}' names the field '{name}', which is not one of {', '.join(FIELDS)}")
    return problems


DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
INLINE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL)
MATH_COMMANDS = {
    "frac", "sum", "int", "prod", "ln", "log", "exp", "max", "min", "mathrm", "text", "left", "right", "cdot", "quad", "qquad", "Longrightarrow", "Rightarrow", "ldots", "cdots", "times", "to", "approx",
    "in", "infty", "partial", "sqrt", "mathbb", "mathbf", "mathcal", "operatorname", "begin", "end", "hat", "bar", "tilde", "vec", "le", "ge", "leq", "geq", "neq", "ne", "sim", "propto", "pm", "mid", "sin",
    "cos", "tanh", "arg", "lim", "det", "ell", "binom", "underbrace", "overline", "dots", "equiv", "cap", "cup", "land", "lor", "neg", "forall", "exists", "subset", "nabla", "langle", "rangle", "Big", "big",
    "lbrace", "rbrace", "vert", "Vert", "lVert", "rVert", "lfloor", "rfloor", "stackrel", "overset", "limits", "bigl", "bigr", "tfrac", "dfrac", "displaystyle", "mathit", "boldsymbol", "scriptstyle", "colon", "prime", "longrightarrow", "longleftarrow", "rightarrow", "leftarrow", "leftrightarrow", "rightleftharpoons", "underset", "Bigl", "Bigr", "biggl", "biggr", "Biggl", "Biggr", "Bigg", "bigg",
    "setminus", "top", "hspace", "vspace", "phantom", "substack", "atop", "choose", "over", "mathop", "mathbin", "mathrel", "boxed", "cancel",
}
UNIVERSAL_SYMBOLS = {"e", "pi", "d", "dt", "dx"}  # the constant e, pi, and the d of a differential, defined in section 0.x and 1.1 and never redefined


def math_symbols(math: str) -> set[str]:
    """The single-letter and Greek symbols of a piece of LaTeX, leaving out commands, words inside \\text and \\mathrm, and digits."""
    cleaned = re.sub(r"\\(?:text|mathrm|operatorname)\{[^}]*\}", " ", math)
    found = set()
    for token in re.findall(r"\\[A-Za-z]+|[A-Za-z]", cleaned):
        name = token.lstrip("\\")
        if token.startswith("\\") and name in MATH_COMMANDS:
            continue
        found.add(name)
    return found


def symbol_problems(cells: list[dict], start: int = 0) -> list[str]:
    """Within a section, every symbol of a display equation is named in prose before or beside the equation: it appears in an inline ``$...$`` of the same cell (or of an earlier cell of the same
    section), or it is one of the universal symbols. An equation that only repeats symbols the text has introduced passes; an equation that uses a letter nobody has said anything about does not.
    A symbol inside an equation that stands inside a sentence is checked like any other because it is read from the inline maths as well."""
    labels, problems, named = section_of_cells(cells), [], {}
    for i, c in enumerate(cells):
        if i < start or c["cell_type"] != "markdown" or c.get("generated"):
            continue
        section = labels[i]
        seen = named.setdefault(section, set())
        text = FENCE.sub("", c["source"])
        for block in re.split(r"(?m)^(?=#{1,2} )", text):  # a symbol named in one section does not carry over to the next heading of the same cell
            prose = DISPLAY.sub(" ", block)
            for m in INLINE.finditer(prose):
                seen |= math_symbols(m.group(1))
            for m in DISPLAY.finditer(block):
                missing = sorted(s for s in math_symbols(m.group(1)) - seen - UNIVERSAL_SYMBOLS)
                if missing:
                    problems.append(f"cell {i} (section {section}): symbols {', '.join(missing)} of an equation are not named in the text of the section before it ('{m.group(1).strip()[:50]}')")
                    seen |= set(missing)  # report each symbol once per section
    return problems


COVERED_GLOBS = {
    "module": ("cypmode/**/*.py",),
    "script": ("scripts/*",),
    "queue job": ("queue/jobs/*.sh", "queue/*.sh"),
    "test": ("tests/test_*.py", "tests/conftest.py"),
    "other": ("app.py", ".github/workflows/*.yml", "environment/*", "notebooks/*.py", "audit/*.md"),
}


def repository_items(root: Path = ROOT) -> dict[str, list[str]]:
    items: dict[str, list[str]] = {}
    for kind, patterns in COVERED_GLOBS.items():
        paths = sorted({p.relative_to(root).as_posix() for pat in patterns for p in root.glob(pat) if p.is_file() and p.name != "__init__.py" and "__pycache__" not in p.parts})
        items[kind] = paths
    return items


def _spellings(path: str, all_paths: list[str]) -> list[str]:
    """The ways a file may be named in the text: the full path, the dotted module, the file name when it is unique, the job number for queue jobs."""
    name = path.rsplit("/", 1)[-1]
    out = [path]
    if path.endswith(".py"):
        out.append(path[:-3].replace("/", "."))
    if sum(p.rsplit("/", 1)[-1] == name for p in all_paths) == 1:
        out.append(name)
        if name.endswith(".py"):
            out.append(name[:-3])
    parent = path.rsplit("/", 2)
    if len(parent) == 3:
        out.append(f"{parent[1]}/{name}")
    if path.startswith("queue/jobs/"):
        out = [path, name, name[:-3]]
    return out


def coverage(cells: list[dict], root: Path = ROOT) -> tuple[dict[str, dict[str, list[str]]], list[str]]:
    """Map every repository item to the sections that name it, and list the items no section names (generated cells do not count)."""
    items = repository_items(root)
    all_paths = [p for paths in items.values() for p in paths]
    labels = section_of_cells(cells)
    texts = [(labels[i], c["source"]) for i, c in enumerate(cells) if not c.get("generated")]
    mapping: dict[str, dict[str, list[str]]] = {}
    missing = []
    for kind, paths in items.items():
        mapping[kind] = {}
        for path in paths:
            patterns = [re.compile(rf"(?<![\w/.]){re.escape(s)}(?![\w])") for s in _spellings(path, all_paths)]
            sections = []
            for label, text in texts:
                if any(p.search(text) for p in patterns) and label not in sections:
                    sections.append(label)
            mapping[kind][path] = sections
            if not sections:
                missing.append(f"{kind}: {path}")
    return mapping, missing


def coverage_cell(mapping: dict[str, dict[str, list[str]]]) -> dict:
    """Appendix D: the map from every file of the repository to the sections that explain it."""
    lines = ["# Appendix D. Coverage map", "", "Every module, script, queue job and test file of the repository and the sections of this notebook that explain it. The notebook builder fails when a file is not named in the text, so the map is complete by construction."]
    for kind, entries in mapping.items():
        lines += ["", f"### {kind.capitalize()}s" if not kind.endswith("s") else f"### {kind.capitalize()}", "", "| file | sections |", "| --- | --- |"]
        lines += [f"| `{path}` | {', '.join(sections)} |" for path, sections in entries.items()]
    return md("\n".join(lines), generated=True, in_contents=True)


# -- writing --------------------------------------------------------------------------------------------------------------------------------

def _cell_id(index: int, source: str) -> str:
    return hashlib.sha1(f"{index}:{source[:80]}".encode()).hexdigest()[:8]


def to_notebook(cells: list[dict]) -> dict:
    out = []
    for i, c in enumerate(cells):
        metadata = {"generated": True} if c.get("generated") else {}
        if c.get("hide_input"):
            metadata["jupyter"] = {"source_hidden": True}
        if c.get("titlepage"):
            metadata["titlepage"] = True
        base = {"cell_type": c["cell_type"], "id": _cell_id(i, c["source"]), "metadata": metadata, "source": c["source"].splitlines(keepends=True)}
        if c["cell_type"] == "code":
            base.update(execution_count=c.get("execution_count"), outputs=c.get("outputs", []))
        out.append(base)
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


BLINDED_ID = re.compile(r"OCNT-\d{7}")
MAX_IDS_PER_CELL = 5  # the rule of the public-readiness scan: five or more blinded compounds in one place is a result for them


def public_problems(executed: list[dict], official: list[str]) -> list[str]:
    """What keeps an executed edition from being public: a cell that names one of the official structure-track ligands (their poses are results) or five or more blinded compounds."""
    problems = []
    official = set(official)
    for i, cell in enumerate(executed):
        text = cell["source"] + json.dumps(cell.get("outputs", []), ensure_ascii=False)
        ids = set(BLINDED_ID.findall(text))
        if ids & official:
            problems.append(f"cell {i} names {len(ids & official)} official structure ligand(s)")
        elif len(ids) >= MAX_IDS_PER_CELL:
            problems.append(f"cell {i} names {len(ids)} compounds of the blinded sets")
    return problems


def build(cells: list[dict], path: str | Path, public: bool = False) -> list[dict]:
    executed = execute(cells, public=public)
    Path(path).write_text(json.dumps(to_notebook(executed), indent=1, ensure_ascii=False) + "\n")
    return executed

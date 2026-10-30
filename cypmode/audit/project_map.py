# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The project map: ``python -m cypmode.audit.project_map`` writes ``docs/map``, ``--check`` compares the files with the code.

The map is four views of one project, each a Markdown page with a Mermaid diagram that GitHub draws and a table that names every element: the code (packages and modules with what they import),
the queue (the jobs, what each waits for and the chain that closes the project), the flow of the data (inputs, jobs, logs, ledger, reports) and the evidence (the families of ideas that were tried).
A fifth file, ``explorer.html``, draws the same four views as one page that can be explored (a graph of the modules, a grid of the jobs, a search box, and for every module and job what it is connected to
and which section of the notebook names it); it is one file with its data inside, filled into the template ``project_map_page.html``, and it is checked like the pages.
Every sentence is the first sentence of a docstring or of the header comment of a job, or a count from the files of the repository. Only tracked sources are read (the modules, the queue scripts and the
tables of ``ledger_kinds``), so a copy of the repository regenerates the same pages without any result.
"""

from __future__ import annotations

import argparse
import ast
import base64
import functools
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / "docs/map"
PAGE = Path(__file__).with_name("project_map_page.html")
BRAND = ROOT / "docs/brand"
TOP = "top level"
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z`(])")
LOG_WRITE = re.compile(r"\$L/([A-Za-z0-9_.\-${}]+\.log)")
GLOB_READER = re.compile(r"""\.glob\(["']\*\.log["']\)""")
MODULE_RUN = re.compile(r"-m ((?:cypmode|notebooks)[\w.]*)")
SETTLED = re.compile(r"need_settled ([^\n;|&]+)")
LOWER = re.compile(r"all_lower_done")
USED_BY_SHOWN = 6


def shorten(sentence: str, limit: int = 420) -> str:
    """A sentence of at most ``limit`` characters that still reads as a sentence: a longer one is cut after its last clause that fits, at a semicolon or a colon, else at a comma."""
    if len(sentence) <= limit:
        return sentence
    for marks in (";:", ","):
        found = [m.start() for m in re.finditer(f"[{marks}]", sentence[40:limit])]
        if found:
            return sentence[: 40 + found[-1]].rstrip(" ,;:") + "."
    return sentence[: sentence.rfind(" ", 0, limit)].rstrip(" ,;:") + "."


def first_sentence(text: str | None) -> str:
    flat = " ".join((text or "").split()).replace("``", "`")
    return shorten(SENTENCE.split(flat, maxsplit=1)[0]) if flat else "no docstring"


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s" if not word.endswith("y") else f"{n} {word[:-1]}ies" if n != 1 else f"{n} {word}"


def cell(text: str) -> str:
    return text.replace("|", "\\|")


def table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|", *("| " + " | ".join(cell(c) for c in row) + " |" for row in rows)])


# -- the code ---------------------------------------------------------------------------------------------------------------------------------------------

def module_paths() -> dict[str, Path]:
    return {".".join(p.relative_to(ROOT).with_suffix("").parts): p for p in sorted((ROOT / "cypmode").rglob("*.py")) if "__pycache__" not in p.parts and p.name != "__init__.py"}


def package_of(module: str) -> str:
    parts = module.split(".")
    return TOP if len(parts) == 2 else parts[1]


def imports_of(module: str, path: Path, known: set[str]) -> set[str]:
    """The modules of the project that a module imports, wherever the import stands (at the top or inside a function)."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = ".".join([*module.split(".")[: -node.level], *([node.module] if node.module else [])])
            found |= {t for t in [base, *(f"{base}.{a.name}" for a in node.names)] if t in known}
        elif isinstance(node, ast.Import):
            found |= {a.name for a in node.names if a.name in known}
    found.discard(module)
    return found


def code_facts() -> dict:
    paths = module_paths()
    graph = {m: imports_of(m, p, set(paths)) for m, p in paths.items()}
    used_by: dict[str, set[str]] = defaultdict(set)
    for module, targets in graph.items():
        for target in targets:
            used_by[target].add(module)
    cross = Counter((package_of(m), package_of(t)) for m, targets in graph.items() for t in targets if package_of(t) != package_of(m))
    summary = {m: first_sentence(ast.get_docstring(ast.parse(p.read_text()))) for m, p in paths.items()}
    return {"paths": paths, "graph": graph, "used_by": used_by, "cross": cross, "summary": summary}


def package_doc(package: str) -> str:
    init = ROOT / "cypmode" / ("__init__.py" if package == TOP else f"{package}/__init__.py")
    doc = ast.get_docstring(ast.parse(init.read_text())) if init.exists() else None
    return first_sentence(doc) if doc else "no package docstring"


def stems(names: set[str]) -> str:
    ordered = sorted(n.split(".", 1)[1] for n in names)  # the package and the module, because a bare module name is not unique
    shown = ", ".join(f"`{n}`" for n in ordered[:USED_BY_SHOWN])
    return shown + (f" and {len(ordered) - USED_BY_SHOWN} more" if len(ordered) > USED_BY_SHOWN else "") if ordered else "none"


def code_diagram(facts: dict) -> str:
    """The Mermaid diagram of the packages: an arrow from a package to a package whose modules it imports, labelled with the number of those imports."""
    packages = Counter(package_of(m) for m in facts["paths"])
    node = lambda p: re.sub(r"\W", "_", p)
    return "\n".join([
        "```mermaid",
        "flowchart LR",
        *(f'  {node(p)}["{p} ({plural(n, "module")})"]' for p, n in sorted(packages.items(), key=lambda x: (-x[1], x[0]))),
        *(f"  {node(a)} -->|{n}| {node(b)}" for (a, b), n in sorted(facts["cross"].items(), key=lambda x: (-x[1], x[0]))),
        "```",
    ])


def render_code() -> str:
    facts = code_facts()
    packages = Counter(package_of(m) for m in facts["paths"])
    edges = sum(len(t) for t in facts["graph"].values())
    lines = [
        "# The code",
        "",
        f"The project has {len(facts['paths'])} modules in {len(packages)} packages, and the modules hold {edges} imports of one another. Each arrow below goes from a package to a package whose modules it imports, and its label counts those imports.",
        "",
        code_diagram(facts),
        "",
        "A module-level drawing of the largest packages would hold hundreds of arrows and say less than the tables, so each package below lists its modules with what they import and what imports them.",
        "",
    ]
    for package, count in sorted(packages.items(), key=lambda x: (-x[1], x[0])):
        rows = []
        for module in sorted(m for m in facts["paths"] if package_of(m) == package):
            rows.append([f"`{facts['paths'][module].relative_to(ROOT).as_posix()}`", facts["summary"][module], stems(facts["graph"][module]), stems(facts["used_by"][module])])
        lines += [f"## {package} ({plural(count, 'module')})", "", package_doc(package), "", table(["module", "what it does", "imports", "imported by"], rows), ""]
    return "\n".join(lines)


# -- the queue --------------------------------------------------------------------------------------------------------------------------------------------

def job_header(text: str) -> str:
    """The comment that opens a job script: every consecutive comment line after the first, joined (a long comment is wrapped over several lines)."""
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("#!"):
            continue
        if line.startswith("#"):
            lines.append(line.lstrip("# ").strip())
        elif lines:
            break
    return " ".join(lines)


def job_facts() -> list[dict]:
    jobs = []
    for path in sorted((ROOT / "queue/jobs").glob("*.sh")):
        text = path.read_text()
        header = job_header(text)
        named = [n for m in SETTLED.findall(text) for n in m.split()]
        jobs.append({
            "id": path.stem.split("_", 1)[0], "name": path.stem, "summary": first_sentence(header), "all_lower": bool(LOWER.search(text)), "named": named,
            "runs": sorted(set(MODULE_RUN.findall(text))), "logs": sorted(set(LOG_WRITE.findall(text))),
        })
    return jobs


def waits_for(job: dict) -> str:
    parts = (["every job with a smaller number"] if job["all_lower"] else []) + ([", ".join(f"`{n}`" for n in job["named"])] if job["named"] else [])
    return " and ".join(parts) or "nothing"


def render_queue() -> str:
    jobs = job_facts()
    families: dict[str, list[dict]] = defaultdict(list)
    for job in jobs:
        families[job["id"][0] + "xx"].append(job)
    chain = [j for j in jobs if int(j["id"]) >= 900]
    lines = [
        "# The queue",
        "",
        f"Every experiment and every build step of the project is a script of `queue/jobs`; the runner (`scripts/queue_runner.sh`) takes them in the order of their numbers. There are {len(jobs)} jobs in {len(families)} families, by the first digit of the number, and {sum(j['all_lower'] for j in jobs)} of them wait until every job with a smaller number has finished.",
        "",
        table(["family", "jobs", "first", "last"], [[f, str(len(js)), f"`{js[0]['name']}`", f"`{js[-1]['name']}`"] for f, js in sorted(families.items())]),
        "",
        f"## The chain that closes the project ({len(chain)} jobs)",
        "",
        "The jobs numbered 900 and above rebuild every report, figure, notebook and submission file from the logs and then check them. The runner takes the jobs in the order of their numbers; an arrow joins two consecutive jobs when the later one also waits, explicitly, for every job before it.",
        "",
        "```mermaid",
        "flowchart LR",
        *(f'  j{j["id"]}["{j["name"]}"]' for j in chain),
        *(f"  j{a['id']} --> j{b['id']}" for a, b in pairwise(chain) if b["all_lower"]),
        "```",
        "",
    ]
    for family, js in sorted(families.items()):
        rows = [[f"`{j['name']}`", j["summary"], waits_for(j)] for j in js]
        lines += [f"## Family {family} ({len(js)} jobs)", "", table(["job", "what it does", "waits for"], rows), ""]
    return "\n".join(lines)


# -- the flow of the data -----------------------------------------------------------------------------------------------------------------------------------

def dataflow_facts() -> dict:
    jobs = job_facts()
    written = sorted({log for j in jobs for log in j["logs"]})
    sources = [p for pattern in ("cypmode/**/*.py", "notebooks/**/*.py") for p in sorted(ROOT.glob(pattern)) if "__pycache__" not in p.parts]
    texts = {p: p.read_text() for p in sources}
    literal = {log for log in written if "$" not in log}
    return {
        "jobs": jobs, "written": written, "literal": literal, "named": sorted(log for log in literal if any(log in text for text in texts.values())),
        "glob_readers": sorted(p.relative_to(ROOT).as_posix() for p, text in texts.items() if GLOB_READER.search(text)), "writers": sum(1 for j in jobs if j["logs"]),
    }


def dataflow_diagram(facts: dict) -> str:
    """The Mermaid diagram of the five stages, with the counts of the jobs, the logs and the readers."""
    return "\n".join([
        "```mermaid",
        "flowchart LR",
        '  inputs["inputs<br/>data/*/inputs"] --> jobs',
        f'  jobs["queue jobs<br/>{facts["writers"]} of {len(facts["jobs"])} write a log"] -->|"{len(facts["written"])} log names"| logs["logs"]',
        '  logs -->|"every log, through gate_rows"| ledger["ledger<br/>cypmode/audit/ledger_kinds.py"]',
        f'  logs -->|"{len(facts["named"])} logs are read by name"| reports',
        '  ledger --> reports["reports, figures, notebook<br/>built by the chain 900 to 960"]',
        '  reports --> submissions["submission files"]',
        "```",
    ])


def render_dataflow() -> str:
    facts = dataflow_facts()
    jobs, literal, named, glob_readers = facts["jobs"], facts["literal"], facts["named"], facts["glob_readers"]
    documents = [(j["name"], r) for j in jobs if int(j["id"]) >= 900 for r in j["runs"]]
    lines = [
        "# The flow of the data",
        "",
        "The data go one way: the inputs of a track enter the jobs of the queue, the jobs write logs, the ledger reads the lines of the gates in the logs, and the reports, the figures, the notebook and the submission files are built from the logs and the ledger.",
        "",
        dataflow_diagram(facts),
        "",
        f"A log is named by a module in {len(named)} of the {len(literal)} cases in which a job writes it under a fixed name. The others are read through the folder, and {plural(len(glob_readers), 'module')} read every log of it that way:",
        "",
        *(f"- `{p}`" for p in glob_readers),
        "",
        "The jobs of the chain that run a module of the project (the others run scripts) are:",
        "",
        table(["job", "module it runs"], [[f"`{a}`", f"`{b}`"] for a, b in documents]),
        "",
    ]
    return "\n".join(lines)


# -- the evidence -----------------------------------------------------------------------------------------------------------------------------------------

def render_evidence() -> str:
    from cypmode.audit.ledger_kinds import KINDS
    from cypmode.audit.method_report import GATE

    described = _kind_descriptions()
    lines = [
        "# The evidence",
        "",
        (
            "Every idea that the project tried passed through a gate written before its result. A gate prints a line that ends in `MET` or `NOT MET`; the ledger collects those lines from the logs, groups them by the family of the log and by the kind of idea, and counts them. "
            f"The pattern that finds a gate line is `{GATE.pattern}`."
        ),
        "",
        "```mermaid",
        "flowchart LR",
                '  ledger["ledger of verdicts"]',
        *(f'  k{i}["{kind} ({plural(len(stems), "family")})"] --> ledger' for i, (kind, stems) in enumerate(KINDS.items())),
        '  ledger --> negative["NEGATIVE_RESULTS.md"]',
        '  ledger --> report["method report"]',
        "```",
        "",
        table(["kind of idea", "families", "what it tested"], [[kind, str(len(stems)), described.get(kind, "")] for kind, stems in KINDS.items()]),
        "",
        "The counts of verdicts and of verdicts that were MET are in the method report and in `NEGATIVE_RESULTS.md`; they are read from the logs, which this map does not read.",
        "",
    ]
    return "\n".join(lines)


def _kind_descriptions() -> dict[str, str]:
    from cypmode.audit import ledger_kinds

    for value in vars(ledger_kinds).values():
        if isinstance(value, dict) and value and set(value) == set(ledger_kinds.KINDS) and all(isinstance(v, str) for v in value.values()):
            return {k: first_sentence(v) for k, v in value.items()}
    return {}


# -- the page that can be explored ------------------------------------------------------------------------------------------------------------------------

def repository_url() -> str:
    found = re.search(r'^repository-code:\s*"([^"]+)"', (ROOT / "CITATION.cff").read_text(), re.MULTILINE)
    return found.group(1) if found else ""


@functools.cache
def notebook_sections() -> tuple[dict[str, list[str]], dict[str, dict[str, str]]]:
    """Which sections of the notebook name each module and job (the coverage that the builder checks), and for every label the text of its heading and the anchor that the contents link to."""
    from notebooks.notebook_builder import coverage, headings, slug
    from notebooks.notebook_cells import ALL_CELLS

    mapping, _ = coverage(ALL_CELLS)
    targets = {}
    for level, text, _ in headings(ALL_CELLS):
        numbered = re.match(r"(\d+\.\d+) ", text) if level == 2 else None
        part = re.match(r"(Part [IVX0]+|Appendix [A-Z])\b", text) if level == 1 else None
        if numbered or part:
            label = (numbered or part).group(1)
            targets[label] = {"label": label, "text": text, "slug": slug(text)}
    return {path: sections for kind in ("module", "queue job") for path, sections in mapping[kind].items()}, targets


@functools.cache
def explorer_data_json() -> str:
    from cypmode.audit.ledger_kinds import KINDS

    code, jobs, flow = code_facts(), job_facts(), dataflow_facts()
    named_in, targets = notebook_sections()
    known = set(code["paths"])

    def taught(path: str) -> list[dict[str, str]]:
        return [targets.get(label, {"label": label, "text": label, "slug": ""}) for label in named_in.get(path, [])]

    tests_of: dict[str, list[str]] = defaultdict(list)
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        for module in imports_of("tests." + path.stem, path, known):
            tests_of[module].append(path.relative_to(ROOT).as_posix())
    run_by: dict[str, list[str]] = defaultdict(list)
    for job in jobs:
        for module in job["runs"]:
            run_by[module].append(job["name"])
    modules = []
    for module, path in sorted(code["paths"].items()):
        rel = path.relative_to(ROOT).as_posix()
        modules.append({"id": module, "path": rel, "package": package_of(module), "summary": code["summary"][module], "imports": sorted(code["graph"][module]), "importedBy": sorted(code["used_by"][module]),
                        "runBy": sorted(run_by[module]), "tests": tests_of[module], "taught": taught(rel)})
    job_rows = [{"id": j["id"], "name": j["name"], "family": j["id"][0] + "xx", "summary": j["summary"], "waitsAll": j["all_lower"], "waits": j["named"], "runs": [m for m in j["runs"] if m in known],
                 "runsOther": [m for m in j["runs"] if m not in known], "logs": j["logs"], "taught": taught(f"queue/jobs/{j['name']}.sh")} for j in jobs]
    described = _kind_descriptions()
    evidence = [{"kind": kind, "what": described.get(kind, ""), "families": list(stems)} for kind, stems in KINDS.items()]
    documents = [r for j in jobs if int(j["id"]) >= 900 for r in j["runs"] if r in known]
    inputs = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.glob("data/*/inputs") if p.is_dir())
    stages = [
        {"id": "inputs", "title": "Inputs", "count": f"{len(inputs)} tracks", "text": "The files that a track starts from: the challenge's training and test tables for the TDI track, the structure inputs for the structure track.",
         "itemsTitle": "Folders", "items": [{"text": p} for p in inputs]},
        {"id": "jobs", "title": "Queue jobs", "count": f"{flow['writers']} of {len(flow['jobs'])} write a log", "itemsTitle": "", "items": [],
         "text": "Every job of queue/jobs is a script and the runner takes them in the order of their numbers. A job that writes a log leaves the lines that the later stages read."},
        {"id": "logs", "title": "Logs", "count": f"{len(flow['written'])} log names", "itemsTitle": "Modules that read every log", "items": [({"module": p[:-3].replace("/", ".")} if p[:-3].replace("/", ".") in known else {"text": p}) for p in flow["glob_readers"]],
         "text": f"{len(flow['named'])} of the {len(flow['literal'])} logs that a job writes under a fixed name are read by name in a module. The others are read through the folder, and these modules read every log of it that way."},
        {"id": "ledger", "title": "Ledger", "count": f"{sum(len(e['families']) for e in evidence)} families in {len(evidence)} kinds", "itemsTitle": "", "items": [],
         "text": "The ledger (cypmode/audit/ledger_kinds.py) collects the line of every gate, MET or NOT MET, from the logs and groups the gates by the kind of idea. The view of the evidence lists the kinds."},
        {"id": "reports", "title": "Reports, figures, notebook", "count": f"{len(documents)} documents built by the chain", "itemsTitle": "Jobs of the chain that run a module (click one to see which)",
         "items": [{"job": j["name"]} for j in jobs if int(j["id"]) >= 900 and any(r in known for r in j["runs"])],
         "text": "The jobs numbered 900 and above rebuild every report, figure, notebook and submission file from the logs. These are the jobs of the chain that run a module of the project."},
        {"id": "submissions", "title": "Submission files", "count": "built last", "itemsTitle": "", "items": [],
         "text": "The last job of the chain writes the submission files from the final candidates and checks them with the rules of the portal. It never uploads anything."},
    ]
    data = {"meta": {"modules": len(modules), "imports": sum(len(m["imports"]) for m in modules), "jobs": len(job_rows), "repository": repository_url()},
            "modules": modules, "jobs": job_rows, "flow": {"stages": stages}, "evidence": evidence}
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")


def explorer_data() -> dict:
    return json.loads(explorer_data_json().replace("<\\/", "</"))


def icon_tags() -> str:
    """The icon of the project as links inside the page (the picture is part of the file, so that it shows in a tab wherever the page is opened from)."""
    svg, png = (base64.b64encode((BRAND / name).read_bytes()).decode() for name in ("cypmode-icon.svg", "favicon-32.png"))
    return f'<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,{svg}">\n<link rel="icon" type="image/png" sizes="32x32" href="data:image/png;base64,{png}">'


def render_explorer() -> str:
    page = PAGE.read_text()
    assert page.count("/*DATA*/null") == 1 and page.count("<!--ICON-->") == 1
    return page.replace("<!--ICON-->", icon_tags()).replace("/*DATA*/null", explorer_data_json())


# -- the index --------------------------------------------------------------------------------------------------------------------------------------------

def render_index() -> str:
    facts = code_facts()
    jobs = job_facts()
    return "\n".join([
        "# Map of the project",
        "",
        "Four views of one project, and one page that draws them so that they can be explored. Every page is written by `python -m cypmode.audit.project_map` from the files of the repository, and `python -m cypmode.audit.project_map --check` fails when a page differs from them, so a page cannot drift from the code.",
        "",
        table(["page", "what it shows", "size"], [
            ["[The code](code.md)", "the packages and modules, what each module does and what it imports", f"{len(facts['paths'])} modules"],
            ["[The queue](queue.md)", "the jobs, what each waits for and the chain that rebuilds and checks the project", f"{len(jobs)} jobs"],
            ["[The flow of the data](dataflow.md)", "from the inputs through the jobs and the logs to the reports and the submission files", "5 stages"],
            ["[The evidence](evidence.md)", "the kinds of idea that were tried and the ledger that counts the verdicts", "one ledger"],
            ["[The interactive map](explorer.html)", "all four views as one page that can be explored: a graph of the modules, a grid of the jobs, a search box, and for each one what it is connected to", "one file"],
        ]),
        "",
        "The interactive map is one file with its data inside: download it and open it in a browser (GitHub shows an HTML file as text), or open it on the Pages site. What the map does not show: the content of any input or result, and what happens while a job runs. It names files and counts them, and it reads no log.",
        "",
    ])


def readme_block() -> str:
    """The section of the README that shows the map: the loop of the interactive page, the links to open it on the Pages site and to download it, and the way into the pages."""
    repository = repository_url()
    account, name = repository.rstrip("/").split("github.com/")[-1].split("/")
    pages, download = f"https://{account}.github.io/{name}/map/", f"{repository}/raw/main/docs/map/explorer.html"
    return "\n".join([
        "## Map of the project",
        "",
        (
            "The pages of [docs/map](docs/map/README.md) show where everything is and what depends on what: the code, the queue, the flow of the data and the evidence. The interactive map draws the same four views "
            "as one page that can be explored: a graph of the modules, a grid of the jobs, a search box, and for each module and job what it imports, what imports it, which job runs it, which test imports it and "
            "which section of the notebook names it. The loop below is the page itself."
        ),
        "",
        (
            f'<p align="center"><a href="{pages}"><img src="docs/map/preview/explorer.gif" alt="The interactive map of the project: a search for a word, a module with what it imports and what imports it, '
            f'the job that runs it, the flow of the data and the kinds of idea" width="90%"></a></p>'
        ),
        "",
        f"**Open the interactive map:** [on the Pages site]({pages}). **Download it:** [explorer.html]({download}) (save the file with a right click and open it in a browser; it is one file and needs no network).",
        "",
        "The pages list every module with what it does and what it imports, and every job of the queue with what it waits for.",
        "",
    ])


def build() -> dict[str, str]:
    return {"README.md": render_index(), "code.md": render_code(), "queue.md": render_queue(), "dataflow.md": render_dataflow(), "evidence.md": render_evidence(), "explorer.html": render_explorer()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="compare docs/map with the code and exit 1 on a difference")
    args = parser.parse_args(argv)
    pages = build()
    if args.check:
        stale = [name for name, text in pages.items() if not (MAP / name).exists() or (MAP / name).read_text() != text]
        stale += [p.name for p in MAP.iterdir() if p.is_file() and not p.name.startswith(".") and p.name not in pages] if MAP.exists() else []
        print("project map: " + ("matches the code" if not stale else f"differs in {', '.join(sorted(stale))}"))
        return 1 if stale else 0
    MAP.mkdir(parents=True, exist_ok=True)
    for stray in MAP.iterdir():
        if stray.is_file() and not stray.name.startswith(".") and stray.name not in pages:
            stray.unlink()
    for name, text in pages.items():
        (MAP / name).write_text(text if text.endswith("\n") else text + "\n")
    print(f"project map: wrote {len(pages)} pages to {MAP.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

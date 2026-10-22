# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Does the checklist match the repository? Every promise must point at something that exists.

The checklist (SUBMISSION_CHECKLIST.md) is executed by the queue (queue/jobs/NNN_*.sh). This module makes the three ways it can
drift from reality checkable by a script instead of by reading:

  * a job number mentioned in the checklist has no job file;
  * a code file mentioned in backticks does not exist (`physics_block.py`, `cypmode/audit/`, ...);
  * a queue job calls a module that does not exist or waits (``need_file``) for a code file that does not exist, which would make it
    answer "not ready" forever.

It also lists every open item (`[ ]` / `[~]`) that has neither a queue job attached nor an explicit tag: a standing rule is tagged ``(standing rule)``, a step that follows the pre-submission gate (the submissions and the form) ``(after the gate)``, and a text, review or action that is done by hand ``(written by hand)``.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKLIST = ROOT / "SUBMISSION_CHECKLIST.md"
JOBS = ROOT / "queue" / "jobs"
ITEM = re.compile(r"^\s*- \[([ x~])\] (.*)$")
ASSIGNEES = {"standing rule": "(standing rule)", "after the gate": "(after the gate)", "written by hand": "(written by hand"}
SKIP_TOKENS = {"python", "git", "README.md", "CONTRIBUTING.md", "CITATION.cff", "pyproject.toml", "app.py", "stacking_significance.py", "amines.py", "oof_bridge.py", "constants.py", "run_batch"}


def items(text: str) -> list[tuple[str, str, int]]:
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        m = ITEM.match(line)
        if m:
            out.append((m.group(1), m.group(2), n))
    return out


def job_ids(text: str) -> set[str]:
    ids: set[str] = set()
    for m in re.finditer(r"\bjobs? ([0-9][0-9/, ()a-z-]*)", text):
        ids.update(re.findall(r"\b(\d{3})\b", m.group(1)))
    return ids


def code_refs(text: str) -> set[str]:
    return {t for t in re.findall(r"`([A-Za-z0-9_./-]+\.py)`", text) if t not in SKIP_TOKENS}


def resolve(token: str) -> bool:
    if "/" in token:
        return (ROOT / token).exists()
    return any(ROOT.glob(f"**/{token}") if False else [p for base in ("cypmode", "scripts", "tests", "queue") for p in (ROOT / base).rglob(token)])


def job_problems() -> list[str]:
    problems = []
    for job in sorted(JOBS.glob("*.sh")):
        text = job.read_text()
        for module in re.findall(r"-m (cypmode(?:\.\w+)+)", text):
            if not (ROOT / (module.replace(".", "/") + ".py")).exists():
                problems.append(f"{job.name}: module {module} does not exist")
        for target in re.findall(r"^need_file (\S+)", text, re.MULTILINE):
            if target.endswith(".py") and not (ROOT / target).exists():
                problems.append(f"{job.name}: waits for code that does not exist: {target}")
        for script in re.findall(r"(?:bash|python3?) (scripts/\S+\.(?:sh|py))", text):
            if not (ROOT / script).exists():
                problems.append(f"{job.name}: script {script} does not exist")
    return problems


def check() -> tuple[list[str], list[str], dict[str, int]]:
    text = CHECKLIST.read_text()
    problems, loose = [], []
    existing = {p.name.split("_", 1)[0] for p in JOBS.glob("*.sh")}
    for jid in sorted(job_ids(text)):
        if jid not in existing:
            problems.append(f"checklist mentions job {jid} but queue/jobs has no such job")
    for token in sorted(code_refs(text)):
        if not resolve(token):
            problems.append(f"checklist mentions `{token}` but no such file exists")
    problems += job_problems()
    counts = {"done": 0, "running": 0, "open": 0}
    for status, body, line in items(text):
        counts["done" if status == "x" else "running" if status == "~" else "open"] += 1
        if status != "x" and not job_ids(body) and not assignee(body):
            loose.append(f"line {line}: {body[:110]}")
    return problems, loose, counts


def assignee(body: str) -> str | None:
    """The tag of an open item that has no queue job (a standing rule, a step after the gate, work by hand), if it has one."""
    return next((name for name, tag in ASSIGNEES.items() if tag in body), None)


def job_state(jid: str) -> str:
    state = ROOT / "queue" / "state"
    names = [p.name for p in JOBS.glob(f"{jid}_*.sh")]
    if not names:
        return "missing"
    name = names[0][:-3]
    current = (state / "current").read_text().strip() if (state / "current").exists() else ""
    if name == current:
        return "running"
    if (state / "done" / name).exists():
        return "done"
    if (state / "failed" / name).exists():
        return "failed"
    return "queued"


def title(body: str) -> str:
    m = re.search(r"\*\*(.+?)\*\*", body)
    return (m.group(1) if m else body)[:95]


def status_report() -> dict[str, list[str]]:
    """Checklist items grouped by what is happening to them: done, running, queued, failed, no queue job."""
    groups: dict[str, list[str]] = {"done": [], "running": [], "queued": [], "failed": [], "jobs finished, item not ticked": [], "standing rule": [], "after the gate": [], "written by hand": [], "no queue job": []}
    for mark, body, line in items(CHECKLIST.read_text()):
        jobs = sorted(job_ids(body))
        states = {j: job_state(j) for j in jobs}
        label = f"{title(body)}" + (f"  [{', '.join(f'{j} {st}' for j, st in states.items())}]" if jobs else "")
        if mark == "x":
            groups["done"].append(label)
        elif "running" in states.values():
            groups["running"].append(label)
        elif "failed" in states.values():
            groups["failed"].append(label)
        elif jobs and any(st in ("queued", "missing") for st in states.values()):
            groups["queued"].append(label)
        elif jobs:
            groups["jobs finished, item not ticked"].append(label)
        elif assignee(body):
            groups[assignee(body)].append(label)
        else:
            groups["no queue job"].append(label)
    return groups


def main() -> int:
    if "--status" in sys.argv:
        for group, rows in status_report().items():
            print(f"\n== {group} ({len(rows)})")
            for row in rows:
                print("  -", row)
        return 0
    problems, loose, counts = check()
    print(f"checklist items: {counts['done']} done, {counts['running']} running, {counts['open']} open")
    print(f"jobs in the queue: {len(list(JOBS.glob('*.sh')))}")
    for p in problems:
        print("PROBLEM:", p)
    for item in loose:
        print("NO QUEUE JOB:", item)
    print("OK: every promise points at something that exists" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

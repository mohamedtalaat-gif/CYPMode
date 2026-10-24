# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The last gate before a submission: one command that says whether everything is finished, consistent and fresh.

``python -m cypmode.audit.pre_submission`` runs after the whole queue, including the final chain (audit, method report, figures, notebook, film, ``submissions/`` sync), and answers
with a table of checks and one verdict. It never submits or pushes anything; "ready" means that nothing blocks the submission.

The checks are of four kinds:
  * state:       the queue has no job left, waiting or failed; the checklist has no open item except the standing rules and the steps that follow the gate; the working tree is committed and the remote is at HEAD;
  * evidence:    the automated audit has no failed check and is newer than the final outputs; the unit tests it ran passed;
  * deliverables: the TDI candidate files and the structure zip in ``submissions/`` pass the portal rules and are byte-identical to the final outputs; every submitted structure passes the
                  scorer's own PoseBusters call with the margin floor;
  * freshness:   the method report, the README, the architecture document, the storyboard, the figure catalog, the notebook (and its HTML and PDF exports) and the film are newer than the last final output they describe.
Exit status 0 when nothing fails (warnings are listed and do not fail), 1 otherwise.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEADLINE = datetime(2026, 11, 3, 23, 59, tzinfo=timezone.utc)
FINAL_TDI = ROOT / "data/tdi_track/outputs/final"
FINAL_STRUCTURE = ROOT / "data/structure_track/outputs/cyp3a4/final"
SUBMISSIONS = ROOT / "submissions"
AUDIT = ROOT / "audit/AUDIT_REPORT.md"
PASS, FAIL, WARN = "PASS", "FAIL", "WARN"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def newest(paths) -> float:
    return max((p.stat().st_mtime for p in paths if p.exists()), default=0.0)


def source_files() -> list[Path]:
    return [p for p in (ROOT / "cypmode").rglob("*.py") if "__pycache__" not in p.parts] + [ROOT / "queue/lib.sh"] + sorted((ROOT / "queue/jobs").glob("*.sh"))


def final_outputs() -> list[Path]:
    return sorted(FINAL_TDI.glob("candidate_*.csv")) + [FINAL_TDI / "manifest.json", FINAL_STRUCTURE / "structures_final.zip"]


def stamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime("%m-%d %H:%M")


def run_checks() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []

    def add(name: str, ok: bool | None, detail: str = "", warn_only: bool = False) -> None:
        rows.append((name, PASS if ok else (WARN if warn_only or ok is None else FAIL), detail))

    # -- state ------------------------------------------------------------------------------------------------------------------------------------------
    jobs = sorted(p.stem for p in (ROOT / "queue/jobs").glob("*.sh"))
    done = {p.name for p in (ROOT / "queue/state/done").iterdir()} if (ROOT / "queue/state/done").exists() else set()
    failed = sorted(p.name for p in (ROOT / "queue/state/failed").iterdir()) if (ROOT / "queue/state/failed").exists() else []
    waiting = [j for j in jobs if j not in done and j not in failed and j != "960_pre_submission_gate"]  # the gate is itself the last job
    add("the queue has no waiting, running or failed job", not waiting and not failed, f"{len(done)} of {len(jobs)} done" + (f"; waiting: {waiting[:4]}" if waiting else "") + (f"; failed: {failed}" if failed else ""))

    from cypmode.audit.checklist import assignee, check, items

    problems, loose, counts = check()
    add("the checklist resolves and every open item has a job or a tag", not problems and not loose, f"{counts['done']} done, {counts['open']} open; {len(problems)} problems, {len(loose)} without a tag")
    text = (ROOT / "SUBMISSION_CHECKLIST.md").read_text()
    blocking = [body[:80] for mark, body, _ in items(text) if mark != "x" and assignee(body) not in ("standing rule", "after the gate")]
    add("no open checklist item is left except the standing rules and the steps after the gate", not blocking, f"{len(blocking)} open: " + "; ".join(blocking[:3]))

    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.splitlines()
    add("the working tree is committed", not dirty, f"{len(dirty)} modified or untracked entries")
    mirror = subprocess.run(["bash", "scripts/mirror_check.sh"], cwd=ROOT, capture_output=True, text=True, check=False)
    add("the remote branch, HEAD and the working tree are identical", mirror.returncode == 0, (mirror.stdout + mirror.stderr).strip().splitlines()[0][:100] if (mirror.stdout + mirror.stderr).strip() else "")

    # -- evidence ---------------------------------------------------------------------------------------------------------------------------------------
    if AUDIT.exists():
        report = AUDIT.read_text()
        counts_line = re.search(r"Counts: PASS = (\d+), FAIL = (\d+)", report)
        fails = int(counts_line.group(2)) if counts_line else -1
        add("the automated audit has no failed check", fails == 0, f"{counts_line.group(0) if counts_line else 'no counts found'}")
        failing = re.findall(r"^\| (.+?) \| FAIL \|", report, re.MULTILINE)
        if failing:
            rows.append(("   failed checks of the audit", WARN, "; ".join(f[:70] for f in failing[:4])))
        add("the audit is newer than every final output and every source file", AUDIT.stat().st_mtime >= max(newest(final_outputs()), newest(source_files())), f"audit {stamp(AUDIT)}")
        tests = re.search(r"all tests pass \| (PASS|FAIL) \| (.+?) \|", report)
        add("the unit tests passed in that audit", bool(tests and tests.group(1) == "PASS"), tests.group(2) if tests else "not found")
    else:
        add("the automated audit exists", False, "audit/AUDIT_REPORT.md is missing")

    # -- deliverables -----------------------------------------------------------------------------------------------------------------------------------
    from cypmode.challenge.submission import (
        portal_validate_classification,
        portal_validate_structure,
    )

    manifest = json.loads((FINAL_TDI / "manifest.json").read_text()) if (FINAL_TDI / "manifest.json").exists() else {}
    for candidate in sorted(FINAL_TDI.glob("candidate_*.csv")):
        theirs = SUBMISSIONS / "tdi_track" / f"tdi_track_{candidate.name}"
        ok, message = portal_validate_classification(theirs) if theirs.exists() else (False, "missing in submissions/")
        same = theirs.exists() and sha256(theirs) == sha256(candidate)
        add(f"{theirs.name}: portal rules and byte-identical to the final output", ok and same, message if not ok else ("identical" if same else "differs from the final output"))
    add("the final manifest lists the candidate files", bool(manifest.get("files")), f"{len(manifest.get('files', []))} files")
    zip_final, zip_sub = FINAL_STRUCTURE / "structures_final.zip", SUBMISSIONS / "structure_track" / "structure_track_submission.zip"
    if zip_final.exists() and zip_sub.exists():
        ok, message = portal_validate_structure(zip_sub)
        add("the submitted structure zip passes the portal rules and equals the final zip", ok and sha256(zip_sub) == sha256(zip_final), message if not ok else ("identical" if sha256(zip_sub) == sha256(zip_final) else "differs from the final zip"))
        try:
            import pandas as pd

            from cypmode.validation.official_posebusters import MARGIN_FLOOR, check_zip

            official = pd.read_csv(ROOT / "data/structure_track/inputs/structure_track_test_blinded.csv")
            reports = check_zip(zip_sub, dict(zip(official.Molecule_Name, official.SMILES, strict=True)))
            smallest = min((r.distance_margin for r in reports.values()), default=float("nan"))
            add("every submitted structure passes the scorer's own PoseBusters call with the margin floor", len(reports) == 20 and all(r.safe for r in reports.values()), f"{sum(r.passed for r in reports.values())} of {len(reports)} pass; smallest relative distance {smallest:.3f} (floor {MARGIN_FLOOR})")
            from cypmode.validation.pose_stereo import zip_reproduces_stereo

            stereo = zip_reproduces_stereo(zip_sub, dict(zip(official.Molecule_Name, official.SMILES, strict=True)))
            add("every submitted structure carries the stereochemistry of the official SMILES", len(stereo) == 20 and all(v is True for v in stereo.values()),
                f"{sum(v is True for v in stereo.values())} of {len(stereo)}" + ("" if all(v is True for v in stereo.values()) else f"; not: {sorted(n for n, v in stereo.items() if v is not True)}"))
        except ImportError:
            rows.append(("scorer-exact PoseBusters check", WARN, "posebusters is not installed"))
    else:
        add("the structure zips exist (final and submitted)", False, f"final {zip_final.exists()}, submitted {zip_sub.exists()}")

    # -- freshness --------------------------------------------------------------------------------------------------------------------------------------
    last = newest(final_outputs())
    for label, path in (("METHOD_REPORT.md", ROOT / "METHOD_REPORT.md"), ("README.md", ROOT / "README.md"), ("CYPMode_Architecture_Specification.md", ROOT / "CYPMode_Architecture_Specification.md"),
                        ("docs/video/STORYBOARD.md", ROOT / "docs/video/STORYBOARD.md"), ("docs/figures/CATALOG.md", ROOT / "docs/figures/CATALOG.md")):
        add(f"{label} is newer than the last final output", path.exists() and path.stat().st_mtime >= last, f"{stamp(path)} against {datetime.fromtimestamp(last, tz=timezone.utc).strftime('%m-%d %H:%M')}" if path.exists() else "missing")
    notebook = ROOT / "notebooks"
    exports = [notebook / "cypmode_notebook.ipynb", notebook / "cypmode_notebook.html", notebook / "cypmode_notebook.pdf"]
    add("the notebook and its HTML and PDF exports exist and are newer than the last final output and the last source change", all(p.exists() for p in exports) and min(p.stat().st_mtime for p in exports) >= max(last, newest(source_files())), ", ".join(f"{p.suffix} {stamp(p)}" for p in exports if p.exists()))
    film = ROOT / "docs/video/cypmode_overview.mp4"
    add("the film is newer than the last final output", film.exists() and film.stat().st_mtime >= last, stamp(film) if film.exists() else "missing")

    from cypmode.audit.artifacts import verify

    for name, status, detail in verify(deep="--shallow" not in sys.argv):  # the card, the poster, the film and the figures against the final numbers (the deep checks recompute their sources)
        rows.append((name, status, detail))

    days = (DEADLINE - datetime.now(timezone.utc)).total_seconds() / 86400
    rows.append(("time to the deadline (2026-11-03 23:59 UTC)", PASS if days > 1 else WARN, f"{days:.1f} days"))
    return rows


def main() -> int:
    rows = run_checks()
    width = max(len(r[0]) for r in rows)
    for name, status, detail in rows:
        print(f"{status:4}  {name:{width}}  {detail}")
    blocking = [r for r in rows if r[1] == FAIL]
    warnings = [r for r in rows if r[1] == WARN]
    print(f"\n{len(rows) - len(blocking) - len(warnings)} passed, {len(warnings)} warnings, {len(blocking)} blocking")
    print("READY: nothing blocks the submission" if not blocking else "NOT READY: " + "; ".join(r[0] for r in blocking[:5]))
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())

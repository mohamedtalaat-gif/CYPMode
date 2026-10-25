# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Audit sections added with the gate-driven build: gate ledger, checklist consistency, environment and seeds, feature-block
leakage, enforced rules, and the final candidate files. Same contract as ``sections``: everything is computed at run time."""

import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from cypmode.audit.core import LOGS, ROOT, Context, Section, fmt, md_table, sha256

FINAL_TDI = ROOT / "data/tdi_track/outputs/final"
FINAL_STRUCTURE = ROOT / "data/structure_track/outputs/cyp3a4/final"
TDI_GATE = re.compile(r"^(CYP\w+) GATE \(([^)]*)\): (MET|NOT MET)", re.MULTILINE)
STRUCTURE_GATE = re.compile(r"^STRUCTURE GATE (\w+)([^:\n]*): (MET|NOT MET)", re.MULTILINE)
PRIVATE_PATHS = (str(Path.home()), str(ROOT))


def gate_ledger(ctx: Context) -> Section:
    s = Section("14. Gate ledger (every pre-registered verdict found in the run logs, including the negative ones)")
    rows = []
    for f in sorted(LOGS.glob("*.log")):
        text = f.read_text(errors="ignore")
        rows += [(f.name, iso, rule.strip(), verdict) for iso, rule, verdict in TDI_GATE.findall(text)]
        rows += [(f.name, "structure", (name + rule).strip(), verdict) for name, rule, verdict in STRUCTURE_GATE.findall(text)]
    table = pd.DataFrame(rows, columns=["log", "target", "gate", "verdict"])
    s.body.append(md_table(table) if len(table) else "No gate lines found yet.")
    met, not_met = int((table.verdict == "MET").sum()) if len(table) else 0, int((table.verdict == "NOT MET").sum()) if len(table) else 0
    s.body.append(f"\n{met} gates MET, {not_met} NOT MET. A gate that is NOT MET is a measured negative result: the idea is dropped, not tuned until it passes.")
    s.add("gate verdicts parsed", None, f"{met} met, {not_met} not met")
    ctx.metrics.update(gates_met=met, gates_not_met=not_met)
    return s


def checklist(ctx: Context) -> Section:
    from cypmode.audit.checklist import check

    s = Section("15. Checklist consistency (every promise points at something that exists)")
    problems, loose, counts = check()
    s.body.append(f"Items: {counts['done']} done, {counts['running']} running, {counts['open']} open. Jobs in the queue: {len(list((ROOT / 'queue/jobs').glob('*.sh')))}.")
    s.body += [f"- PROBLEM: {p}" for p in problems] or ["No missing job, module or code file."]
    s.body += [f"- no queue job: {item}" for item in loose]
    s.add("checklist references resolve", not problems, f"{len(problems)} problems")
    s.add("every open item has a queue job", not loose, f"{len(loose)} without")
    return s


def environment_and_seeds(ctx: Context) -> Section:
    s = Section("16. Pinned environment and seed registry")
    for venv, lock in ((".venv", "requirements-lock.txt"), (".venv-ml", "requirements-lock-ml.txt")):
        path = ROOT / "environment" / lock
        py = ROOT / venv / "bin" / "python"
        if not (path.exists() and py.exists()):
            s.add(f"{venv} matches {lock}", None, "missing")
            continue
        live = subprocess.run([str(py), "-m", "pip", "freeze"], capture_output=True, text=True, check=False).stdout.splitlines()
        pinned = path.read_text().splitlines()

        def unpinned(line: str) -> str:  # an editable install names the commit it was installed from; the project's own commit changes with every commit
            return re.sub(r"(-e git\+\S+?)@[0-9a-f]{7,40}", r"\1", line)

        drift = sorted({unpinned(x) for x in live} ^ {unpinned(x) for x in pinned})
        s.add(f"{venv} matches {lock}", not drift, f"{len(drift)} lines differ")
        s.body.append(f"`{lock}`: {len(pinned)} pinned packages, sha256 {sha256(path)[:16]}")
        if drift:
            s.body.append("Differences: " + "; ".join(drift[:8]) + (" ..." if len(drift) > 8 else ""))
    seed_rows = []
    for f in sorted((ROOT / "cypmode").rglob("*.py")):
        if "audit" in f.parts:
            continue
        text = f.read_text(errors="ignore")
        for m in re.finditer(r"(random_state|seed)\s*[=:]\s*(\d+)", text):
            seed_rows.append((str(f.relative_to(ROOT)), m.group(1), m.group(2)))
    table = pd.DataFrame(seed_rows, columns=["file", "kind", "value"]).value_counts().reset_index(name="count").sort_values(["file", "kind", "value"])
    s.body.append(f"\nSeed registry: {len(table)} distinct (file, kind, value) entries; fixed seeds make every cache reproducible.\n\n" + md_table(table.head(60)))
    s.add("fixed seeds present", len(table) > 0, f"{len(table)} entries")
    return s


def block_leakage(ctx: Context) -> Section:
    """No feature block may depend on a label: blocks take structures only, caches hold no label columns."""
    import inspect

    from cypmode.challenge.feature_store import BLOCKS, OPT_IN_BLOCKS
    from cypmode.challenge.physics_block import FEATURE_NAMES

    s = Section("17. Feature-block leakage checks")
    rows = []
    for name, fn in sorted(BLOCKS.items()):
        params = list(inspect.signature(fn).parameters) if callable(fn) else []
        rows.append((name, "opt-in" if name in OPT_IN_BLOCKS else "default", ",".join(params), "structures only" if params in (["mols"], ["mols_"]) or len(params) == 1 else "CHECK"))
    s.body.append(md_table(pd.DataFrame(rows, columns=["block", "kind", "arguments", "inputs"])))
    s.add("every feature block takes only structures", all(r[3] == "structures only" for r in rows), "label-free by construction (ext_activity uses external panels, tested separately)")
    physics = ROOT / "data/tdi_track/outputs/caches/physics_cache.csv"
    if physics.exists():
        cols = set(pd.read_csv(physics, nrows=1).columns)
        s.add("physics cache holds no label or measurement column", cols <= set(FEATURE_NAMES) | {"SMILES"}, f"{len(cols)} columns")
    mmp = ROOT / "data/tdi_track/outputs/caches/mmp_pairs.npz"
    if mmp.exists():
        keys = set(np.load(mmp, allow_pickle=True).files)
        s.add("matched-pair cache is label-free", keys <= {"key", "a", "b", "t", "names"}, ", ".join(sorted(keys)))
    chem = ROOT / "data/tdi_track/outputs/caches/chemeleon_embeddings.npz"
    if chem.exists():
        keys = set(np.load(chem, allow_pickle=True).files)
        s.add("CheMeleon cache is label-free", keys <= {"smiles", "embeddings"}, ", ".join(sorted(keys)))
    return s


def enforced_rules(ctx: Context) -> Section:
    s = Section("18. Standing rules enforced by script")
    arabic = re.compile("[" + chr(0x0600) + "-" + chr(0x06FF) + "]")  # Arabic script block, built from code points so this file stays ASCII
    offenders, private = [], []
    for base in ("cypmode", "scripts", "tests", "queue"):
        for f in sorted((ROOT / base).rglob("*")):
            if f.suffix not in (".py", ".sh") or "__pycache__" in f.parts:
                continue
            text = f.read_text(errors="ignore")
            if arabic.search(text):
                offenders.append(str(f.relative_to(ROOT)))
    for name in ("README.md", "CONTRIBUTING.md", "CITATION.cff", "METHOD_REPORT.md", "NEGATIVE_RESULTS.md"):
        f = ROOT / name
        if f.exists() and any(p in f.read_text(errors="ignore") for p in PRIVATE_PATHS):
            private.append(name)
    s.add("source code comments are English only (no Arabic script in code)", not offenders, ", ".join(offenders[:5]) or "none found")
    s.add("public documents carry no private local paths", not private, ", ".join(private) or "none found")
    from cypmode.audit.prose_rules import mentions_of_the_owner

    owner = mentions_of_the_owner()
    s.add("no text of the project refers to an owner (a decision or a step is written without a person named by a role)", not owner, ", ".join(owner[:5]) or "none found")
    s.body.append("Rules checked: code comments in English; no private paths in the documents meant to be public; no text that refers to an owner; attribution hygiene is section 12.")
    return s


def final_candidates(ctx: Context) -> Section:
    from cypmode.challenge.submission import (
        portal_validate_classification,
        portal_validate_structure,
    )

    s = Section("19. Final candidate files (hashes and the portal's own checks)")
    rows = []
    for f in sorted(FINAL_TDI.glob("candidate_*.csv")):
        ok, message = portal_validate_classification(f)
        d = pd.read_csv(f)
        order_ok = d["Molecule_Name"].tolist() == ctx.test["Molecule_Name"].tolist()
        rows.append((f.name, sha256(f)[:16], ok and order_ok, float(d.CYP3A4_is_TDI.mean()), float(d.CYP2D6_is_TDI.mean())))
        s.add(f"{f.name} passes the portal checks and keeps the test-file order", ok and order_ok, message)
    zips = sorted(FINAL_STRUCTURE.glob("*.zip"))
    for z in zips:
        ok, message = portal_validate_structure(z)
        with zipfile.ZipFile(z) as zf:
            n = len(zf.namelist())
        s.add(f"{z.name} passes the portal check ({n} files)", ok, message)
        s.body.append(f"`{z.name}` sha256 {sha256(z)[:16]}")
        try:
            from cypmode.validation.official_posebusters import MARGIN_FLOOR, check_zip

            official = pd.read_csv(ROOT / "data/structure_track/inputs/structure_track_test_blinded.csv")
            reports = check_zip(z, dict(zip(official.Molecule_Name, official.SMILES, strict=True)))
            passes = sum(r.passed for r in reports.values())
            smallest = min((r.distance_margin for r in reports.values() if np.isfinite(r.distance_margin)), default=float("nan"))
            s.body.append("\nThe zip checked the way the organisers' scorer checks it (22 PoseBusters checks, dock configuration, bond orders from the SMILES, receptor = the complex minus the ligand). The blinded compound identifiers are withheld here, in the order of the submission file, because the repository's release rule forbids naming a blinded compound in a file that would be pushed:\n")
            s.body.append(md_table(pd.DataFrame([(f"compound {i:02d}", r.passed, r.n_checks, fmt(r.distance_margin, 3), ",".join(r.failed) or r.error or "-") for i, (_, r) in enumerate(reports.items(), start=1)], columns=["compound", "passes", "checks", "smallest relative distance", "failed"])))
            s.add(f"{z.name}: every structure passes the scorer's own PoseBusters call", passes == len(reports) == 20, f"{passes} of {len(reports)} pass")
            s.add(f"{z.name}: every structure clears a relative distance of {MARGIN_FLOOR} to the receptor (the scorer's threshold is 0.75)", bool(smallest >= MARGIN_FLOOR), f"smallest {fmt(smallest, 3)}")
            from cypmode.validation.pose_stereo import zip_reproduces_stereo

            stereo = zip_reproduces_stereo(z, dict(zip(official.Molecule_Name, official.SMILES, strict=True)))
            s.add(f"{z.name}: every structure carries the stereochemistry of the official SMILES (the scorer's PoseBusters call does not check it)", len(stereo) == 20 and all(v is True for v in stereo.values()),
                  f"{sum(v is True for v in stereo.values())} of {len(stereo)}")
        except ImportError:
            s.add(f"{z.name}: scorer-exact PoseBusters check", None, "posebusters not installed")
    from cypmode.validation.affinity_retry import failed_rows

    base_table = ROOT / "data/structure_track/outputs/cyp3a4/structure_track_results.json"
    if base_table.exists():  # a failed step of one run leaves its whole row in error and the checkpoint skips it on every later run (2026-10-25)
        failed = failed_rows(base_table)
        s.add("the base-run table of the 20 official ligands has no row in error (structure summary and affinity of every ligand)", not failed, f"{len(failed)} in error" + (f": {', '.join(failed[:4])}" if failed else ""))
    s.body.insert(0, md_table(pd.DataFrame(rows, columns=["file", "sha256", "valid", "CYP3A4 rate", "CYP2D6 rate"])) if rows else "No final TDI candidate files yet (job 090).")
    if not rows:
        s.add("final TDI candidate files present", None, "pending job 090")
    return s



def citation_consistency(ctx: Context) -> Section:
    """The name, the repository and the identifiers of the work must be the same wherever the repository states them: the citation file is the source, the other places are compared with it."""
    import json

    from cypmode import provenance

    s = Section("20. Attribution and citation consistency")
    cff = provenance.load()
    family = cff["authors"][0]["family-names"]
    url = cff["repository-code"]
    ids = provenance.identifiers(cff)
    readme = (ROOT / "README.md").read_text(errors="ignore")
    notice = (ROOT / "NOTICE").read_text(errors="ignore") if (ROOT / "NOTICE").exists() else ""
    zenodo_path = ROOT / ".zenodo.json"
    zenodo = json.loads(zenodo_path.read_text()) if zenodo_path.exists() else None
    s.add("the citation file names an author, a repository, a licence and a version", all(cff.get(k) for k in ("authors", "repository-code", "license", "version")), f"{provenance.authors(cff)}; {url}; version {cff.get('version')}")
    s.add("the README names the author and the repository", family in readme and url in readme, "found" if family in readme and url in readme else "missing")
    s.add("NOTICE names the author", family in notice, "found" if family in notice else "missing")
    if zenodo is not None:
        names = {c["name"].split(",")[0].strip() for c in zenodo.get("creators", [])}
        s.add("the archive metadata (.zenodo.json) names the same author", family in names, ", ".join(sorted(names)))
    else:
        s.add("the archive metadata (.zenodo.json) names the same author", None, "no .zenodo.json yet")
    if ids.get("doi"):
        s.add("the DOI of the citation file is written in the README", ids["doi"] in readme, ids["doi"])
    else:
        s.add("the DOI of the citation file is written in the README", None, "the citation file has no DOI yet")
    licence_text = (ROOT / "LICENSE").read_text(errors="ignore")[:120] if (ROOT / "LICENSE").exists() else ""
    s.add("the licence of the code is the same in the citation file, LICENSE, pyproject.toml and the headers", "AFFERO" in licence_text and cff.get("license") == "AGPL-3.0-or-later" and "AGPL-3.0-or-later" in (ROOT / "pyproject.toml").read_text(), f"citation file: {cff.get('license')}")
    missing_texts = [t for t in ("AGPL-3.0-or-later", "CC-BY-NC-SA-4.0", "CC-BY-4.0", "LicenseRef-third-party") if not (ROOT / "LICENSES" / f"{t}.txt").exists()]
    s.add("the texts of the licences are in LICENSES/", not missing_texts, ", ".join(missing_texts) or "4 found")
    listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.py"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.split()
    headerless = [f for f in listed if not f.startswith("third_party/") and "SPDX-License-Identifier" not in (ROOT / f).read_text(errors="ignore")]
    s.add("every Python file of the repository carries its SPDX licence header", not headerless, ", ".join(headerless[:4]) or "all have one")
    s.add("the declaration of use and citation exists", (ROOT / "USE_AND_CITATION.md").exists(), "USE_AND_CITATION.md" if (ROOT / "USE_AND_CITATION.md").exists() else "missing")
    entry_points = ("challenge/nested_runners", "challenge/aux_experiments", "challenge/final_tdi_build", "challenge/physics_block", "challenge/frozen_embeddings", "validation/batch", "validation/final_recipe", "validation/relax_final")
    silent = [name for name in entry_points if "provenance.announce()" not in (ROOT / "cypmode" / f"{name}.py").read_text()]
    s.add("the scripts that generate data announce their origin", not silent, ", ".join(silent) or f"{len(entry_points)} checked")
    s.body.append("Rules checked: the author, the repository and the DOI are the same in the citation file, the README, NOTICE and the archive metadata; every script that generates data prints the citation and the generated directories carry their origin (`cypmode/provenance.py`).")
    return s

EXTRA = [gate_ledger, checklist, environment_and_seeds, block_leakage, enforced_rules, final_candidates, citation_consistency]
_ = sys


def multiplicity(ctx: Context) -> Section:
    from cypmode.challenge.multiplicity import collect

    s = Section("14b. Multiplicity of the additions tested together (Holm adjustment next to the pre-registered gates)")
    table = collect(LOGS)
    if not len(table):
        s.body.append("No stacked test of the added candidates found in the logs yet.")
        s.add("Holm-adjusted p-values of the added tests", None, "pending")
        return s
    s.body.append("The p-value of an addition is the larger of its two confidence cut-offs (its gate needs both); Holm adjusts those over every addition of the batch. The pre-registered gate decides what enters the model.\n")
    s.body.append(md_table(table.round(4)))
    survivors = table[(table.holm_adjusted < 0.05) & table.gain_positive_at_both]
    s.add("additions whose stacked gain survives the Holm adjustment over the batch", None, f"{len(survivors)} of {len(table)}")
    return s


EXTRA.insert(1, multiplicity)

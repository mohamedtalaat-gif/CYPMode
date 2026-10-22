# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Force-field relaxation of the 20 submitted poses, under a rule that was registered before any official pose was relaxed.

``relax_gate`` tested the relaxation of ``relax`` on 90 poses of 30 ligands whose crystal pose is public and found it MET (validity, margin and fidelity). This module applies it to the 20 poses
that ``final_recipe`` chose (``final/_pdb_files``, which it never modifies), and rebuilds the final zip from the relaxed poses that pass the rule below.

RULE (registered 2026-10-16, before any of the 20 poses was relaxed; the two bounds come from the validation poses of ``relax_gate`` and not from the official ones). A relaxed pose replaces the
original only when all of these hold; otherwise the original stays:

  1. the original and the relaxed pose both pass the scorer's 22 PoseBusters checks (``official_posebusters``);
  2. the smallest relative distance to the receptor rises, and the relaxed pose is at or above the margin floor (``MARGIN_FLOOR``);
  3. the ligand moved by at most ``MAX_MOVEMENT_A`` (validation poses: median 0.46 A, largest 0.95 A);
  4. when a ligand nitrogen coordinates the iron in the original (closer than 3.0 A), it still does and the distance changed by at most ``MAX_FE_N_CHANGE_A`` (validation poses: largest change 0.13 A).

The validation gate also required that the LDDT-PLI to the crystal does not fall by more than 0.01. That cannot be measured for a blinded ligand, which has no crystal. The gate
established it on average (mean change +0.0035 over the validation poses, fifth percentile -0.017, worst -0.032), and criterion 3 is the per-pose guard that stands in for it here.

``python -m cypmode.validation.relax_final`` writes ``final/relaxation/`` (the relaxed poses, ``decisions.csv`` with one row per ligand and the reason, ``stamp.json``), ``final/structures_unrelaxed.zip``
(the recipe's zip, for rollback) and the new ``final/structures_final.zip``, checks the candidate zip with the portal's rules and the scorer's own PoseBusters call before it replaces anything,
and records the relaxation in ``final/manifest.json``. It is idempotent: a relaxation is reused only while the original pose is byte-identical.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

from cypmode.challenge.structure_submission import write_deterministic_zip
from cypmode.validation.official_posebusters import MARGIN_FLOOR

OUT = Path("data/structure_track/outputs/cyp3a4")
FINAL = OUT / "final"
WORK = FINAL / "relaxation"
OFFICIAL = Path("data/structure_track/inputs/structure_track_test_blinded.csv")
GATE_LOG = OUT / "logs" / "relaxation_gate.log"
MAX_MOVEMENT_A = 1.0
MAX_FE_N_CHANGE_A = 0.2
GATE_LINE = re.compile(r"^STRUCTURE GATE force_field_relaxation[^:\n]*: (MET|NOT MET)", re.MULTILINE)


def gate_met(log: Path = GATE_LOG) -> bool:
    """The verdict of ``relax_gate`` (the relaxation is used only when it is MET)."""
    found = GATE_LINE.findall(log.read_text()) if log.exists() else []
    return bool(found) and found[-1] == "MET"


def adopt(before: dict, after: dict, info: dict) -> tuple[bool, str]:
    """The registered rule: ``before`` and ``after`` hold ``passed`` and ``margin`` of the original and the relaxed pose, ``info`` the record that ``relax`` wrote (``rmsd``, ``fe_n_*``, ``error``)."""
    if info.get("error"):
        return False, f"the relaxation failed: {info['error']}"
    if not before["passed"]:
        return False, "the original pose does not pass the checks"
    if not after["passed"]:
        return False, "the relaxed pose fails a check"
    if not after["margin"] > before["margin"]:
        return False, "the margin does not rise"
    if after["margin"] < MARGIN_FLOOR:
        return False, "the relaxed pose is below the margin floor"
    if not info.get("rmsd", math.inf) <= MAX_MOVEMENT_A:
        return False, "the ligand moved too far"
    fe_before, fe_after = info.get("fe_n_before"), info.get("fe_n_after")
    if fe_before is not None and math.isfinite(fe_before) and (fe_after is None or not math.isfinite(fe_after) or abs(fe_after - fe_before) > MAX_FE_N_CHANGE_A):
        return False, "the iron-nitrogen coordination changed"
    return True, "adopted"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    from cypmode.challenge.submission import portal_validate_structure
    from cypmode.validation import relax_gate
    from cypmode.validation.official_posebusters import check_pdb, check_zip

    if not gate_met(GATE_LOG):
        print("STRUCTURE RELAXATION not used: the gate force_field_relaxation is not MET")
        return 0
    official = pd.read_csv(OFFICIAL)
    names = official.Molecule_Name.tolist()
    smiles = dict(zip(official.Molecule_Name, official.SMILES, strict=True))
    originals = {n: FINAL / "_pdb_files" / f"{n}.pdb" for n in names}
    missing = [n for n, p in originals.items() if not p.exists()]
    if missing:
        print(f"the recipe's poses are missing for {missing}")
        return 1
    jobs = []
    for n in names:  # a relaxed pose is reused only while its original is byte-identical
        out = WORK / "relaxed" / f"{n}.pdb"
        marker = Path(str(out) + ".source")
        if out.exists() and (not marker.exists() or marker.read_text() != sha256(originals[n])):
            for stale in (out, Path(str(out) + ".json"), marker):
                stale.unlink(missing_ok=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        jobs.append({"pdb": str(originals[n]), "smiles": smiles[n], "out": str(out)})
    relax_gate.relax_all(jobs)
    rows = []
    for n, job in zip(names, jobs, strict=True):
        out = Path(job["out"])
        info_path = Path(job["out"] + ".json")
        info = json.loads(info_path.read_text()) if info_path.exists() else {"error": "no result"}
        if out.exists():
            Path(job["out"] + ".source").write_text(sha256(originals[n]))
            after_report = check_pdb(out, smiles[n])
            after = {"passed": after_report.passed, "margin": after_report.distance_margin, "failed": ";".join(after_report.failed) or after_report.error}
        else:
            after = {"passed": False, "margin": float("nan"), "failed": "no relaxed file"}
        before_report = check_pdb(originals[n], smiles[n])
        before = {"passed": before_report.passed, "margin": before_report.distance_margin}
        ok, reason = adopt(before, after, info)
        rows.append({"compound": n, "passed_before": before["passed"], "passed_after": after["passed"], "margin_before": before["margin"], "margin_after": after["margin"], "movement_A": info.get("rmsd"),
                     "fe_n_before": info.get("fe_n_before"), "fe_n_after": info.get("fe_n_after"), "adopted": ok, "reason": reason})
        print(f"{n}: pass {before['passed']} -> {after['passed']}, margin {before['margin']:.3f} -> {after['margin']:.3f}, moved {info.get('rmsd', float('nan')):.2f} A, "
              f"Fe-N {info.get('fe_n_before', float('nan')):.2f} -> {info.get('fe_n_after', float('nan')):.2f}: {reason}", flush=True)
    decisions = pd.DataFrame(rows)
    decisions.to_csv(WORK / "decisions.csv", index=False)

    unrelaxed = write_deterministic_zip([originals[n] for n in names], FINAL / "structures_unrelaxed.zip")
    chosen = [WORK / "relaxed" / f"{n}.pdb" if row.adopted else originals[n] for n, row in zip(names, decisions.itertuples(), strict=True)]
    candidate = write_deterministic_zip(chosen, WORK / "structures_candidate.zip", names=[f"{n}.pdb" for n in names])
    ok, message = portal_validate_structure(candidate)
    reports = check_zip(candidate, smiles)
    before_reports = check_zip(unrelaxed, smiles)
    smallest = min(r.distance_margin for r in reports.values())
    smallest_before = min(r.distance_margin for r in before_reports.values())
    if not (ok and len(reports) == len(names) and all(r.passed for r in reports.values()) and smallest >= smallest_before):
        print(f"the candidate zip is rejected (portal: {ok} {message}; all pass: {all(r.passed for r in reports.values())}; smallest margin {smallest:.3f} against {smallest_before:.3f}); nothing was replaced")
        return 1
    final = FINAL / "structures_final.zip"
    stamp_path = WORK / "stamp.json"
    stamp = json.loads(stamp_path.read_text()) if stamp_path.exists() else {}
    allowed = {sha256(unrelaxed), stamp.get("final_sha256")}
    if final.exists() and sha256(final) not in allowed:
        print("structures_final.zip is neither the recipe's zip nor the one written by an earlier run of this module; nothing was replaced")
        return 1
    shutil.copyfile(candidate, final)
    adopted = int(decisions.adopted.sum())
    summary = {"adopted": adopted, "of": len(names), "rule": {"max_movement_A": MAX_MOVEMENT_A, "max_fe_n_change_A": MAX_FE_N_CHANGE_A, "margin_floor": MARGIN_FLOOR},
               "unrelaxed_sha256": sha256(unrelaxed), "final_sha256": sha256(final), "smallest_margin_before": smallest_before, "smallest_margin_after": smallest}
    stamp_path.write_text(json.dumps(summary, indent=2))
    manifest_path = FINAL / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["recipe_zip_sha256"] = summary["unrelaxed_sha256"]
        manifest["sha256"] = summary["final_sha256"]
        manifest["relaxation"] = summary
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    moved = decisions[decisions.adopted]
    print(f"STRUCTURE RELAXATION applied to {adopted} of {len(names)} poses (rule of relax_final.py); smallest relative distance {smallest_before:.3f} -> {smallest:.3f}; "
          f"mean movement of the adopted poses {moved.movement_A.mean() if len(moved) else float('nan'):.2f} A; zip sha256 {summary['final_sha256'][:16]}")
    for reason, count in decisions[~decisions.adopted].reason.value_counts().items():
        print(f"  not adopted ({count}): {reason}")
    return 0


if __name__ == "__main__":
    from cypmode import provenance

    provenance.announce()
    sys.exit(main())

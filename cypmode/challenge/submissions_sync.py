"""Brings ``submissions/`` to the state of the final results: the files that would be uploaded to the challenge portal, each checked with the portal's own rules, with checksums
and a README per track. Nothing is uploaded; the choice of which file to upload, and the upload, stay manual.

TDI track: every candidate file of ``data/tdi_track/outputs/final`` is copied (name prefixed with the track), re-validated with ``validate_submission`` and
``portal_validate_classification``, and described. Structure track: the final zip of ``data/structure_track/outputs/cyp3a4/final`` is copied, checked with
``portal_validate_structure`` and unpacked next to itself so the 20 complexes can be opened. Files that were in a track folder before and are not part of the new set are moved to
``submissions/_superseded`` and never deleted.

Usage: python -m cypmode.challenge.submissions_sync [--check-only]
"""

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

from cypmode.challenge.submission import (
    portal_validate_classification,
    portal_validate_structure,
    validate_submission,
)

SUBMISSIONS = Path("submissions")
FINAL_TDI = Path("data/tdi_track/outputs/final")
FINAL_STRUCTURE = Path("data/structure_track/outputs/cyp3a4/final")
POSEBUSTERS = Path("data/structure_track/outputs/cyp3a4/posebusters_validity_official20.csv")
TEST_SET = Path("data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv")
STRUCTURE_ZIP_NAME = "structure_track_submission.zip"
ARCHIVE = "_superseded"
CANDIDATE_MEANING = {
    "candidate_A_hits_only": "the stack fitted on the compounds that were measured (the scored population is assumed to be the measured compounds)",
    "candidate_B_selection_aware": "for CYP2D6, the stack fitted on the measured compounds plus the never-measured compounds of the training panel as negatives (the scored population is assumed to look like the training panel); CYP3A4 is the same as in A",
    "candidate_C_hedged_blend": "the rank average of A and B for CYP2D6, written only when its own gate was MET",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe_tdi_candidate(path: Path, test_names: set[str]) -> dict:
    frame = pd.read_csv(path)
    schema_ok, schema_errors = validate_submission(frame, test_names)
    portal_ok, portal_message = portal_validate_classification(path)
    return {
        "file": f"tdi_track_{path.stem}.csv", "source": str(path), "candidate": path.stem, "sha256": sha256(path), "rows": len(frame),
        "CYP3A4_positive_rate": float(frame["CYP3A4_is_TDI"].mean()), "CYP2D6_positive_rate": float(frame["CYP2D6_is_TDI"].mean()),
        "schema_valid": bool(schema_ok), "schema_errors": schema_errors, "portal_valid": bool(portal_ok), "portal_message": portal_message,
    }


def differing_calls(first: Path, second: Path) -> dict[str, int]:
    a, b = pd.read_csv(first), pd.read_csv(second)
    return {column: int((a[column] != b[column]).sum()) for column in ("CYP3A4_is_TDI", "CYP2D6_is_TDI")}


def render_tdi_readme(entries: list[dict], manifest: dict, differences: dict[str, int] | None) -> str:
    lines = ["# TDI track: submission files", "",
             "The files below are the candidates of the final build (`cypmode.challenge.final_tdi_build`, files and hashes in `data/tdi_track/outputs/final/manifest.json`), copied here",
             "unchanged and checked again with the portal's own rules. **None of them has been uploaded.** The portal takes one file per submission; which one to upload is decided before submitting.", "",
             "## Required format (read from the challenge portal's `config.py` and `submission.py`, 2026-09-26)", "",
             "- A `.parquet` or `.csv` file with exactly **750 rows** and exactly these 4 columns, in this order, case-sensitive: `SMILES`, `Molecule_Name`, `CYP2D6_is_TDI`, `CYP3A4_is_TDI`.",
             "- The two endpoint columns are boolean (`True`/`False` or `1`/`0`), without missing values, and neither may hold a single class.", "",
             "## Files", "", "| file | candidate | rows | CYP3A4 positive rate | CYP2D6 positive rate | schema check | portal check | sha256 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for e in entries:
        lines.append(f"| `{e['file']}` | {e['candidate']} | {e['rows']} | {e['CYP3A4_positive_rate']:.3f} | {e['CYP2D6_positive_rate']:.3f} | {'valid' if e['schema_valid'] else 'INVALID'} | "
                     f"{'valid' if e['portal_valid'] else 'INVALID: ' + e['portal_message']} | `{e['sha256']}` |")
    lines += ["", "## What the candidates are", ""]
    for e in entries:
        lines.append(f"- `{e['candidate']}`: {CANDIDATE_MEANING.get(e['candidate'], 'see the final report')}.")
    if differences:
        lines += ["", f"The first two candidates differ in {differences['CYP3A4_is_TDI']} CYP3A4 calls and {differences['CYP2D6_is_TDI']} CYP2D6 calls."]
    lines += ["", "## How they were made", ""]
    for iso in ("CYP3A4", "CYP2D6"):
        info = manifest["isoforms"][iso]
        lines.append(f"- {iso}: stack `{info['stack']}` of {info['n_candidates']} candidate models; the top {info['called_rate']:.1%} of the test compounds are called "
                     f"({info['call_rate_multiplier']:g} times the prevalence {info['target_prevalence']:.3f} that the public leaderboard implies).")
    lines += ["", "The method, every experiment that was tried and the gate each one met or missed are in the project's method report and in `data/tdi_track/outputs/final/final_report.md`.", ""]
    return "\n".join(lines)


def render_structure_readme(zip_hash: str, members: list[str], failures: int, recipe: dict, portal_message: str, relaxation: dict | None = None) -> str:
    lines = ["# Structure track: submission file", "",
             f"`{STRUCTURE_ZIP_NAME}` is the final zip of `cypmode.validation.final_recipe`{', with the relaxation of `cypmode.validation.relax_final` applied' if relaxation else ''}, copied unchanged. **It has not been uploaded.** `pdb_files/` holds the same 20 complexes, unpacked from it for inspection.", "",
             "## Required format (quoted from the portal's submission form, 2026-09-26)", "",
             "> Submit a `.zip` archive containing exactly **20 `.pdb` files**, one per compound, named after the compound identifier (e.g. `x00011-1.pdb`). Each file must be a full protein-ligand complex with the ligand residue named `LIG`.", "",
             "## Checks", "", f"- portal check (a `.zip` with exactly 20 files): {portal_message}",
             f"- PoseBusters on the 20 complexes: {failures} of {len(members)} fail any check (a failing structure scores zero on the structure metrics, so none is handed in)",
             "- the zip is built with fixed timestamps, so rebuilding it from the same predictions gives the same bytes", f"- sha256 of the zip: `{zip_hash}`", "",
             "## How the poses were chosen", ""]
    lines += [f"- {reason}" for reason in recipe.get("recipe", {}).get("reasons", [])]
    if relaxation:
        rule = relaxation["rule"]
        lines += ["", "## Relaxation of the chosen poses", "",
                  (
                      f"The ligand of each chosen pose was relaxed in its frozen pocket with a force field (`cypmode.validation.relax`), after a gate on 30 ligands with public crystal poses had found it to raise validity and margin without moving the poses away from the crystal. "
                      f"A relaxed pose replaced the original only when both pass the 22 checks, the smallest relative distance to the receptor rises and stays at or above {rule['margin_floor']}, the ligand moved by at most {rule['max_movement_A']} A and a coordinated iron-nitrogen distance changed by at most {rule['max_fe_n_change_A']} A. "
                      f"{relaxation['adopted']} of {relaxation['of']} poses were replaced; the smallest relative distance in the zip went from {relaxation['smallest_margin_before']:.3f} to {relaxation['smallest_margin_after']:.3f}. "
                      f"The zip of the unrelaxed poses has sha256 `{relaxation['unrelaxed_sha256']}`."
                  )]
    lines += ["", "## Contents", "", *[f"- `{name}`" for name in sorted(members)], ""]
    return "\n".join(lines)


def render_top_readme() -> str:
    return """# Submissions

The files that would be uploaded to the [OpenADMET CYP Inhibition Blind Challenge](https://huggingface.co/spaces/openadmet/cyp-challenge), kept apart from `data/*/outputs/`, which holds the project's
own experiments. Each track folder has its own README with the required format, the checks every file passed and the file hashes, and is rewritten by `cypmode.challenge.submissions_sync`
from the final results. **No file here has been uploaded to the portal.**

- [`tdi_track/`](tdi_track/README.md): time-dependent inhibition (classification), CYP3A4 and CYP2D6.
- [`structure_track/`](structure_track/README.md): structure prediction, 20 CYP3A4 complexes.

## Mechanics that apply to both tracks (read from the portal's `submission.py`, 2026-09-26)

- One submission per track every 12 hours; only the latest valid submission of a track counts for the leaderboard.
- Required at submission: the Hugging Face `username`, a `user_alias`, an `anonymous` flag and the track.
- Optional: `model_report_link` (a reachable HTTPS URL, required when `open_source_code` is ticked), `include_in_publication`, `used_proprietary_data`, `open_source_code` and contact fields.
- The upload itself is a manual action on the portal.

`_superseded/` (when present) holds files that earlier versions of these folders contained and that are not part of the final set.
"""


def archive_stale(folder: Path, keep: set[str]) -> list[str]:
    """Move everything in a track folder that is not in ``keep`` (and not the README) to ``submissions/_superseded/<track>/``; returns the names moved."""
    moved = []
    target = SUBMISSIONS / ARCHIVE / folder.name
    for item in sorted(folder.iterdir()):
        if item.name in keep or item.name in ("README.md", ".DS_Store"):
            continue
        target.mkdir(parents=True, exist_ok=True)
        destination = target / item.name
        if destination.exists():
            shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
        shutil.move(str(item), str(destination))
        moved.append(item.name)
    return moved


def sync_tdi(check_only: bool) -> list[dict]:
    test_names = set(pd.read_csv(TEST_SET)["Molecule_Name"])
    candidates = sorted(FINAL_TDI.glob("candidate_*.csv"))
    entries = [describe_tdi_candidate(path, test_names) for path in candidates]
    bad = [e["file"] for e in entries if not (e["schema_valid"] and e["portal_valid"])]
    if not entries or bad:
        raise SystemExit(f"TDI candidates that are missing or fail the checks: {bad or 'none found'}")
    if check_only:
        return entries
    folder = SUBMISSIONS / "tdi_track"
    folder.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        shutil.copyfile(entry["source"], folder / entry["file"])
    manifest = json.loads((FINAL_TDI / "manifest.json").read_text())
    differences = differing_calls(candidates[0], candidates[1]) if len(candidates) > 1 else None
    (folder / "README.md").write_text(render_tdi_readme(entries, manifest, differences))
    archive_stale(folder, {e["file"] for e in entries})
    return entries


def sync_structure(check_only: bool) -> dict:
    source = FINAL_STRUCTURE / "structures_final.zip"
    ok, message = portal_validate_structure(source)
    if not ok:
        raise SystemExit(f"the final structure zip fails the portal check: {message}")
    with zipfile.ZipFile(source) as archive:
        members = sorted(archive.namelist())
    failures = int((~pd.read_csv(POSEBUSTERS)["all_passed"].astype(bool)).sum())
    info = {"file": STRUCTURE_ZIP_NAME, "sha256": sha256(source), "members": len(members), "posebusters_failures": failures, "portal_message": message}
    if check_only:
        return info
    folder = SUBMISSIONS / "structure_track"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, folder / STRUCTURE_ZIP_NAME)
    pdb_dir = folder / "pdb_files"
    if pdb_dir.exists():
        shutil.rmtree(pdb_dir)
    pdb_dir.mkdir()
    with zipfile.ZipFile(source) as archive:
        for name in members:
            (pdb_dir / name).write_bytes(archive.read(name))
    recipe = json.loads((FINAL_STRUCTURE / "recipe.json").read_text())
    stamp = FINAL_STRUCTURE / "relaxation" / "stamp.json"
    relaxation = json.loads(stamp.read_text()) if stamp.exists() and json.loads(stamp.read_text()).get("final_sha256") == info["sha256"] else None
    (folder / "README.md").write_text(render_structure_readme(info["sha256"], members, failures, recipe, message, relaxation))
    archive_stale(folder, {STRUCTURE_ZIP_NAME, "pdb_files"})
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="validate the final files and print what would be written, without touching submissions/")
    args = parser.parse_args()
    entries = sync_tdi(args.check_only)
    info = sync_structure(args.check_only)
    if not args.check_only:
        (SUBMISSIONS / "README.md").write_text(render_top_readme())
        manifest = {"tdi_track": [{k: e[k] for k in ("file", "sha256", "rows", "CYP3A4_positive_rate", "CYP2D6_positive_rate")} for e in entries],
                    "structure_track": {k: info[k] for k in ("file", "sha256", "members", "posebusters_failures")}}
        (SUBMISSIONS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for e in entries:
        print(f"{e['file']}: {e['rows']} rows, schema {'valid' if e['schema_valid'] else 'INVALID'}, portal {'valid' if e['portal_valid'] else 'INVALID'}, sha256 {e['sha256'][:16]}")
    print(f"{info['file']}: {info['members']} files, PoseBusters failures {info['posebusters_failures']}, portal {info['portal_message']}, sha256 {info['sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

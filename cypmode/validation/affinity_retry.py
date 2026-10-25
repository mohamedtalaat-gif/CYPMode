# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Complete the base-run table of the official ligands when the affinity step of a run was skipped for memory (checklist item "Affinity step of the corrected base runs", 2026-10-25).

Boltz-2 runs its affinity head after the structure. When the affinity head reaches the memory cap of the guards (``PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.3``, ``scripts/guards.sh``) it prints
"ran out of memory, skipping batch" and writes no ``affinity_*.json``. The pipeline then raises ``no affinity_*.json`` and the batch runner records ``{"error": ...}`` for the whole ligand, so the row of the
table holds no structure summary (coordination, iron-nitrogen distance, confidence) either, although the structure of the run exists and is valid. This happened to the three base runs made again from the
corrected SMILES (job 275, 2026-10-24 and 25); the checkpoint then skips the ligand on every later run, so nothing retried it.

``--prepare`` records the SHA-256 of the structure files of every run that failed in this way and removes its entry from the checkpoint (the original is kept beside the checkpoint).
``--complete`` runs when the retry did not finish for a run (an error entry or none is left): it writes the summary of the structure without the affinity and records why the affinity is absent.
``--verify`` runs after ``python -m cypmode.validation.batch ... --no-override --accelerator cpu`` has made the missing step: it checks that the structure files are byte-identical to the recorded ones,
that no row of the table is left in error, and prints the affinity of each retried ligand next to the one of the racemic run that was set aside. The structure is never predicted again by this procedure
(``--no-override`` makes Boltz-2 skip a finished prediction and run only the missing step), so the poses that the final recipe worked with stay as they are.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

OUT = Path("data/structure_track/outputs")
BOLTZ = OUT / "out" / "cyp3a4"
RESULTS = OUT / "cyp3a4" / "structure_track_results.json"
FINAL = OUT / "cyp3a4" / "final"
DIGESTS = FINAL / "affinity_retry_digests.json"
BEFORE = FINAL / "structure_track_results.before_affinity_retry.json"
SET_ASIDE_CHECKPOINT = OUT / "out" / "superseded_racemic" / "checkpoints" / "structure_track_results.json"
OFFICIAL = Path("data/structure_track/inputs/structure_track_test_blinded.csv")
AFFINITY_ERROR = "no affinity_*.json"


def failed_rows(results_path: Path = RESULTS) -> list[str]:
    """The names whose entry of the checkpoint is an error (any error)."""
    if not Path(results_path).exists():
        return []
    return sorted(name for name, entry in json.loads(Path(results_path).read_text()).items() if "error" in entry)


def affinity_failures(results: dict) -> list[str]:
    """The names whose entry is the error of a missing affinity file."""
    return sorted(name for name, entry in results.items() if AFFINITY_ERROR in str(entry.get("error", "")))


def structure_files(name: str, boltz: Path = BOLTZ) -> list[Path]:
    """The prediction files of the structure of a base run: the models and their confidence summaries."""
    folder = boltz / f"boltz_results_cyp3a4_{name}" / "predictions" / f"cyp3a4_{name}"
    return sorted([*folder.glob("*.cif"), *folder.glob("confidence_*.json")])


def digests(names: list[str], boltz: Path = BOLTZ) -> dict[str, dict[str, str]]:
    return {name: {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in structure_files(name, boltz)} for name in names}


def prepare(results_path: Path = RESULTS, boltz: Path = BOLTZ, digest_path: Path = DIGESTS, backup: Path = BEFORE) -> list[str]:
    """Record the digests of the runs that failed for a missing affinity file and remove them from the checkpoint; returns their names."""
    results = json.loads(Path(results_path).read_text())
    names = affinity_failures(results)
    if not names:
        return []
    recorded = digests(names, boltz)
    empty = [name for name, files in recorded.items() if not files]
    if empty:
        raise RuntimeError(f"no structure files for {', '.join(empty)}: the failure is not the one of a skipped affinity step")
    Path(digest_path).parent.mkdir(parents=True, exist_ok=True)
    if not Path(backup).exists():
        shutil.copyfile(results_path, backup)
    Path(digest_path).write_text(json.dumps(recorded, indent=2) + "\n")
    Path(results_path).write_text(json.dumps({k: v for k, v in results.items() if k not in names}, indent=2) + "\n")
    return names


def unchanged(digest_path: Path = DIGESTS, boltz: Path = BOLTZ) -> dict[str, bool]:
    """{name: the structure files are byte-identical to the ones recorded by ``prepare``}."""
    recorded = json.loads(Path(digest_path).read_text())
    now = digests(list(recorded), boltz)
    return {name: now[name] == files for name, files in recorded.items()}


def complete(results_path: Path = RESULTS, boltz: Path = BOLTZ, digest_path: Path = DIGESTS, official: Path = OFFICIAL) -> list[str]:
    """For every run recorded by ``prepare`` whose entry is missing or still an error, write the summary of its structure without the affinity (the structure is intact, only the affinity file is missing) and say why
    the affinity is absent; returns their names. Runs whose retry finished are left as they are."""
    import pandas as pd

    from cypmode.validation.batch import to_dataframe
    from cypmode.validation.structures import summarize_compound

    recorded = json.loads(Path(digest_path).read_text())
    results = json.loads(Path(results_path).read_text())
    table = pd.read_csv(official)
    smiles = dict(zip(table.Molecule_Name, table.SMILES, strict=True))
    completed = []
    for name in recorded:
        if name in results and "error" not in results[name]:
            continue
        entry = summarize_compound(boltz / f"boltz_results_cyp3a4_{name}", smiles[name], expect_affinity=False)
        entry["affinity_note"] = "not computed: the affinity step ran out of memory on the GPU and the retry on the processor did not finish"
        results[name] = entry
        completed.append(name)
    if completed:
        Path(results_path).write_text(json.dumps(results, indent=2) + "\n")
        frame = to_dataframe(results, dict(smiles))
        frame.to_csv(Path(results_path).with_suffix(".csv"), index=False)
        try:
            frame.to_parquet(Path(results_path).with_suffix(".parquet"), index=False)
        except ImportError:
            pass
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--complete", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        names = prepare()
        print(f"AFFINITY RETRY {len(names)} runs to complete: {', '.join(names) or 'none'}")
        return 0
    if args.complete:
        names = complete()
        print(f"AFFINITY RETRY {len(names)} rows completed without the affinity: {', '.join(names) or 'none'}")
        return 0
    same = unchanged()
    for name, ok in same.items():
        print(f"AFFINITY RETRY {name}: structure files {'byte-identical to the recorded ones' if ok else 'CHANGED'}")
    results = json.loads(RESULTS.read_text())
    old = json.loads(SET_ASIDE_CHECKPOINT.read_text()) if SET_ASIDE_CHECKPOINT.exists() else {}
    for name in same:
        entry = results.get(name, {})
        racemic = old.get(name, {}).get("affinity_probability_binary")
        if entry.get("affinity_probability_binary") is None:
            print(f"  {name}: affinity not computed ({entry.get('affinity_note', 'no entry')})")
            continue
        print(f"  {name}: affinity probability {entry['affinity_probability_binary']:.3f}, value {entry['affinity_pred_value']:.3f}" + (f" (the racemic run: {racemic:.3f})" if racemic is not None else ""))
    failed = failed_rows()
    print(f"AFFINITY RETRY base table: {len(results)} rows, {len(failed)} in error" + (f" ({', '.join(failed)})" if failed else ""))
    return 0 if all(same.values()) and not failed else 1


if __name__ == "__main__":
    from cypmode import provenance

    provenance.announce()
    sys.exit(main())

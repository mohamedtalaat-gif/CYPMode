# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Set aside the runs that were made from an outdated official SMILES, so that they are regenerated from the corrected one (job 275_stereo_official_rerun, registered on 2026-10-24).

The organisers replaced the SMILES of three official ligands, first published as racemates, by the SMILES of the models used for scoring, which carry the stereochemistry. A run is *stale* when the SMILES of its
Boltz input (``data/structure_track/inputs/cyp3a4_<run>.yaml``, rewritten by the pipeline every time a run starts) differs, as a canonical isomeric SMILES, from the one of its ligand in the official file. The
runs of a ligand are its base run, the unguided, restrained and extra runs and the lever runs (``pose_margins.candidate_runs``).

``python -m cypmode.validation.stereo_rerun`` moves the output folder of every stale run, and the OpenFold3 folder of a ligand whose base run is stale, to ``data/structure_track/outputs/out/superseded_racemic/``
(unchanged, with a ledger of what was moved and why), and removes the entries of the stale runs from the checkpoints that make the batch runner skip a finished run (``structure_track_results.json`` and
``final/runs/*_results.json``; the originals are copied to the same folder first). It changes nothing else: the next run of the batch runner and of ``final_recipe`` finds the runs missing and makes them from the
corrected SMILES, and rewrites the input YAMLs. It is idempotent, and ``--dry-run`` only lists what would be set aside.

``--report BEFORE.csv`` compares the poses chosen before (the ``consensus.csv`` that the recipe wrote earlier, copied before the rerun) with the ones in ``final/consensus.csv`` now: for every ligand whose
pose changed, the pocket-aligned ligand RMSD between the two picks and the iron-nitrogen distance of each. It reports and does not judge: a change of pose is the expected result of a new draw.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from rdkit import Chem

from cypmode.validation.final_recipe import FINAL, OFFICIAL
from cypmode.validation.pose_margins import candidate_runs

INPUTS = Path("data/structure_track/inputs")
OUT = Path("data/structure_track/outputs")
BOLTZ = OUT / "out" / "cyp3a4"
OPENFOLD3 = OUT / "openfold3" / "cyp3a4"
BASE_RESULTS = OUT / "cyp3a4" / "structure_track_results.json"
RUN_RESULTS = FINAL / "runs"
SUPERSEDED = OUT / "out" / "superseded_racemic"


def canonical(smiles: str) -> str:
    """The canonical isomeric SMILES (stereochemistry kept)."""
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def input_smiles(run: str, inputs: Path = INPUTS) -> str | None:
    """The ligand SMILES of the Boltz input of a run, or ``None`` when the run has no input file."""
    path = inputs / f"cyp3a4_{run}.yaml"
    if not path.exists():
        return None
    for entry in yaml.safe_load(path.read_text()).get("sequences", []):
        if "ligand" in entry and "smiles" in entry["ligand"]:
            return entry["ligand"]["smiles"]
    return None


def stale_runs(official: pd.DataFrame, inputs: Path = INPUTS) -> dict[str, list[str]]:
    """{ligand: [run names]} of the runs whose input SMILES is not the official one; ligands without a stale run are left out."""
    out: dict[str, list[str]] = {}
    for ligand, smiles in zip(official.Molecule_Name, official.SMILES, strict=True):
        wanted = canonical(smiles)
        stale = [run for run in candidate_runs(ligand) if (used := input_smiles(run, inputs)) is not None and canonical(used) != wanted]
        if stale:
            out[ligand] = stale
    return out


def _move(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"{dst} exists already; the earlier set-aside run must be looked at by hand")
    shutil.move(str(src), str(dst))


def supersede(stale: dict[str, list[str]], boltz: Path = BOLTZ, openfold3: Path = OPENFOLD3, base_results: Path = BASE_RESULTS, run_results: Path = RUN_RESULTS,
              target: Path = SUPERSEDED, dry_run: bool = False) -> dict:
    """Set the stale runs aside; returns the ledger ``{moved, checkpoints, ligands}``."""
    moved, pruned = [], []
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    for ligand, runs in stale.items():
        for run in runs:
            src = boltz / f"boltz_results_cyp3a4_{run}"
            if src.exists():
                moved.append({"ligand": ligand, "what": f"boltz run {run}", "from": str(src), "to": str(target / "boltz" / src.name)})
                if not dry_run:
                    _move(src, target / "boltz" / src.name)
        if ligand in runs:  # OpenFold3 reads the input of the base run
            src = openfold3 / ligand
            if src.exists():
                moved.append({"ligand": ligand, "what": "openfold3 samples", "from": str(src), "to": str(target / "openfold3" / ligand)})
                if not dry_run:
                    _move(src, target / "openfold3" / ligand)
    stale_names = {run for runs in stale.values() for run in runs}
    for path in [base_results, *sorted(Path(run_results).glob("*_results.json"))]:
        if not path.exists():
            continue
        entries = json.loads(path.read_text())
        gone = sorted(set(entries) & stale_names)
        if not gone:
            continue
        pruned.append({"file": str(path), "entries": gone})
        if not dry_run:
            backup = target / "checkpoints" / path.name
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                shutil.copyfile(path, backup)
            path.write_text(json.dumps({k: v for k, v in entries.items() if k not in stale_names}, indent=2) + "\n")
    ledger = {"moved": moved, "checkpoints": pruned, "ligands": {k: v for k, v in stale.items()}, "reason": "the input SMILES of the run differs from the corrected official SMILES", "stamp": stamp}
    if not dry_run and (moved or pruned):
        old = json.loads((target / "ledger.json").read_text()) if (target / "ledger.json").exists() else []
        (target / "ledger.json").parent.mkdir(parents=True, exist_ok=True)
        (target / "ledger.json").write_text(json.dumps([*old, ledger], indent=2))
    return ledger


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pick_report(before: pd.DataFrame, after: pd.DataFrame, smiles_of: dict[str, str], boltz: Path = BOLTZ, target: Path = SUPERSEDED) -> pd.DataFrame:
    """One row per ligand: whether the chosen pose is the same file as before, and for a changed pose the pocket-aligned ligand RMSD between the two picks and the iron-nitrogen distance of each.

    ``before`` and ``after`` are ``consensus.csv`` tables (``compound``, ``chosen_cif``); an earlier pick that was set aside is looked up in ``target``.
    """
    from cypmode.validation.pose_consensus import aligned_ligand_rmsd, pose_metrics
    from cypmode.validation.structures import load_structure

    old_of = dict(zip(before.compound, before.chosen_cif, strict=True))
    rows = []
    for ligand, new_path in zip(after.compound, after.chosen_cif, strict=True):
        old = Path(old_of[ligand])
        if old.is_relative_to(boltz) and (target / "boltz" / old.relative_to(boltz)).exists():  # the run was set aside: the same path now holds the new draw
            old = target / "boltz" / old.relative_to(boltz)
        new = Path(new_path)
        row = {"ligand": ligand, "old_pick": str(old_of[ligand]), "new_pick": str(new), "changed": True, "ligand_rmsd": float("nan"), "fe_n_old": float("nan"), "fe_n_new": float("nan")}
        if old.exists() and new.exists() and _digest(old) == _digest(new):
            row["changed"] = False
        elif old.exists() and new.exists():
            old_structure, new_structure = load_structure(old), load_structure(new)
            row["ligand_rmsd"] = aligned_ligand_rmsd(old_structure, new_structure, smiles_of[ligand])["ligand_rmsd"]
            row["fe_n_old"] = pose_metrics(old_structure, smiles_of[ligand])["closest_motif_n_fe"] or float("nan")
            row["fe_n_new"] = pose_metrics(new_structure, smiles_of[ligand])["closest_motif_n_fe"] or float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--official", default=str(OFFICIAL))
    parser.add_argument("--report", metavar="BEFORE_CSV", help="compare the picks of this consensus.csv with final/consensus.csv and exit")
    args = parser.parse_args()
    official = pd.read_csv(args.official)
    if args.report:
        table = pick_report(pd.read_csv(args.report), pd.read_csv(FINAL / "consensus.csv"), dict(zip(official.Molecule_Name, official.SMILES, strict=True)))
        table.to_csv(FINAL / "stereo_picks.csv", index=False)
        changed = table[table.changed]
        print(f"STEREO RERUN picks: {len(changed)} of {len(table)} ligands changed pose" + (f" ({', '.join(changed.ligand)})" if len(changed) else ""))
        for row in changed.itertuples():
            print(f"  {row.ligand}: ligand RMSD between the picks {row.ligand_rmsd:.2f} A, iron-nitrogen distance {row.fe_n_old:.2f} -> {row.fe_n_new:.2f} A")
        return 0
    stale = stale_runs(official)
    if not stale:
        print("STEREO RERUN nothing to set aside: every run was made from the official SMILES")
        return 0
    for ligand, runs in stale.items():
        print(f"{ligand}: {len(runs)} stale runs ({', '.join(runs)})")
    ledger = supersede(stale, dry_run=args.dry_run)
    verb = "would be set aside" if args.dry_run else "set aside"
    print(f"STEREO RERUN {len(ledger['moved'])} output folders {verb} in {SUPERSEDED}; {sum(len(c['entries']) for c in ledger['checkpoints'])} checkpoint entries removed from {len(ledger['checkpoints'])} files")
    for item in ledger["moved"]:
        print(f"  {item['what']} of {item['ligand']}")
    return 0


if __name__ == "__main__":
    from cypmode import provenance

    provenance.announce()
    sys.exit(main())

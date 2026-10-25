# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""OpenStructure's LDDT-PLI and BiSyRMSD for every pose that a pool gate was judged on.

``ost_official`` found that the LDDT-PLI of the project does not agree with the organisers' scorer closely enough (mean absolute difference 0.066 against the registered 0.05, pose ordering 77.3% against 90%) and
registered that the gates that rest on the local measure are then repeated with OpenStructure's. The gates of the pools (minimum Bayes risk, engine consensus, expected-improvement sampling, the heme-frame
prior, the series term, the three pose terms, the restraint and the three pilots) are judged by ``true_rmsd``, the pocket-aligned symmetry-aware RMSD of the project to the crystal pose, whose official counterpart is
OpenStructure's BiSyRMSD; three exploratory gates are judged by the LDDT-PLI of the project. This module scores every pose of the tables of those gates (``YARDSTICK_TABLES``) with OpenStructure, in the conda
environment ``ost``, exactly as ``ost_official`` does for the confirmation poses; ``ost_pool_gates`` then repeats the gates with the new values.

``python -m cypmode.validation.ost_pools [--max-minutes N]`` writes ``data/structure_track/outputs/cyp3a4/ost_pools.csv`` (one row per pose; resumable: scored poses are kept) and prints the number
of poses still to score on its last line, so that the queue can run it in time-boxed chunks.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/structure_track/outputs/cyp3a4")
TABLE = OUT / "ost_pools.csv"
ENGINES = ("boltz", "openfold3")
YARDSTICK_TABLES = (
    "benchmark_engine_pose_validation.csv",
    "benchmark_engine_pose_validation_unguided_boltz.csv",
    "benchmark_engine_pose_validation_unguided9.csv",
    "benchmark_engine_pose_validation_febond.csv",
    "benchmark_engine_pose_validation_fecontact.csv",
    "diversity_pilot_poses.csv",
    "diversity_pilot_poses_cryoem.csv",
    "series_pose_validation.csv",
)


def pose_keys(directory: Path = OUT) -> list[tuple[str, str, int]]:
    """(compound, engine, sample) of every pose that has a ``true_rmsd`` in one of the yardstick tables, and of every pose of the LDDT-PLI table."""
    keys: set[tuple[str, str, int]] = set()
    lddt_table = directory / "pose_lddt_to_crystal.csv"  # the poses of the three exploratory gates that are judged by LDDT-PLI
    if lddt_table.exists():
        table = pd.read_csv(lddt_table)
        keys |= {(c, e, int(s)) for c, e, s in zip(table["compound"], table["engine"], table["sample"], strict=True)}
    for name in YARDSTICK_TABLES:
        path = directory / name
        if path.exists():
            table = pd.read_csv(path).dropna(subset=["true_rmsd"])
            engines = table["engine"] if "engine" in table else pd.Series(["boltz"] * len(table), index=table.index)
            keys |= {(c, e, int(s)) for c, e, s in zip(table["compound"], engines, table["sample"], strict=True)}
    return sorted(keys)


def inventory(keys: list[tuple[str, str, int]]) -> list[dict]:
    """The CIF path and the ligand SMILES of every pose (the SMILES is that of the compound's own Boltz input)."""
    from cypmode.validation.openfold3 import read_boltz_inputs
    from cypmode.validation.pose_pools import pool_paths

    rows = []
    for compound in sorted({k[0] for k in keys}):
        paths = {(engine, sample): path for engine, sample, path in pool_paths(compound, ENGINES)}
        smiles = read_boltz_inputs(compound)["ligand_smiles"]
        for _, engine, sample in (k for k in keys if k[0] == compound):
            if (engine, sample) in paths and paths[(engine, sample)].exists():
                rows.append({"compound": compound, "engine": engine, "sample": sample, "cif": paths[(engine, sample)], "smiles": smiles})
    return rows


def score_copies(model_pdb: Path, smiles: str, records: list[dict], workdir: Path) -> dict:
    """OpenStructure's LDDT-PLI and BiSyRMSD of a pose against every crystal copy of its ligand: the largest LDDT-PLI and the smallest BiSyRMSD, as the local measures take the best copy."""
    from cypmode.validation.ost_official import ost_score, write_reference_pdb

    lddt, rmsd, error = [], [], ""
    for k, record in enumerate(records):
        reference = workdir / f"ref_{k}.pdb"
        try:
            write_reference_pdb(record, reference)
        except Exception as exc:
            error = error or f"reference {record['pdb_id']}: {exc}"
            continue
        result = ost_score(model_pdb, reference, smiles)
        if "lddt_pli" in result:
            lddt.append(result["lddt_pli"])
            if np.isfinite(result.get("bisyrmsd", np.nan)):
                rmsd.append(result["bisyrmsd"])
        else:
            error = error or result.get("error", "")
    return {"ost_lddt_pli": max(lddt) if lddt else np.nan, "ost_bisyrmsd": min(rmsd) if rmsd else np.nan, "error": "" if lddt else (error or "no reference copy could be scored")}


def main() -> int:
    from cypmode.challenge.structure_submission import convert_cif_to_pdb
    from cypmode.validation.ost_official import reference_records

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-minutes", type=float, default=None, help="stop after this many minutes (the poses scored so far are kept)")
    args = parser.parse_args()
    keys = pose_keys()
    todo = inventory(keys)
    done = pd.read_csv(TABLE) if TABLE.exists() else pd.DataFrame(columns=["compound", "engine", "sample"])
    finished = set(zip(done["compound"], done["engine"], done["sample"], strict=True))
    rows = done.to_dict("records")
    pending = [r for r in todo if (r["compound"], r["engine"], r["sample"]) not in finished]
    print(f"{len(keys)} poses in the yardstick tables, {len(todo)} with a structure file, {len(pending)} still to score", flush=True)
    started = time.monotonic()
    records: dict[str, list[dict]] = {}
    for row in pending:
        if args.max_minutes is not None and (time.monotonic() - started) / 60 > args.max_minutes:
            break
        smiles = row["smiles"]
        if smiles not in records:
            records[smiles] = reference_records(smiles)
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            model = workdir / "model.pdb"
            convert_cif_to_pdb(row["cif"], model)
            result = score_copies(model, smiles, records[smiles], workdir)
        rows.append({"compound": row["compound"], "engine": row["engine"], "sample": row["sample"], **result})
        pd.DataFrame(rows).to_csv(TABLE, index=False)
        print(f"{row['compound']} {row['engine']} {row['sample']}: LDDT-PLI {rows[-1]['ost_lddt_pli']:.3f}, BiSyRMSD {rows[-1]['ost_bisyrmsd']:.2f}", flush=True)
    remaining = len(pending) - (len(rows) - len(done))
    print(remaining)
    return 0


if __name__ == "__main__":
    sys.exit(main())

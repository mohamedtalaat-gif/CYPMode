# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Margin audit of every pose in the pools of the official ligands, under the challenge scorer's own PoseBusters call.

``python -m cypmode.validation.pose_margins`` checks every predicted pose of every official ligand (all runs: the base run, the restrained run, the
seeds and conformer levers, the extra samples) with ``official_posebusters``, writes one row per pose to
``data/structure_track/outputs/cyp3a4/final/official_posebusters_pools.csv`` and prints, per ligand, how many poses pass, how many pass with a safe
margin, and the margin of the pose the final recipe chose. The rows are cached by file path, size and modification time, so a second run only checks
poses that changed. The column ``stereo_ok`` says whether the ligand of the pose reproduces the stereochemistry of the official SMILES (``pose_stereo``): the
scorer's PoseBusters call has no chirality check, so this is the only place where the wrong enantiomer of a chiral ligand is noticed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cypmode.validation.final_recipe import FINAL, LEVER_RUNS, OFFICIAL
from cypmode.validation.official_posebusters import (
    MARGIN_FLOOR,
    OfficialReport,
    check_cif,
)
from cypmode.validation.pose_pools import pool_paths
from cypmode.validation.pose_stereo import cif_reproduces_stereo

TABLE = FINAL / "official_posebusters_pools.csv"
MAX_EXTRA_RUNS = 9


def candidate_runs(ligand: str) -> list[str]:
    """Every run name that can hold poses of a ligand (the recipe's runs and its extra samples)."""
    return [ligand, f"{ligand}-u", f"{ligand}-rs", *(f"{ligand}-seed{k}" for k in range(1, LEVER_RUNS + 1)), *(f"{ligand}-cf{k}" for k in range(1, LEVER_RUNS + 1)), f"{ligand}-ss",
            *(f"{ligand}-ex{k}" for k in range(1, MAX_EXTRA_RUNS + 1))]


def pose_signature(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{int(stat.st_mtime)}"


def row_of(ligand: str, run: str, engine: str, sample: int, path: Path, report: OfficialReport, stereo_ok: bool) -> dict:
    return {"ligand": ligand, "run": run, "engine": engine, "sample": sample, "cif": str(path), "signature": pose_signature(path), "passed": report.passed, "stereo_ok": stereo_ok, "failed": ";".join(report.failed),
            "n_checks": report.n_checks, "distance_margin": report.distance_margin, **{f"dist_{k}": v for k, v in report.distances.items()}, **{f"slack_{k}": v for k, v in report.margins.items()},
            "error": report.error}


def audit(smiles_of: dict[str, str], engines: tuple[str, ...] = ("boltz",), table: Path = TABLE) -> pd.DataFrame:
    """One row per pose of every ligand; poses already audited (same path, size and modification time) are kept from the table."""
    old = pd.read_csv(table) if table.exists() else pd.DataFrame()
    known = {(r.cif, r.signature): r._asdict() for r in old.itertuples(index=False)} if len(old) else {}
    rows = []
    for ligand, smiles in smiles_of.items():
        for run in candidate_runs(ligand):
            for engine, sample, path in pool_paths(run, engines):
                if not path.exists():
                    continue
                key = (str(path), pose_signature(path))
                if key in known:
                    row = known[key]
                    if row.get("stereo_ok") is None or pd.isna(row.get("stereo_ok")):  # a table written before the stereochemistry was checked
                        row = {**row, "stereo_ok": cif_reproduces_stereo(path, smiles) is True}
                    rows.append(row)
                    continue
                rows.append(row_of(ligand, run, engine, sample, path, check_cif(path, smiles), cif_reproduces_stereo(path, smiles) is True))
    out = pd.DataFrame(rows)
    table.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(table, index=False)
    return out


def summary(pools: pd.DataFrame, chosen: dict[str, str]) -> pd.DataFrame:
    rows = []
    for ligand, group in pools.groupby("ligand", sort=True):
        valid = group[group.passed]
        safe = valid[valid.distance_margin >= MARGIN_FLOOR]
        picked = group[group.cif == chosen.get(ligand, "")]
        rows.append({"ligand": ligand, "poses": len(group), "pass": len(valid), "stereo": int(group.stereo_ok.astype(bool).sum()), "safe": len(safe), "chosen_margin": float(picked.distance_margin.iloc[0]) if len(picked) else float("nan"),
                     "best_margin": float(valid.distance_margin.max()) if len(valid) else float("nan")})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engines", nargs="*", default=["boltz"])
    args = parser.parse_args()
    official = pd.read_csv(OFFICIAL)
    smiles_of = dict(zip(official.Molecule_Name, official.SMILES, strict=True))
    pools = audit(smiles_of, tuple(args.engines))
    consensus = FINAL / "consensus.csv"
    chosen = dict(pd.read_csv(consensus)[["compound", "chosen_cif"]].itertuples(index=False)) if consensus.exists() else {}
    table = summary(pools, chosen)
    print(table.round(3).to_string(index=False))
    thin = table[(table.chosen_margin < MARGIN_FLOOR) | table.chosen_margin.isna()]
    print(f"\n{len(pools)} poses audited; {int(pools.passed.sum())} pass all checks, {int((pools.passed & (pools.distance_margin >= MARGIN_FLOOR)).sum())} pass with a relative distance of at least {MARGIN_FLOOR}")
    print(f"ligands whose chosen pose is below the margin floor: {len(thin)}; of these with a safe valid alternative in the pool: {int((thin.safe > 0).sum())}")


if __name__ == "__main__":
    main()

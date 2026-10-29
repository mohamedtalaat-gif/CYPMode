# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Check of the binding-site representation (declared on 2026-10-23 before any code ran).

The fields and the sites of ``pocket_pharmacophore`` are holo: they are read from ligands in crystal complexes and say where ligands sit in the frame of the heme. They say nothing of an apo pocket or of
a cryptic pocket that nobody has seen. Three measurements, none a gate (no blinded compound is involved):

1. the variability of the site: for the crystal ligand copies of the CYP3A4 reference set, the RMSD of the pocket C-alpha atoms (the residues present in at least ``CORE_SHARE`` of the copies, superposed
   by the Kabsch method) of each copy to the nearest copy with a different PDB code and a different ligand; the 95th percentile of this distribution is the flag of an unseen pocket conformation;
2. the pocket of every predicted complex of the pools of ``pose_pharmacophore`` (job 272), on the same residues, its RMSD to the nearest reference pocket (the references of the ligand itself left out), the
   share beyond the flag and the correlation with the true RMSD of the pose;
3. the support of the fields: for each crystal copy, the share of its heavy atoms where the occupancy field built without that compound is below ``pose_pharmacophore.SUPPORT_MIN``; the 95th percentile
   over the copies is the cut at which a pose is said to leave the region where ligands have been seen, and the share of the predicted poses beyond it and the correlation with the true RMSD are printed.

The two flags are written to ``site_representation_flags.json`` for the official ligands of job 925, whose poses carry both. A flag marks a pose to be read with care and corrects nothing. Limits: the references
are holo and the entries of one enzyme; a pocket that a co-folding model shapes around a ligand can be plausible or spurious and the predicted complex alone cannot say which; a cryptic pocket that no entry shows
is by construction outside what these fields can represent.

Usage: python -m cypmode.validation.site_representation      (job 274, after job 272)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CORE_SHARE = 0.9
MIN_SHARED = 40
FLAG_PERCENTILE = 95
OUT_DIR = Path("data/structure_track/outputs/cyp3a4")
FLAGS = OUT_DIR / "site_representation_flags.json"


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """RMSD of two point sets of the same length after the best rigid superposition."""
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    sign = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, sign]) @ vt
    return float(np.sqrt(((a @ rotation - b) ** 2).sum(axis=1).mean()))


def pockets_of(records) -> list[dict[int, np.ndarray]]:
    """The pocket C-alpha atoms of each record as {residue number: xyz}."""
    return [{int(k): np.asarray(v, dtype=float) for k, v in r["pocket_ca"].items()} for r in records]


def core_residues(pockets: list[dict[int, np.ndarray]], share: float = CORE_SHARE) -> set[int]:
    """The residues that at least ``share`` of the pockets have."""
    counts: dict[int, int] = {}
    for pocket in pockets:
        for residue in pocket:
            counts[residue] = counts.get(residue, 0) + 1
    return {residue for residue, n in counts.items() if n >= share * len(pockets)}


def pocket_rmsd(a: dict[int, np.ndarray], b: dict[int, np.ndarray], core: set[int]) -> tuple[float, int]:
    """(Kabsch RMSD over the core residues that both pockets have, their number); NaN when there are fewer than ``MIN_SHARED``."""
    shared = sorted(core & a.keys() & b.keys())
    if len(shared) < MIN_SHARED:
        return float("nan"), len(shared)
    return kabsch_rmsd(np.array([a[r] for r in shared]), np.array([b[r] for r in shared])), len(shared)


def nearest_other(index: int, pockets: list, keys: list, ids: list, core: set[int]) -> float:
    """The smallest pocket RMSD of copy ``index`` to a copy with a different PDB code and a different ligand (NaN when there is none)."""
    best = float("nan")
    for j, other in enumerate(pockets):
        if j == index or ids[j] == ids[index] or (keys[j] is not None and keys[j] == keys[index]):
            continue
        value, _ = pocket_rmsd(pockets[index], other, core)
        if np.isfinite(value) and not (value >= best):
            best = value
    return best


def nearest_reference(pocket: dict[int, np.ndarray], pockets: list, keys: list, own_key: str | None, core: set[int]) -> tuple[float, int]:
    """(the smallest RMSD of a predicted pocket to a reference pocket whose ligand is not the compound itself, the index of that reference)."""
    best, arg = float("nan"), -1
    for j, other in enumerate(pockets):
        if own_key is not None and keys[j] == own_key:
            continue
        value, _ = pocket_rmsd(pocket, other, core)
        if np.isfinite(value) and not (value >= best):
            best, arg = value, j
    return best, arg


def crystal_table(refs: tuple[dict, ...]) -> pd.DataFrame:
    """One row per crystal copy: its pocket RMSD to the nearest other entry and the share of its heavy atoms outside the support of the fields built without its compound."""
    import gemmi

    from cypmode.challenge import pocket_pharmacophore as pp
    from cypmode.validation import pose_pharmacophore as pose
    from cypmode.validation.pose_rescoring import connectivity_key

    pockets = pockets_of(refs)
    core = core_residues(pockets)
    keys = [connectivity_key(r["smiles"]) for r in refs]
    ids = [r["pdb_id"] for r in refs]
    rows = []
    for i, record in enumerate(refs):
        row = {"pdb_id": record["pdb_id"], "comp_id": record["comp_id"], "pocket_nearest_rmsd": nearest_other(i, pockets, keys, ids, core), "outside": np.nan}
        path = pp.PDB_CACHE / f"{record['pdb_id']}.cif"
        if path.exists() and record.get("smiles"):
            iron, xyz = np.array(record["fe"], dtype=float), np.array(record["coords"], dtype=float)
            frame = pp.heme_frame(gemmi.read_structure(str(path)), iron, xyz)
            if frame is not None:
                fields, _ = pose.fields_and_sites(pose.kept_indices(record["smiles"], "compound", refs))
                row["outside"] = float((pp.lookup(fields["occupancy"], pp.to_frame(xyz, iron, frame)) < pose.SUPPORT_MIN).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def predicted_table(poses: list[tuple[str, str, int, float, Path]], refs: tuple[dict, ...], smiles: dict[str, str], flag: float, cut: float, outside: dict[tuple[str, str, int], float]) -> pd.DataFrame:
    """One row per predicted pose of the pools: the RMSD of its pocket to the nearest reference pocket, whether it lies beyond the flag, and the share of its atoms outside the support."""
    from cypmode.validation.pose_rescoring import connectivity_key, structure_arrays
    from cypmode.validation.structures import load_structure

    pockets = pockets_of(refs)
    core = core_residues(pockets)
    keys = [connectivity_key(r["smiles"]) for r in refs]
    rows = []
    for ligand, run, sample, rmsd, cif in poses:
        _, _, pocket_ca = structure_arrays(load_structure(cif))
        value, _ = nearest_reference({int(k): v for k, v in pocket_ca.items()}, pockets, keys, connectivity_key(smiles[ligand]), core)
        out = outside.get((ligand, run, sample), np.nan)
        rows.append({"compound": ligand, "run": run, "sample": sample, "true_rmsd": rmsd, "pocket_nearest_rmsd": value, "beyond_pocket_flag": bool(value > flag) if np.isfinite(value) else False, "outside": out,
                     "beyond_support_cut": bool(out > cut) if np.isfinite(out) else False})
    return pd.DataFrame(rows)


def percentile(values: pd.Series, q: float = FLAG_PERCENTILE) -> float:
    values = values.dropna()
    return float(np.percentile(values, q)) if len(values) else float("nan")


def correlation(table: pd.DataFrame, column: str) -> tuple[float, float]:
    """(Spearman rho with the true RMSD over all poses, the mean of the per-ligand rho)."""
    from scipy.stats import spearmanr

    ok = table[[column, "true_rmsd"]].dropna()
    pooled = float(spearmanr(ok[column], ok["true_rmsd"]).statistic) if len(ok) >= 3 and ok[column].nunique() > 1 else float("nan")
    per = []
    for _, group in table.groupby("compound"):
        g = group[[column, "true_rmsd"]].dropna()
        if len(g) >= 4 and g[column].nunique() > 1 and g["true_rmsd"].nunique() > 1:
            per.append(float(spearmanr(g[column], g["true_rmsd"]).statistic))
    return pooled, float(np.mean(per)) if per else float("nan")


def summarize(crystal: pd.DataFrame, predicted: pd.DataFrame, flag: float, cut: float, n_core: int) -> list[str]:
    nn = crystal.pocket_nearest_rmsd.dropna()
    lines = [(f"variability of the site: {len(crystal)} crystal copies, {n_core} core residues (present in at least {CORE_SHARE:.0%} of the copies); pocket RMSD to the nearest copy with a different PDB code and ligand: "
              f"median {nn.median():.2f} A, 90th percentile {nn.quantile(0.9):.2f} A, {FLAG_PERCENTILE}th percentile {flag:.2f} A (the flag of an unseen pocket conformation)")]
    near = predicted.pocket_nearest_rmsd.dropna()
    pooled, per = correlation(predicted, "pocket_nearest_rmsd")
    lines.append(f"predicted pockets: {len(near)} of {len(predicted)} poses compared; RMSD to the nearest reference pocket (own compound left out) median {near.median():.2f} A, {predicted.beyond_pocket_flag.mean():.0%} beyond the flag; "
                 f"Spearman rho with the true RMSD of the pose {pooled:+.3f} over all poses, mean per ligand {per:+.3f}")
    out = crystal.outside.dropna()
    lines.append(f"support of the fields: share of heavy atoms outside the support, crystal copies (fields built without their compound) median {out.median():.3f}, {FLAG_PERCENTILE}th percentile {cut:.3f} (the cut)")
    pooled, per = correlation(predicted, "outside")
    lines.append(f"predicted poses beyond the cut: {predicted.beyond_support_cut.mean():.0%} of {len(predicted)}; Spearman rho of the outside share with the true RMSD {pooled:+.3f} over all poses, mean per ligand {per:+.3f}")
    lines.append(f"SITE CHECK flags: pocket RMSD flag {flag:.2f} A, support cut {cut:.3f} (written to {FLAGS.name}; information, no gate)")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--directory", default=str(OUT_DIR), help="where the pose tables of pose_awareness are read from")
    parser.add_argument("--output", default=str(OUT_DIR), help="where the tables and the flags are written")
    parser.add_argument("--poses", default=str(OUT_DIR / "pose_pharmacophore_poses.csv"), help="the pose table of job 272 (regime compound), which holds the outside share of every pose")
    args = parser.parse_args()
    from cypmode.validation import pose_pharmacophore as pose

    refs = pose.references()
    crystal = crystal_table(refs)
    n_core = len(core_residues(pockets_of(refs)))
    flag, cut = percentile(crystal.pocket_nearest_rmsd), percentile(crystal.outside)
    scored = pd.read_csv(args.poses)
    scored = scored[scored.regime == "compound"]
    outside = {(r.compound, r.run, int(r.sample)): float(r.outside) for r in scored.itertuples()}
    smiles = pose.benchmark_smiles()
    poses = pose.pooled_poses(Path(args.directory))
    predicted = predicted_table(poses, refs, smiles, flag, cut, outside)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    crystal.to_csv(output / "site_representation_crystal.csv", index=False)
    predicted.to_csv(output / "site_representation_poses.csv", index=False)
    (output / FLAGS.name).write_text(json.dumps({"pocket_rmsd_flag": flag, "support_cut": cut, "core_residues": n_core, "percentile": FLAG_PERCENTILE}, indent=2) + "\n")
    print("\n".join(summarize(crystal, predicted, flag, cut, n_core)))


if __name__ == "__main__":
    from cypmode import provenance

    provenance.announce()
    main()

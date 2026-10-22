# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Physics-, molecule- and pharmacophore-aware terms for choosing among predicted poses (jobs 168 / 169).

Boltz-2 reports a confidence of 0.91-0.96 even for poses 5-6 A from the crystal pose, so its own score cannot rank a pool. The terms below
describe a pose with knowledge the co-folding model does not use. None of them needs a new structure prediction; each is computed from
the pose pool and the ligand's SMILES and judged on the crystal-anchored benchmark.

Molecule and physics (the ligand itself)
    conf_rmsd         heavy-atom RMSD (symmetry aware) of the pose to the nearest member of the ligand's low-energy conformer ensemble:
                      ETKDG conformers, MMFF-relaxed, scored by GFN2-xTB with implicit water, kept within ``ENSEMBLE_KCAL`` of the minimum;
    conf_energy       relative GFN2-xTB energy (kcal/mol) of that nearest conformer: a pose that resembles only a high-energy conformer is
                      strained;
Physics (the pocket, from coordinates)
    burial            fraction of the ligand surface buried in the complex (``pocket_geometry.ligand_burial_sasa``);
    contacts          protein heavy atoms within 4.5 A of the ligand per ligand heavy atom (packing);
    clashes           ligand-protein heavy-atom pairs closer than 3.0 A that cannot be hydrogen bonds, and polar pairs closer than 2.4 A;
Pharmacophore (what the ligand should do at the heme)
    donor distance    for ligands with a Type II donor nitrogen: |d(Fe, donor N) - 2.2 A|, the dative bond length of solved Type II structures;
    site distance     for the others: how far the ligand's most easily oxidised carbon (lowest GFN2 hydrogen-abstraction energy,
                      ``physics_block``) lies outside the 3.5-6.0 A window typical of substrate-bound CYP structures.

Scores are mean within-pool ranks of the terms (no fitted weights): physics = conf_rmsd, conf_energy, burial, contacts, clashes;
pharmacophore = the distance term that applies to the ligand; combined = mean rank of both and of the MBR consensus (``pose_pools``).
Gates (fixed in advance, same bar as the heme-frame prior): over the pooled benchmark poses the Spearman correlation between a score and
minus the true RMSD, averaged over ligands, exceeds 0.3 and is positive for at least 60% of the ligands.

tblite is used for the conformer energies and the oxidation-site ranking; this module never imports LightGBM.

Usage: python -m cypmode.validation.pose_awareness [--table CSV]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ENSEMBLE_KCAL = 8.0
N_CONFORMERS = 40
CONTACT_RADIUS_A = 4.5
CLASH_A, POLAR_CLASH_A = 3.0, 2.4
DONOR_FE_A = 2.2
SITE_WINDOW_A = (3.5, 6.0)
MIN_MEAN_RHO, MIN_FRACTION_POSITIVE = 0.3, 0.6
TERMS = ("conf_rmsd", "conf_energy", "burial", "contacts", "clashes", "donor_error", "site_error")


def window_penalty(distance: float, low: float = SITE_WINDOW_A[0], high: float = SITE_WINDOW_A[1]) -> float:
    """0 inside [low, high], the distance to the nearest edge outside it."""
    return float(max(0.0, low - distance, distance - high))


def contact_terms(ligand_xyz: np.ndarray, ligand_elements: list[str], protein_xyz: np.ndarray, protein_elements: list[str]) -> dict:
    """Packing and clash descriptors from heavy-atom coordinates."""
    from scipy.spatial import cKDTree

    tree = cKDTree(protein_xyz)
    near = {j for i in tree.query_ball_point(ligand_xyz, CONTACT_RADIUS_A) for j in i}
    clashes = 0
    for i, neighbours in enumerate(tree.query_ball_point(ligand_xyz, CLASH_A)):
        for j in neighbours:
            polar_pair = ligand_elements[i] in ("N", "O") and protein_elements[j] in ("N", "O")
            d = float(np.linalg.norm(ligand_xyz[i] - protein_xyz[j]))
            if (not polar_pair and d < CLASH_A) or (polar_pair and d < POLAR_CLASH_A):
                clashes += 1
    return {"contacts": len(near) / max(len(ligand_xyz), 1), "clashes": float(clashes)}


def conformer_ensemble(smiles: str, n: int = N_CONFORMERS, window_kcal: float = ENSEMBLE_KCAL):
    """(RDKit mol with the kept conformers, relative GFN2-xTB energies in kcal/mol); hydrogens are present on the mol."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from cypmode.challenge import physics_block as pb

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed, params.numThreads, params.pruneRmsThresh = 7, 1, 0.3
    ids = list(AllChem.EmbedMultipleConfs(mol, n, params))
    if not ids:
        raise ValueError(f"no conformer for {smiles}")
    energies = np.array([e for _, e in AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000, numThreads=1)]) if AllChem.MMFFHasAllMoleculeParams(mol) else np.zeros(len(ids))
    keep = [i for i in np.argsort(energies) if energies[i] - energies.min() <= window_kcal + 4.0][:12]
    numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    charge = Chem.GetFormalCharge(mol)
    xtb = []
    for i in keep:
        result = pb._safe(pb._single_point, numbers, mol.GetConformer(ids[i]).GetPositions(), charge, "water")
        xtb.append(float(result.get("energy")) * pb.HARTREE_KCAL if result is not None else np.nan)
    xtb = np.array(xtb)
    ok = np.isfinite(xtb)
    rel = xtb - np.nanmin(xtb)
    chosen = [k for k in range(len(keep)) if ok[k] and rel[k] <= window_kcal]
    out = Chem.Mol(mol)
    out.RemoveAllConformers()
    for k in chosen:
        out.AddConformer(Chem.Conformer(mol.GetConformer(ids[keep[k]])), assignId=True)
    return out, np.array([rel[k] for k in chosen])


def conformer_terms(pose_mol, ensemble, relative_energies: np.ndarray) -> dict:
    """Nearest ensemble member to the pose (heavy atoms, symmetry aware): its RMSD and its relative energy."""
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign

    pose = Chem.RemoveHs(pose_mol)
    heavy = Chem.RemoveHs(ensemble)
    best = (np.inf, np.nan)
    for conf, energy in zip(heavy.GetConformers(), relative_energies, strict=False):
        probe = Chem.Mol(heavy)
        probe.RemoveAllConformers()
        probe.AddConformer(Chem.Conformer(conf), assignId=True)
        try:
            rmsd = float(rdMolAlign.GetBestRMS(probe, Chem.Mol(pose)))
        except Exception:
            continue
        if rmsd < best[0]:
            best = (rmsd, float(energy))
    return {"conf_rmsd": best[0] if np.isfinite(best[0]) else np.nan, "conf_energy": best[1]}


def pharmacophore_terms(pose_mol, smiles: str, fe_xyz: np.ndarray, donor: tuple | None, site_atom: int | None) -> dict:
    """Distances to the iron: the donor error for Type II donors, otherwise the site error of the best oxidation site."""
    from rdkit import Chem

    template = Chem.MolFromSmiles(smiles)
    match = pose_mol.GetSubstructMatch(template)
    xyz = pose_mol.GetConformer().GetPositions()
    out = {"donor_error": np.nan, "site_error": np.nan}
    if not match:
        return out
    if donor is not None:
        out["donor_error"] = abs(float(np.linalg.norm(xyz[match[donor[0]]] - fe_xyz)) - DONOR_FE_A)
    elif site_atom is not None:
        out["site_error"] = window_penalty(float(np.linalg.norm(xyz[match[site_atom]] - fe_xyz)))
    return out


def best_oxidation_site(smiles: str) -> int | None:
    """Heavy-atom index of the carbon with the lowest GFN2 hydrogen-abstraction energy (tier A)."""
    from cypmode.challenge import physics_block as pb

    _, rows = pb.compute_molecule(smiles)
    table = pd.DataFrame(rows).dropna(subset=["hat_kcal"]) if rows else pd.DataFrame()
    return int(table.sort_values("hat_kcal").iloc[0]["atom"]) if len(table) else None


def ligand_context(smiles: str) -> dict:
    """Everything about a ligand that the pose terms need and that does not depend on the pose (conformer ensemble, donor, oxidation site)."""
    from cypmode.validation.build_input import heme_donor_atom

    ensemble, energies = conformer_ensemble(smiles)
    return {"ensemble": ensemble, "energies": energies, "donor": heme_donor_atom(smiles), "site": best_oxidation_site(smiles)}


def pose_terms(cif: Path, smiles: str, context: dict) -> dict | None:
    """The term columns of one predicted pose, or None when its ligand bond orders cannot be assigned (a distorted pose)."""
    from cypmode.validation.pocket_geometry import ligand_burial_sasa
    from cypmode.validation.structures import (
        _STANDARD_RESIDUES,
        _ligand_atoms,
        _ligand_mol_3d,
        find_heme_iron,
        load_structure,
    )

    structure = load_structure(cif)
    try:
        pose_mol, _ = _ligand_mol_3d(structure, smiles)
    except ValueError:
        return None
    protein = [a for a in structure.get_atoms() if a.get_parent().get_resname() in _STANDARD_RESIDUES]
    ligand_atoms = _ligand_atoms(structure)
    row = conformer_terms(pose_mol, context["ensemble"], context["energies"])
    row.update(contact_terms(np.array([a.coord for a in ligand_atoms]), [a.element.strip() for a in ligand_atoms], np.array([a.coord for a in protein]), [a.element.strip() for a in protein]))
    row["burial"] = ligand_burial_sasa(structure)["burial_fraction"]
    row.update(pharmacophore_terms(pose_mol, smiles, np.asarray(find_heme_iron(structure).coord, dtype=float), context["donor"], context["site"]))
    return row


def within_pool_scores(pool: pd.DataFrame) -> pd.DataFrame:
    """Add physics, pharmacophore and combined scores (mean within-pool ranks; higher is better) to a per-pose term table."""
    ranks = {}
    sign = {"conf_rmsd": -1, "conf_energy": -1, "burial": 1, "contacts": 1, "clashes": -1}
    for term, s in sign.items():
        values = pool[term].to_numpy(dtype=float) * s
        ranks[term] = pd.Series(values).rank(pct=True).to_numpy() if np.isfinite(values).any() else np.full(len(pool), np.nan)
    out = pool.copy()
    out["physics"] = np.nanmean(np.column_stack(list(ranks.values())), axis=1)
    distance = pool["donor_error"].where(pool["donor_error"].notna(), pool["site_error"]).to_numpy(dtype=float)
    out["pharmacophore"] = pd.Series(-distance).rank(pct=True).to_numpy() if np.isfinite(distance).any() else np.nan
    parts = [out["physics"].to_numpy(), out["pharmacophore"].to_numpy(), pool["mbr"].rank(pct=True).to_numpy()]
    out["combined"] = np.nanmean(np.column_stack(parts), axis=1)
    return out


def gate(per_ligand: pd.DataFrame, column: str) -> dict:
    rho = per_ligand[column].dropna()
    mean, fraction = float(rho.mean()) if len(rho) else float("nan"), float((rho > 0).mean()) if len(rho) else float("nan")
    return {"mean_rho": mean, "fraction_positive": fraction, "n": len(rho), "met": bool(len(rho) >= 3 and mean > MIN_MEAN_RHO and fraction >= MIN_FRACTION_POSITIVE)}


def evaluate(poses: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-pose scores with Spearman rho per ligand and the rule outcomes. ``poses`` has compound, true_rmsd, mbr and the term columns."""
    from scipy.stats import spearmanr

    scored, rows = [], []
    for compound, pool in poses.groupby("compound", sort=True):
        s = within_pool_scores(pool.reset_index(drop=True))
        scored.append(s)
        row = {"compound": compound, "poses": len(s), "mean_pool": float(s.true_rmsd.mean()), "mbr_pick": float(s.loc[s.mbr.idxmax(), "true_rmsd"])}
        for name in ("physics", "pharmacophore", "combined"):
            valid = s[name].notna()
            row[f"rho_{name}"] = float(spearmanr(s.loc[valid, name], -s.loc[valid, "true_rmsd"]).statistic) if valid.sum() >= 3 and s.loc[valid, name].nunique() > 1 else np.nan
            row[f"{name}_pick"] = float(s.loc[s[name].idxmax(), "true_rmsd"]) if valid.any() else np.nan
        rows.append(row)
    return pd.concat(scored, ignore_index=True), pd.DataFrame(rows)


POOLED_TABLES = (
    "benchmark_engine_pose_validation_unguided_boltz.csv",
    "benchmark_engine_pose_validation_febond.csv",
    "benchmark_engine_pose_validation_fecontact.csv",
    "diversity_pilot_poses.csv",
)
_VARIANT_SUFFIX = __import__("re").compile(r"-(febond|fecontact|seed\d+|cf\d+|ss)$")


def base_ligand(run_name: str) -> str:
    return _VARIANT_SUFFIX.sub("", run_name)


def pooled_pose_list(directory: Path, table_names: tuple[str, ...] = POOLED_TABLES) -> list[tuple[str, str, int, float]]:
    """[(ligand, run name, sample, true RMSD)] of every Boltz pose of every benchmark run of a ligand (unguided, restrained, pilot levers)."""
    out = []
    for name in table_names:
        path = directory / name
        if not path.exists():
            continue
        t = pd.read_csv(path)
        t = t[(t.engine == "boltz")].dropna(subset=["true_rmsd"])
        out += [(base_ligand(r.compound), r.compound, int(r.sample), float(r.true_rmsd)) for r in t.itertuples()]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pooled", "single"), default="pooled", help="pooled: every benchmark run of a ligand is one pool (the gate); single: one table, one run per ligand (information)")
    parser.add_argument("--table", default=None, help="pose table for --mode single")
    parser.add_argument("--out", default="data/structure_track/outputs/cyp3a4/pose_awareness_poses.csv")
    parser.add_argument("--directory", default="data/structure_track/outputs/cyp3a4", help="where the pose tables of --mode pooled are read from")
    args = parser.parse_args()
    from cypmode.validation.pose_pools import pairwise_lddt, pool_paths
    from cypmode.validation.selection_validation import DEFAULT_TABLES

    directory = Path(args.directory)
    if args.mode == "pooled":
        listing = pooled_pose_list(directory)
    else:
        path = args.table or next((t for t in DEFAULT_TABLES if Path(t).exists()), None)
        if path is None:
            raise SystemExit("no pose table found (run job 160 first)")
        t = pd.read_csv(path)
        listing = [(r.compound, r.compound, int(r.sample), float(r.true_rmsd)) for r in t[t.engine == "boltz"].dropna(subset=["true_rmsd"]).itertuples()]
    smiles = dict(zip(*[pd.read_csv("data/structure_track/inputs/structure_benchmark.csv")[c] for c in ("Molecule_Name", "SMILES")], strict=True))
    by_ligand: dict[str, list[tuple[str, int, float, Path]]] = {}
    for ligand, run, sample, rmsd in listing:
        paths = {s: p for e, s, p in pool_paths(run, ("boltz",))}
        if ligand in smiles and sample in paths and paths[sample].exists():
            by_ligand.setdefault(ligand, []).append((run, sample, rmsd, paths[sample]))
    records = []
    for ligand, poses in sorted(by_ligand.items()):
        if len(poses) < 3:
            continue
        matrix = pairwise_lddt([p for *_, p in poses])
        mbr = [float(np.mean([matrix[i, j] for j in range(len(poses)) if j != i])) for i in range(len(poses))]
        context = ligand_context(smiles[ligand])
        for k, (run, sample, rmsd, cif) in enumerate(poses):
            row = pose_terms(cif, smiles[ligand], context)
            if row is not None:
                records.append({"compound": ligand, "run": run, "sample": sample, "true_rmsd": rmsd, "mbr": mbr[k], **row})
    scored, per_ligand = evaluate(pd.DataFrame(records))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.out, index=False)
    print(f"mode {args.mode}: {len(scored)} poses of {len(per_ligand)} ligands")
    print(per_ligand.round(3).to_string(index=False))
    for name in ("physics", "pharmacophore", "combined"):
        g = gate(per_ligand, f"rho_{name}")
        picks = per_ligand[f"{name}_pick"].mean()
        print(f"{name}: mean Spearman rho {g['mean_rho']:+.3f}, positive for {g['fraction_positive']:.0%} of {g['n']} ligands, mean RMSD of the top-scored pose {picks:.3f} (MBR {per_ligand.mbr_pick.mean():.3f}, random pose {per_ligand.mean_pool.mean():.3f})")
        label = "STRUCTURE GATE" if args.mode == "pooled" else "information (not a gate)"
        print(f"{label} pose_{name} (mean rho > {MIN_MEAN_RHO} and positive for >= {MIN_FRACTION_POSITIVE:.0%} of ligands): {'MET' if g['met'] else 'NOT MET'}")


if __name__ == "__main__":
    main()

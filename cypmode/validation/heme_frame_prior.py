"""Heme-frame pose prior from crystal ligands (checklist item S1, job 175).

Where does a ligand sit relative to the heme? Every pose is described in the heme's own frame, with the iron as origin and
the porphyrin normal as axis, by seven rotation-invariant numbers (the four pyrrole nitrogens make the heme nearly
four-fold symmetric, so the azimuth around the axis is not used):

    d_fe        distance from the iron to the nearest ligand heavy atom
    z_near      height of that atom above the porphyrin plane       r_near   its radial distance from the axis
    z_cent      height of the ligand centroid                       r_cent   its radial distance from the axis
    cos_long    |cos| between the ligand's long axis (first principal axis) and the heme normal
    cos_short   |cos| between the ligand's shortest principal axis (the plane normal of a flat ligand) and the heme normal

The distribution of these descriptors over the experimental CYP3A4 active-site ligands (the reference set, heme atoms read
from the cached mmCIFs) is a Gaussian kernel density in standardised coordinates; its log density is a knowledge-based
pose-scoring term. Leakage control: when a benchmark ligand is scored, every crystal reference of the same compound (by
InChIKey) is excluded from the density, so a pose is never scored against its own answer.

Gate (fixed in advance): over the pooled benchmark poses, the Spearman correlation between the prior and minus the true
RMSD, averaged over ligands, must exceed 0.3 and be positive for at least 60% of the ligands. A combined rule (mean rank of
the MBR consensus score and of the prior within each pool) is reported next to MBR alone.

Usage: python -m cypmode.validation.heme_frame_prior [--table CSV]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HEME_NITROGENS = ("NA", "NB", "NC", "ND")
DESCRIPTOR_NAMES = ["d_fe", "z_near", "r_near", "z_cent", "r_cent", "cos_long", "cos_short"]
MIN_MEAN_RHO, MIN_FRACTION_POSITIVE = 0.3, 0.6
PDB_CACHE = Path("data/structure_track/inputs/pdb_cache")


def heme_axis(pyrrole_nitrogens: np.ndarray) -> np.ndarray:
    """Unit normal of the best-fit plane through the pyrrole nitrogens (sign arbitrary)."""
    centred = pyrrole_nitrogens - pyrrole_nitrogens.mean(axis=0)
    return np.linalg.svd(centred)[2][-1]


def descriptor(ligand_xyz: np.ndarray, fe: np.ndarray, axis: np.ndarray) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    rel = ligand_xyz - fe
    z = rel @ axis
    radial = np.linalg.norm(rel - np.outer(z, axis), axis=1)
    dist = np.linalg.norm(rel, axis=1)
    nearest = int(np.argmin(dist))
    centroid = rel.mean(axis=0)
    zc = float(centroid @ axis)
    rc = float(np.linalg.norm(centroid - zc * axis))
    # the sign of the axis is arbitrary: put the ligand on the positive side so the heights are comparable
    sign = 1.0 if zc >= 0 else -1.0
    centred = rel - centroid
    if len(rel) >= 3:
        vectors = np.linalg.svd(centred, full_matrices=False)[2]
        long_axis, short_axis = vectors[0], vectors[-1]
    else:
        long_axis = short_axis = rel[0] / max(np.linalg.norm(rel[0]), 1e-9)
    return np.array([dist[nearest], sign * z[nearest], radial[nearest], sign * zc, rc, abs(long_axis @ axis), abs(short_axis @ axis)])


def structure_descriptor(structure) -> np.ndarray:
    from cypmode.validation.structures import _ligand_atoms, find_heme_iron

    fe = find_heme_iron(structure)
    heme = fe.get_parent()
    nitrogens = np.array([heme[n].coord for n in HEME_NITROGENS if n in heme], dtype=float)
    if len(nitrogens) < 4:
        raise ValueError("heme pyrrole nitrogens NA-ND not found")
    ligand = np.array([a.coord for a in _ligand_atoms(structure) if a.element.strip().upper() not in ("H", "D")], dtype=float)
    return descriptor(ligand, np.asarray(fe.coord, dtype=float), heme_axis(nitrogens))


def crystal_descriptors(references: list[dict], cache: Path = PDB_CACHE) -> tuple[np.ndarray, list[str], list[str]]:
    """Descriptors of the crystal ligands: (matrix, InChIKeys, PDB ids); entries whose heme cannot be read are skipped."""
    from Bio.PDB import MMCIFParser

    parser = MMCIFParser(QUIET=True)
    rows, keys, ids, structures = [], [], [], {}
    for record in references:
        path = cache / f"{record['pdb_id']}.cif"
        if not path.exists():
            continue
        if record["pdb_id"] not in structures:
            structures[record["pdb_id"]] = parser.get_structure(record["pdb_id"], str(path))
        fe = np.array(record["fe"], dtype=float)
        best = None
        for residue in structures[record["pdb_id"]][0].get_residues():
            if residue.resname == "HEM" and "FE" in residue and np.linalg.norm(residue["FE"].coord - fe) < 1.0 and all(n in residue for n in HEME_NITROGENS):
                best = residue
                break
        if best is None:
            continue
        coords = np.array(record["coords"], dtype=float)
        rows.append(descriptor(coords, np.asarray(best["FE"].coord, dtype=float), heme_axis(np.array([best[n].coord for n in HEME_NITROGENS], dtype=float))))
        keys.append(record["inchikey"])
        ids.append(record["pdb_id"])
    return np.array(rows), keys, ids


class HemePrior:
    """Gaussian kernel density over the descriptors of reference ligands (excluding chosen compounds)."""

    def __init__(self, descriptors: np.ndarray):
        self.mean, self.std = descriptors.mean(axis=0), descriptors.std(axis=0) + 1e-6
        self.z = (descriptors - self.mean) / self.std
        self.bandwidth = len(descriptors) ** (-1.0 / (descriptors.shape[1] + 4))  # Scott's rule on standardised coordinates

    def log_density(self, x: np.ndarray) -> np.ndarray:
        q = (np.atleast_2d(x) - self.mean) / self.std
        d2 = ((q[:, None, :] - self.z[None, :, :]) ** 2).sum(axis=2) / self.bandwidth**2
        m = (-0.5 * d2).max(axis=1)
        return m + np.log(np.exp(-0.5 * d2 - m[:, None]).sum(axis=1))


def evaluate(pools: dict[str, list[tuple[np.ndarray, float, float]]], crystal: np.ndarray, crystal_keys: list[str], inchikey_of: dict[str, str]) -> pd.DataFrame:
    """pools[compound] = [(pose descriptor, true RMSD, MBR expected LDDT)]; returns per-ligand Spearman rho and picks."""
    from scipy.stats import rankdata, spearmanr

    rows = []
    keys = np.array(crystal_keys)
    for compound, poses in pools.items():
        if len(poses) < 3:
            continue
        prior = HemePrior(crystal[keys != inchikey_of.get(compound, "")])
        x = np.array([p[0] for p in poses])
        rmsd = np.array([p[1] for p in poses])
        mbr = np.array([p[2] for p in poses])
        score = prior.log_density(x)
        rho = float(spearmanr(score, -rmsd).statistic) if np.ptp(score) > 0 else 0.0
        combined = rankdata(mbr) + rankdata(score)
        rows.append({"compound": compound, "poses": len(poses), "rho_prior_vs_minus_rmsd": rho, "prior_pick": float(rmsd[int(np.argmax(score))]),
                     "mbr_pick": float(rmsd[int(np.argmax(mbr))]), "mbr_plus_prior_pick": float(rmsd[int(np.argmax(combined))]), "mean_pool": float(rmsd.mean())})
    return pd.DataFrame(rows)


def gate(per_ligand: pd.DataFrame) -> dict:
    rho = per_ligand["rho_prior_vs_minus_rmsd"]
    mean, fraction = float(rho.mean()), float((rho > 0).mean())
    return {"mean_rho": mean, "fraction_positive": fraction, "met": bool(len(rho) >= 3 and mean > MIN_MEAN_RHO and fraction >= MIN_FRACTION_POSITIVE)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default=None)
    parser.add_argument("--out", default="data/structure_track/outputs/cyp3a4/heme_frame_prior_outcomes.csv")
    args = parser.parse_args()
    from rdkit import Chem

    from cypmode.validation.pose_pools import pairwise_lddt, pool_paths
    from cypmode.validation.reference_ligands import load_reference_set
    from cypmode.validation.selection_validation import DEFAULT_TABLES
    from cypmode.validation.structures import load_structure

    path = args.table or next((t for t in DEFAULT_TABLES if Path(t).exists()), None)
    if path is None:
        raise SystemExit("no pose table found (run job 160 first)")
    table = pd.read_csv(path)
    crystal, keys, _ = crystal_descriptors(load_reference_set())
    print(f"crystal reference descriptors: {len(crystal)} ligands from {len(set(keys))} distinct compounds")
    print("descriptor means: " + ", ".join(f"{n} {v:.2f}" for n, v in zip(DESCRIPTOR_NAMES, crystal.mean(axis=0), strict=True)))
    smiles = dict(zip(*[pd.read_csv("data/structure_track/inputs/structure_benchmark.csv")[c] for c in ("Molecule_Name", "SMILES")], strict=True))
    inchikey_of = {c: Chem.MolToInchiKey(Chem.MolFromSmiles(s)) for c, s in smiles.items()}
    truth = table.set_index(["compound", "engine", "sample"])["true_rmsd"]
    pools = {}
    for compound in sorted(table["compound"].unique()):
        poses = [(e, s, p) for e, s, p in pool_paths(compound, ("boltz",)) if p.exists()]
        if len(poses) < 3:
            continue
        matrix = pairwise_lddt([p for _, _, p in poses])
        expected = [float(np.mean([matrix[i, j] for j in range(len(poses)) if j != i])) for i in range(len(poses))]
        pools[compound] = [(structure_descriptor(load_structure(p)), truth.get((compound, e, s), np.nan), expected[i]) for i, (e, s, p) in enumerate(poses)]
        pools[compound] = [x for x in pools[compound] if not np.isnan(x[1])]
    per_ligand = evaluate(pools, crystal, keys, inchikey_of)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    per_ligand.to_csv(args.out, index=False)
    print(per_ligand.round(3).to_string(index=False))
    g = gate(per_ligand)
    print(f"mean RMSD of the picked pose: MBR {per_ligand.mbr_pick.mean():.3f}, prior alone {per_ligand.prior_pick.mean():.3f}, MBR + prior {per_ligand.mbr_plus_prior_pick.mean():.3f}, random pose {per_ligand.mean_pool.mean():.3f}")
    print(f"mean Spearman rho(prior, -RMSD) {g['mean_rho']:+.3f}, positive for {g['fraction_positive']:.0%} of {len(per_ligand)} ligands")
    print(f"STRUCTURE GATE heme_frame_prior (mean rho > {MIN_MEAN_RHO} and positive for >= {MIN_FRACTION_POSITIVE:.0%} of ligands, leave-compound-out): {'MET' if g['met'] else 'NOT MET'}")


if __name__ == "__main__":
    main()

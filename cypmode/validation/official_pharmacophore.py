# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The expected pharmacophore of the 20 official ligands (checklist item S8, part b, declared on 2026-10-23 before any code ran). HELD BACK: the files are predictions for blinded ligands.

For each official ligand, from the final submitted pose (``structures_final.zip``) and from the pool of its valid predicted poses (``official_posebusters_pools.csv``), in the frame of the heme:

1. the representation of the ligand in the pocket: its typed features with their coordinates, the iron-donor distance of a Type II donor or the distance of the most easily oxidised carbon to the iron,
   and how far the pose lies from the low-energy conformers of ``pose_awareness`` (RMSD and energy of the nearest one);
2. the site model against the pose (``pose_pharmacophore.score_pose``: field score, recall, matched sites, precision, outside share), with the fields and sites built without the references that are the
   compound or an analogue of it (the regime ``novel`` of S8 a), and which sites the pose matches;
3. the consensus of the pool, so that one pose is not taken for the whole: the share of the poses of the pool that match each site and the spread of the feature positions across the pool;
4. the receptor side of the final complex: for every donor and acceptor the nearest polar protein atom within ``PARTNER_A`` (or the iron within ``IRON_A``), for every aromatic ring and hydrophobe the residues within
   ``CONTACT_A``, and a flag for a buried polar feature without a partner (``BURIED_ATOMS`` protein heavy atoms within ``BURIED_RADIUS``);
5. a common-feature model of the 20 final poses: the sites at which at least ``COMMON_MIN`` of the 20 place a feature of the same family within ``COMMON_RADIUS``, and the pairwise similarity of the 20 as the
   cosine of their feature fields, ordered by hierarchical clustering so that the chemotypes of the set can be read;
and the two flags of ``site_representation`` (pocket RMSD to the nearest crystal pocket beyond the flag; heavy atoms outside the support beyond the cut). A flag marks a pose to be read with care.

Everything is written under ``pharmacophore_official20/`` (HELD BACK in ``.gitignore``) and nothing of it is pushed before 2026-11-03 23:59 UTC.

Usage: python -m cypmode.validation.official_pharmacophore      (job 935, in the final chain)
"""

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from cypmode.challenge import pocket_pharmacophore as pp
from cypmode.validation import pose_pharmacophore as pose

FINAL = Path("data/structure_track/outputs/cyp3a4/final")
OUT = Path("data/structure_track/outputs/cyp3a4/pharmacophore_official20")
FLAGS = Path("data/structure_track/outputs/cyp3a4/site_representation_flags.json")
PARTNER_A, IRON_A, CONTACT_A = 3.5, 2.6, 4.5
BURIED_ATOMS, BURIED_RADIUS = 25, 6.0
COMMON_MIN, COMMON_RADIUS, COMMON_SEPARATION = 14, 1.5, 2.5
STRAINED_KCAL, OUTSIDE_ENSEMBLE_A = 10.0, 1.0
POLAR = ("Donor", "Acceptor")


# -- the structures -------------------------------------------------------------------------------------------------------------------------------------------

def load_pdb(text: str):
    from Bio.PDB import PDBParser

    return PDBParser(QUIET=True).get_structure("final", io.StringIO(text))


def final_structures(path: Path = FINAL / "structures_final.zip") -> dict[str, object]:
    """{ligand: Bio.PDB structure} of the final submitted poses."""
    with zipfile.ZipFile(path) as archive:
        return {Path(name).stem: load_pdb(archive.read(name).decode()) for name in archive.namelist() if name.endswith(".pdb")}


def pool_poses(table: pd.DataFrame, ligand: str) -> list[Path]:
    """The CIF files of the valid (PoseBusters) poses of the pool of one official ligand."""
    rows = table[(table.ligand == ligand) & (table.passed.astype(str) == "True")]
    return [Path(c) for c in rows.cif if Path(c).exists()]


# -- the receptor side ----------------------------------------------------------------------------------------------------------------------------------------

def protein_atoms(structure) -> pd.DataFrame:
    """The heavy atoms of the protein and of the heme: coordinates, element, residue label and atom name."""
    from cypmode.validation.structures import _STANDARD_RESIDUES

    rows = []
    for atom in structure.get_atoms():
        residue = atom.get_parent()
        if residue.resname in _STANDARD_RESIDUES or residue.resname == "HEM":
            rows.append((*atom.coord, atom.element.strip().upper(), f"{residue.resname}{residue.id[1]}", atom.get_name()))
    return pd.DataFrame(rows, columns=["x", "y", "z", "element", "residue", "atom"])


def receptor_side(features_raw: dict[str, np.ndarray], atoms: pd.DataFrame) -> pd.DataFrame:
    """One row per typed feature of a pose given in the frame of the structure: the nearest polar partner of a donor or acceptor, the residues in contact with an aromatic ring or hydrophobe,
    the number of protein atoms around it and whether it is a buried polar feature without a partner. A heme iron within ``IRON_A`` counts as the partner of an acceptor."""
    xyz = atoms[["x", "y", "z"]].to_numpy(dtype=float)
    polar = atoms.element.isin(["N", "O"]).to_numpy()
    iron = (atoms.element == "FE").to_numpy()
    rows = []
    for family, positions in features_raw.items():
        for k, position in enumerate(positions):
            distance = np.linalg.norm(xyz - position, axis=1)
            near = int((distance <= BURIED_RADIUS).sum())
            row = {"family": family, "index": k, "protein_atoms_within_6A": near, "buried": near >= BURIED_ATOMS, "partner": "", "partner_distance": np.nan, "contacts": "", "unsatisfied": False}
            if family in POLAR:
                candidates = np.flatnonzero(polar)
                if family == "Acceptor" and iron.any() and distance[iron].min() <= IRON_A:
                    j = int(np.flatnonzero(iron)[distance[iron].argmin()])
                    row.update(partner=f"{atoms.residue[j]}:{atoms.atom[j]}", partner_distance=float(distance[j]))
                elif len(candidates) and distance[candidates].min() <= PARTNER_A:
                    j = int(candidates[distance[candidates].argmin()])
                    row.update(partner=f"{atoms.residue[j]}:{atoms.atom[j]}", partner_distance=float(distance[j]))
                row["unsatisfied"] = bool(row["buried"] and not row["partner"])
            elif family in ("Aromatic", "Hydrophobe"):
                close = pd.Series(distance[distance <= CONTACT_A], index=atoms.residue[distance <= CONTACT_A].to_numpy()).groupby(level=0).min().sort_values()
                row["contacts"] = " ".join(f"{name}({value:.1f})" for name, value in close.head(6).items())
            rows.append(row)
    return pd.DataFrame(rows)


# -- sites matched, the pool and the common model ---------------------------------------------------------------------------------------------------------------

def matched_sites(features: dict[str, np.ndarray], sites: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per site: its family, position in the frame of the heme, weight and whether a feature of that family lies in its sphere."""
    rows = []
    for family, family_sites in sites.items():
        for site in family_sites:
            positions = features[family]
            hit = bool(len(positions) and (np.linalg.norm(positions - site[:3], axis=1) <= pp.SITE_RADIUS).any())
            rows.append({"family": family, "x": site[0], "y": site[1], "z": site[2], "weight": site[3], "matched": hit})
    return pd.DataFrame(rows)


def feature_spread(final: dict[str, np.ndarray], others: list[dict[str, np.ndarray]]) -> float:
    """The median, over the features of the final pose and over the other poses of the pool that have a feature of that family, of the distance to the nearest such feature: how far the pharmacophore moves across the pool."""
    values = []
    for family, positions in final.items():
        for position in positions:
            for other in others:
                if len(other[family]):
                    values.append(float(np.linalg.norm(other[family] - position, axis=1).min()))
    return float(np.median(values)) if values else float("nan")


def same_features(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> bool:
    """Whether two poses have the same features at the same positions (the final pose is usually one of the poses of its pool and must not be compared with itself)."""
    return all(a[f].shape == b[f].shape and np.allclose(a[f], b[f], atol=1e-2) for f in pp.FAMILIES)


def common_sites(features_by_ligand: dict[str, dict[str, np.ndarray]], minimum: int = COMMON_MIN) -> pd.DataFrame:
    """The sites at which at least ``minimum`` ligands place a feature of the same family within ``COMMON_RADIUS``: grid points that are local maxima of the count (the summed Gaussian density breaks
    ties), at least ``COMMON_SEPARATION`` apart, with the number of ligands."""
    axes = [pp.GRID_ORIGIN[k] + np.arange(pp.GRID_SHAPE[k]) for k in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    rows = []
    for family in pp.FAMILIES:
        count = np.zeros(len(grid))
        density = np.zeros(len(grid))
        for features in features_by_ligand.values():
            positions = features[family]
            if len(positions):
                nearest = np.linalg.norm(grid[:, None, :] - positions[None, :, :], axis=2).min(axis=1)
                count += nearest <= COMMON_RADIUS
                density += np.exp(-(nearest**2) / (2 * pp.SIGMA_FEATURE**2))
        for i in np.lexsort((-density, -count)):
            if count[i] < minimum:
                break
            if all(np.linalg.norm(grid[i] - np.array([r["x"], r["y"], r["z"]])) >= COMMON_SEPARATION for r in rows if r["family"] == family):
                rows.append({"family": family, "x": grid[i][0], "y": grid[i][1], "z": grid[i][2], "ligands": int(count[i])})
    return pd.DataFrame(rows, columns=["family", "x", "y", "z", "ligands"])


def similarity(features_by_ligand: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    """The cosine similarity of the feature fields (six families, sigma of the fields) of every pair of ligands, the rows and columns ordered by average-linkage clustering."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    names = list(features_by_ligand)
    vectors = []
    for features in features_by_ligand.values():
        vectors.append(np.concatenate([pp.gaussian_field(features[f], np.ones(len(features[f])), pp.SIGMA_FEATURE).ravel() if len(features[f]) else np.zeros(int(np.prod(pp.GRID_SHAPE))) for f in pp.FAMILIES]))
    matrix = np.array(vectors)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    cosine = (matrix / np.maximum(norm, 1e-12)) @ (matrix / np.maximum(norm, 1e-12)).T
    order = leaves_list(linkage(squareform(np.clip(1 - cosine, 0, None), checks=False), "average")) if len(names) > 2 else np.arange(len(names))
    return pd.DataFrame(cosine[np.ix_(order, order)], index=[names[i] for i in order], columns=[names[i] for i in order])


# -- one ligand ----------------------------------------------------------------------------------------------------------------------------------------------

def analyse(ligand: str, smiles: str, structure, pool: list[Path], flags: dict, references) -> dict:
    """Every table row of one ligand: the summary, the features, the sites matched, the pool consensus and the receptor side."""
    from cypmode.validation.pose_awareness import (
        conformer_terms,
        ligand_context,
        pharmacophore_terms,
    )
    from cypmode.validation.pose_rescoring import connectivity_key, structure_arrays
    from cypmode.validation.structures import load_structure

    fields, sites = pose.fields_and_sites(pose.kept_indices(smiles, "novel", references))
    frame = pose.pose_in_frame(structure, smiles)
    if frame is None:
        return {"summary": {"ligand": ligand, "error": "the final pose could not be read"}, "features": [], "sites": pd.DataFrame(), "receptor": pd.DataFrame(), "frame": None}
    score = pose.score_pose(frame["heavy"], frame["features"], fields, sites)
    matches = matched_sites(frame["features"], sites)
    context = ligand_context(smiles)
    geometry = pharmacophore_terms(frame["mol"], smiles, frame["iron"], context["donor"], context["site"])
    strain = conformer_terms(frame["mol"], context["ensemble"], context["energies"])
    pool_features, pool_scores, pool_matches = [], [], []
    for cif in pool:
        other = pose.pose_in_frame(load_structure(cif), smiles)
        if other is None:
            continue
        pool_features.append(other["features"])
        pool_scores.append(pose.score_pose(other["heavy"], other["features"], fields, sites))
        pool_matches.append(matched_sites(other["features"], sites).matched.to_numpy())
    share = np.mean(pool_matches, axis=0) if pool_matches else np.full(len(matches), np.nan)
    receptor = receptor_side(frame["features_raw"], protein_atoms(structure))
    _, _, pocket_ca = structure_arrays(structure)
    from cypmode.validation import site_representation as site

    pockets = site.pockets_of(references)
    keys = [connectivity_key(r["smiles"]) for r in references]
    pocket_rmsd, _ = site.nearest_reference({int(k): v for k, v in pocket_ca.items()}, pockets, keys, connectivity_key(smiles), site.core_residues(pockets))
    recall = np.array([s["recall"] for s in pool_scores]) if pool_scores else np.array([np.nan])
    summary = {"ligand": ligand, "features": int(sum(len(v) for v in frame["features"].values())), **{f"n_{f.lower()}": len(frame["features"][f]) for f in pp.FAMILIES},
               "donor_error": geometry["donor_error"], "site_error": geometry["site_error"], **score, "conf_rmsd": strain["conf_rmsd"], "conf_energy": strain["conf_energy"],
               "strained": bool(strain["conf_energy"] > STRAINED_KCAL), "outside_ensemble": bool(strain["conf_rmsd"] > OUTSIDE_ENSEMBLE_A), "pocket_rmsd": pocket_rmsd,
               "beyond_pocket_flag": bool(pocket_rmsd > flags["pocket_rmsd_flag"]), "beyond_support_cut": bool(score["outside"] > flags["support_cut"]),
               "pool_poses": len(pool_scores), "pool_recall_mean": float(np.nanmean(recall)), "pool_recall_sd": float(np.nanstd(recall)),
               "pool_spread": feature_spread(frame["features"], [f for f in pool_features if not same_features(f, frame["features"])]), "unsatisfied_polar": int(receptor.unsatisfied.sum())}
    features = [{"ligand": ligand, "family": f, "x": p[0], "y": p[1], "z": p[2]} for f, v in frame["features"].items() for p in v]
    return {"summary": summary, "features": features, "sites": matches.assign(ligand=ligand, pool_share=share), "receptor": receptor.assign(ligand=ligand), "frame": frame["features"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--flags", default=str(FLAGS), help="the flags written by site_representation")
    parser.add_argument("--limit", type=int, default=0, help="only the first N ligands (a test run)")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    flags = json.loads(Path(args.flags).read_text())
    refs = pose.references()
    structures = final_structures()
    pools = pd.read_csv(FINAL / "official_posebusters_pools.csv")
    official = pd.read_csv("data/structure_track/inputs/structure_track_test_blinded.csv")
    ligands = list(zip(official.Molecule_Name, official.SMILES, strict=True))[: args.limit or None]
    results = {name: analyse(name, smiles, structures[name], pool_poses(pools, name), flags, refs) for name, smiles in ligands if name in structures}
    summary = pd.DataFrame([r["summary"] for r in results.values()])
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame([row for r in results.values() for row in r["features"]]).to_csv(out / "features.csv", index=False)
    pd.concat([r["sites"] for r in results.values() if len(r["sites"])], ignore_index=True).to_csv(out / "site_matches.csv", index=False)
    pd.concat([r["receptor"] for r in results.values() if len(r["receptor"])], ignore_index=True).to_csv(out / "receptor_side.csv", index=False)
    frames = {name: r["frame"] for name, r in results.items() if r["frame"] is not None}
    common = common_sites(frames)
    common.to_csv(out / "common_sites.csv", index=False)
    similarity(frames).to_csv(out / "similarity.csv")
    print(f"{len(summary)} official ligands; common-feature sites (at least {COMMON_MIN} of {len(frames)} ligands): {len(common)}; flags: pocket beyond the flag for {int(summary.beyond_pocket_flag.sum())}, "
          f"outside the support beyond the cut for {int(summary.beyond_support_cut.sum())}, outside the low-energy ensemble for {int(summary.outside_ensemble.sum())}, strained for {int(summary.strained.sum())}")


if __name__ == "__main__":
    from cypmode import provenance

    provenance.announce()
    main()

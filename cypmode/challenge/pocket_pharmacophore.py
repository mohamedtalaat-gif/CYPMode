# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Pocket-level pharmacophore block (checklist item "Pocket-level pharmacophore block", declared on 2026-10-23 before any result).

Every other pharmacophore representation of the project describes the ligand alone. This one describes a compound relative to the pocket of one enzyme.

Pocket. The active-site ligands of the public crystal complexes of an enzyme (``reference_ligands.py``: UniProt P08684 for CYP3A4, P10635 for CYP2D6) are put in the frame of
their heme: the iron is the origin, z is the porphyrin normal pointing to the side of the ligand, x points to the pyrrole nitrogen NA, y completes the frame. The positions of
the RDKit feature families (donor, acceptor, aromatic ring, hydrophobe, positive and negative ionisable) become Gaussian density fields (sigma 1.5 A, a grid of 1 A, maximum 1), a ligand
counting 1 / (its number of copies) so that a ligand with many entries does not dominate; the heavy atoms give an occupancy field (sigma 1.2 A). No label is used.

Compound. Up to 12 ETKDG conformers within 5 kcal/mol (MMFF, seed 42) are placed in the frame by an anchor atom and 1080 fixed orientations (60 directions in a cone of 70 degrees around z, 18
spins of 20 degrees). In the donor mode the Type II donor nitrogen is the anchor, 2.2 A above the iron; in the substrate mode each non-aromatic carbon that bears a hydrogen is the anchor in
turn, 4.5 A above it. A pose is refused when another heavy atom is lower than 1.5 A above the porphyrin plane or closer than 2.0 A to the iron. A pose scores the mean occupancy of its heavy
atoms plus the mean, over the families that the compound has, of the mean field value at its features (read at the nearest grid point); no weight is fitted. For each mode that applies the block
holds the best score, its occupancy part, its six family parts, the share of accepted poses, the height of the centroid and the distance of the nearest heavy atom at the best pose (11 columns).

Site model (checklist item "Pharmacophore modelling", declared on 2026-10-23 before this code ran). The same fields give a discrete structure-based pharmacophore model per enzyme: in each of the six feature
fields the local maxima of at least 0.35 (the maximum of a field is 1), taken in order of value and at least 2.5 A apart, at most 4 per family, are the sites of the model, each weighted by its field value
and with a tolerance sphere of 1.5 A. A placed compound matches a site when a feature of that family lies in the sphere. The columns, for each mode that applies: the largest weight of matched sites over the
weight of all sites (recall), the number of sites matched at that pose and the share of the compound's typed features that lie in a sphere of a site of their family (precision); the pose has the largest recall,
the larger precision breaking a tie.

Usage: python -m cypmode.challenge.pocket_pharmacophore --extract      fields of both pockets, then the block and the site model for every compound of both files
"""

import argparse
import functools
import json
import os
from pathlib import Path

import numpy as np

FAMILIES = ("Donor", "Acceptor", "Aromatic", "Hydrophobe", "PosIonizable", "NegIonizable")
POCKETS = {"cyp3a4": Path("data/structure_track/inputs/cyp3a4_reference_ligands.json"), "cyp2d6": Path("data/structure_track/inputs/cyp2d6_reference_ligands.json")}
PDB_CACHE = Path("data/structure_track/inputs/pdb_cache")
FIELDS_DIR = Path("data/tdi_track/outputs/caches")
GRID_ORIGIN = np.array([-14.0, -14.0, -3.0])
GRID_SHAPE = (29, 29, 20)  # 1 A spacing: x and y from -14 to 14 A, z from -3 to 16 A
SIGMA_FEATURE, SIGMA_ATOM = 1.5, 1.2
N_CONFORMERS, ENERGY_WINDOW = 12, 5.0
CONFORMER_SEED = 42  # not 0: with this RDKit, EmbedMultipleConfs(..., randomSeed=0) returns the same conformer N times
N_DIRECTIONS, CONE_DEGREES, N_SPINS = 60, 70.0, 18
ANCHOR_HEIGHT = {"donor": 2.2, "substrate": 4.5}
MIN_HEIGHT, MIN_IRON_DISTANCE = 1.5, 2.0
MODES = ("donor", "substrate")
PART_COLUMNS = ("best", "occupancy", *(f.lower() for f in FAMILIES), "accepted", "z_centroid", "d_nearest")
SITE_MIN_VALUE, SITE_SEPARATION, SITES_PER_FAMILY, SITE_RADIUS = 0.35, 2.5, 4, 1.5
SITE_PARTS = ("recall", "matched", "precision")


# -- the frame of a heme ------------------------------------------------------------------------------------------------------------------------------------

def heme_frame(structure, iron: np.ndarray, ligand_xyz: np.ndarray) -> np.ndarray | None:
    """Rows x, y, z of the frame of the heme whose iron is ``iron`` (a gemmi structure): z is the porphyrin normal on the side of the ligand, x points to the pyrrole nitrogen NA."""
    nitrogens = {}
    for chain in structure[0]:
        for residue in chain:
            if residue.name == "HEM":
                for atom in residue:
                    position = np.array(atom.pos.tolist())
                    if atom.name in ("NA", "NB", "NC", "ND") and np.linalg.norm(position - iron) < 3.0:
                        nitrogens[atom.name] = position
    return frame_from_nitrogens(nitrogens, iron, ligand_xyz)


def frame_from_nitrogens(nitrogens: dict[str, np.ndarray], iron: np.ndarray, ligand_xyz: np.ndarray) -> np.ndarray | None:
    """The frame of a heme from the positions of its four pyrrole nitrogens (keys NA, NB, NC, ND), whatever structure reader found them; None when one is missing."""
    if len(nitrogens) < 4:
        return None
    ring = np.array([nitrogens[n] for n in ("NA", "NB", "NC", "ND")])
    z = np.linalg.svd(ring - ring.mean(axis=0))[2][-1]
    if (ligand_xyz.mean(axis=0) - iron) @ z < 0:
        z = -z
    x = nitrogens["NA"] - iron
    x = x - (x @ z) * z
    x /= np.linalg.norm(x)
    return np.array([x, np.cross(z, x), z])


def to_frame(xyz: np.ndarray, iron: np.ndarray, frame: np.ndarray) -> np.ndarray:
    return (np.asarray(xyz) - iron) @ frame.T


# -- features -----------------------------------------------------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def feature_factory():
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures

    return ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))


def features_of(mol, factory, conf_id: int = -1) -> dict[str, np.ndarray]:
    """{family: positions (n, 3)} of the RDKit feature families of a molecule with a conformer."""
    out = {f: [] for f in FAMILIES}
    for feature in factory.GetFeaturesForMol(mol, confId=conf_id):
        family = feature.GetFamily()
        if family in out:
            position = feature.GetPos()
            out[family].append([position.x, position.y, position.z])
    return {f: np.array(v, dtype=float).reshape(-1, 3) for f, v in out.items()}


def crystal_pose(record: dict):
    """The crystal ligand as an RDKit molecule with its template's bond orders and its crystal coordinates (heavy atoms), or None when the template does not fit."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    lines = [f"HETATM{i + 1:5d} {e:<4s} LIG A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {e:>2s}" for i, (e, (x, y, z)) in enumerate(zip(record["elements"], record["coords"], strict=True))]
    pose = Chem.MolFromPDBBlock("\n".join(lines) + "\nEND\n", removeHs=True, proximityBonding=True)
    template = Chem.MolFromSmiles(record["smiles"]) if record.get("smiles") else None
    if pose is None or template is None:
        return None
    try:
        return AllChem.AssignBondOrdersFromTemplate(template, pose)
    except Exception:
        return None


# -- the fields of a pocket ----------------------------------------------------------------------------------------------------------------------------------

def gaussian_field(points: np.ndarray, weights: np.ndarray, sigma: float) -> np.ndarray:
    """The sum of weighted Gaussians on the grid, scaled to a maximum of 1 (zeros when there is no point)."""
    axes = [GRID_ORIGIN[k] + np.arange(GRID_SHAPE[k]) for k in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    field = np.zeros(len(grid))
    for point, weight in zip(points, weights, strict=True):
        field += weight * np.exp(-((grid - point) ** 2).sum(axis=1) / (2 * sigma**2))
    field = field.reshape(GRID_SHAPE)
    return field / field.max() if field.max() > 0 else field


def build_fields(records: list[dict], cache: Path = PDB_CACHE) -> dict[str, np.ndarray]:
    """The density fields of one pocket from its crystal ligand copies: one per feature family and the occupancy of the heavy atoms."""
    import gemmi
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")  # the template assignment prints a notice for every ligand with symmetric atoms
    copies = {}
    for record in records:
        copies[record["comp_id"]] = copies.get(record["comp_id"], 0) + 1
    factory = feature_factory()
    points = {f: [] for f in FAMILIES}
    weights = {f: [] for f in FAMILIES}
    atoms, atom_weights = [], []
    for record in records:
        pose = crystal_pose(record)
        path = cache / f"{record['pdb_id']}.cif"
        if pose is None or not path.exists():
            continue
        structure = gemmi.read_structure(str(path))
        iron = np.array(record["fe"], dtype=float)
        xyz = np.array(record["coords"], dtype=float)
        frame = heme_frame(structure, iron, xyz)
        if frame is None:
            continue
        weight = 1.0 / copies[record["comp_id"]]
        found = features_of(pose, factory)
        for family in FAMILIES:
            if len(found[family]):
                points[family].append(to_frame(found[family], iron, frame))
                weights[family].append(np.full(len(found[family]), weight))
        atoms.append(to_frame(xyz, iron, frame))
        atom_weights.append(np.full(len(xyz), weight))
    fields = {}
    for family in FAMILIES:
        fields[family] = gaussian_field(np.concatenate(points[family]), np.concatenate(weights[family]), SIGMA_FEATURE) if points[family] else np.zeros(GRID_SHAPE)
    fields["occupancy"] = gaussian_field(np.concatenate(atoms), np.concatenate(atom_weights), SIGMA_ATOM)
    return fields


def lookup(field: np.ndarray, points: np.ndarray) -> np.ndarray:
    """The field value at the nearest grid point of each position; zero outside the grid."""
    index = np.rint(points - GRID_ORIGIN).astype(int)
    inside = ((index >= 0) & (index < np.array(GRID_SHAPE))).all(axis=-1)
    out = np.zeros(points.shape[:-1])
    clipped = np.clip(index, 0, np.array(GRID_SHAPE) - 1)
    values = field[clipped[..., 0], clipped[..., 1], clipped[..., 2]]
    out[inside] = values[inside]
    return out


# -- the sites of the model -----------------------------------------------------------------------------------------------------------------------------------

def sites_of(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """{family: (n, 4) array of x, y, z in the heme frame and the weight}: the sites of the structure-based model, see the module docstring. No label is read."""
    out = {}
    shape = np.array(GRID_SHAPE)
    for family in FAMILIES:
        field = fields[family]
        padded = np.pad(field, 1, constant_values=-1.0)
        neighbour_max = np.full(field.shape, -1.0)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx or dy or dz:
                        shifted = padded[1 + dx : 1 + dx + shape[0], 1 + dy : 1 + dy + shape[1], 1 + dz : 1 + dz + shape[2]]
                        neighbour_max = np.maximum(neighbour_max, shifted)
        points = np.argwhere((field >= neighbour_max) & (field >= SITE_MIN_VALUE))
        values = field[tuple(points.T)] if len(points) else np.zeros(0)
        kept: list[np.ndarray] = []
        for i in np.argsort(-values, kind="stable"):
            xyz = GRID_ORIGIN + points[i]
            if all(np.linalg.norm(xyz - site[:3]) >= SITE_SEPARATION for site in kept):
                kept.append(np.append(xyz, values[i]))
            if len(kept) == SITES_PER_FAMILY:
                break
        out[family] = np.array(kept, dtype=float).reshape(-1, 4)
    return out


def site_match(moved: np.ndarray | None, owner: np.ndarray | None, present: list[int], sites: dict[str, np.ndarray], n_poses: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(recall, matched, precision) of every pose: ``moved`` are the positions (poses, features, 3) of the compound's typed features, ``owner`` the family index of each feature (both None
    when the compound has no typed feature, which then matches no site and has no precision)."""
    recall, matched = np.zeros(n_poses), np.zeros(n_poses)
    hits = []
    for k in present:
        family_sites = sites[FAMILIES[k]]
        positions = moved[:, owner == k]
        if not len(family_sites):
            hits.append(np.zeros(positions.shape[:2], dtype=bool))
            continue
        near = np.linalg.norm(positions[:, :, None, :] - family_sites[None, None, :, :3], axis=3) <= SITE_RADIUS  # (poses, features, sites)
        site_hit = near.any(axis=1)
        recall += (site_hit * family_sites[:, 3]).sum(axis=1)
        matched += site_hit.sum(axis=1)
        hits.append(near.any(axis=2))
    total = sum(float(s[:, 3].sum()) for s in sites.values())
    precision = np.concatenate(hits, axis=1).mean(axis=1) if hits else np.full(n_poses, np.nan)
    return (recall / total if total > 0 else np.full(n_poses, np.nan)), matched, precision


# -- the placement of a compound -----------------------------------------------------------------------------------------------------------------------------

def orientations() -> np.ndarray:
    """(N_DIRECTIONS * N_SPINS, 3, 3) rotations: a direction in the cone around z, then a spin about it; applied to a molecule whose centroid direction is z."""
    golden = np.pi * (3 - np.sqrt(5))
    cos_max = np.cos(np.radians(CONE_DEGREES))
    rotations = []
    for i in range(N_DIRECTIONS):
        cos_theta = 1 - (1 - cos_max) * (i + 0.5) / N_DIRECTIONS
        theta, phi = np.arccos(cos_theta), golden * i
        axis = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), cos_theta])
        tilt = rotation_between(np.array([0.0, 0.0, 1.0]), axis)
        for j in range(N_SPINS):
            rotations.append(axis_angle(axis, 2 * np.pi * j / N_SPINS) @ tilt)
    return np.array(rotations)


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The rotation that takes the unit vector a to the unit vector b."""
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else axis_angle(np.array([1.0, 0.0, 0.0]), np.pi)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + (k @ k) * (1 / (1 + c))


ROTATIONS = None


def score_poses(heavy: np.ndarray, anchor: int, feats: dict[str, np.ndarray], mode: str, fields: dict[str, np.ndarray], sites: dict[str, np.ndarray] | None = None) -> dict | None:
    """Best pose of one conformer in one mode: the parts of its score, or None when no pose is accepted. ``heavy`` are the heavy-atom coordinates, ``feats`` the feature positions of the conformer.
    With ``sites`` the result also holds the site model at its own best pose: ``site_recall``, ``site_matched`` and ``site_precision``."""
    global ROTATIONS
    if ROTATIONS is None:
        ROTATIONS = orientations()
    others = np.arange(len(heavy)) != anchor
    relative = heavy - heavy[anchor]
    centroid = relative.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm < 1e-6:
        return None
    to_z = rotation_between(centroid / norm, np.array([0.0, 0.0, 1.0]))
    base = np.array([0.0, 0.0, ANCHOR_HEIGHT[mode]])
    atoms = np.einsum("kij,nj->kni", ROTATIONS, relative @ to_z.T) + base  # (poses, atoms, 3)
    keep = (atoms[:, others, 2] >= MIN_HEIGHT).all(axis=1) & (np.linalg.norm(atoms[:, others], axis=2) >= MIN_IRON_DISTANCE).all(axis=1)
    if not keep.any():
        return None
    atoms = atoms[keep]
    occupancy = lookup(fields["occupancy"], atoms).mean(axis=1)
    present = [k for k, family in enumerate(FAMILIES) if len(feats[family])]
    family_parts = np.zeros((len(atoms), len(FAMILIES)))
    moved = owner = None
    if present:
        stacked = np.concatenate([feats[FAMILIES[k]] for k in present])
        owner = np.concatenate([np.full(len(feats[FAMILIES[k]]), k) for k in present])
        moved = np.einsum("kij,nj->kni", ROTATIONS[keep], (stacked - heavy[anchor]) @ to_z.T) + base
        for k in present:
            family_parts[:, k] = lookup(fields[FAMILIES[k]], moved[:, owner == k]).mean(axis=1)
    total = occupancy + (family_parts[:, present].mean(axis=1) if present else 0.0)
    best = int(np.argmax(total))
    parts = {"best": float(total[best]), "occupancy": float(occupancy[best]), "accepted": float(keep.mean())}
    for k, family in enumerate(FAMILIES):
        parts[family.lower()] = float(family_parts[best, k]) if k in present else float("nan")
    parts["z_centroid"] = float(atoms[best].mean(axis=0)[2])
    parts["d_nearest"] = float(np.linalg.norm(atoms[best], axis=1).min())
    if sites is not None:
        recall, matched, precision = site_match(moved, owner, present, sites, len(atoms))
        pick = int(np.lexsort((np.nan_to_num(precision, nan=-1.0), np.nan_to_num(recall, nan=-1.0)))[-1])
        parts.update(site_recall=float(recall[pick]), site_matched=float(matched[pick]), site_precision=float(precision[pick]))
    return parts


def conformers(mol, seed: int = CONFORMER_SEED):
    """(molecule without hydrogens that carries the conformers, their ids): up to N_CONFORMERS within ENERGY_WINDOW kcal/mol of the lowest, MMFF-relaxed."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(mol)
    ids = list(AllChem.EmbedMultipleConfs(mol, N_CONFORMERS, randomSeed=seed))
    if not ids:
        return None, []
    results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
    energies = np.array([e for _, e in results])
    ids = [cid for cid, e in zip(ids, energies, strict=True) if e <= energies.min() + ENERGY_WINDOW]
    return Chem.RemoveHs(mol), ids


def compound_columns(smiles: str, fields: dict[str, np.ndarray], pocket: str, sites: dict[str, np.ndarray] | None = None) -> dict[str, float]:
    """The 22 columns of the pocket block of one compound for one pocket (11 per mode) and the 6 of the site model (3 per mode), NaN for a mode that does not apply or that accepts no pose."""
    from rdkit import Chem

    from cypmode.validation.build_input import heme_donor_atom

    sites = sites_of(fields) if sites is None else sites
    columns = {f"pp_{pocket}_{mode}_{part}": float("nan") for mode in MODES for part in PART_COLUMNS}
    columns.update({f"st_{pocket}_{mode}_{part}": float("nan") for mode in MODES for part in SITE_PARTS})
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        return columns
    mol, ids = conformers(template)
    if mol is None:
        return columns
    factory = feature_factory()
    donor = heme_donor_atom(smiles)
    anchors = {"donor": [donor[0]] if donor else [], "substrate": [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C" and not a.GetIsAromatic() and a.GetTotalNumHs() > 0]}
    best, best_site = {}, {}
    for mode in MODES:
        for cid in ids:
            heavy = mol.GetConformer(cid).GetPositions()
            feats = features_of(mol, factory, cid)
            for anchor in anchors[mode]:
                result = score_poses(heavy, anchor, feats, mode, fields, sites)
                if result is None:
                    continue
                if mode not in best or result["best"] > best[mode]["best"]:
                    best[mode] = result
                key = (np.nan_to_num(result["site_recall"], nan=-1.0), np.nan_to_num(result["site_precision"], nan=-1.0))
                if mode not in best_site or key > best_site[mode][0]:
                    best_site[mode] = (key, result)
    for mode, result in best.items():
        for part in PART_COLUMNS:
            columns[f"pp_{pocket}_{mode}_{part}"] = result[part]
    for mode, (_, result) in best_site.items():
        for part in SITE_PARTS:
            columns[f"st_{pocket}_{mode}_{part}"] = result[f"site_{part}"]
    return columns


# -- extraction ---------------------------------------------------------------------------------------------------------------------------------------------

def fields_path(pocket: str) -> Path:
    return FIELDS_DIR / f"pocket_fields_{pocket}.npz"


def load_fields(pocket: str) -> dict[str, np.ndarray]:
    with np.load(fields_path(pocket)) as archive:
        return {name: archive[name] for name in archive.files}


def all_smiles() -> list[str]:
    """The canonical SMILES of every compound of the training and the test file, in file order without repeats (the keys of the feature store)."""
    import pandas as pd
    from rdkit import Chem

    smiles = list(pd.read_csv("data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv").SMILES) + list(pd.read_csv("data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv").SMILES)
    return list(dict.fromkeys(Chem.MolToSmiles(m) for m in (Chem.MolFromSmiles(s) for s in smiles) if m is not None))


def extract(workers: int = 4) -> None:
    import multiprocessing

    import pandas as pd

    FIELDS_DIR.mkdir(parents=True, exist_ok=True)
    for pocket, reference in POCKETS.items():
        records = json.loads(reference.read_text())["ligands"]
        fields = build_fields(records)
        np.savez_compressed(fields_path(pocket), **fields)
        print(f"{pocket}: {len(records)} ligand copies, fields {sorted(fields)}", flush=True)
    unique = all_smiles()
    with multiprocessing.Pool(workers, initializer=_init) as pool:
        rows = pool.map(_row, unique, chunksize=20)
    table = pd.DataFrame(rows, index=unique)
    table.index.name = "SMILES"
    table.to_csv(FIELDS_DIR / "pocket_pharmacophore_cache.csv")
    print(f"wrote {len(table)} compounds x {table.shape[1]} columns", flush=True)


_FIELDS: dict[str, dict[str, np.ndarray]] = {}


_SITES: dict[str, dict[str, np.ndarray]] = {}


def _init() -> None:
    for pocket in POCKETS:
        _FIELDS[pocket] = load_fields(pocket)
        _SITES[pocket] = sites_of(_FIELDS[pocket])


def _row(smiles: str) -> dict[str, float]:
    row = {}
    for pocket in POCKETS:
        row.update(compound_columns(smiles, _FIELDS[pocket], pocket, _SITES[pocket]))
    return row


def pocket_columns(pocket: str) -> list[str]:
    return [f"pp_{pocket}_{mode}_{part}" for mode in MODES for part in PART_COLUMNS]


def site_columns(pocket: str) -> list[str]:
    return [f"st_{pocket}_{mode}_{part}" for mode in MODES for part in SITE_PARTS]


def _read(smiles: list[str], columns: list[str], cache: Path) -> np.ndarray:
    import pandas as pd

    table = pd.read_csv(cache, index_col="SMILES")
    missing = [s for s in smiles if s not in table.index]
    if missing:
        raise KeyError(f"{len(missing)} compounds are not in {cache}; run `python -m cypmode.challenge.pocket_pharmacophore --extract`")
    return table.loc[smiles, columns].to_numpy(dtype=float)


def load_pocket_pharmacophore(smiles: list[str], pocket: str, cache: Path = FIELDS_DIR / "pocket_pharmacophore_cache.csv") -> np.ndarray:
    """The 22 columns of one pocket for the given canonical SMILES, read from the cache that ``--extract`` wrote; a compound that is not in it is an error, a missing value stays NaN."""
    return _read(smiles, pocket_columns(pocket), cache)


def load_site_model(smiles: list[str], pocket: str, cache: Path = FIELDS_DIR / "pocket_pharmacophore_cache.csv") -> np.ndarray:
    """The 6 columns of the site model of one pocket, from the same cache."""
    return _read(smiles, site_columns(pocket), cache)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not args.extract:
        raise SystemExit("give --extract")
    extract(args.workers)


if __name__ == "__main__":
    from cypmode import provenance

    provenance.announce()
    main()

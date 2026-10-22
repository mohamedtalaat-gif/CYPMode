# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Basic-centre block: where the molecule carries its positive charge at physiological pH, and how far that charge sits from the aromatic rings.

CYP2D6 binds substrates and inhibitors that carry a protonated basic nitrogen a few angstrom from a flat aromatic ring: the cation meets Asp301 and Glu216 while the ring
sits over the heme. A fingerprint sees the substructure; it does not know that the nitrogen is protonated in the assay buffer, how many sites could be, or the distance
between the charge and the ring in a real conformation. The block gives the model those quantities.

Per compound (label-free, computed from the structure alone):
  * protonation states at pH 6.4 to 8.4 (Dimorphite-DL 2.1, Apache-2.0): the number of states, the share of states that carry a positive nitrogen, the largest number of
    positive charges, and the class counts of the most protonated state (primary, secondary and tertiary ammonium, amidinium or guanidinium, aromatic N-H cation,
    quaternary);
  * topology: the shortest bond path from a basic centre to the nearest aromatic atom, the mean over centres, and the number of rotatable bonds of the molecule (capped at 12);
  * geometry: ten ETKDG conformers of the most protonated state; the distance from each basic nitrogen to every aromatic ring centroid, summarised as the mean nearest and farthest
    distance, the share of conformers with a ring centroid 4.5 to 7.5 angstrom away (the window of the CYP2D6 pharmacophore), and the distance of the nitrogen to the
    centre of the molecule in units of its radius of gyration.

A compound without a basic centre gets the flag 0 and zeros in the other columns; a compound without an aromatic ring gets -1 in the ring columns.

``python -m cypmode.challenge.basic_centre --extract`` computes the block for the training and blinded test compounds into ``CACHE`` (a CSV keyed by SMILES; the feature store
reads it without importing Dimorphite-DL).
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)

CACHE = Path("data/tdi_track/outputs/caches/basic_centre.csv")
TRAIN_CSV = "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
TEST_CSV = "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv"
PH_MIN, PH_MAX, MAX_VARIANTS = 6.4, 8.4, 16
N_CONFORMERS = 10
EMBED_TIMEOUT_S = 20
WINDOW = (4.5, 7.5)  # angstrom between the basic nitrogen and an aromatic ring centroid in the CYP2D6 pharmacophore
COLUMNS = [
    "n_states", "share_charged_states", "max_positive_charges", "has_basic_centre", "n_basic_centres", "n_primary", "n_secondary", "n_tertiary", "n_amidinium", "n_aromatic_cation", "n_quaternary",
    "path_min_to_aromatic", "path_mean_to_aromatic", "n_rotatable_bonds", "n_aromatic_rings", "ring_nearest_mean", "ring_farthest_mean", "share_in_window", "centre_distance_over_rg", "geometry_ok",
]


def positive_nitrogens(mol: Chem.Mol) -> list[tuple[int, str]]:
    """The basic centres of a protonated molecule: (atom index, class) of every positively charged nitrogen that is not a nitro group or an N-oxide."""
    out = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 1:
            continue
        neighbours = list(atom.GetNeighbors())
        if any(n.GetSymbol() == "O" for n in neighbours):
            continue
        hydrogens = atom.GetTotalNumHs()
        amidinium = any(n.GetSymbol() == "C" and any(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(n).GetSymbol() == "N" for b in n.GetBonds()) for n in neighbours) or any(
            b.GetBondTypeAsDouble() == 2 for b in atom.GetBonds()
        )
        if amidinium:
            kind = "amidinium"
        elif atom.GetIsAromatic():
            kind = "aromatic_cation"
        elif hydrogens == 0:
            kind = "quaternary"
        else:
            kind = {3: "primary", 2: "secondary", 1: "tertiary"}.get(hydrogens, "tertiary")
        out.append((atom.GetIdx(), kind))
    return out


def protonation_states(smiles: str) -> list[str]:
    from dimorphite_dl import protonate_smiles

    try:
        states = [s.split()[0].split(",")[0] for s in protonate_smiles(smiles, ph_min=PH_MIN, ph_max=PH_MAX, precision=1.0, max_variants=MAX_VARIANTS, validate_output=True)]
    except Exception:
        states = []
    return states or [smiles]


def _aromatic_rings(mol: Chem.Mol) -> list[tuple[int, ...]]:
    return [tuple(r) for r in mol.GetRingInfo().AtomRings() if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]


def _geometry(mol: Chem.Mol, centres: list[int], rings: list[tuple[int, ...]]) -> tuple[float, float, float, float, float]:
    """Mean nearest and farthest ring-centroid distance, share of conformers with a ring in the window, and the centre distance over the radius of gyration."""
    withh = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed, params.numThreads, params.timeout = 42, 1, EMBED_TIMEOUT_S  # a molecule whose embedding does not converge is skipped, never waited for
    ids = list(AllChem.EmbedMultipleConfs(withh, N_CONFORMERS, params))
    if not ids:
        params.useRandomCoords = True
        ids = list(AllChem.EmbedMultipleConfs(withh, N_CONFORMERS, params))
    ids = [c.GetId() for c in withh.GetConformers()]  # a timeout can leave fewer conformers than were requested
    if not ids:
        return (np.nan,) * 5
    nearest, farthest, inside, radial = [], [], [], []
    for cid in ids:
        xyz = withh.GetConformer(cid).GetPositions()
        heavy = np.array([a.GetIdx() for a in withh.GetAtoms() if a.GetAtomicNum() > 1])
        centre = xyz[heavy].mean(axis=0)
        rg = float(np.sqrt(((xyz[heavy] - centre) ** 2).sum(axis=1).mean())) or 1.0
        if rings:
            centroids = np.array([xyz[list(r)].mean(axis=0) for r in rings])
            d = np.array([[np.linalg.norm(xyz[n] - c) for c in centroids] for n in centres])
            nearest.append(float(d.min(axis=1).mean()))
            farthest.append(float(d.max(axis=1).mean()))
            inside.append(float(((d >= WINDOW[0]) & (d <= WINDOW[1])).any()))
        radial.append(float(np.mean([np.linalg.norm(xyz[n] - centre) for n in centres]) / rg))
    if not rings:
        return -1.0, -1.0, -1.0, float(np.mean(radial)), 1.0
    return float(np.mean(nearest)), float(np.mean(farthest)), float(np.mean(inside)), float(np.mean(radial)), 1.0


def basic_centre_row(smiles: str) -> list[float]:
    """The block of one compound, in the order of ``COLUMNS``."""
    states = protonation_states(smiles)
    mols = [m for m in (Chem.MolFromSmiles(s) for s in states) if m is not None] or [Chem.MolFromSmiles(smiles)]
    centres_of = [positive_nitrogens(m) for m in mols]
    charged = [bool(c) for c in centres_of]
    most = max(range(len(mols)), key=lambda i: (len(centres_of[i]), i))
    mol, centres = mols[most], centres_of[most]
    head = [float(len(mols)), float(np.mean(charged)), float(max(len(c) for c in centres_of))]
    if not centres:
        return head + [0.0] * 8 + [-1.0, -1.0, 0.0, float(len(_aromatic_rings(mol)))] + [-1.0, -1.0, -1.0, -1.0, 0.0]
    classes = [k for _, k in centres]
    rings = _aromatic_rings(mol)
    aromatic_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
    dist = Chem.GetDistanceMatrix(mol)
    paths = [float(min((dist[i, j] for j in aromatic_atoms if j != i), default=-1.0)) for i, _ in centres]
    rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    near, far, share, radial, ok = _geometry(mol, [i for i, _ in centres], rings)
    if not np.isfinite(radial):
        near = far = share = radial = -1.0
        ok = 0.0
    counts = [float(classes.count(k)) for k in ("primary", "secondary", "tertiary", "amidinium", "aromatic_cation", "quaternary")]
    return head + [1.0, float(len(centres)), *counts] + [float(min(paths)), float(np.mean(paths)), float(min(rotatable, 12)), float(len(rings)), near, far, share, radial, ok]


def extract(smiles: Sequence[str], path: Path = CACHE, workers: int = 3) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor

    unique = {}
    for s in smiles:
        unique.setdefault(Chem.MolToSmiles(Chem.MolFromSmiles(s)), s)  # the feature store looks compounds up by canonical SMILES
    keys, inputs = list(unique), list(unique.values())
    rows = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (key, row) in enumerate(zip(keys, pool.map(basic_centre_row, inputs, chunksize=16), strict=True), 1):
            rows[key] = row
            if i % 500 == 0:
                logger.info("%d / %d compounds", i, len(keys))
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=COLUMNS)
    frame.index.name = "smiles"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)
    return frame


def load_basic_centre_features(smiles: Sequence[str], path: Path = CACHE) -> np.ndarray:
    """The cached block for ``smiles`` (a read-only lookup; run ``--extract`` first)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} is missing: run `python -m cypmode.challenge.basic_centre --extract`")
    table = pd.read_csv(path, index_col="smiles")
    missing = [s for s in smiles if s not in table.index]
    if missing:
        raise KeyError(f"{len(missing)} compounds are not in the basic-centre cache, e.g. {missing[0]!r}")
    return np.nan_to_num(table.loc[list(smiles), COLUMNS].to_numpy(dtype=np.float32), nan=-1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.extract:
        smiles = list(pd.read_csv(TRAIN_CSV).SMILES) + list(pd.read_csv(TEST_CSV).SMILES)
        frame = extract(smiles, workers=args.workers)
        print(f"wrote {CACHE}: {len(frame)} compounds, {int(frame.has_basic_centre.sum())} with a basic centre, {int((frame.geometry_ok == 1).sum())} with a geometry")


if __name__ == "__main__":
    main()

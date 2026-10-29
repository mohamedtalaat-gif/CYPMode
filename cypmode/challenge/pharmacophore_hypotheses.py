# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Ligand-based pharmacophore hypothesis model (declared on 2026-10-23 before this code ran).

Hypotheses. A hypothesis is a subset of two or three pharmacophore features of pmapper (github.com/DrrDom/pmapper, default feature definitions, bin step 1 A) with the binned distances
between them, identified by the md5 signature that pmapper computes for the subset, so that nothing is hashed into a shared bit as in the fingerprint of ``pharmacophore_fingerprint``. A
compound realises a hypothesis when any of its conformers does: the up to 12 ETKDG conformers within 5 kcal/mol of the lowest (MMFF, seed 42) of ``pocket_pharmacophore.conformers``.
The vocabulary is every hypothesis that at least 10 of the compounds of the training and the test file realise (structures only, no label); it is cached once as a sparse matrix.

Model. Inside a task the model is fitted on the training rows of the task only. For each hypothesis the number of positive rows a and of negative rows b that realise it must satisfy a + b >= 10;
its log-likelihood ratio is ln((a + 1) / (n1 + 2)) - ln((b + 1) / (n0 + 2)) (Laplace correction) and its rank is the chi-square statistic of the 2 x 2 table. The model keeps the 60 hypotheses
with the largest statistic among those with a positive ratio (the pharmacophore of the positives) and the 60 among those with a negative ratio (the exclusion hypotheses). A compound gets five
columns: the sum of the ratios of the kept hypotheses it realises (``hyp_score``), the number of positive (``hyp_npos``) and of negative (``hyp_nneg``) kept hypotheses it realises, the largest
ratio it realises, 0 when none (``hyp_max``), and the share of the 5 strongest positive hypotheses that it realises (``hyp_top5``).

Label-free columns. The training rows get the columns out of fold (the model, including the choice of the hypotheses, refitted on the other 4 of 5 stratified folds of the training rows) and the
evaluation rows get them from the model fitted on all training rows, so that a training row never meets a hypothesis that was chosen with its own label. The control permutes the labels within
the training rows before any step, so that no label information can reach the columns.

Usage: python -m cypmode.challenge.pharmacophore_hypotheses --extract      the sparse matrix of the hypotheses realised by every compound of both files
"""

import argparse
import functools
from pathlib import Path

import numpy as np

CACHE = Path("data/tdi_track/outputs/caches/pharmacophore_hypotheses.npz")
MIN_FEATURES, MAX_FEATURES = 2, 3
MIN_COMPOUNDS = 10  # a hypothesis enters the vocabulary when this many compounds of both files realise it
MIN_SUPPORT = 10  # ... and a model when this many rows of its training rows do
KEEP_PER_SIDE = 60
TOP_POSITIVE = 5
INNER_FOLDS = 5
COLUMNS = ("hyp_score", "hyp_npos", "hyp_nneg", "hyp_max", "hyp_top5")
LABEL_NAMES = {"a": "aromatic", "A": "acceptor", "D": "donor", "H": "hydrophobe", "P": "positive", "N": "negative"}  # the letters of pmapper: lower case a is the aromatic ring, upper case A the hydrogen-bond acceptor


# -- the hypotheses a compound realises --------------------------------------------------------------------------------------------------------------------

def _conformer_molecules(smiles: str):
    """One hydrogen-complete copy of the compound per retained conformer, or []."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from cypmode.challenge.pocket_pharmacophore import (
        CONFORMER_SEED,
        ENERGY_WINDOW,
        N_CONFORMERS,
    )

    parsed = Chem.MolFromSmiles(smiles)
    if parsed is None:
        return []
    mol = Chem.AddHs(Chem.MolFromSmiles(Chem.MolToSmiles(parsed)))  # the atom order of the canonical SMILES, so that a conformer does not depend on how the compound was written
    ids = list(AllChem.EmbedMultipleConfs(mol, N_CONFORMERS, randomSeed=CONFORMER_SEED))
    if not ids:
        return []
    results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
    energies = np.array([e for _, e in results])
    return [Chem.Mol(mol, False, cid) for cid, e in zip(ids, energies, strict=True) if e <= energies.min() + ENERGY_WINDOW]


def _signature(md5_hex: str) -> int:
    return int(md5_hex[:16], 16)  # 64 bits of the md5: a collision among the 10^6 hypotheses of the files has a probability of about 10^-7


def realised(smiles: str) -> np.ndarray:
    """The sorted unique 64-bit signatures of the hypotheses that any conformer of the compound realises; an empty array when no conformer can be made."""
    from pmapper.pharmacophore import Pharmacophore

    found: set[int] = set()
    for conformer in _conformer_molecules(smiles):
        pharmacophore = Pharmacophore()
        pharmacophore.load_from_mol(conformer)
        found.update(_signature(h) for h in pharmacophore.iterate_pharm(MIN_FEATURES, MAX_FEATURES, 0, False))
    return np.array(sorted(found), dtype=np.uint64)


def describe(smiles: str, signature: int) -> str | None:
    """A readable account of one hypothesis, taken from the first conformer of the compound that realises it: the feature types and the distances between them (as found in that conformer)."""
    from pmapper.pharmacophore import Pharmacophore

    for conformer in _conformer_molecules(smiles):
        pharmacophore = Pharmacophore()
        pharmacophore.load_from_mol(conformer)
        for md5_hex, ids in pharmacophore.iterate_pharm(MIN_FEATURES, MAX_FEATURES, 0, True):
            if _signature(md5_hex) == signature:
                features = pharmacophore.get_feature_coords(ids)
                names = [LABEL_NAMES.get(label, label) for label, _ in features]
                points = np.array([xyz for _, xyz in features])
                distances = [np.linalg.norm(points[i] - points[j]) for i in range(len(points)) for j in range(i + 1, len(points))]
                return f"{' + '.join(names)} at {', '.join('the same site' if d < 0.5 else f'{d:.0f} A' for d in distances)}"
    return None


def locate(smiles: str, signature: int) -> dict | None:
    """Where one hypothesis lies in a compound: for the first conformer that realises it, the names of its features, the atoms of each (heavy-atom indices of the compound as written by RDKit from its canonical SMILES),
    their positions and the distances between them, with the heavy-atom positions, elements and bonds of that conformer (to draw it in 3D). None when no conformer realises it."""
    from pmapper import pharmacophore as pmapper_module
    from pmapper.pharmacophore import Pharmacophore

    for conformer in _conformer_molecules(smiles):
        pharmacophore = Pharmacophore()
        pharmacophore.load_from_mol(conformer)
        for md5_hex, ids in pharmacophore.iterate_pharm(MIN_FEATURES, MAX_FEATURES, 0, True):
            if _signature(md5_hex) != signature:
                continue
            atom_groups = pharmacophore._get_features_atom_ids(conformer, pmapper_module._smarts_patterns)
            ordered = [(label, group) for label, groups in atom_groups.items() for group in groups]  # the order in which pmapper numbers its features
            heavy = conformer.GetNumHeavyAtoms()
            positions = conformer.GetConformer().GetPositions()
            features = [{"name": LABEL_NAMES.get(ordered[i][0], ordered[i][0]), "atoms": tuple(a for a in ordered[i][1] if a < heavy), "xyz": positions[list(ordered[i][1])].mean(axis=0)} for i in ids]
            points = np.array([f["xyz"] for f in features])
            return {"features": features, "distances": [float(np.linalg.norm(points[i] - points[j])) for i in range(len(points)) for j in range(i + 1, len(points))],
                    "xyz": positions[:heavy], "elements": [a.GetSymbol() for a in conformer.GetAtoms()][:heavy],
                    "bonds": [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in conformer.GetBonds() if max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) < heavy]}
    return None


def _init_worker() -> None:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")


def extract(workers: int = 4, cache: Path = CACHE) -> None:
    import multiprocessing

    from cypmode.challenge.pocket_pharmacophore import all_smiles

    smiles = all_smiles()
    with multiprocessing.Pool(workers, initializer=_init_worker) as pool:
        per_compound = pool.map(realised, smiles, chunksize=20)
    every = np.concatenate(per_compound) if per_compound else np.zeros(0, dtype=np.uint64)
    values, counts = np.unique(every, return_counts=True)
    vocabulary = values[counts >= MIN_COMPOUNDS]
    indptr, indices = [0], []
    for signatures in per_compound:
        position = np.searchsorted(vocabulary, signatures)
        member = vocabulary[np.minimum(position, len(vocabulary) - 1)] == signatures if len(vocabulary) else np.zeros(len(signatures), dtype=bool)
        found = position[member]
        indices.append(found)
        indptr.append(indptr[-1] + len(found))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, smiles=np.array(smiles), vocabulary=vocabulary, indptr=np.array(indptr, dtype=np.int64), indices=np.concatenate(indices).astype(np.int32) if indices else np.zeros(0, np.int32))
    print(f"{len(smiles)} compounds, {len(every)} realised hypotheses in all, {len(vocabulary)} in the vocabulary (at least {MIN_COMPOUNDS} compounds), "
          f"{int((np.diff(indptr) == 0).sum())} compounds realise none", flush=True)


@functools.lru_cache(maxsize=2)
def load(smiles: tuple[str, ...], cache: str = str(CACHE)):
    """(csr matrix of the hypotheses realised by each of the given SMILES, in that order, and the vocabulary); the SMILES are canonicalised first, as the keys of the cache are, and every compound must be in it."""
    from rdkit import Chem
    from scipy.sparse import csr_matrix

    with np.load(cache, allow_pickle=False) as archive:
        keys = list(archive["smiles"])
        vocabulary, indptr, indices = archive["vocabulary"], archive["indptr"], archive["indices"]
    row = {s: i for i, s in enumerate(keys)}
    canonical = [Chem.MolToSmiles(m) if (m := Chem.MolFromSmiles(s)) is not None else s for s in smiles]
    missing = [s for s in canonical if s not in row]
    if missing:
        raise KeyError(f"{len(missing)} compounds are not in {cache}; run `python -m cypmode.challenge.pharmacophore_hypotheses --extract`")
    full = csr_matrix((np.ones(len(indices), dtype=np.uint8), indices, indptr), shape=(len(keys), len(vocabulary)))
    return full[[row[s] for s in canonical]], vocabulary


# -- the model -----------------------------------------------------------------------------------------------------------------------------------------------

class HypothesisModel:
    """The kept hypotheses of one fit: column indices into the vocabulary (positive ones first, strongest first), their ratios, and how many are positive."""

    def __init__(self, index: np.ndarray, ratio: np.ndarray, n_positive: int):
        self.index, self.ratio, self.n_positive = index, ratio, n_positive


def fit_model(x, y: np.ndarray) -> HypothesisModel:
    """Fit on the rows ``x`` (csr, rows x vocabulary) with the 0/1 labels ``y``."""
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    a = np.asarray(x[y].sum(axis=0), dtype=float).ravel()
    b = np.asarray(x[~y].sum(axis=0), dtype=float).ravel()
    ratio = np.log((a + 1) / (n1 + 2)) - np.log((b + 1) / (n0 + 2))
    n = n1 + n0
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = n * (a * (n0 - b) - (n1 - a) * b) ** 2 / ((a + b) * (n - a - b) * n1 * n0)
    statistic = np.nan_to_num(statistic, nan=0.0, posinf=0.0)
    supported = (a + b) >= MIN_SUPPORT

    def strongest(selected: np.ndarray) -> np.ndarray:
        candidates = np.flatnonzero(selected & supported)
        return candidates[np.argsort(-statistic[candidates], kind="stable")][:KEEP_PER_SIDE]

    positive, negative = strongest(ratio > 0), strongest(ratio < 0)
    index = np.concatenate([positive, negative]).astype(int)
    return HypothesisModel(index, ratio[index], len(positive))


def model_columns(model: HypothesisModel, x) -> np.ndarray:
    """(rows, 5) columns of ``COLUMNS`` for the rows ``x`` under a fitted model."""
    if not len(model.index):
        return np.zeros((x.shape[0], len(COLUMNS)))
    present = np.asarray(x[:, model.index].todense(), dtype=float)
    score = present @ model.ratio
    n_positive = present[:, : model.n_positive].sum(axis=1)
    n_negative = present[:, model.n_positive :].sum(axis=1)
    largest = np.where(present > 0, model.ratio, -np.inf).max(axis=1)
    largest[~np.isfinite(largest)] = 0.0
    top = present[:, : min(TOP_POSITIVE, model.n_positive)]
    share = top.mean(axis=1) if top.shape[1] else np.zeros(x.shape[0])
    return np.column_stack([score, n_positive, n_negative, largest, share])


def hypothesis_columns(x_all, train_rows: np.ndarray, y_train: np.ndarray, eval_rows: np.ndarray, seed: int, permute: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(columns of the training rows, out of fold; columns of the evaluation rows, from the model fitted on all training rows). ``permute`` shuffles the labels within the training rows first."""
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y_train).astype(int)
    if permute:
        y = np.random.default_rng(seed ^ 0x5EED).permutation(y)
    train_x, eval_x = x_all[train_rows], x_all[eval_rows]
    evaluation = model_columns(fit_model(train_x, y), eval_x)
    training = np.zeros((len(train_rows), len(COLUMNS)))
    for fit_rows, held_rows in StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed % 2**31).split(np.zeros(len(y)), y):
        training[held_rows] = model_columns(fit_model(train_x[fit_rows], y[fit_rows]), train_x[held_rows])
    return training, evaluation


def fitted(isoform: str, cache: str = str(CACHE)):
    """The model fitted on all the training rows of one enzyme, with what is needed to read it: (model, vocabulary, the cache rows of the pool, the labels of the pool, the SMILES of the training file, the matrix
    of the pool)."""
    import pandas as pd

    from cypmode.challenge.tdi_targets import BOTH_ARMS, training_view

    train = pd.read_csv("data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv")
    view = training_view(train, isoform)
    mask = (view["population"] == BOTH_ARMS).to_numpy()
    rows, y = np.flatnonzero(mask), view["label"][mask].astype(int).to_numpy()
    x, vocabulary = load(tuple(train["SMILES"]), cache)
    pool = x[rows]
    return fit_model(pool, y), vocabulary, rows, y, list(train["SMILES"]), pool


def summary(isoform: str, per_side: int = 8, cache: str = str(CACHE)) -> dict:
    """The model fitted on all the training rows of one enzyme, described: the sizes of the two classes, and for the strongest ``per_side`` positive and exclusion hypotheses the ratio, the
    number of positive (a) and negative (b) rows that realise it and a readable account taken from one compound that realises it (the realising compound of the class with the largest label confidence; its SMILES and row are returned with the signature). A description of the model, not a measure of performance."""
    import pandas as pd

    from cypmode.challenge.label_confidence import confidence_z

    model, vocabulary, rows, y, smiles, pool = fitted(isoform, cache)
    confidence = confidence_z(pd.read_csv("data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"), isoform)
    out = {"isoform": isoform, "n_positive": int(y.sum()), "n_negative": int((1 - y).sum()), "vocabulary": len(vocabulary), "kept_positive": int(model.n_positive), "kept_exclusion": int(len(model.index) - model.n_positive), "hypotheses": []}
    chosen = list(range(min(per_side, model.n_positive))) + list(range(model.n_positive, min(model.n_positive + per_side, len(model.index))))
    for k in chosen:
        column = model.index[k]
        present = np.asarray(pool[:, column].todense()).ravel() > 0
        candidates = np.flatnonzero(present & (y == (1 if k < model.n_positive else 0)))
        example = rows[candidates[np.nanargmax(np.nan_to_num(confidence[rows[candidates]], nan=-1.0))]]  # the realising compound of the class with the largest label confidence
        out["hypotheses"].append({"side": "positive" if k < model.n_positive else "exclusion", "ratio": float(model.ratio[k]), "a": int(present[y == 1].sum()), "b": int(present[y == 0].sum()),
                                  "description": describe(smiles[example], int(vocabulary[column])), "signature": int(vocabulary[column]), "smiles": smiles[example], "row": int(example)})
    return out


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

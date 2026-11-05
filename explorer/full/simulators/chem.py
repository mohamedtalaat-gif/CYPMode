# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Parts III and IV: molecules as numbers and the first models on them. Folding of a fingerprint (3.1), the blocks of the feature store (3.4, 3.7), a three-point pharmacophore key (3.5), series (3.6), the
nearest neighbours with the Tanimoto weights (4.5), the arms with noise (4.6), how far a message reaches in a graph (4.8) and a principal-component analysis of descriptors (4.9)."""

import itertools
from functools import cache

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, ChemicalFeatures, RDConfig, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

from cypmode.challenge import feature_store as fs
from cypmode.challenge.tdi_targets import ASSAY_FLOOR, SHIFT_THRESHOLD
from simulators import store

CHEAP_BLOCKS = ("morgan_bits", "morgan_c2", "morgan_c3", "fmorgan_c2", "atompair_c", "torsion_c", "maccs", "avalon", "erg", "rdkit_desc", "motif", "tdi_alerts", "tdi_alert_context", "amine")


def morgan_folding(smiles: str, radius: int, n_bits: int) -> dict:
    """The environments of a molecule (unfolded) against the bits that they set once folded into n_bits: the difference is the collisions."""
    mol = Chem.MolFromSmiles(smiles)
    environments = list(rdFingerprintGenerator.GetMorganGenerator(radius=radius).GetSparseFingerprint(mol).GetOnBits())
    bits = {int(h) % n_bits for h in environments}
    return {"environments": len(environments), "bits set": len(bits), "environments that share a bit": len(environments) - len(bits)}


def block_widths(smiles: str) -> pd.DataFrame:
    mol = Chem.MolFromSmiles(smiles)
    rows = []
    for name in CHEAP_BLOCKS:
        vector = np.asarray(fs.BLOCKS[name]([mol]))[0]
        rows.append({"block": name, "width": int(vector.size), "non-zero for this molecule": int(np.count_nonzero(vector))})
    return pd.DataFrame(rows)


@cache
def _feature_factory():
    return ChemicalFeatures.BuildFeatureFactory(f"{RDConfig.RDDataDir}/BaseFeatures.fdef")


def pharmacophore_keys(smiles: str, bin_width: float) -> dict:
    """A teaching version of the three-point pharmacophore key: the features of one embedded conformer (RDKit's feature definitions), every triple of them, the three distances in bins of ``bin_width`` angstrom."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
        return {"features": {}, "triples": 0, "distinct keys": 0}
    features = [f for f in _feature_factory().GetFeaturesForMol(mol) if f.GetFamily() != "LumpedHydrophobe"]
    coordinates = [np.array(f.GetPos()) for f in features]
    keys = set()
    n_triples = 0
    for a, b, c in itertools.combinations(range(len(features)), 3):
        n_triples += 1
        edges = sorted([(features[i].GetFamily(), features[j].GetFamily(), int(np.linalg.norm(coordinates[i] - coordinates[j]) // bin_width)) for i, j in ((a, b), (a, c), (b, c))])
        keys.add(tuple(edges))
    families = pd.Series([f.GetFamily() for f in features]).value_counts().to_dict()
    return {"features": families, "triples": n_triples, "distinct keys": len(keys)}


@cache
def _fingerprints(n: int):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    table = store.table("compounds").head(n)
    return [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in table["SMILES"]], table


def series(n: int, similarity: float) -> dict:
    """Butina clusters of the first n compounds at a Tanimoto similarity, and the Murcko scaffolds of the same compounds."""
    fps, table = _fingerprints(n)
    distances = [1.0 - s for i in range(1, len(fps)) for s in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])]
    clusters = Butina.ClusterData(distances, len(fps), 1.0 - similarity, isDistData=True)
    sizes = sorted((len(c) for c in clusters), reverse=True)
    scaffolds = pd.Series([MurckoScaffold.MurckoScaffoldSmiles(s) for s in table["SMILES"]])
    return {"compounds": len(fps), "clusters": len(sizes), "singletons": sum(1 for s in sizes if s == 1), "largest cluster": sizes[0], "distinct scaffolds": int(scaffolds.nunique()), "compounds without a ring system": int((scaffolds == "").sum())}


def labelled(isoform: str) -> pd.DataFrame:
    table = store.table("compounds").reset_index(drop=True)
    column = f"{isoform}_is_TDI"
    labels = table[column].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    return table.assign(label=labels)[labels.notna()]


def nearest(isoform: str, row: int, k: int, weighted: bool) -> tuple[pd.DataFrame, float]:
    """The k labelled compounds most similar to the compound ``row`` of the labelled set (itself left out) and the estimate sum(s y) / sum(s), or the plain vote."""
    table = labelled(isoform).head(800).reset_index(drop=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in table["SMILES"]]
    sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[row], fps))
    sims[row] = -1.0
    top = np.argsort(-sims)[:k]
    found = table.iloc[top][["Molecule_Name", "label"]].assign(similarity=sims[top])
    weights = found["similarity"].to_numpy() if weighted else np.ones(k)
    return found, float((weights * found["label"].to_numpy()).sum() / weights.sum())


def noisy_rule_accuracy(points: pd.DataFrame, sd_d: float, sd_t: float, clip: bool, seed: int) -> float:
    """The share of labels that the rule gives back when it is applied to the two arms with a normal error each; with ``clip`` the direct arm is floored at the assay floor first, as the project does."""
    rng = np.random.default_rng(seed)
    d = points["d"].to_numpy() + rng.normal(0, sd_d, len(points))
    t = points["t"].to_numpy() + rng.normal(0, sd_t, len(points))
    shift = t - (np.maximum(d, ASSAY_FLOOR) if clip else d)
    return float(((shift > SHIFT_THRESHOLD).astype(int) == points["label"].astype(int).to_numpy()).mean())


def within_bonds(smiles: str, atom: int, steps: int) -> tuple[int, ...]:
    mol = Chem.MolFromSmiles(smiles)
    distance = Chem.GetDistanceMatrix(mol)[atom]
    return tuple(int(i) for i in np.where(distance <= steps)[0])


@cache
def _descriptor_matrix(isoform: str, n: int):
    table = labelled(isoform).head(n)
    matrix = np.asarray(fs.BLOCKS["rdkit_desc"]([Chem.MolFromSmiles(s) for s in table["SMILES"]]), dtype=float)
    return np.nan_to_num(matrix), table["label"].to_numpy()


def pca_of_descriptors(isoform: str, n: int):
    """The principal components of the standardised RDKit descriptors of the first n labelled compounds: the scores, the variance ratios and the labels."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    matrix, labels = _descriptor_matrix(isoform, n)
    keep = matrix.std(axis=0) > 0
    pca = PCA().fit(StandardScaler().fit_transform(matrix[:, keep]))
    return pca.transform(StandardScaler().fit_transform(matrix[:, keep])), pca.explained_variance_ratio_, labels


def knn_leave_one_out_auc(isoform: str, k: int, weighted: bool, n: int = 400) -> float:
    """The AUC of the nearest-neighbour estimate when each of the first n labelled compounds is scored by the others."""
    from sklearn.metrics import roc_auc_score

    table = labelled(isoform).head(n).reset_index(drop=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in table["SMILES"]]
    labels = table["label"].to_numpy()
    scores = []
    for i, f in enumerate(fps):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(f, fps))
        sims[i] = -1.0
        top = np.argsort(-sims)[:k]
        w = sims[top] if weighted else np.ones(k)
        scores.append((w * labels[top]).sum() / max(w.sum(), 1e-9))
    return float(roc_auc_score(labels, scores))

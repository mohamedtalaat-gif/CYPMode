# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part IX, the logic: a selected population in miniature (9.1), pairs of near neighbours (9.2), the weights that make training resemble the test set (9.5), the redundancy of a candidate (9.12) and the functional groups
that go with a label (9.15)."""

from functools import cache

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Fragments, rdFingerprintGenerator
from scipy import stats

from cypmode.challenge.multiplicity import holm
from simulators import chem, store


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def selection_auc(non_hit_share: float, seed: int = 0, n: int = 24000) -> dict:
    """A population in which only the compounds that an earlier screen promoted (the hits) have a label. A model fitted on the hits alone and the same model multiplied by a gate P(hit) are scored on a set that holds the
    given share of non-hits. The data are made up so that the true label is known for everybody; the structure of the problem is that of CYP2D6."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    hit = rng.random(n) < _sigmoid(2.0 * x[:, 0] - 0.5)
    label = (0.8 * x[:, 0] + 0.6 * x[:, 1] + rng.normal(0, 0.7, n) > 0.6)
    train, scored = np.arange(n) < n // 2, np.arange(n) >= n // 2
    model = LogisticRegression().fit(x[train & hit], label[train & hit])
    gate = LogisticRegression().fit(x[train], hit[train])
    hits, non_hits = np.where(scored & hit)[0], np.where(scored & ~hit)[0]
    k = 2000
    pick = np.concatenate([rng.choice(hits, round(k * (1 - non_hit_share))), rng.choice(non_hits, round(k * non_hit_share))])
    plain = model.predict_proba(x[pick])[:, 1]
    return {"fitted on the hits alone": float(roc_auc_score(label[pick], plain)), "multiplied by the gate P(hit)": float(roc_auc_score(label[pick], plain * gate.predict_proba(x[pick])[:, 1]))}


def similar_pairs(isoform: str, threshold: float, n: int = 800) -> tuple[pd.DataFrame, dict]:
    """The pairs of labelled compounds that are at least ``threshold`` similar, and how often the two labels differ."""
    table = chem.labelled(isoform).head(n).reset_index(drop=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in table["SMILES"]]
    rows = []
    for i in range(1, len(fps)):
        for j, s in enumerate(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])):
            if s >= threshold:
                rows.append({"compound": table.loc[i, "Molecule_Name"], "its neighbour": table.loc[j, "Molecule_Name"], "similarity": s, "labels differ": bool(table.loc[i, "label"] != table.loc[j, "label"]), "SMILES 1": table.loc[i, "SMILES"], "SMILES 2": table.loc[j, "SMILES"]})
    pairs = pd.DataFrame(rows)
    return pairs, {"pairs": len(pairs), "pairs with different labels": int(pairs["labels differ"].sum()) if len(pairs) else 0}


@cache
def _fingerprint_matrix(bits: int = 512):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=bits)
    train, test = store.table("compounds"), store.table("test_names")
    matrix = np.array([generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in list(train["SMILES"]) + list(test["SMILES"])], dtype=np.uint8)
    return matrix, len(train), len(test)


def test_likeness_weights(c: float) -> dict:
    """A classifier of test against training compounds on the fingerprints alone gives each training compound the weight p / (1 - p), corrected for the sizes of the two sets; the effective sample size is (sum w)^2 / sum w^2."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict

    matrix, n_train, n_test = _fingerprint_matrix()
    domain = np.r_[np.zeros(n_train), np.ones(n_test)].astype(int)
    model = LogisticRegression(C=c, max_iter=300)
    p = cross_val_predict(model, matrix, domain, cv=3, method="predict_proba")[:, 1]
    weights = (p[:n_train] / (1 - p[:n_train])) * (n_train / n_test)
    return {"AUC test against training": float(roc_auc_score(domain, p)), "effective sample size": float(weights.sum() ** 2 / (weights**2).sum()), "training compounds": n_train, "weights": weights}


def redundancy(oof: pd.DataFrame, candidate: str) -> dict:
    """How much of a candidate the other candidates already hold: its rank correlation with their mean score, its AUC alone, the AUC of the others, and the AUC of the others with it."""
    from sklearn.metrics import roc_auc_score

    y = oof["y"].to_numpy().astype(int)
    columns = [c for c in oof.columns if c.endswith("|0") and not c.startswith("fold")]
    others = [c for c in columns if c != f"{candidate}|0"]
    rest = oof[others].rank().mean(axis=1).to_numpy()
    own = oof[f"{candidate}|0"].to_numpy()
    return {"rank correlation with the others": float(stats.spearmanr(own, rest)[0]), "AUC alone": float(roc_auc_score(y, own)), "AUC of the others": float(roc_auc_score(y, rest)), "AUC of the others with it": float(roc_auc_score(y, (rest + pd.Series(own).rank().to_numpy()) / 2))}


@cache
def _groups(isoform: str, n: int):
    table = chem.labelled(isoform).head(n)
    names = [name for name in dir(Fragments) if name.startswith("fr_")]
    counts = np.array([[getattr(Fragments, name)(Chem.MolFromSmiles(s)) for name in names] for s in table["SMILES"]])
    return names, counts > 0, table["label"].to_numpy().astype(bool)


def group_enrichment(isoform: str, n: int, min_count: int) -> pd.DataFrame:
    """For each functional group of RDKit: how many compounds have it, the share of positives among them against the rest, the one-sided Fisher p and the p after the correction of Holm for all the groups."""
    names, present, label = _groups(isoform, n)
    rows = []
    for k, name in enumerate(names):
        flag = present[:, k]
        if flag.sum() < min_count or (~flag).sum() == 0:
            continue
        table = [[int((flag & label).sum()), int((flag & ~label).sum())], [int((~flag & label).sum()), int((~flag & ~label).sum())]]
        rows.append({"group": name, "compounds": int(flag.sum()), "positives with it": table[0][0] / flag.sum(), "positives without it": table[1][0] / max((~flag).sum(), 1), "p": stats.fisher_exact(table, alternative="two-sided")[1]})
    frame = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    frame["Holm p"] = holm(frame["p"].tolist()) if len(frame) else []
    return frame

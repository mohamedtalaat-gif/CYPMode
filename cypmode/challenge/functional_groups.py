# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The functional groups of the pharmacophore atlas (checklist item "Pharmacophore atlas", declared on 2026-10-23 before any code of it ran).

A functional group is a small, named arrangement of atoms that chemists read as a unit and that carries one or more pharmacophore features: an amide is a hydrogen-bond donor and acceptor, an aryl chlorine is a
hydrophobe, a carboxylic acid is a negative ionisable group. The table below is written once, before any rate is looked at, and is not changed after: ``GROUPS`` maps a name to its SMARTS patterns (a compound
carries the group when any of them matches), the pharmacophore families that it contributes (the six families of ``pocket_pharmacophore.FAMILIES``) and, where the literature names one, why it matters for a
cytochrome P450. The heme-binding motifs are those of ``cypmode.metrics.motifs``.

``groups_of`` lists the atoms of each group found in a molecule (for drawing), ``group_matrix`` flags the groups of many compounds, and ``group_rates`` is the table behind the SAR figure: for every group the
share of positive compounds among those that carry it, with the Wilson interval, against the share among all compounds.
"""

import numpy as np
import pandas as pd
from rdkit import Chem

from cypmode.metrics.motifs import TYPE_II_MOTIFS

GROUPS: dict[str, dict] = {
    "azole or pyridine (heme binder)": {"smarts": tuple(TYPE_II_MOTIFS.values()), "families": ("Acceptor", "Aromatic"), "note": "its free nitrogen can bind the heme iron (Type II)"},
    "basic amine": {"smarts": ("[NX3;!$(N-C=[O,S,N]);!$(N-a);!$(N-[S,P]);!$(N-[N,O]);!$(N-C#N)]([#6])",), "families": ("PosIonizable", "Donor"), "note": "protonated at physiological pH, binds acidic residues"},
    "carboxylic acid": {"smarts": ("[CX3](=O)[OX2H1,OX1-]",), "families": ("NegIonizable", "Acceptor"), "note": "negatively charged at physiological pH"},
    "sulfonamide": {"smarts": ("[SX4](=O)(=O)[NX3]",), "families": ("Acceptor", "Donor"), "note": ""},
    "amide": {"smarts": ("[#6][CX3](=O)[NX3]",), "families": ("Donor", "Acceptor"), "note": ""},
    "urea or carbamate": {"smarts": ("[NX3][CX3](=O)[OX2,NX3]",), "families": ("Donor", "Acceptor"), "note": ""},
    "ester": {"smarts": ("[#6][CX3](=O)[OX2][#6]",), "families": ("Acceptor",), "note": ""},
    "aryl halide": {"smarts": ("a[F,Cl,Br,I]",), "families": ("Hydrophobe",), "note": ""},
    "phenol": {"smarts": ("[OX2H]a",), "families": ("Donor", "Acceptor", "Aromatic"), "note": ""},
    "aliphatic hydroxyl": {"smarts": ("[OX2H][CX4]",), "families": ("Donor", "Acceptor"), "note": ""},
    "aryl ether": {"smarts": ("a[OX2][CX4]",), "families": ("Acceptor", "Hydrophobe"), "note": ""},
    "nitrile": {"smarts": ("[CX2]#[NX1]",), "families": ("Acceptor",), "note": ""},
    "nitro": {"smarts": ("[N+](=O)[O-]",), "families": ("Acceptor", "NegIonizable"), "note": ""},
    "tert-butyl": {"smarts": ("[CX4]([CH3])([CH3])[CH3]",), "families": ("Hydrophobe",), "note": ""},
    "methylenedioxyphenyl": {"smarts": ("c1cc2OCOc2cc1",), "families": ("Aromatic", "Acceptor"), "note": "a known mechanism-based inactivator of P450s"},
    "furan": {"smarts": ("c1ccoc1",), "families": ("Aromatic",), "note": "bioactivated by P450s to a reactive intermediate"},
    "thiophene": {"smarts": ("c1ccsc1",), "families": ("Aromatic",), "note": "bioactivated by P450s to a reactive intermediate"},
    "terminal alkyne": {"smarts": ("[CX2]#[CX2;H1]",), "families": ("Hydrophobe",), "note": "a known mechanism-based inactivator of P450s"},
}

EXAMPLES = {  # a small molecule that shows each group, for the drawings of the atlas
    "azole or pyridine (heme binder)": "Cn1ccnc1",
    "basic amine": "CCN(CC)CC",
    "carboxylic acid": "CCC(=O)O",
    "sulfonamide": "c1ccccc1S(=O)(=O)NC",
    "amide": "c1ccccc1C(=O)NC",
    "urea or carbamate": "c1ccccc1NC(=O)OC",
    "ester": "c1ccccc1C(=O)OC",
    "aryl halide": "Clc1ccccc1",
    "phenol": "Oc1ccccc1",
    "aliphatic hydroxyl": "CCCO",
    "aryl ether": "COc1ccccc1",
    "nitrile": "c1ccccc1C#N",
    "nitro": "c1ccccc1[N+](=O)[O-]",
    "tert-butyl": "CC(C)(C)c1ccccc1",
    "methylenedioxyphenyl": "c1cc2OCOc2cc1",
    "furan": "c1ccoc1",
    "thiophene": "c1ccsc1",
    "terminal alkyne": "C#Cc1ccccc1",
}

_PATTERNS = {name: [Chem.MolFromSmarts(s) for s in spec["smarts"]] for name, spec in GROUPS.items()}


def groups_of(mol) -> dict[str, list[tuple[int, ...]]]:
    """{group: [atom indices of each match]} for the groups that a molecule carries, the matches of all patterns of a group together, each atom set once."""
    found: dict[str, list[tuple[int, ...]]] = {}
    for name, patterns in _PATTERNS.items():
        seen: set[frozenset[int]] = set()
        matches = []
        for pattern in patterns:
            for match in mol.GetSubstructMatches(pattern):
                if frozenset(match) not in seen:
                    seen.add(frozenset(match))
                    matches.append(tuple(match))
        if matches:
            found[name] = matches
    return found


def group_matrix(smiles: list[str]) -> pd.DataFrame:
    """One row per SMILES, one boolean column per group: whether the compound carries it. A SMILES that does not parse carries none."""
    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        present = groups_of(mol) if mol is not None else {}
        rows.append({name: name in present for name in GROUPS})
    return pd.DataFrame(rows, index=list(smiles))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval of a proportion k / n (the interval that stays inside 0 and 1 and keeps its coverage for small n)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return float(centre - half), float(centre + half)


def group_rates(flags: pd.DataFrame, labels: np.ndarray, min_compounds: int = 10) -> pd.DataFrame:
    """For each group carried by at least ``min_compounds`` of the compounds: how many carry it, how many of those are positive, the rate with its Wilson interval and the rate among the compounds that do not
    carry it, plus the base rate of all. ``flags`` is the output of ``group_matrix`` and ``labels`` the 0/1 label of each row."""
    labels = np.asarray(labels).astype(int)
    rows = []
    for name in flags.columns:
        carry = flags[name].to_numpy()
        n, k = int(carry.sum()), int(labels[carry].sum())
        if n < min_compounds:
            continue
        lo, hi = wilson(k, n)
        rest = labels[~carry]
        rows.append({"group": name, "n": n, "positives": k, "rate": k / n, "low": lo, "high": hi, "rate_without": float(rest.mean()) if len(rest) else float("nan"), "base_rate": float(labels.mean()),
                     "families": "+".join(GROUPS[name]["families"])})
    return pd.DataFrame(rows).sort_values("rate", ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=["group", "n", "positives", "rate", "low", "high", "rate_without", "base_rate", "families"])


def group_gain_pairs(smiles: list[str], pairs: tuple, rows: np.ndarray, labels: np.ndarray, min_pairs: int = 5) -> tuple[pd.DataFrame, dict]:
    """The SAR of a change at the level of the functional group. ``pairs`` are the directed matched pairs of ``mmp_transfer`` (source row, target row, transformation id, transformation strings) over the list
    ``smiles``; ``rows`` and ``labels`` are the labelled compounds (row in the list, 0/1 label). A pair that has both compounds labelled counts for a group when the second compound has it, the first does not and no
    other group of the table changes, so that the change of label (0 = not an inhibitor, 1 = inhibitor) can be put down to that group. Returned: the table of the groups carried by at least ``min_pairs`` such pairs
    (group, index of the group, number of pairs, mean change of the label, its standard error) and the pairs themselves (arrays a, b, t, the group each gains or -1, the change of the label, the labels by row)."""
    a, b, t, names = pairs
    y = np.full(len(smiles), np.nan)
    y[np.asarray(rows)] = np.asarray(labels)
    keep = ~np.isnan(y[a]) & ~np.isnan(y[b])
    a, b, t = a[keep], b[keep], t[keep]
    involved = np.unique(np.concatenate([a, b]))
    flags = group_matrix([smiles[i] for i in involved])
    position = {int(r): k for k, r in enumerate(involved)}
    matrix = flags.to_numpy()
    gained_group = np.full(len(a), -1)
    for i in range(len(a)):
        before, after = matrix[position[int(a[i])]], matrix[position[int(b[i])]]
        gained, lost = after & ~before, before & ~after
        if gained.sum() == 1 and lost.sum() == 0:
            gained_group[i] = int(np.flatnonzero(gained)[0])
    delta = y[b] - y[a]
    table = []
    for k, name in enumerate(flags.columns):
        selected = gained_group == k
        n = int(selected.sum())
        if n >= min_pairs:
            table.append({"group": name, "k": k, "pairs": n, "mean": float(delta[selected].mean()), "se": float(delta[selected].std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")})
    table = pd.DataFrame(table, columns=["group", "k", "pairs", "mean", "se"]).sort_values(["pairs", "mean"], ascending=[False, False]).reset_index(drop=True)
    return table, {"a": a, "b": b, "t": t, "names": names, "gained": gained_group, "delta": delta, "y": y, "smiles": smiles}

# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part V: the screens for heme binding. A flag compared with a label (5.3, 5.7), a rank test on potencies (5.4) and the alert of the amines (5.2); the tests are scipy's and the patterns are the project's."""

import numpy as np
from rdkit import Chem
from scipy import stats

from cypmode.metrics import amines


def two_by_two(flag, label) -> dict:
    """The counts of a yes-or-no flag against a yes-or-no label, the odds ratio, the one-sided Fisher p of an excess of positives among the flagged, and the precision and recall of the flag."""
    flag, label = np.asarray(flag, dtype=bool), np.asarray(label, dtype=bool)
    tp, fp, fn, tn = int((flag & label).sum()), int((flag & ~label).sum()), int((~flag & label).sum()), int((~flag & ~label).sum())
    odds = (tp * tn) / (fp * fn) if fp * fn else float("inf")
    p = stats.fisher_exact([[tp, fp], [fn, tn]], alternative="greater")[1]
    return {"flagged and positive": tp, "flagged and negative": fp, "not flagged and positive": fn, "not flagged and negative": tn, "odds ratio": odds, "p (Fisher, one-sided)": float(p),
            "precision": tp / (tp + fp) if tp + fp else float("nan"), "recall": tp / (tp + fn) if tp + fn else float("nan")}


def rank_test(positive, negative) -> dict:
    """The one-sided Mann-Whitney test that the first group has higher values, with the medians."""
    u, p = stats.mannwhitneyu(positive, negative, alternative="greater")
    return {"U": float(u), "p": float(p), "median of the first group": float(np.median(positive)), "median of the second group": float(np.median(negative)), "probability that a value of the first group is higher": float(u / (len(positive) * len(negative)))}


def amine_alert(smiles: str) -> dict:
    """Whether the molecule has a secondary or a tertiary trialkylamine nitrogen, and the atoms of the first nitrogen that the pattern accepts."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"alert": False, "atoms": (), "valid": False}
    for name, pattern in amines._TRIALKYLAMINE_PATTERNS.items():
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            return {"alert": True, "atoms": tuple(matches[0]), "valid": True, "pattern": name, "matches": len(matches)}
    return {"alert": False, "atoms": (), "valid": True}

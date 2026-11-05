# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Section 1.10: the structural alerts of reactive metabolites, read by the same patterns as the ``tdi_alerts`` block of the feature store."""

from rdkit import Chem
from rdkit.Chem import Crippen

from cypmode.challenge import feature_store as fs


def alerts_found(smiles: str) -> list[dict]:
    """Every alert pattern that the molecule matches, with the atoms of its first match and whether the alert is one of the groups that a P450 can turn into a reactive intermediate."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    rows = []
    for name, pattern in fs._ALERT_PATTERNS.items():
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            rows.append({"alert": name, "matches": len(matches), "atoms": tuple(matches[0]), "can be bioactivated": name in fs._BIOACTIVATION})
    return rows


def context(smiles: str) -> dict:
    """What the context block adds to a flat alert: the lipophilicity, the sites that metabolism reaches easily and the fluorine that can block them."""
    mol = Chem.MolFromSmiles(smiles)
    labile = sum(len(mol.GetSubstructMatches(p)) for p in fs._LABILE_PATTERNS)
    return {"logP": round(float(Crippen.MolLogP(mol)), 2), "labile sites": int(labile), "fluorine atoms": len(mol.GetSubstructMatches(fs._FLUORINE)), "CF3 groups": len(mol.GetSubstructMatches(fs._CF3))}

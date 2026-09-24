"""Trialkylamine detector -- a second, independent structural-alert screen
alongside cypmode.metrics.motifs, for the mechanism-based (time-dependent)
inactivation route that screen was never built to catch.

A tertiary or secondary *trialkylamine* nitrogen is the classic substrate
for CYP-mediated oxidative N-dealkylation, which can generate a reactive
iminium/nitroso intermediate -- a real, documented route to time-dependent
inhibition that has nothing to do with direct azole/pyridine/thiazole heme
coordination (see cypmode/metrics/motifs.py's docstring).

Verified against cypmode/validation/tdi_reference.py's 6-drug real TDI
panel before use: an unrefined "any tertiary amine" SMARTS false-positives
on ketoconazole (a real, documented *reversible* Type II inhibitor, not
MBI) via its N-aryl piperazine nitrogen -- that nitrogen's lone pair is
aryl-delocalized, chemically distinct from a trialkylamine's. Excluding
any nitrogen bonded to an aromatic atom (`!$(N-a)`) removes that false
positive while still matching verapamil, diltiazem, and troleandomycin --
the panel's three real trialkylamine-bearing TDI-positive drugs.
"""

from rdkit import Chem

_TRIALKYLAMINE_PATTERNS = {
    "tertiary_trialkylamine": Chem.MolFromSmarts("[NX3;H0;!$(NC=[O,S]);!$(N=*);!a;!$(N-a)]"),
    "secondary_trialkylamine": Chem.MolFromSmarts("[NX3;H1;!$(NC=[O,S]);!$(N=*);!a;!$(N-a)]"),
}


def has_trialkylamine(smiles: str) -> bool:
    """True if smiles has a secondary or tertiary trialkylamine nitrogen --
    a real correlate of CYP-mediated N-dealkylation-route TDI, not azole
    heme coordination. Raises ValueError on unparseable SMILES, matching
    cypmode.metrics.motifs.detect_motifs's behavior.
    """
    if not smiles or not smiles.strip():
        raise ValueError("smiles must be a non-empty string")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smiles!r}")
    return any(mol.HasSubstructMatch(pattern) for pattern in _TRIALKYLAMINE_PATTERNS.values())

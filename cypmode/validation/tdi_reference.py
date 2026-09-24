"""Cross-references real TDI-shift measurements against the motif screen and
this project's own structural coordination results.

Source: OpenADMET/Octant's CYP Data Engine blog post companion repo
(github.com/OpenADMET/Octant_CYP_blog_post, data/ under CC BY 4.0) -- a
small (6-compound) but high-quality, named-drug panel with real
preincubation-vs-direct pIC50 shifts (delta_mean, with 95% CI), unlike
TDC's binary labels or the Octant HF blog-release's unnamed diversity-
library batches. Two of the six (ketoconazole, azamulin) are also in this
project's own 4-compound structural reference set (cypmode/validation),
so their motif call AND static heme-coordination geometry are both
already known -- this module is what lets those be checked against a
real, quantitative TDI measurement instead of prose. See README.md's
"TDI reference panel" section for the actual result.
"""

import json
from pathlib import Path

import pandas as pd

from cypmode.metrics.motifs import detect_motifs

TDI_REFERENCE_PATH = Path("data/octant_cache/tdi_reference/tdi_pic50_shift.tsv")
VALIDATION_SUMMARY_PATH = Path("data/boltz_test/validation_summary.json")

# tdi_pic50_shift.tsv gives drug names only, no SMILES -- each verified
# against ChEMBL (compound_search) individually, not guessed.
REFERENCE_SMILES = {
    "azamulin": "CC[C@]1(C)C[C@@H](OC(=O)CSc2nnc(N)[nH]2)[C@]2(C)[C@H](C)CC[C@]3(CCC(=O)[C@H]32)[C@@H](C)[C@@H]1O",
    "ketoconazole": "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1",
    "clotrimazole": "Clc1ccccc1C(c1ccccc1)(c1ccccc1)n1ccnc1",
    "troleandomycin": (
        "CO[C@H]1C[C@H](O[C@H]2[C@H](C)[C@@H](O[C@@H]3O[C@H](C)C[C@H](N(C)C)[C@H]3OC(C)=O)"
        "[C@@H](C)C[C@@]3(CO3)C(=O)[C@H](C)[C@@H](OC(C)=O)[C@@H](C)[C@@H](C)OC(=O)[C@@H]2C)"
        "O[C@@H](C)[C@@H]1OC(C)=O"
    ),
    "verapamil": "COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC",
    "diltiazem": "COc1ccc([C@@H]2Sc3ccccc3N(CCN(C)C)C(=O)[C@@H]2OC(C)=O)cc1",
}

# The challenge's own TDI-positive threshold (log10(2), a >2-fold shift) --
# see README.md's "OpenADMET CYP Blind Challenge" section.
TDI_SHIFT_LOG2_THRESHOLD = 0.301053


def load_tdi_reference(path: str | Path = TDI_REFERENCE_PATH) -> pd.DataFrame:
    """Load the 6-drug TDI shift panel, annotated with this project's own
    motif call for each drug.
    """
    df = pd.read_csv(path, sep="\t")
    df["smiles"] = df["drug"].map(REFERENCE_SMILES)
    missing = df.loc[df["smiles"].isna(), "drug"].tolist()
    if missing:
        raise ValueError(f"no reference SMILES for: {missing} -- add to REFERENCE_SMILES")
    df["motifs"] = df["smiles"].apply(detect_motifs)
    df["has_motif"] = df["motifs"].apply(bool)
    df["tdi_positive"] = df["delta_mean"] > TDI_SHIFT_LOG2_THRESHOLD
    return df


def with_structural_coordination(
    df: pd.DataFrame, validation_summary_path: str | Path = VALIDATION_SUMMARY_PATH
) -> pd.DataFrame:
    """Left-join this project's own structural coordination call onto
    whichever of these drugs are already in the structural reference set
    (data/boltz_test/validation_summary.json) -- currently ketoconazole
    and azamulin. Drugs not yet run structurally get null coordination
    columns, not an error.
    """
    summary = json.loads(Path(validation_summary_path).read_text())
    out = df.copy()
    out["coordinated"] = out["drug"].map(lambda d: summary.get(d, {}).get("coordinated"))
    out["closest_motif_nitrogen_distance_angstrom"] = out["drug"].map(
        lambda d: summary.get(d, {}).get("closest_motif_nitrogen_distance_angstrom")
    )
    return out

"""Panel-wide CYP450 inhibition data, annotated with Type II mechanism motifs.

TDC's Veith-derived binary inhibition labels (CYP1A2/2C9/2C19/2D6/3A4) say
whether a compound inhibits a given CYP isoform, but nothing about how. This
module loads those labels and joins them against the motif screen in
cypmode.metrics.motifs, so downstream code and the app can show, per
isoform, what fraction of labeled inhibitors carry no heme-coordinating
motif at all -- the mechanism information the raw TDC label throws away.
"""

from pathlib import Path

import pandas as pd

from cypmode.metrics.motifs import classify_mechanism, detect_motifs

CYP_ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]

_TDC_LABEL_NAMES = {
    "CYP1A2": "CYP1A2_Veith",
    "CYP2C9": "CYP2C9_Veith",
    "CYP2C19": "CYP2C19_Veith",
    "CYP2D6": "CYP2D6_Veith",
    "CYP3A4": "CYP3A4_Veith",
}


def load_isoform(isoform: str, cache_path: str | Path = "data/tdc_cache") -> pd.DataFrame:
    """Load one isoform's binary inhibition labels from TDC (ADME/CYP*_Veith)."""
    if isoform not in CYP_ISOFORMS:
        raise ValueError(f"unknown isoform {isoform!r}; expected one of {CYP_ISOFORMS}")
    from tdc.single_pred import ADME

    data = ADME(name=_TDC_LABEL_NAMES[isoform], path=str(cache_path))
    df = data.get_data()
    df = df.rename(columns={"Drug_ID": "drug_id", "Drug": "smiles", "Y": "inhibitor"})
    df["isoform"] = isoform
    return df


def annotate_mechanism(df: pd.DataFrame) -> pd.DataFrame:
    """Add motifs/mechanism columns; rows with unparseable SMILES get mechanism='unparseable SMILES'."""
    motifs_col = []
    mechanism_col = []
    for smiles in df["smiles"]:
        try:
            motifs = detect_motifs(smiles)
            mechanism = classify_mechanism(smiles)
        except ValueError:
            motifs, mechanism = None, "unparseable SMILES"
        motifs_col.append(motifs)
        mechanism_col.append(mechanism)
    out = df.copy()
    out["motifs"] = motifs_col
    out["mechanism"] = mechanism_col
    return out


def load_panel(cache_path: str | Path = "data/tdc_cache") -> pd.DataFrame:
    """Load and mechanism-annotate all 5 major CYP isoforms into one long DataFrame."""
    frames = [annotate_mechanism(load_isoform(isoform, cache_path)) for isoform in CYP_ISOFORMS]
    return pd.concat(frames, ignore_index=True)


def summarize_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Per-isoform breakdown of labeled inhibitors that do/don't carry a Type II motif."""
    inhibitors = df[df["inhibitor"] == 1]
    rows = []
    for isoform, group in inhibitors.groupby("isoform"):
        has_motif = group["motifs"].apply(bool).sum()
        n = len(group)
        rows.append(
            {
                "isoform": isoform,
                "n_labeled_inhibitors": n,
                "n_with_type_ii_motif": int(has_motif),
                "frac_with_type_ii_motif": has_motif / n if n else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("isoform").reset_index(drop=True)

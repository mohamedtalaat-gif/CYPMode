"""Independent, large-N check of the Type II motif screen against real inhibition potency.

cypmode.data.panel checks the motif screen against TDC's Veith labels -- a
binary inhibitor/non-inhibitor call with no potency information, and no
compound outside a hand-curated drug set. This module checks a different
axis on a different, independent dataset: does motif presence correlate
with how *potent* an inhibitor actually is, across a real high-throughput
screen the motif screen was never tuned against?

Source: OpenADMET/Octant's "Building the OpenADMET Data Engine" blog
release (https://huggingface.co/datasets/openadmet/Octant_CYP_inhibition_reactivity_blog_release,
CC BY 4.0) -- a ~1,200-compound diversity-library CYP3A4 dose-response
screen with a 30-minute active-enzyme pre-incubation, so measured potency
reflects combined reversible + time-dependent inhibition, not reversible
inhibition alone. This is drawn from a diversity chemical library, not
marketed drugs, so it has essentially no overlap with either the 5
structurally-validated reference compounds in cypmode.validation or TDC's
Veith-derived panel in cypmode.data.panel: confirmed directly by canonical
SMILES comparison, only vardenafil (one of the 5 structural reference
compounds) appears in this library at all.

This screen is not a potency predictor and isn't claimed to be one --
cypmode/metrics/motifs.py's docstring is explicit that a motif match flags
a compound as *capable* of heme coordination, not confirmed to coordinate.
A real, positive, statistically significant correlation with potency here
is still a meaningful independent check: it means the structural feature
the motif screen looks for tracks a real pharmacological property on data
it never saw, not just the 5 compounds it was validated against
structurally. See README.md's "Independent large-N potency check" section
for the actual result and its honest limits.
"""

import urllib.request
from pathlib import Path

import pandas as pd
from scipy import stats

from cypmode.metrics.motifs import detect_motifs

_PARQUET_URL = (
    "https://huggingface.co/api/datasets/openadmet/"
    "Octant_CYP_inhibition_reactivity_blog_release/parquet/inhibition/train/0.parquet"
)


def load_inhibition_data(cache_path: str | Path = "data/octant_cache/inhibition.parquet") -> pd.DataFrame:
    """Load Octant's CYP3A4 dose-response summary (one row per compound).

    Downloads and caches on first use; a copy is checked into the repo so
    CI and offline use don't need network access.
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_PARQUET_URL, cache_path)
    return pd.read_parquet(cache_path)


def annotate_motifs(df: pd.DataFrame) -> pd.DataFrame:
    """Add motifs/has_motif columns via cypmode.metrics.motifs.detect_motifs.

    Unlike cypmode.data.panel.annotate_mechanism, unparseable SMILES are not
    caught and relabeled -- this dataset's standardized_smiles column is
    machine-curated and confirmed to parse in full (0/1340 failures); a
    parse failure here would mean the upstream data changed underneath this
    analysis, which should surface as an error, not a silently dropped row.
    """
    out = df.copy()
    out["motifs"] = [detect_motifs(smiles) for smiles in df["standardized_smiles"]]
    out["has_motif"] = out["motifs"].apply(bool)
    return out


def compare_potency_by_motif(df: pd.DataFrame) -> dict:
    """Compare CYP3A4_pIC50 between motif-positive and motif-negative compounds.

    Restricted to QC-passed dose-response curves (drc_qc_status == 'PASS')
    with a fitted pIC50 -- curves that failed QC, or didn't reach a
    fittable endpoint, don't carry a reliable potency value to compare.

    Uses a one-sided Mann-Whitney U test (motif-positive > motif-negative):
    the motif screen's mechanistic claim is that a coordinating nitrogen
    tightens binding, not that it produces any particular potency
    distribution shape, so a rank-based test is the honest choice over
    assuming normality for a t-test.
    """
    qc_passed = df[df["drc_qc_status"] == "PASS"]
    scored = qc_passed.dropna(subset=["CYP3A4_pIC50"])

    positive = scored.loc[scored["has_motif"], "CYP3A4_pIC50"]
    negative = scored.loc[~scored["has_motif"], "CYP3A4_pIC50"]

    u_statistic, p_value = stats.mannwhitneyu(positive, negative, alternative="greater")

    return {
        "n_qc_passed_with_pic50": int(len(scored)),
        "n_motif_positive": int(len(positive)),
        "n_motif_negative": int(len(negative)),
        "motif_positive_mean_pIC50": float(positive.mean()),
        "motif_negative_mean_pIC50": float(negative.mean()),
        "motif_positive_median_pIC50": float(positive.median()),
        "motif_negative_median_pIC50": float(negative.median()),
        "mannwhitneyu_statistic": float(u_statistic),
        "p_value_one_sided_greater": float(p_value),
    }


def main() -> None:
    import json

    df = annotate_motifs(load_inhibition_data())
    result = compare_potency_by_motif(df)

    out_path = Path("data/octant_cache/motif_potency_validation.json")
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

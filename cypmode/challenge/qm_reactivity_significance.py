"""Repeated-CV + Tukey HSD significance test for the QM-reactivity
feature (cypmode.challenge.qm_reactivity_features) against M3 -- kept as
a fully separate module/process on purpose, same reasoning as
chemberta_significance.py: this file imports lightgbm, sklearn, and
statsmodels, and never imports torch or qmdesc.

Run `python -m cypmode.challenge.qm_reactivity_features` first to
populate the feature cache (a separate, torch-only invocation), then
this module as its own invocation afterward. Reads the cache directly;
does not compute any QM features itself.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cypmode.challenge.knn_features import _morgan_fp
from cypmode.challenge.qm_reactivity_features import load_cached_qm_reactivity_features
from cypmode.challenge.stacking_significance import (
    paired_comparison_summary,
    repeated_cv_stacking_comparison,
)
from cypmode.challenge.tdi_ablation import add_ablation_features
from cypmode.challenge.tdi_baseline import (
    TDI_ISOFORMS,
    compute_descriptors,
    load_tdi_training_data,
)


def _main() -> None:
    tdi_df = add_ablation_features(compute_descriptors(load_tdi_training_data()))
    tdi_df = load_cached_qm_reactivity_features(tdi_df)

    n_failed = tdi_df["qm_reactivity_features"].apply(lambda f: np.isnan(f).all()).sum()
    print(f"{n_failed}/{len(tdi_df)} compounds have a fully-failed cached QM feature vector")

    all_results = []
    all_summaries = []
    for cyp in TDI_ISOFORMS:
        col = f"{cyp}_is_TDI"
        cyp_df = tdi_df.dropna(subset=[col]).reset_index(drop=True)
        y = cyp_df[col].astype(int).to_numpy()
        m3_features = np.stack(cyp_df["m3_no_amine_features"])
        fps = [_morgan_fp(s) for s in cyp_df["SMILES"]]
        # NaN-safe: a compound with no trialkylamine nitrogen has NaN in
        # the 4 qm_basicN_* columns (real missingness, not zero -- see
        # qm_reactivity_features.py's own docstring) but real values in
        # the 4 qm_mol_* columns, so LightGBM's native NaN handling (same
        # reliance as cypmode.challenge.tier3_features) is the right
        # fit here, not row-dropping.
        qm_features = np.stack(cyp_df["qm_reactivity_features"])
        aux_features = {"QM-Reactivity": qm_features}

        print(f"\nRunning 5x5=25-fold repeated comparison for {cyp}...")
        results = repeated_cv_stacking_comparison(m3_features, y, fps, aux_features, n_repeats=5, n_splits=5)
        results.insert(0, "isoform", cyp)
        all_results.append(results)

        print(f"-- {cyp}: mean MCC per variant (25 folds each) --")
        print(results.groupby("variant")["mcc"].agg(["mean", "std"]).to_string())

        summary = paired_comparison_summary(results)
        summary.insert(0, "isoform", cyp)
        all_summaries.append(summary)
        print(f"-- {cyp}: Tukey HSD pairwise comparison --")
        print(summary.to_string(index=False))

    raw_path = Path("data/tdi_track/outputs/comparisons/qm_reactivity_significance_raw.csv")
    pd.concat(all_results, ignore_index=True).to_csv(raw_path, index=False)
    print(f"\nwrote {raw_path}")

    tukey_path = Path("data/tdi_track/outputs/comparisons/qm_reactivity_significance_tukey.csv")
    pd.concat(all_summaries, ignore_index=True).to_csv(tukey_path, index=False)
    print(f"wrote {tukey_path}")


if __name__ == "__main__":
    _main()

"""Repeated-CV + Tukey HSD significance test for the ChemBERTa feature
(cypmode.challenge.chemberta_features) against M3 -- kept as a fully
separate module/process from chemberta_features.py on purpose: this
file imports lightgbm, sklearn, and statsmodels, and NEVER imports
torch or transformers, so it is always safe to run regardless of
process history. See chemberta_features.py's own module docstring for
the real, confirmed reason (torch+LightGBM in one process segfaults).

Run `python -m cypmode.challenge.chemberta_features` first to populate
the embedding cache (a separate, torch-only invocation), then this
module as its own invocation afterward. Reads the cache directly; does
not compute any embeddings itself.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cypmode.challenge.chemberta_features import (
    load_cached_chemberta_embeddings,
)
from cypmode.challenge.knn_features import _morgan_fp
from cypmode.challenge.stacking_significance import (
    repeated_cv_stacking_comparison,
    tukey_hsd_summary,
)
from cypmode.challenge.tdi_ablation import add_ablation_features
from cypmode.challenge.tdi_baseline import (
    TDI_ISOFORMS,
    compute_descriptors,
    load_tdi_training_data,
)


def _main() -> None:
    tdi_df = add_ablation_features(compute_descriptors(load_tdi_training_data()))
    tdi_df = load_cached_chemberta_embeddings(tdi_df)

    n_failed = tdi_df["chemberta_embedding"].apply(lambda e: np.isnan(e).any()).sum()
    print(f"{n_failed}/{len(tdi_df)} compounds have a failed cached embedding (NaN)")

    all_results = []
    all_summaries = []
    for cyp in TDI_ISOFORMS:
        col = f"{cyp}_is_TDI"
        cyp_df = tdi_df.dropna(subset=[col]).reset_index(drop=True)
        valid = ~cyp_df["chemberta_embedding"].apply(lambda e: np.isnan(e).any())
        cyp_df = cyp_df[valid].reset_index(drop=True)
        y = cyp_df[col].astype(int).to_numpy()
        m3_features = np.stack(cyp_df["m3_features"])
        fps = [_morgan_fp(s) for s in cyp_df["SMILES"]]
        aux_features = {"ChemBERTa": np.stack(cyp_df["chemberta_embedding"])}

        print(f"\nRunning 5x5=25-fold repeated comparison for {cyp}...")
        results = repeated_cv_stacking_comparison(m3_features, y, fps, aux_features, n_repeats=5, n_splits=5)
        results.insert(0, "isoform", cyp)
        all_results.append(results)

        print(f"-- {cyp}: mean MCC per variant (25 folds each) --")
        print(results.groupby("variant")["mcc"].agg(["mean", "std"]).to_string())

        summary = tukey_hsd_summary(results)
        summary.insert(0, "isoform", cyp)
        all_summaries.append(summary)
        print(f"-- {cyp}: Tukey HSD pairwise comparison --")
        print(summary.to_string(index=False))

    raw_path = Path("data/tdi_track/outputs/chemberta_significance_raw.csv")
    pd.concat(all_results, ignore_index=True).to_csv(raw_path, index=False)
    print(f"\nwrote {raw_path}")

    tukey_path = Path("data/tdi_track/outputs/chemberta_significance_tukey.csv")
    pd.concat(all_summaries, ignore_index=True).to_csv(tukey_path, index=False)
    print(f"wrote {tukey_path}")


if __name__ == "__main__":
    _main()

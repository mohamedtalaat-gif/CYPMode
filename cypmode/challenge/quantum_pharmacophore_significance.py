"""Ablation + repeated-CV/Tukey HSD significance test for the quantum-
electronic feature (cypmode.challenge.quantum_pharmacophore) against M3
-- kept as a fully separate module/process from quantum_pharmacophore.py
on purpose: this file imports lightgbm, sklearn, and statsmodels, and
NEVER imports tblite, so it is always safe to run regardless of process
history. See quantum_pharmacophore.py's own module docstring for the
real, confirmed reason (tblite+LightGBM in one process deadlocks, not
just crashes -- confirmed by reproducing it directly).

Run `python -m cypmode.challenge.quantum_pharmacophore` first to
populate the feature cache (a separate, tblite-only invocation), then
this module as its own invocation afterward. Reads the cache directly
via load_cached_quantum_features; does not compute any quantum features
itself.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

from cypmode.challenge.knn_features import _morgan_fp
from cypmode.challenge.quantum_pharmacophore import load_cached_quantum_features
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
    tdi_df = load_cached_quantum_features(tdi_df)

    n_failed = tdi_df["quantum_features"].apply(lambda f: np.isnan(f).any()).sum()
    print(f"{n_failed}/{len(tdi_df)} compounds have a failed cached QM calculation (NaN)")

    all_results = []
    for cyp in TDI_ISOFORMS:
        col = f"{cyp}_is_TDI"
        cyp_df = tdi_df.dropna(subset=[col]).reset_index(drop=True)
        valid = ~cyp_df["quantum_features"].apply(lambda f: np.isnan(f).any())
        cyp_df = cyp_df[valid].reset_index(drop=True)
        y = cyp_df[col].astype(int).to_numpy()

        m3_features = np.stack(cyp_df["m3_features"])
        quantum_features = np.stack(cyp_df["quantum_features"])
        m3_plus_q_features = np.hstack([m3_features, quantum_features])

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rows = []
        for fold, (train_idx, val_idx) in enumerate(kfold.split(m3_features, y)):
            m3_model = LGBMClassifier(verbose=-1, random_state=42)
            m3_model.fit(m3_features[train_idx], y[train_idx])
            m3_pred = m3_model.predict(m3_features[val_idx])

            m3q_model = LGBMClassifier(verbose=-1, random_state=42)
            m3q_model.fit(m3_plus_q_features[train_idx], y[train_idx])
            m3q_pred = m3q_model.predict(m3_plus_q_features[val_idx])

            rows.append(
                {
                    "isoform": cyp,
                    "fold": fold,
                    "n_val": len(val_idx),
                    "mcc_m3": matthews_corrcoef(y[val_idx], m3_pred),
                    "mcc_m3_plus_quantum": matthews_corrcoef(y[val_idx], m3q_pred),
                }
            )
        results = pd.DataFrame(rows)
        all_results.append(results)
        print(
            f"{cyp}: mean MCC M3={results['mcc_m3'].mean():.4f}, "
            f"M3+quantum={results['mcc_m3_plus_quantum'].mean():.4f} (n={len(cyp_df)})"
        )

    combined = pd.concat(all_results, ignore_index=True)
    out_path = Path("data/tdi_track/outputs/quantum_pharmacophore_ablation.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")

    # The single-split result above showed a mixed/negative point estimate
    # for both isoforms -- per this project's own standing rule (a
    # surprising result needs the same repeated-CV + Tukey HSD rigor
    # already applied to KNN/BindingDB before being trusted either way,
    # see CYPMode_Architecture_Specification.md), not just accepted at
    # face value on one split.
    all_sig_results = []
    all_sig_summaries = []
    for cyp in TDI_ISOFORMS:
        col = f"{cyp}_is_TDI"
        cyp_df = tdi_df.dropna(subset=[col]).reset_index(drop=True)
        valid = ~cyp_df["quantum_features"].apply(lambda f: np.isnan(f).any())
        cyp_df = cyp_df[valid].reset_index(drop=True)
        y = cyp_df[col].astype(int).to_numpy()
        m3_features = np.stack(cyp_df["m3_features"])
        fps = [_morgan_fp(s) for s in cyp_df["SMILES"]]
        aux_features = {"Quantum": np.stack(cyp_df["quantum_features"])}

        print(f"\nRunning 5x5=25-fold repeated comparison for {cyp} (quantum features)...")
        sig_results = repeated_cv_stacking_comparison(m3_features, y, fps, aux_features, n_repeats=5, n_splits=5)
        sig_results.insert(0, "isoform", cyp)
        all_sig_results.append(sig_results)

        print(f"-- {cyp}: mean MCC per variant (25 folds each) --")
        print(sig_results.groupby("variant")["mcc"].agg(["mean", "std"]).to_string())

        sig_summary = tukey_hsd_summary(sig_results)
        sig_summary.insert(0, "isoform", cyp)
        all_sig_summaries.append(sig_summary)
        print(f"-- {cyp}: Tukey HSD pairwise comparison --")
        print(sig_summary.to_string(index=False))

    sig_raw_path = Path("data/tdi_track/outputs/quantum_pharmacophore_significance_raw.csv")
    pd.concat(all_sig_results, ignore_index=True).to_csv(sig_raw_path, index=False)
    print(f"\nwrote {sig_raw_path}")

    sig_tukey_path = Path("data/tdi_track/outputs/quantum_pharmacophore_significance_tukey.csv")
    pd.concat(all_sig_summaries, ignore_index=True).to_csv(sig_tukey_path, index=False)
    print(f"wrote {sig_tukey_path}")


if __name__ == "__main__":
    _main()

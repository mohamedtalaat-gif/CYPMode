# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Repeated-CV + Tukey HSD significance test for the two external
auxiliary-activity sources that showed a real, consistent single-split
bump for CYP2D6 -- cypmode.challenge.pubchem_auxiliary's AID 1851 panel
(CYP2D6 single-split MCC 0.072->0.104) and cypmode.challenge.
chembl_auxiliary's ChEMBL pull (0.072->0.104, essentially the same
magnitude from an independent source) -- while cypmode.challenge.
bindingdb_auxiliary's BindingDB feature went the other way (0.072->
0.037, actively hurts). Two independent external sources landing on the
same real effect size is a much stronger signal than any single one of
this project's other CYP2D6 feature tests has shown, but every other
promising-looking CYP2D6 single-split number so far either didn't
survive repeated CV (the OOF-bridge retraction) or came back null once
retested properly (KNN, Tier3, ChemBERTa, quantum pharmacophore,
pharmacophore fingerprint) -- so this gets the same treatment before
being trusted, not an exception.

AID 1851 and ChEMBL are both external, fully-safe-to-precompute features
(see their own module docstrings for why no fold-internal OOF treatment
is needed -- fit once on the external data, apply identically to every
TDI compound, train and validation alike, same as an RDKit descriptor).
That makes this simpler than the KNN significance tests: the aux proba
columns are computed ONCE before the CV loop, not rebuilt every fold.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

N_REPEATS = 5
N_SPLITS = 5
BASE_SEED = 42


def add_aux_columns(tdi_df: pd.DataFrame, isoform: str) -> pd.DataFrame:
    from cypmode.challenge.chembl_auxiliary import (
        CHEMBL_ISOFORM_TARGETS,
        fetch_chembl_isoform,
        load_chembl_isoform,
    )
    from cypmode.challenge.chembl_auxiliary import (
        exclude_exact_overlap as exclude_chembl_overlap,
    )
    from cypmode.challenge.chembl_auxiliary import (
        fit_auxiliary_model as fit_chembl_model,
    )
    from cypmode.challenge.chembl_auxiliary import (
        predict_auxiliary_activity as predict_chembl_activity,
    )
    from cypmode.challenge.pubchem_auxiliary import (
        exclude_exact_overlap as exclude_aid_overlap,
    )
    from cypmode.challenge.pubchem_auxiliary import fit_auxiliary_model as fit_aid_model
    from cypmode.challenge.pubchem_auxiliary import load_aid1851
    from cypmode.challenge.pubchem_auxiliary import (
        predict_auxiliary_activity as predict_aid_activity,
    )

    aid_df = exclude_aid_overlap(load_aid1851(), tdi_df["SMILES"])
    aid_model = fit_aid_model(aid_df, isoform)
    tdi_df = tdi_df.copy()
    tdi_df["aux_aid1851"] = predict_aid_activity(aid_model, tdi_df)

    chembl_raw = fetch_chembl_isoform(isoform, CHEMBL_ISOFORM_TARGETS[isoform])
    chembl_df = exclude_chembl_overlap(load_chembl_isoform(chembl_raw), tdi_df["SMILES"])
    chembl_model = fit_chembl_model(chembl_df)
    tdi_df["aux_chembl"] = predict_chembl_activity(chembl_model, tdi_df)

    return tdi_df


def repeated_cv_comparison(
    m3_features: np.ndarray,
    aid1851: np.ndarray,
    chembl: np.ndarray,
    y: np.ndarray,
    n_repeats: int = N_REPEATS,
    n_splits: int = N_SPLITS,
    base_seed: int = BASE_SEED,
) -> pd.DataFrame:
    rows = []
    for repeat in range(n_repeats):
        seed = base_seed + repeat
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(kfold.split(m3_features, y)):
            variants = {
                "M3": m3_features,
                "M3+AID1851": np.hstack([m3_features, aid1851.reshape(-1, 1)]),
                "M3+ChEMBL": np.hstack([m3_features, chembl.reshape(-1, 1)]),
                "M3+AID1851+ChEMBL": np.hstack([m3_features, aid1851.reshape(-1, 1), chembl.reshape(-1, 1)]),
            }
            y_val = y[val_idx]
            can_score = len(set(y_val)) > 1
            for name, X in variants.items():
                model = LGBMClassifier(verbose=-1, random_state=seed)
                model.fit(X[train_idx], y[train_idx])
                pred = model.predict(X[val_idx])
                rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "variant": name,
                        "mcc": matthews_corrcoef(y_val, pred) if can_score else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def _main() -> None:
    import argparse

    from cypmode.challenge.stacking_significance import paired_comparison_summary
    from cypmode.challenge.tdi_ablation import add_ablation_features
    from cypmode.challenge.tdi_baseline import (
        compute_descriptors,
        load_tdi_training_data,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isoform", default="CYP2D6", choices=["CYP3A4", "CYP2D6"])
    args = parser.parse_args()
    isoform = args.isoform

    tdi_df = add_ablation_features(compute_descriptors(load_tdi_training_data()))
    tdi_df = add_aux_columns(tdi_df, isoform)

    label_col = f"{isoform}_is_TDI"
    cyp_df = tdi_df.dropna(subset=[label_col]).reset_index(drop=True)
    y = cyp_df[label_col].astype(int).to_numpy()
    m3_features = np.stack(cyp_df["m3_no_amine_features"])
    aid1851 = cyp_df["aux_aid1851"].to_numpy()
    chembl = cyp_df["aux_chembl"].to_numpy()

    print(f"Running {N_REPEATS}x{N_SPLITS}={N_REPEATS * N_SPLITS}-fold repeated comparison for {isoform}...")
    results = repeated_cv_comparison(m3_features, aid1851, chembl, y)
    print("\n-- mean MCC per variant --")
    print(results.groupby("variant")["mcc"].agg(["mean", "std", "count"]).to_string())

    isoform_dir = Path(f"data/tdi_track/outputs/{isoform.lower()}")
    isoform_dir.mkdir(parents=True, exist_ok=True)
    out_path = isoform_dir / "external_auxiliary_significance_raw.csv"
    results.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")

    scoreable = results.dropna(subset=["mcc"])
    n_dropped = len(results) - len(scoreable)
    if n_dropped:
        print(f"({n_dropped} repeat/fold rows had an undefined MCC -- excluded from the Tukey test)")

    summary = paired_comparison_summary(scoreable)
    print("\n-- Tukey HSD pairwise comparison (all variant pairs) --")
    print(summary.to_string(index=False))
    summary_path = isoform_dir / "external_auxiliary_significance_tukey.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    _main()

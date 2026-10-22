# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Checks whether a train/test split is actually informative before trusting
any MCC computed on it -- a validation set that's chemically close to
training (near-duplicates/analogs) can make a CV split look far more
reliable than performance on genuinely novel chemistry actually is, a
known general risk in cheminformatics benchmarking.

Applies that check to our own data: our own CV folds (i.i.d. from the
training population, expected to look "easy"/similar) against the real
750-compound blinded TDI test set (the one that actually matters -- tells
us whether our cross-validated MCC is likely to transfer to it).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cypmode.challenge.reliability import tanimoto_ood_distance

BLINDED_TEST_PATH = Path("data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv")


def train_test_similarity_summary(train_smiles: pd.Series, apply_smiles: pd.Series) -> dict:
    """max-Tanimoto(each apply compound, full train set), summarized as
    median/mean similarity and the fraction at similarity >=0.5.
    """
    distances = tanimoto_ood_distance(train_smiles, apply_smiles)
    similarities = 1.0 - distances
    return {
        "n": len(apply_smiles),
        "median_similarity": float(np.median(similarities)),
        "mean_similarity": float(np.mean(similarities)),
        "frac_similarity_ge_0.5": float((similarities >= 0.5).mean()),
        "frac_similarity_ge_0.7": float((similarities >= 0.7).mean()),
    }


def _main() -> None:
    from sklearn.model_selection import StratifiedKFold

    from cypmode.challenge.tdi_baseline import load_tdi_training_data

    train_df = load_tdi_training_data()
    blinded_df = pd.read_csv(BLINDED_TEST_PATH)

    rows = []

    # Sanity check: our own i.i.d. CV fold vs its own training portion --
    # expected to look "easy" (same population, random split), the
    # baseline this project's own reported CV numbers are measured against.
    y = train_df["CYP3A4_is_TDI"].fillna(0).astype(int).to_numpy()
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_train_idx, cv_val_idx = next(kfold.split(train_df, y))
    cv_summary = train_test_similarity_summary(
        train_df.loc[cv_train_idx, "SMILES"], train_df.loc[cv_val_idx, "SMILES"]
    )
    rows.append({"comparison": "own_cv_fold_vs_own_training_portion", **cv_summary})

    # The one that actually matters: the real blinded TDI test set vs this
    # project's full real training set.
    blind_summary = train_test_similarity_summary(train_df["SMILES"], blinded_df["SMILES"])
    rows.append({"comparison": "real_blinded_test_vs_full_training_set", **blind_summary})

    result = pd.DataFrame(rows)
    print(result.to_string(index=False))

    out_path = Path("data/tdi_track/outputs/comparisons/split_validity.csv")
    result.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    _main()

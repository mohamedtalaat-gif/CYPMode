"""Checks whether a train/test split is actually informative before trusting
any MCC computed on it -- inspired directly by a real finding from another
OpenADMET CYP Challenge team (450nm, TDI leaderboard rank 16): they measured
max-Tanimoto(test, train)=0.598 with 95.7% of their test compounds at
similarity >=0.5, i.e. their test set was close analog-expansion of known
hits rather than genuinely novel chemistry -- meaning a CV split drawn from
train alone can look far more reliable than performance on the real test
set actually is.

This applies the same diagnostic to our own data: our own CV folds (i.i.d.
from the training population, expected to look "easy"/similar) against the
real 750-compound blinded TDI test set (the one that actually matters --
tells us whether our cross-validated MCC is likely to transfer, the same
honest self-check JacksonBurns reported for their own OOF-to-blind gap).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cypmode.challenge.reliability import tanimoto_ood_distance

BLINDED_TEST_PATH = Path("data/cyp_challenge/tdi_track/cyp-challenge-TEST-BLINDED.csv")


def train_test_similarity_summary(train_smiles: pd.Series, apply_smiles: pd.Series) -> dict:
    """max-Tanimoto(each apply compound, full train set), summarized the
    same way 450nm reported theirs -- median/mean similarity and the
    fraction at similarity >=0.5 -- so the two are directly comparable.
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

    out_path = Path("data/cyp_challenge/tdi_track/split_validity.csv")
    result.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    _main()

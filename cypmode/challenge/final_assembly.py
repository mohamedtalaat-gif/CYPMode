# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The final assembled TDI model: combines the settled, evidence-based
per-isoform architecture choice into one pipeline that produces real
predictions on the real 750-row blinded test set and writes a validated
submission file.

CYP3A4: independent-head LightGBM, m3_no_amine_features + KNN read-across,
fold-internal-style tuned threshold. Real evidence
(data/tdi_track/outputs/): M3 alone tuned=0.387, M3+KNN naive=0.315
(vs. M3 naive=0.295) -- the established "practical default" from
CYPMode_Architecture_Specification.md's Component-by-component section,
simplest of the statistically-tied options, no external-API dependency
at inference time.

A real retraction belongs here: an earlier version of this module also
added the OOF Direct-Inhibition Bridge as a CYP3A4 feature, based on a
single-split naive-threshold read that looked like the single largest
feature effect found in this whole project (tuned MCC 0.387 -> 0.470).
That result did not survive honest re-checking. Root cause: ~34.8% of
CYP3A4-labeled training compounds have NO real direct-inhibition
measurement at all, and those compounds are TDI-negative essentially by
construction (the label formula needs a direct-inhibition baseline to
compute a "shift" from -- see CYPMode_Architecture_Specification.md's
"Official scoring harness" section). The old feature encoded that fact
via an explicit "missing_flag" the model could read as a near-perfect
shortcut ("missing_flag=1" -> ~0% positive in training) -- but the real
blind test set NEVER exposes whether a compound has real
direct-inhibition data (SMILES + Molecule_Name only, for every row),
so every test row got missing_flag=0 and the shortcut simply didn't
fire, pushing predictions to 60.9% positive against a real 21.3%
training base rate. Re-verified honestly via CV that actually simulates
deployment (validation-fold compounds scored with zero awareness of
their own real missingness status, using only a fold-trained
regressor's plain prediction): OOF-bridge's tuned MCC drops to 0.378 --
*lower* than plain M3 alone (0.387), the difference well inside normal
fold-to-fold noise for this project (std ~0.03-0.06 at this scale).
Once the leak is closed, OOF-bridge joins the same "statistically
indistinguishable from plain M3" pattern as KNN, BindingDB, AID1851,
quantum-electronic, and pharmacophore-fingerprint features -- not a
real win, and not included here. Kept as a real, documented negative
result (like the ChemBERTa finding), not quietly dropped.

CYP2D6: shared trunk, m3_no_amine_features only -- see
cypmode.challenge.final_assembly_cyp2d6's own docstring for why KNN is
deliberately left out (tested and shown to hurt that architecture
specifically).

CRITICAL: never import cypmode.challenge.multitask (or anything that
imports torch) into this process -- see that module's own docstring for
the real, confirmed lightgbm+torch segfault. This module reads
cypmode.challenge.final_assembly_cyp2d6's cached predictions from disk
instead of calling it directly; run that module first, as a completely
separate process (see _main()'s own check).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from cypmode.challenge.knn_features import (
    _morgan_fp,
    add_knn_label_features_for_fold,
    knn_label_features,
)
from cypmode.challenge.submission import (
    TEST_SET_PATH,
    build_submission,
    load_test_set,
    validate_submission,
)
from cypmode.challenge.tdi_ablation import add_ablation_features
from cypmode.challenge.tdi_baseline import compute_descriptors, load_tdi_training_data
from cypmode.challenge.threshold_tuning import _inner_oof_threshold

FEATURE_COL = "m3_no_amine_features"
KNN_K = 10
INNER_SPLITS = 3
RANDOM_STATE = 42
CYP2D6_CACHE_PATH = Path("data/tdi_track/outputs/cyp2d6/final_assembly_predictions.csv")
SUBMISSION_PATH = Path("submissions/tdi_track/tdi_track_submission.csv")


def fit_predict_cyp3a4(train_df: pd.DataFrame, test_df: pd.DataFrame, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    label_col = "CYP3A4_is_TDI"
    cyp_df = train_df.dropna(subset=[label_col]).reset_index(drop=True)
    y = cyp_df[label_col].astype(int).to_numpy()

    base_features = np.stack(cyp_df[FEATURE_COL])

    train_fps = [_morgan_fp(s) for s in cyp_df["SMILES"]]
    test_fps = [_morgan_fp(s) for s in test_df["SMILES"]]

    all_idx = np.arange(len(cyp_df))
    empty_idx = np.array([], dtype=int)
    train_knn = add_knn_label_features_for_fold(train_fps, y, all_idx, empty_idx, k=KNN_K)
    test_knn = knn_label_features(test_fps, train_fps, y, k=KNN_K)

    train_features = np.hstack([base_features, train_knn])
    test_features = np.hstack([np.stack(test_df[FEATURE_COL]), test_knn])

    tuned_threshold = _inner_oof_threshold(train_features, y, inner_splits=INNER_SPLITS, random_state=random_state)

    model = LGBMClassifier(verbose=-1, random_state=random_state)
    model.fit(train_features, y)
    test_proba = model.predict_proba(test_features)[:, 1]

    return pd.DataFrame(
        {
            "Molecule_Name": test_df["Molecule_Name"].to_numpy(),
            "CYP3A4_proba": test_proba,
            "CYP3A4_is_TDI": test_proba >= tuned_threshold,
            "CYP3A4_threshold_used": tuned_threshold,
        }
    )


def _main() -> None:
    if not CYP2D6_CACHE_PATH.exists():
        raise SystemExit(
            f"{CYP2D6_CACHE_PATH} doesn't exist -- run "
            "`python -m cypmode.challenge.final_assembly_cyp2d6` first, as a separate process "
            "(it imports torch; this module must never share a process with it, see module docstring)"
        )
    cyp2d6_predictions = pd.read_csv(CYP2D6_CACHE_PATH)

    train_df = add_ablation_features(compute_descriptors(load_tdi_training_data()))
    test_df = add_ablation_features(compute_descriptors(pd.read_csv(TEST_SET_PATH)))

    cyp3a4_predictions = fit_predict_cyp3a4(train_df, test_df)
    print(f"CYP3A4: {cyp3a4_predictions['CYP3A4_is_TDI'].sum()}/{len(cyp3a4_predictions)} predicted positive")
    print(f"threshold used: {cyp3a4_predictions['CYP3A4_threshold_used'].iloc[0]:.3f}")

    merged = test_df[["Molecule_Name", "SMILES"]].merge(cyp3a4_predictions, on="Molecule_Name", how="left")
    merged = merged.merge(cyp2d6_predictions[["Molecule_Name", "CYP2D6_is_TDI"]], on="Molecule_Name", how="left")

    real_test_df = load_test_set()
    submission = build_submission(
        real_test_df,
        cyp2d6_predictions=merged.set_index("Molecule_Name").loc[real_test_df["Molecule_Name"], "CYP2D6_is_TDI"].to_numpy(),
        cyp3a4_predictions=merged.set_index("Molecule_Name").loc[real_test_df["Molecule_Name"], "CYP3A4_is_TDI"].to_numpy(),
    )

    is_valid, errors = validate_submission(submission, expected_molecule_names=set(real_test_df["Molecule_Name"]))
    if not is_valid:
        raise SystemExit(f"submission failed validation: {errors}")

    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nvalid submission written to {SUBMISSION_PATH} ({len(submission)} rows)")
    print(f"CYP3A4 positive rate: {submission['CYP3A4_is_TDI'].mean():.1%}")
    print(f"CYP2D6 positive rate: {submission['CYP2D6_is_TDI'].mean():.1%}")


if __name__ == "__main__":
    _main()

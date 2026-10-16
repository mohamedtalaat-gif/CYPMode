"""CYP2D6's deployed model: the shared trunk, m3_no_amine_features only.

Real evidence behind this exact configuration (data/tdi_track/outputs/):
- shared_trunk_threshold_tuning_no_amine.csv: CYP2D6 tuned MCC 0.163,
  the best number found for CYP2D6 across every architecture/feature
  combination tried, including the shared trunk + KNN (0.135 -- KNN
  actively hurts here, see shared_trunk_knn_no_amine.csv) and OOF-bridge
  (0.075 on the LightGBM architecture -- negligible, and CYP2D6 doesn't
  even use that architecture; not re-tested on the shared trunk given
  the LightGBM read was already flat and OOF-bridge coverage there is
  near-total anyway, so a floor/no-floor split like CYP3A4's doesn't
  apply the same way -- CYP2D6 uses a different, label-free mass-spec
  assay, see CYPMode_Architecture_Specification.md's "Official scoring
  harness" section for why that assay difference plausibly explains
  this).
- cyp2d6/architecture_significance_tukey.csv: shared trunk vs.
  independent-head (the same LightGBM family cypmode.challenge.
  final_assembly uses for CYP3A4), properly paired this time -- the
  raw 0.163-vs-0.088 numbers above came from two single-split runs
  built with DIFFERENT fold-splitting strategies (plain KFold over the
  full multi-isoform dataframe vs StratifiedKFold over CYP2D6-only
  rows), so that comparison was confounded, not just a small sample.
  cypmode.challenge.architecture_significance fixes this: both
  architectures are scored on the exact same held-out CYP2D6 compounds
  each fold, 5x5=25 folds, repeated CV + Tukey HSD. Result: shared
  trunk 0.158 vs independent-head 0.080, meandiff=0.0785, p-adj<0.0001,
  reject=True -- a real, statistically significant effect, not noise.
  This is the one architecture/feature choice in either isoform's
  pipeline that clears that bar on its own (everything else tried for
  CYP2D6 -- KNN, Tier3 structural features, ChemBERTa, quantum
  pharmacophore, pharmacophore fingerprint, BindingDB -- came back
  statistically indistinguishable from plain M3).

CRITICAL: this module must never be imported in the same process as
LightGBM (see cypmode.challenge.multitask's own docstring for the real,
confirmed segfault). Writes CYP2D6 test predictions to a cache CSV;
cypmode.challenge.final_assembly (LightGBM-only) reads that cache
rather than importing this module directly.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cypmode.challenge.multitask import (
    ISOFORMS,
    _inner_oof_threshold_shared_trunk,
    _standardize,
    predict_proba,
    train_shared_trunk,
)
from cypmode.challenge.submission import TEST_SET_PATH
from cypmode.challenge.tdi_ablation import add_ablation_features
from cypmode.challenge.tdi_baseline import compute_descriptors, load_tdi_training_data

CACHE_PATH = Path("data/tdi_track/outputs/cyp2d6/final_assembly_predictions.csv")
FEATURE_COL = "m3_no_amine_features"
INNER_SPLITS = 3
EPOCHS = 200
RANDOM_STATE = 42


def fit_predict_cyp2d6(
    train_df: pd.DataFrame, test_df: pd.DataFrame, epochs: int = EPOCHS, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Fits the shared trunk on ALL labeled training data (both isoforms,
    masked loss) and predicts CYP2D6 probabilities/labels for every test
    row, using a threshold tuned the same way as every fold in
    cross_validate_shared_trunk_with_threshold_tuning -- an inner K-fold
    over the (here: full) training portion, never touching test labels
    (which don't exist anyway).
    """
    X_train_raw = np.stack(train_df[FEATURE_COL]).astype(np.float32)
    X_test_raw = np.stack(test_df[FEATURE_COL]).astype(np.float32)

    y, mask = {}, {}
    for cyp in ISOFORMS:
        col = f"{cyp}_is_TDI"
        mask[cyp] = train_df[col].notna().to_numpy().astype(np.float32)
        y[cyp] = train_df[col].fillna(0).astype(int).to_numpy().astype(np.float32)

    tuned_thresholds = _inner_oof_threshold_shared_trunk(X_train_raw, y, mask, INNER_SPLITS, epochs, random_state)

    mean, std = X_train_raw.mean(axis=0), X_train_raw.std(axis=0) + 1e-8
    X_train = _standardize(X_train_raw, mean, std)
    X_test = _standardize(X_test_raw, mean, std)

    model = train_shared_trunk(X_train, y, mask, epochs=epochs, seed=random_state)
    test_proba = predict_proba(model, X_test)["CYP2D6"]
    threshold = tuned_thresholds["CYP2D6"]

    return pd.DataFrame(
        {
            "Molecule_Name": test_df["Molecule_Name"].to_numpy(),
            "CYP2D6_proba": test_proba,
            "CYP2D6_is_TDI": test_proba >= threshold,
            "CYP2D6_threshold_used": threshold,
        }
    )


def _main() -> None:
    train_df = add_ablation_features(compute_descriptors(load_tdi_training_data()))
    test_df = add_ablation_features(compute_descriptors(pd.read_csv(TEST_SET_PATH)))

    predictions = fit_predict_cyp2d6(train_df, test_df)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(CACHE_PATH, index=False)
    print(f"CYP2D6: {predictions['CYP2D6_is_TDI'].sum()}/{len(predictions)} predicted positive")
    print(f"threshold used: {predictions['CYP2D6_threshold_used'].iloc[0]:.3f}")
    print(f"wrote {CACHE_PATH}")


if __name__ == "__main__":
    _main()

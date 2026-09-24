"""Reproduces the OpenADMET CYP Blind Challenge tutorial's own TDI baseline:
one LightGBM classifier per isoform (CYP3A4, CYP2D6) on RDKit descriptors,
evaluated over repeated stratified splits.

This exists to get a real, verified number to compare against before
changing anything about CYPMode's own approach -- rather than trusting a
recalled baseline MCC, this project computes its own from the real
training data (data/cyp_challenge/tdi_track/), matching the tutorial's own
evaluation procedure (activity_prediction.ipynb / TDI_prediction.ipynb,
github.com/OpenADMET/CYP-Challenge-Tutorial) exactly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import useful_rdkit_utils as uru
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

TDI_ISOFORMS = ["CYP3A4", "CYP2D6"]
TDI_TRAIN_PATH = Path("data/cyp_challenge/tdi_track/cyp-challenge-TRAIN_TDI.csv")


def load_tdi_training_data(path: str | Path = TDI_TRAIN_PATH) -> pd.DataFrame:
    """Molecule_Name, SMILES, and the two is_TDI label columns, restricted to
    compounds with at least one isoform's label (matching the tutorial).
    """
    df = pd.read_csv(path)
    label_cols = [f"{cyp}_is_TDI" for cyp in TDI_ISOFORMS]
    model_df = df[["Molecule_Name", "SMILES"] + label_cols].copy()
    return model_df.dropna(subset=label_cols, how="all").reset_index(drop=True)


def compute_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a `descriptors` column (RDKit descriptor vector per compound),
    via the same useful_rdkit_utils.RDKitDescriptors the official tutorial
    uses -- not a different featurization that would make results
    incomparable to the published baseline.
    """
    rdkit_desc = uru.RDKitDescriptors()
    out = df.copy()
    out["descriptors"] = [rdkit_desc.calc_smiles(s) for s in out["SMILES"]]
    return out


def cross_validate_baseline(df: pd.DataFrame, n_iterations: int = 10) -> pd.DataFrame:
    """Reproduces the tutorial's own evaluation exactly: per isoform,
    n_iterations independent stratified 75/25 splits, one fresh LightGBM
    classifier per split, reporting Accuracy/F1/MCC/ROC-AUC. `df` must
    already carry a `descriptors` column (see compute_descriptors).
    """
    rows = []
    for cyp in TDI_ISOFORMS:
        col = f"{cyp}_is_TDI"
        cyp_df = df.dropna(subset=[col]).reset_index(drop=True)
        for _ in range(n_iterations):
            train, test = train_test_split(cyp_df, stratify=cyp_df[col].astype(int), test_size=0.25)
            y_train, y_test = train[col].astype(int), test[col].astype(int)

            model = LGBMClassifier(verbose=-1)
            model.fit(np.stack(train["descriptors"]), y_train)
            proba = model.predict_proba(np.stack(test["descriptors"]))[:, 1]
            pred = (proba >= 0.5).astype(int)

            rows.append(
                {
                    "isoform": cyp,
                    "n_train": len(train),
                    "n_test": len(test),
                    "accuracy": accuracy_score(y_test, pred),
                    "f1": f1_score(y_test, pred, zero_division=0),
                    "mcc": matthews_corrcoef(y_test, pred),
                    "roc_auc": roc_auc_score(y_test, proba),
                }
            )
    return pd.DataFrame(rows)


def cross_validate_fixed_seed(
    df: pd.DataFrame, n_splits: int = 5, random_state: int = 42, feature_col: str = "descriptors"
) -> pd.DataFrame:
    """A second, independent evaluation distinct from cross_validate_baseline:
    a fixed-seed StratifiedKFold rather than the tutorial's own repeated
    unseeded random splits. The tutorial's own number is useful as a sanity
    check that this project's pipeline matches published behavior, but it
    isn't reproducible run to run and isn't enough on its own for comparing
    CYPMode's later additions against -- this is the number those
    comparisons should actually use, same folds every time.

    feature_col names the column holding each row's feature vector -- lets
    the same CV logic score different feature sets (see
    cypmode/challenge/tdi_ablation.py) without duplicating it. With the same
    random_state and an unchanged row order, different feature_col values
    still see identical folds, so results across feature sets are a fair,
    paired comparison.
    """
    rows = []
    for cyp in TDI_ISOFORMS:
        col = f"{cyp}_is_TDI"
        cyp_df = df.dropna(subset=[col]).reset_index(drop=True)
        y = cyp_df[col].astype(int)

        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for fold, (train_idx, test_idx) in enumerate(kfold.split(cyp_df, y)):
            train, test = cyp_df.iloc[train_idx], cyp_df.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = LGBMClassifier(verbose=-1, random_state=random_state)
            model.fit(np.stack(train[feature_col]), y_train)
            proba = model.predict_proba(np.stack(test[feature_col]))[:, 1]
            pred = (proba >= 0.5).astype(int)

            rows.append(
                {
                    "isoform": cyp,
                    "fold": fold,
                    "n_train": len(train),
                    "n_test": len(test),
                    "accuracy": accuracy_score(y_test, pred),
                    "f1": f1_score(y_test, pred, zero_division=0),
                    "mcc": matthews_corrcoef(y_test, pred),
                    "roc_auc": roc_auc_score(y_test, proba),
                }
            )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return results.groupby("isoform")[["accuracy", "f1", "mcc", "roc_auc"]].mean()


def _main() -> None:
    df = compute_descriptors(load_tdi_training_data())
    results = cross_validate_baseline(df)
    summary = summarize(results)
    print(summary.to_string())

    out_path = Path("data/cyp_challenge/tdi_track/baseline_reproduction.csv")
    results.to_csv(out_path, index=False)
    print(f"\nwrote per-iteration results to {out_path}")


if __name__ == "__main__":
    _main()

"""Out-of-fold direct-inhibition pIC50 bridge for the TDI model.

The blind test set has no direct-inhibition pIC50 exposed (see
CYPMode-TDI_DATA_AUDIT.md's Feature Eligibility Matrix), so a TDI model
that wants potency as a feature can't just read it off the test rows.
This module trains a separate regressor and produces out-of-fold (OOF)
predictions on the training data -- each compound's predicted pIC50 comes
from a model that never saw that compound during training -- so the
feature distribution the TDI model learns from at training time matches
what it will actually see at inference time on the real blind test set
(via predict_test, one model fit on all training data).

Originally planned as a Tobit/censored-regression loss (borrowed from
github.com/OpenADMET/moal), matching that project's primary-screen
inequality-censored labels. Checked against the real data first: direct-
inhibition pIC50 below the assay floor (pIC50=4) is NOT hard-censored --
there's no pile-up at a floor value, values continue smoothly down to
~1.9. What IS real is heteroscedastic measurement noise: the reported std
column jumps 5-10x below pIC50=4 (confirmed directly on the real data).
Tobit assumes the wrong noise model here; inverse-variance weighted
regression uses the real, measured std instead of an assumed censoring
mechanism, which is both a better fit to this specific data and requires
no unverified assumption about how sub-floor values were generated.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import useful_rdkit_utils as uru
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

DIRECT_INHIBITION_ISOFORMS = ["CYP3A4", "CYP2D6"]
TDI_TRAIN_PATH = Path("data/cyp_challenge/tdi_track/cyp-challenge-TRAIN_TDI.csv")


def load_direct_inhibition_data(path: str | Path = TDI_TRAIN_PATH) -> pd.DataFrame:
    """Molecule_Name, SMILES, and each isoform's direct-inhibition pIC50 +
    std, restricted to compounds with at least one isoform's measurement.
    """
    df = pd.read_csv(path)
    value_cols = [f"{cyp}_pIC50_direct_inhibition" for cyp in DIRECT_INHIBITION_ISOFORMS]
    std_cols = [f"{cyp}_pIC50_direct_inhibition_std" for cyp in DIRECT_INHIBITION_ISOFORMS]
    model_df = df[["Molecule_Name", "SMILES"] + value_cols + std_cols].copy()
    return model_df.dropna(subset=value_cols, how="all").reset_index(drop=True)


def compute_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    rdkit_desc = uru.RDKitDescriptors()
    out = df.copy()
    out["descriptors"] = [rdkit_desc.calc_smiles(s) for s in out["SMILES"]]
    return out


def _inverse_variance_weights(std: pd.Series, floor: float = 1e-3) -> np.ndarray:
    """1/std**2, with a floor so a near-zero std can't produce an
    unbounded weight (none are exactly zero in the real data, but this
    keeps the function safe for any future input, not just today's file).
    """
    return 1.0 / np.maximum(std.to_numpy(), floor) ** 2


def compute_oof_predictions(
    df: pd.DataFrame, n_splits: int = 5, random_state: int = 42
) -> pd.DataFrame:
    """Fold-internal OOF direct-inhibition pIC50 predictions per isoform:
    for each fold, a fresh LGBMRegressor is fit on the training portion
    only (inverse-variance weighted), and predicts the held-out portion.
    Every compound's OOF prediction comes from a model that never saw it,
    same discipline as the TDI selection mask elsewhere in this pipeline.
    Returns df with new `{isoform}_pIC50_oof` and `{isoform}_pIC50_oof_abs_error`
    columns (the latter only where the real value exists, for diagnostics).
    """
    out = df.copy()
    for cyp in DIRECT_INHIBITION_ISOFORMS:
        value_col = f"{cyp}_pIC50_direct_inhibition"
        std_col = f"{cyp}_pIC50_direct_inhibition_std"
        oof_col = f"{cyp}_pIC50_oof"

        cyp_df = out.dropna(subset=[value_col]).reset_index(drop=False)  # keep original index
        weights = _inverse_variance_weights(cyp_df[std_col])

        oof_preds = np.full(len(cyp_df), np.nan)
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for train_idx, test_idx in kfold.split(cyp_df):
            model = LGBMRegressor(verbose=-1, random_state=random_state)
            model.fit(
                np.stack(cyp_df.iloc[train_idx]["descriptors"]),
                cyp_df.iloc[train_idx][value_col],
                sample_weight=weights[train_idx],
            )
            oof_preds[test_idx] = model.predict(np.stack(cyp_df.iloc[test_idx]["descriptors"]))

        out.loc[cyp_df["index"], oof_col] = oof_preds
        out.loc[cyp_df["index"], f"{oof_col}_abs_error"] = np.abs(oof_preds - cyp_df[value_col].to_numpy())
    return out


def fit_final_model(df: pd.DataFrame, isoform: str) -> LGBMRegressor:
    """The model that actually runs on the blind test set: fit on ALL
    available training data for one isoform (inverse-variance weighted),
    no held-out fold -- there's no OOF concern at inference time since the
    test set has no ground truth to leak.
    """
    value_col = f"{isoform}_pIC50_direct_inhibition"
    std_col = f"{isoform}_pIC50_direct_inhibition_std"
    cyp_df = df.dropna(subset=[value_col]).reset_index(drop=True)
    weights = _inverse_variance_weights(cyp_df[std_col])

    model = LGBMRegressor(verbose=-1, random_state=42)
    model.fit(np.stack(cyp_df["descriptors"]), cyp_df[value_col], sample_weight=weights)
    return model


def predict_test(model: LGBMRegressor, test_df: pd.DataFrame) -> np.ndarray:
    """Apply a fit_final_model model to compute_descriptors(test_df)."""
    return model.predict(np.stack(test_df["descriptors"]))


def _main() -> None:
    df = compute_descriptors(load_direct_inhibition_data())
    df = compute_oof_predictions(df)

    for cyp in DIRECT_INHIBITION_ISOFORMS:
        errors = df[f"{cyp}_pIC50_oof_abs_error"].dropna()
        print(f"{cyp}: OOF MAE = {errors.mean():.3f} (n={len(errors)})")

    out_path = Path("data/cyp_challenge/tdi_track/oof_direct_inhibition.csv")
    df.drop(columns=["descriptors"]).to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    _main()

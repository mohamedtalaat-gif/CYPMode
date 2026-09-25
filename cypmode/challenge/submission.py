"""TDI submission formatting and validation against the official schema.

Confirmed directly from the live challenge portal (not the tutorial):
a .parquet or .csv with exactly 750 rows and 4 columns -- SMILES (string),
Molecule_Name (string), CYP2D6_is_TDI (bool), CYP3A4_is_TDI (bool), in
that column order. See CYPMode-TDI_Architecture_Specification.md.

This module only formats and validates; it doesn't generate predictions
itself. Building it now, independent of having a final trained model,
means the submission step isn't a rushed, unverified afterthought once
predictions exist -- the format is already checked against the real
blinded test set.
"""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["SMILES", "Molecule_Name", "CYP2D6_is_TDI", "CYP3A4_is_TDI"]
EXPECTED_ROW_COUNT = 750
TEST_SET_PATH = Path("data/cyp_challenge/tdi_track/cyp-challenge-TEST-BLINDED.csv")


def build_submission(
    test_df: pd.DataFrame, cyp2d6_predictions, cyp3a4_predictions
) -> pd.DataFrame:
    """Assembles a submission DataFrame from the blinded test set (must
    have SMILES, Molecule_Name columns, in the official test set's own
    row order) and two boolean prediction arrays, one per isoform.
    """
    if len(cyp2d6_predictions) != len(test_df) or len(cyp3a4_predictions) != len(test_df):
        raise ValueError(
            f"prediction length mismatch: test_df has {len(test_df)} rows, "
            f"got {len(cyp2d6_predictions)} CYP2D6 and {len(cyp3a4_predictions)} CYP3A4 predictions"
        )
    return pd.DataFrame(
        {
            "SMILES": test_df["SMILES"].to_numpy(),
            "Molecule_Name": test_df["Molecule_Name"].to_numpy(),
            "CYP2D6_is_TDI": pd.Series(cyp2d6_predictions).astype(bool).to_numpy(),
            "CYP3A4_is_TDI": pd.Series(cyp3a4_predictions).astype(bool).to_numpy(),
        }
    )


def validate_submission(
    submission: pd.DataFrame, expected_molecule_names: set[str] | None = None
) -> tuple[bool, list[str]]:
    """Checks submission against every rule stated on the official portal.
    Returns (is_valid, error_messages) -- never raises, so this can be
    used as a real pre-flight check on any candidate submission file.
    """
    errors = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in submission.columns]
    if missing_cols:
        errors.append(f"missing required column(s): {missing_cols}")
        return False, errors  # can't check anything else meaningfully without the columns

    extra_cols = [c for c in submission.columns if c not in REQUIRED_COLUMNS]
    if extra_cols:
        errors.append(f"unexpected extra column(s): {extra_cols}")

    if len(submission) != EXPECTED_ROW_COUNT:
        errors.append(f"expected exactly {EXPECTED_ROW_COUNT} rows, got {len(submission)}")

    if submission["Molecule_Name"].duplicated().any():
        n_dupes = submission["Molecule_Name"].duplicated().sum()
        errors.append(f"{n_dupes} duplicate Molecule_Name value(s)")

    for col in ["SMILES", "Molecule_Name", "CYP2D6_is_TDI", "CYP3A4_is_TDI"]:
        if col in submission.columns and submission[col].isna().any():
            errors.append(f"column {col!r} has {submission[col].isna().sum()} missing value(s)")

    for col in ["CYP2D6_is_TDI", "CYP3A4_is_TDI"]:
        if col in submission.columns:
            non_bool = ~submission[col].dropna().apply(lambda v: isinstance(v, (bool,))).all()
            if non_bool:
                errors.append(f"column {col!r} contains non-boolean value(s)")

    if expected_molecule_names is not None and "Molecule_Name" in submission.columns:
        actual_names = set(submission["Molecule_Name"])
        missing = expected_molecule_names - actual_names
        unexpected = actual_names - expected_molecule_names
        if missing:
            errors.append(f"{len(missing)} expected Molecule_Name value(s) missing from submission")
        if unexpected:
            errors.append(f"{len(unexpected)} unexpected Molecule_Name value(s) not in the real test set")

    return len(errors) == 0, errors


def load_test_set(path: str | Path = TEST_SET_PATH) -> pd.DataFrame:
    return pd.read_csv(path)

# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import pandas as pd
import pytest

from cypmode.challenge.submission import (
    EXPECTED_ROW_COUNT,
    REQUIRED_COLUMNS,
    build_submission,
    load_test_set,
    validate_submission,
)


def test_load_test_set_matches_the_real_blinded_test_set():
    df = load_test_set()
    assert len(df) == EXPECTED_ROW_COUNT
    assert {"SMILES", "Molecule_Name"} <= set(df.columns)


def _fake_test_df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({"SMILES": ["CCO"] * n, "Molecule_Name": [f"m{i}" for i in range(n)]})


def test_build_submission_assembles_correct_columns_and_order():
    test_df = _fake_test_df()
    submission = build_submission(test_df, cyp2d6_predictions=[True, False, True, False, True],
                                   cyp3a4_predictions=[False, False, True, True, False])
    assert list(submission.columns) == REQUIRED_COLUMNS
    assert submission["CYP2D6_is_TDI"].dtype == bool
    assert submission["CYP3A4_is_TDI"].dtype == bool


def test_build_submission_rejects_length_mismatch():
    test_df = _fake_test_df(n=5)
    with pytest.raises(ValueError):
        build_submission(test_df, cyp2d6_predictions=[True, False], cyp3a4_predictions=[True, False])


def _valid_submission(n: int = EXPECTED_ROW_COUNT) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SMILES": ["CCO"] * n,
            "Molecule_Name": [f"m{i}" for i in range(n)],
            "CYP2D6_is_TDI": [i % 2 == 0 for i in range(n)],
            "CYP3A4_is_TDI": [i % 3 == 0 for i in range(n)],
        }
    )


def test_validate_submission_accepts_a_correct_file():
    submission = _valid_submission()
    is_valid, errors = validate_submission(submission, expected_molecule_names=set(submission["Molecule_Name"]))
    assert is_valid
    assert errors == []


def test_validate_submission_rejects_wrong_row_count():
    submission = _valid_submission(n=100)
    is_valid, errors = validate_submission(submission)
    assert not is_valid
    assert any("750" in e for e in errors)


def test_validate_submission_rejects_missing_column():
    submission = _valid_submission().drop(columns=["CYP3A4_is_TDI"])
    is_valid, errors = validate_submission(submission)
    assert not is_valid
    assert any("missing required column" in e for e in errors)


def test_validate_submission_rejects_duplicate_molecule_names():
    submission = _valid_submission()
    submission.loc[1, "Molecule_Name"] = submission.loc[0, "Molecule_Name"]
    is_valid, errors = validate_submission(submission)
    assert not is_valid
    assert any("duplicate" in e for e in errors)


def test_validate_submission_rejects_missing_values():
    submission = _valid_submission()
    # object dtype with a real None, matching how a malformed CSV read
    # (missing cell in a boolean column) actually shows up
    submission["CYP2D6_is_TDI"] = submission["CYP2D6_is_TDI"].astype(object)
    submission.loc[0, "CYP2D6_is_TDI"] = None
    is_valid, errors = validate_submission(submission)
    assert not is_valid
    assert any("missing value" in e for e in errors)


def test_validate_submission_rejects_wrong_molecule_names():
    submission = _valid_submission()
    is_valid, errors = validate_submission(submission, expected_molecule_names={"totally_different_name"})
    assert not is_valid
    assert any("unexpected Molecule_Name" in e for e in errors)


def test_validate_submission_real_test_set_names_roundtrip():
    """The real test set's own names, used as a submission, must validate clean."""
    test_df = load_test_set()
    submission = build_submission(
        test_df,
        cyp2d6_predictions=[False] * len(test_df),
        cyp3a4_predictions=[False] * len(test_df),
    )
    is_valid, errors = validate_submission(submission, expected_molecule_names=set(test_df["Molecule_Name"]))
    assert is_valid, errors

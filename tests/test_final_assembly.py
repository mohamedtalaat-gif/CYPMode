# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from cypmode.challenge.final_assembly import FEATURE_COL, fit_predict_cyp3a4

KETOCONAZOLE = "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1"
VERAPAMIL = "COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC"
ETHANOL = "CCO"
ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


def _synthetic_train_df(n_per_class: int = 15, n_features: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    smiles_pool = [KETOCONAZOLE, VERAPAMIL, ETHANOL, ASPIRIN, CAFFEINE]
    rows = []
    for label in (0, 1):
        for i in range(n_per_class):
            rows.append(
                {
                    "Molecule_Name": f"train_{label}_{i}",
                    "SMILES": smiles_pool[i % len(smiles_pool)],
                    "CYP3A4_is_TDI": bool(label),
                    FEATURE_COL: rng.normal(loc=float(label), size=n_features),
                }
            )
    return pd.DataFrame(rows)


def _synthetic_test_df(n: int = 8, n_features: int = 6, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    smiles_pool = [KETOCONAZOLE, VERAPAMIL, ETHANOL, ASPIRIN, CAFFEINE]
    rows = [
        {
            "Molecule_Name": f"test_{i}",
            "SMILES": smiles_pool[i % len(smiles_pool)],
            FEATURE_COL: rng.normal(size=n_features),
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows)


def test_fit_predict_cyp3a4_returns_one_row_per_test_compound():
    train_df = _synthetic_train_df()
    test_df = _synthetic_test_df()

    result = fit_predict_cyp3a4(train_df, test_df)

    assert len(result) == len(test_df)
    assert list(result["Molecule_Name"]) == list(test_df["Molecule_Name"])
    assert set(result.columns) >= {"Molecule_Name", "CYP3A4_proba", "CYP3A4_is_TDI", "CYP3A4_threshold_used"}


def test_fit_predict_cyp3a4_predictions_are_real_booleans_from_a_valid_threshold():
    train_df = _synthetic_train_df()
    test_df = _synthetic_test_df()

    result = fit_predict_cyp3a4(train_df, test_df)

    assert result["CYP3A4_is_TDI"].dtype == bool
    assert (result["CYP3A4_proba"].to_numpy() >= 0).all() and (result["CYP3A4_proba"].to_numpy() <= 1).all()
    threshold = result["CYP3A4_threshold_used"].iloc[0]
    assert 0.0 <= threshold <= 1.0
    # threshold_used is one fold-internal-tuned value shared across the whole
    # (single, full-data) fit -- not a per-row artifact.
    assert result["CYP3A4_threshold_used"].nunique() == 1


def test_fit_predict_cyp3a4_prediction_matches_its_own_reported_threshold():
    train_df = _synthetic_train_df()
    test_df = _synthetic_test_df()

    result = fit_predict_cyp3a4(train_df, test_df)

    expected = result["CYP3A4_proba"] >= result["CYP3A4_threshold_used"]
    pd.testing.assert_series_equal(result["CYP3A4_is_TDI"], expected, check_names=False)


def test_fit_predict_cyp3a4_never_leaks_train_labels_into_predict_at_the_source_level():
    """The final classifier is fit once on ALL training data -- there's no
    fold-holdout for the deployed model. The real leakage discipline this
    function must preserve is different: no piece of test_df ever informs
    the KNN reference pool or the fitted model beyond its own features. A
    cheap proxy for that: predictions must depend only on test_df's own
    feature columns, not on row order or on any test-only side channel.
    """
    train_df = _synthetic_train_df()
    test_df = _synthetic_test_df()

    result_a = fit_predict_cyp3a4(train_df, test_df)
    shuffled_test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    result_b = fit_predict_cyp3a4(train_df, shuffled_test_df)

    a_by_name = result_a.set_index("Molecule_Name")["CYP3A4_proba"]
    b_by_name = result_b.set_index("Molecule_Name")["CYP3A4_proba"]
    pd.testing.assert_series_equal(a_by_name.sort_index(), b_by_name.sort_index(), check_names=False)

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

# Real, reproducible segfault otherwise: torch and lightgbm loaded in the
# same process crash (OpenMP runtime conflict, confirmed directly on this
# machine) -- run this file in its own pytest invocation, never alongside
# the rest of the suite. See pyproject.toml's `torch` marker, and
# cypmode.challenge.final_assembly_cyp2d6's own module docstring.
pytestmark = pytest.mark.torch

from cypmode.challenge.final_assembly_cyp2d6 import FEATURE_COL, fit_predict_cyp2d6


def _synthetic_train_df(n_per_class: int = 15, n_features: int = 6, seed: int = 0) -> pd.DataFrame:
    """Real production shape: one row per compound, with independent (and
    sometimes missing) CYP3A4_is_TDI / CYP2D6_is_TDI labels -- not two
    separate per-isoform populations glued together.
    """
    rng = np.random.default_rng(seed)
    n = n_per_class * 2
    rows = []
    for i in range(n):
        label_a = bool(i % 2)
        label_b = bool((i // 2) % 2)
        features = rng.normal(loc=float(label_a or label_b), size=n_features)
        rows.append(
            {
                "Molecule_Name": f"train_{i}",
                FEATURE_COL: features,
                "CYP3A4_is_TDI": label_a,
                "CYP2D6_is_TDI": label_b,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_test_df(n: int = 8, n_features: int = 6, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Molecule_Name": [f"test_{i}" for i in range(n)],
            FEATURE_COL: [rng.normal(size=n_features) for _ in range(n)],
        }
    )


def test_fit_predict_cyp2d6_returns_one_row_per_test_compound():
    train_df = _synthetic_train_df()
    test_df = _synthetic_test_df()

    result = fit_predict_cyp2d6(train_df, test_df, epochs=20)

    assert len(result) == len(test_df)
    assert list(result["Molecule_Name"]) == list(test_df["Molecule_Name"])
    assert set(result.columns) >= {"Molecule_Name", "CYP2D6_proba", "CYP2D6_is_TDI", "CYP2D6_threshold_used"}


def test_fit_predict_cyp2d6_predictions_are_real_booleans_matching_the_threshold():
    train_df = _synthetic_train_df()
    test_df = _synthetic_test_df()

    result = fit_predict_cyp2d6(train_df, test_df, epochs=20)

    assert result["CYP2D6_is_TDI"].dtype == bool
    assert (result["CYP2D6_proba"].to_numpy() >= 0).all() and (result["CYP2D6_proba"].to_numpy() <= 1).all()
    expected = result["CYP2D6_proba"] >= result["CYP2D6_threshold_used"]
    pd.testing.assert_series_equal(result["CYP2D6_is_TDI"], expected, check_names=False)


def test_fit_predict_cyp2d6_tolerates_missing_cyp3a4_labels_via_masking():
    """Real production shape: most rows have a CYP2D6 label but no CYP3A4
    one (or vice versa). masked_bce_loss must handle that without raising
    or silently treating the missing label as a real 0/negative.
    """
    train_df = _synthetic_train_df()
    train_df["CYP3A4_is_TDI"] = train_df["CYP3A4_is_TDI"].astype(object)
    train_df.loc[: len(train_df) // 2, "CYP3A4_is_TDI"] = np.nan
    test_df = _synthetic_test_df()

    result = fit_predict_cyp2d6(train_df, test_df, epochs=20)
    assert len(result) == len(test_df)
    assert result["CYP2D6_proba"].notna().all()

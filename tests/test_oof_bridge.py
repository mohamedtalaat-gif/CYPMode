import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from cypmode.challenge.oof_bridge import (
    DIRECT_INHIBITION_ISOFORMS,
    compute_oof_predictions,
    fit_final_model,
    load_direct_inhibition_data,
    predict_test,
)


def test_load_direct_inhibition_data_matches_real_file():
    df = load_direct_inhibition_data()
    assert {"Molecule_Name", "SMILES", "CYP3A4_pIC50_direct_inhibition", "CYP2D6_pIC50_direct_inhibition"} <= set(
        df.columns
    )
    assert len(df) > 0


def _synthetic_regression_df(n: int = 150, n_features: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cyp in DIRECT_INHIBITION_ISOFORMS:
        for i in range(n):
            desc = rng.normal(size=n_features)
            rows.append(
                {
                    "Molecule_Name": f"m_{cyp}_{i}",
                    "descriptors": desc,
                    f"{cyp}_pIC50_direct_inhibition": float(desc.sum() + rng.normal(scale=0.1)),
                    f"{cyp}_pIC50_direct_inhibition_std": abs(rng.normal(scale=0.1)) + 0.01,
                }
            )
    return pd.DataFrame(rows)


def test_compute_oof_predictions_every_compound_predicted_by_a_model_that_never_saw_it():
    df = _synthetic_regression_df()
    result = compute_oof_predictions(df, n_splits=3)

    for cyp in DIRECT_INHIBITION_ISOFORMS:
        oof_col = f"{cyp}_pIC50_oof"
        labeled = result[result[f"{cyp}_pIC50_direct_inhibition"].notna()]
        assert labeled[oof_col].notna().all()
        # predictions should correlate with the real (noisy linear) signal --
        # LightGBM approximating a linear sum from limited samples per fold
        # won't be a tight fit, so this checks the OOF wiring works, not
        # model accuracy (that's what the real-data run reports separately)
        corr = np.corrcoef(labeled[oof_col], labeled[f"{cyp}_pIC50_direct_inhibition"])[0, 1]
        assert corr > 0.3


def test_fit_final_model_and_predict_test_roundtrip():
    df = _synthetic_regression_df()
    model = fit_final_model(df, "CYP3A4")

    test_df = pd.DataFrame({"descriptors": [np.zeros(5), np.ones(5)]})
    preds = predict_test(model, test_df)
    assert len(preds) == 2
    assert np.isfinite(preds).all()


def test_inverse_variance_weighting_reduces_influence_of_noisy_points():
    # A single very-high-std outlier shouldn't dominate the fit the way
    # unweighted regression would let it.
    rng = np.random.default_rng(1)
    n = 40
    desc = [rng.normal(size=3) for _ in range(n)]
    values = [float(d.sum()) for d in desc]
    stds = [0.05] * n

    # inject one wildly wrong, high-std (unreliable) point
    desc.append(np.array([0.0, 0.0, 0.0]))
    values.append(100.0)
    stds.append(5.0)

    df = pd.DataFrame(
        {
            "Molecule_Name": [f"m{i}" for i in range(n + 1)],
            "descriptors": desc,
            "CYP3A4_pIC50_direct_inhibition": values,
            "CYP3A4_pIC50_direct_inhibition_std": stds,
            "CYP2D6_pIC50_direct_inhibition": [np.nan] * (n + 1),
            "CYP2D6_pIC50_direct_inhibition_std": [np.nan] * (n + 1),
        }
    )
    model = fit_final_model(df, "CYP3A4")
    pred_at_origin = model.predict(np.array([[0.0, 0.0, 0.0]]))[0]
    # a model that gave the outlier full weight would predict close to 100
    assert pred_at_origin < 50

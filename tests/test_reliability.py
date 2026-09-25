import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from cypmode.challenge.reliability import (
    compute_reliability_score,
    ensemble_disagreement,
    tanimoto_ood_distance,
    validate_reliability_correlation,
)

KETOCONAZOLE = "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1"
ETHANOL = "CCO"
BENZENE = "c1ccccc1"


def test_ensemble_disagreement_is_zero_for_a_deterministic_signal():
    rng = np.random.default_rng(0)
    train_features = rng.normal(size=(100, 4))
    train_labels = (train_features[:, 0] > 0).astype(int)  # trivially separable on one feature
    apply_features = rng.normal(size=(10, 4))

    disagreement = ensemble_disagreement(train_features, train_labels, apply_features, n_models=5)
    assert len(disagreement) == 10
    assert (disagreement >= 0).all()
    # an easy, trivially-separable signal should give low disagreement on average
    assert disagreement.mean() < 0.3


def test_tanimoto_ood_distance_zero_for_identical_compound():
    train_smiles = pd.Series([KETOCONAZOLE, ETHANOL])
    apply_smiles = pd.Series([KETOCONAZOLE])  # identical to a training compound
    distances = tanimoto_ood_distance(train_smiles, apply_smiles)
    assert distances[0] == pytest.approx(0.0, abs=1e-9)


def test_tanimoto_ood_distance_positive_for_dissimilar_compound():
    train_smiles = pd.Series([ETHANOL])
    apply_smiles = pd.Series([KETOCONAZOLE])  # structurally very different
    distances = tanimoto_ood_distance(train_smiles, apply_smiles)
    assert distances[0] > 0.8


def test_compute_reliability_score_combines_components_without_forcing_a_single_number():
    df = compute_reliability_score(
        ensemble_disagreement_values=np.array([0.1, 0.2]),
        ood_distance_values=np.array([0.3, 0.4]),
    )
    assert list(df.columns) == ["ensemble_disagreement", "ood_distance"]
    assert len(df) == 2


def test_compute_reliability_score_includes_structural_confidence_when_given():
    df = compute_reliability_score(
        ensemble_disagreement_values=np.array([0.1]),
        ood_distance_values=np.array([0.3]),
        structural_confidence_values=np.array([0.9]),
    )
    assert "structural_confidence" in df.columns


def test_validate_reliability_correlation_detects_a_real_relationship():
    rng = np.random.default_rng(0)
    n = 200
    error = rng.integers(0, 2, size=n)
    # a component perfectly aligned with error should show a strong positive correlation
    aligned_component = error + rng.normal(scale=0.05, size=n)
    unrelated_component = rng.normal(size=n)

    reliability_df = pd.DataFrame({"aligned": aligned_component, "unrelated": unrelated_component})
    result = validate_reliability_correlation(reliability_df, error)

    aligned_row = result[result["component"] == "aligned"].iloc[0]
    unrelated_row = result[result["component"] == "unrelated"].iloc[0]
    assert aligned_row["point_biserial_r"] > 0.8
    assert abs(unrelated_row["point_biserial_r"]) < 0.3


def test_validate_reliability_correlation_handles_missing_values():
    reliability_df = pd.DataFrame({"partial": [0.1, np.nan, 0.3, 0.4, np.nan, 0.6, 0.7, 0.8]})
    error = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    result = validate_reliability_correlation(reliability_df, error)
    assert result.iloc[0]["n"] == 6  # 2 NaNs dropped

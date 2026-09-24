import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from cypmode.challenge.tdi_baseline import cross_validate_baseline, load_tdi_training_data, summarize


def test_load_tdi_training_data_matches_the_real_official_file():
    df = load_tdi_training_data()
    assert {"Molecule_Name", "SMILES", "CYP3A4_is_TDI", "CYP2D6_is_TDI"} <= set(df.columns)
    # 6145 rows in the raw file, but not all carry an is_TDI label for either
    # isoform -- confirmed directly against the official HF dataset.
    assert len(df) == 4822


def _synthetic_labeled_descriptors(n_per_class: int = 30, n_features: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cyp in ["CYP3A4", "CYP2D6"]:
        for label in (0, 1):
            center = rng.normal(size=n_features) * (3 if label else -3)
            for _ in range(n_per_class):
                rows.append(
                    {
                        "Molecule_Name": f"synthetic_{len(rows)}",
                        f"{cyp}_is_TDI": bool(label),
                        "descriptors": center + rng.normal(scale=0.5, size=n_features),
                    }
                )
    # each row only carries a label for the isoform it was generated for, matching real sparsity
    df = pd.DataFrame(rows)
    return df


def test_cross_validate_baseline_returns_metrics_per_isoform():
    df = _synthetic_labeled_descriptors()
    results = cross_validate_baseline(df, n_iterations=2)

    assert set(results["isoform"]) == {"CYP3A4", "CYP2D6"}
    assert len(results) == 4  # 2 isoforms x 2 iterations
    assert {"accuracy", "f1", "mcc", "roc_auc"} <= set(results.columns)
    assert results["mcc"].between(-1.0, 1.0).all()


def test_summarize_averages_per_isoform():
    df = _synthetic_labeled_descriptors()
    results = cross_validate_baseline(df, n_iterations=2)
    summary = summarize(results)
    assert set(summary.index) == {"CYP3A4", "CYP2D6"}
    assert {"accuracy", "f1", "mcc", "roc_auc"} <= set(summary.columns)

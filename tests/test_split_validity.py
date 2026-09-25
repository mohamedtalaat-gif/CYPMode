import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from cypmode.challenge.split_validity import train_test_similarity_summary

KETOCONAZOLE = "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1"
ETHANOL = "CCO"
BENZENE = "c1ccccc1"


def test_similarity_summary_is_high_when_apply_set_equals_train_set():
    train_smiles = pd.Series([KETOCONAZOLE, ETHANOL, BENZENE])
    summary = train_test_similarity_summary(train_smiles, train_smiles)
    assert summary["n"] == 3
    assert summary["median_similarity"] == pytest.approx(1.0)
    assert summary["frac_similarity_ge_0.5"] == pytest.approx(1.0)


def test_similarity_summary_is_low_for_chemically_distant_compounds():
    train_smiles = pd.Series([ETHANOL])
    apply_smiles = pd.Series([KETOCONAZOLE])  # structurally very different, low similarity expected
    summary = train_test_similarity_summary(train_smiles, apply_smiles)
    assert summary["median_similarity"] < 0.2
    assert summary["frac_similarity_ge_0.5"] == pytest.approx(0.0)


def test_similarity_summary_reports_fraction_above_threshold_correctly():
    # one compound identical to train (similarity=1.0), one very different (near 0)
    train_smiles = pd.Series([KETOCONAZOLE])
    apply_smiles = pd.Series([KETOCONAZOLE, ETHANOL])
    summary = train_test_similarity_summary(train_smiles, apply_smiles)
    assert summary["n"] == 2
    assert summary["frac_similarity_ge_0.5"] == pytest.approx(0.5)

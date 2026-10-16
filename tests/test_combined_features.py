import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from cypmode.challenge.combined_features import cross_validate_combined

KETOCONAZOLE = "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1"
CLOTRIMAZOLE = "Clc1ccccc1C(c1ccccc1)(c1ccccc1)n1ccnc1"
ETHANOL = "CCO"
BENZENE = "c1ccccc1"


def _synthetic_df(n: int = 150, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    real_smiles = [KETOCONAZOLE, CLOTRIMAZOLE, ETHANOL, BENZENE]
    rows = []
    for i in range(n):
        rows.append(
            {
                "SMILES": real_smiles[i % 4],
                "m3_no_amine_features": rng.normal(size=5),
                "aux_aid1851_CYP3A4": rng.uniform(0, 1),
                "aux_bindingdb_CYP3A4": rng.uniform(0, 1),
                "CYP3A4_is_TDI": bool(rng.random() > 0.5),
            }
        )
    return pd.DataFrame(rows)


def test_cross_validate_combined_covers_every_compound_once():
    df = _synthetic_df()
    results = cross_validate_combined(df, label_col="CYP3A4_is_TDI", isoform="CYP3A4", n_splits=5)
    assert results["n_val"].sum() == len(df)
    assert len(results) == 5
    assert {"mcc_m3", "mcc_combined"} <= set(results.columns)


def test_cross_validate_combined_requires_the_expected_aux_columns():
    df = _synthetic_df().drop(columns=["aux_bindingdb_CYP3A4"])
    with pytest.raises(KeyError):
        cross_validate_combined(df, label_col="CYP3A4_is_TDI", isoform="CYP3A4", n_splits=5)

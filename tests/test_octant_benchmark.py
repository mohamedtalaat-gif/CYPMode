import pandas as pd

from cypmode.data.octant_benchmark import (
    annotate_motifs,
    compare_potency_by_motif,
    load_inhibition_data,
)

KETOCONAZOLE_SMILES = "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1"  # imidazole motif
ETHANOL_SMILES = "CCO"  # no motif


def test_annotate_motifs_adds_columns():
    df = pd.DataFrame({"standardized_smiles": [KETOCONAZOLE_SMILES, ETHANOL_SMILES]})
    annotated = annotate_motifs(df)
    assert list(annotated["has_motif"]) == [True, False]
    assert annotated.loc[0, "motifs"] == ["imidazole_Nsub"]
    assert annotated.loc[1, "motifs"] == []


def test_compare_potency_by_motif_on_synthetic_data():
    df = pd.DataFrame(
        {
            "drc_qc_status": ["PASS", "PASS", "PASS", "PASS", "FAIL"],
            "CYP3A4_pIC50": [7.0, 6.5, 4.0, 4.5, 9.0],
            "has_motif": [True, True, False, False, True],
        }
    )
    result = compare_potency_by_motif(df)

    assert result["n_qc_passed_with_pic50"] == 4  # the FAIL row is excluded
    assert result["n_motif_positive"] == 2
    assert result["n_motif_negative"] == 2
    assert result["motif_positive_mean_pIC50"] == 6.75
    assert result["motif_negative_mean_pIC50"] == 4.25
    assert 0.0 <= result["p_value_one_sided_greater"] <= 1.0


def test_compare_potency_excludes_missing_pic50():
    df = pd.DataFrame(
        {
            "drc_qc_status": ["PASS", "PASS", "PASS"],
            "CYP3A4_pIC50": [7.0, None, 5.0],
            "has_motif": [True, True, False],
        }
    )
    result = compare_potency_by_motif(df)
    assert result["n_qc_passed_with_pic50"] == 2
    assert result["n_motif_positive"] == 1


def test_load_inhibition_data_uses_checked_in_cache():
    df = load_inhibition_data()
    assert {"standardized_smiles", "CYP3A4_pIC50", "drc_qc_status"} <= set(df.columns)
    assert len(df) > 1000


def test_load_inhibition_data_does_not_redownload_when_cached(tmp_path, monkeypatch):
    cache = tmp_path / "inhibition.parquet"
    real_df = load_inhibition_data()
    real_df.to_parquet(cache)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("should not re-download when the cache already exists")

    monkeypatch.setattr("cypmode.data.octant_benchmark.urllib.request.urlretrieve", _fail_if_called)

    df = load_inhibition_data(cache_path=cache)
    assert len(df) == len(real_df)

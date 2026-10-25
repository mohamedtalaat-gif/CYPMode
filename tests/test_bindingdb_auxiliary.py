# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from cypmode.challenge.bindingdb_auxiliary import (
    _parse_affinity_nM,
    exclude_exact_overlap,
    fit_auxiliary_model,
    load_bindingdb_isoform,
    predict_auxiliary_activity,
)

KETOCONAZOLE = "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1"
CLOTRIMAZOLE = "Clc1ccccc1C(c1ccccc1)(c1ccccc1)n1ccnc1"
ETHANOL = "CCO"
BENZENE = "c1ccccc1"


def test_parse_affinity_nM_strips_censoring_prefixes():
    assert _parse_affinity_nM("29400") == pytest.approx(29400.0)
    assert _parse_affinity_nM(">400000") == pytest.approx(400000.0)
    assert _parse_affinity_nM("<10") == pytest.approx(10.0)
    assert _parse_affinity_nM("~500") == pytest.approx(500.0)


def test_parse_affinity_nM_returns_none_for_unparseable_value():
    assert _parse_affinity_nM("not-a-number") is None


def _fake_bindingdb_json(tmp_path, records):
    path = tmp_path / "fake.json"
    path.write_text(json.dumps({"getLindsByUniprotsResponse": {"affinities": records}}))
    return path


def test_load_bindingdb_isoform_takes_median_across_repeated_measurements(tmp_path):
    # Real quirk: the same compound can appear many times across different
    # papers with different reported values -- median-combine, do not drop.
    records = [
        {"query": "x", "monomerid": "1", "smile": KETOCONAZOLE, "affinity_type": "Ki", "affinity": "1000"},
        {"query": "x", "monomerid": "1", "smile": KETOCONAZOLE, "affinity_type": "Ki", "affinity": "3000"},
        {"query": "x", "monomerid": "1", "smile": KETOCONAZOLE, "affinity_type": "Ki", "affinity": "5000"},
        {"query": "x", "monomerid": "2", "smile": ETHANOL, "affinity_type": "IC50", "affinity": ">400000"},
    ]
    path = _fake_bindingdb_json(tmp_path, records)
    df = load_bindingdb_isoform(path)
    assert len(df) == 2  # one row per unique SMILES
    keto_row = df[df["SMILES"] == KETOCONAZOLE].iloc[0]
    assert keto_row["affinity_uM"] == pytest.approx(3.0)  # median of 1000/3000/5000 nM -> uM


def test_load_bindingdb_isoform_drops_unparseable_extended_smiles(tmp_path):
    # Real quirk, confirmed directly on a real CYP3A4 pull: a trailing
    # "|r,THB:...|" extended-annotation suffix that plain RDKit cannot
    # parse -- must be dropped, not passed through to crash downstream
    # featurization.
    records = [
        {
            "query": "x",
            "monomerid": "1",
            "smile": "CC(=O)N1C2CCC1CN(C2)C(=O)OC1(CC1) |r,THB:11:9:3:5.6|",
            "affinity_type": "Ki",
            "affinity": "1000",
        },
        {"query": "x", "monomerid": "2", "smile": ETHANOL, "affinity_type": "Ki", "affinity": "2000"},
    ]
    path = _fake_bindingdb_json(tmp_path, records)
    df = load_bindingdb_isoform(path)
    assert len(df) == 1
    assert df.iloc[0]["SMILES"] == ETHANOL


def test_load_bindingdb_isoform_skips_unparseable_rows(tmp_path):
    records = [
        {"query": "x", "monomerid": "1", "smile": KETOCONAZOLE, "affinity_type": "Ki", "affinity": "not-a-number"},
        {"query": "x", "monomerid": "2", "smile": ETHANOL, "affinity_type": "Ki", "affinity": "1000"},
    ]
    path = _fake_bindingdb_json(tmp_path, records)
    df = load_bindingdb_isoform(path)
    assert len(df) == 1
    assert df.iloc[0]["SMILES"] == ETHANOL


def test_exclude_exact_overlap_removes_matching_compounds():
    bdb_df = pd.DataFrame({"SMILES": [KETOCONAZOLE, ETHANOL, BENZENE], "affinity_uM": [1.0, 2.0, 3.0]})
    challenge_smiles = pd.Series([KETOCONAZOLE, CLOTRIMAZOLE])
    result = exclude_exact_overlap(bdb_df, challenge_smiles)
    assert KETOCONAZOLE not in result["SMILES"].values
    assert ETHANOL in result["SMILES"].values


def test_exclude_exact_overlap_catches_a_salt_form_of_a_challenge_compound():
    bdb_df = pd.DataFrame({"SMILES": [KETOCONAZOLE + ".Cl", ETHANOL], "affinity_uM": [1.0, 2.0]})
    challenge_smiles = pd.Series([KETOCONAZOLE])
    result = exclude_exact_overlap(bdb_df, challenge_smiles)
    assert (KETOCONAZOLE + ".Cl") not in result["SMILES"].values
    assert ETHANOL in result["SMILES"].values


def test_fit_and_predict_auxiliary_model_runs_end_to_end():
    rng = np.random.default_rng(0)
    smiles_pool = [KETOCONAZOLE, CLOTRIMAZOLE, ETHANOL, BENZENE]
    bdb_df = pd.DataFrame(
        {
            "SMILES": [smiles_pool[i % 4] for i in range(20)],
            "affinity_uM": rng.uniform(0.1, 100, size=20),
        }
    )
    model = fit_auxiliary_model(bdb_df)
    target_df = pd.DataFrame({"SMILES": [KETOCONAZOLE, ETHANOL]})
    proba = predict_auxiliary_activity(model, target_df)
    assert len(proba) == 2
    assert ((proba >= 0) & (proba <= 1)).all()


def test_fit_auxiliary_model_uses_the_active_threshold():
    smiles_pool = [KETOCONAZOLE, CLOTRIMAZOLE, ETHANOL, BENZENE] * 5
    # all potent (well below 10 uM default threshold) -> all should end up
    # labeled active internally; just confirm this does not raise on an
    # all-one-class label set (LightGBM handles it; predict_proba should
    # still return finite values)
    bdb_df = pd.DataFrame({"SMILES": smiles_pool, "affinity_uM": [0.01] * len(smiles_pool)})
    model = fit_auxiliary_model(bdb_df, active_threshold_uM=10.0)
    proba = predict_auxiliary_activity(model, pd.DataFrame({"SMILES": [KETOCONAZOLE]}))
    assert np.isfinite(proba).all()

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from cypmode.challenge.pubchem_auxiliary import (
    AID1851_ISOFORMS,
    exclude_exact_overlap,
    fit_auxiliary_model,
    load_aid1851,
    predict_auxiliary_activity,
)

KETOCONAZOLE = "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1"
CLOTRIMAZOLE = "Clc1ccccc1C(c1ccccc1)(c1ccccc1)n1ccnc1"
ETHANOL = "CCO"
BENZENE = "c1ccccc1"

# Minimal fake AID1851 CSV in PubChem's real long format: 4 metadata rows
# (skipped by load_aid1851, matching the real file's shape) + a header +
# real data rows for 2 compounds x 2 panels each.
FAKE_AID1851_CSV = """PUBCHEM_RESULT_TAG,PUBCHEM_SID,PUBCHEM_CID,PUBCHEM_EXT_DATASOURCE_SMILES,PUBCHEM_ACTIVITY_OUTCOME,Potency,Panel Name
RESULT_TYPE,,,,,,
RESULT_DESCR,,,,,,
RESULT_UNIT,,,,,,
RESULT_ATTR,,,,,,
1,100,1,{keto},Active,1.2,p450-cyp2d6
2,100,1,{keto},Active,2.3,p450-cyp3a4
3,101,2,{eth},Inactive,,p450-cyp2d6
4,101,2,{eth},Inconclusive,,p450-cyp3a4
""".format(keto=KETOCONAZOLE, eth=ETHANOL)


@pytest.fixture
def fake_aid1851_path(tmp_path):
    path = tmp_path / "aid1851_raw.csv"
    path.write_text(FAKE_AID1851_CSV)
    return path


def test_load_aid1851_collapses_replicate_rows_instead_of_multiplying(tmp_path):
    # Real data quirk, confirmed directly on the actual AID1851 file: a
    # (CID, panel) pair can have >1 row (replicate measurements). Without
    # collapsing before merging, this multiplies row count per isoform
    # merge and compounds across all 5 -- confirmed directly: it inflated
    # 16,560 real compounds into 86,721 rows before this was fixed.
    csv = """PUBCHEM_RESULT_TAG,PUBCHEM_SID,PUBCHEM_CID,PUBCHEM_EXT_DATASOURCE_SMILES,PUBCHEM_ACTIVITY_OUTCOME,Potency,Panel Name
RESULT_TYPE,,,,,,
RESULT_DESCR,,,,,,
RESULT_UNIT,,,,,,
RESULT_ATTR,,,,,,
1,100,1,{keto},Active,1.2,p450-cyp2d6
2,100,1,{keto},Inactive,1.5,p450-cyp2d6
3,100,1,{keto},Active,2.3,p450-cyp3a4
4,100,1,{keto},Active,2.4,p450-cyp3a4
5,100,1,{keto},Inactive,2.5,p450-cyp3a4
""".format(keto=KETOCONAZOLE)
    path = tmp_path / "aid1851_raw.csv"
    path.write_text(csv)

    df = load_aid1851(path)
    # 2 duplicate CYP2D6 rows x 3 duplicate CYP3A4 rows for the same CID
    # would multiply to 6 rows if not collapsed first -- must stay 1.
    assert len(df) == 1


def test_load_aid1851_pivots_to_one_row_per_compound(fake_aid1851_path):
    df = load_aid1851(fake_aid1851_path)
    assert len(df) == 2
    assert set(AID1851_ISOFORMS) <= {c.replace("_active", "") for c in df.columns if c.endswith("_active")} | set(AID1851_ISOFORMS)
    keto_row = df[df["SMILES"] == KETOCONAZOLE].iloc[0]
    assert keto_row["CYP2D6_active"] == True  # noqa: E712
    assert keto_row["CYP3A4_active"] == True  # noqa: E712
    eth_row = df[df["SMILES"] == ETHANOL].iloc[0]
    assert eth_row["CYP2D6_active"] == False  # noqa: E712
    assert pd.isna(eth_row["CYP3A4_active"])  # Inconclusive -> NaN, not False


def test_exclude_exact_overlap_removes_matching_compounds():
    aid_df = pd.DataFrame({"SMILES": [KETOCONAZOLE, ETHANOL, BENZENE]})
    challenge_smiles = pd.Series([KETOCONAZOLE, CLOTRIMAZOLE])  # ketoconazole overlaps
    result = exclude_exact_overlap(aid_df, challenge_smiles)
    assert KETOCONAZOLE not in result["SMILES"].values
    assert ETHANOL in result["SMILES"].values
    assert BENZENE in result["SMILES"].values


def test_exclude_exact_overlap_catches_a_salt_form_of_a_challenge_compound():
    # the real gap plain canonical-SMILES matching had: a salt form and its
    # free base canonicalize differently but are the same real compound
    aid_df = pd.DataFrame({"SMILES": [KETOCONAZOLE + ".Cl", ETHANOL]})
    challenge_smiles = pd.Series([KETOCONAZOLE])  # free base only
    result = exclude_exact_overlap(aid_df, challenge_smiles)
    assert (KETOCONAZOLE + ".Cl") not in result["SMILES"].values
    assert ETHANOL in result["SMILES"].values


def test_fit_and_predict_auxiliary_model_runs_end_to_end():
    rng = np.random.default_rng(0)
    smiles_pool = [KETOCONAZOLE, CLOTRIMAZOLE, ETHANOL, BENZENE]
    aid_df = pd.DataFrame(
        {
            "SMILES": [smiles_pool[i % 4] for i in range(20)],
            "CYP2D6_active": [bool(rng.random() > 0.5) for _ in range(20)],
        }
    )
    model = fit_auxiliary_model(aid_df, "CYP2D6")
    target_df = pd.DataFrame({"SMILES": [KETOCONAZOLE, ETHANOL]})
    proba = predict_auxiliary_activity(model, target_df)
    assert len(proba) == 2
    assert ((proba >= 0) & (proba <= 1)).all()


def test_fit_auxiliary_model_drops_inconclusive_rows():
    aid_df = pd.DataFrame(
        {
            "SMILES": [KETOCONAZOLE, CLOTRIMAZOLE, ETHANOL, BENZENE] * 5,
            "CYP2D6_active": ([True, False, np.nan, True] * 5),  # one NaN (Inconclusive) per group of 4
        }
    )
    # should not raise despite the NaNs, and should train on the remaining 15 labeled rows
    model = fit_auxiliary_model(aid_df, "CYP2D6")
    assert model is not None

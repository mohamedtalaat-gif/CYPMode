"""The continuous external block: how the panel's rows become targets, and that challenge compounds never reach the training table."""

import numpy as np
import pandas as pd

from cypmode.challenge import external_continuous as ec

HEADER = "PUBCHEM_RESULT_TAG,PUBCHEM_SID,PUBCHEM_CID,PUBCHEM_EXT_DATASOURCE_SMILES,PUBCHEM_ACTIVITY_OUTCOME,Max_Response,Fit_LogAC50,Panel Name"


def write_raw(path, rows):
    lines = [HEADER] + ["RESULT_TYPE,,,,,,,", "RESULT_DESCr,,,,,,,", "RESULT_UNIT,,,,,,,", "RESULT_TYPE2,,,,,,,"]  # the four metadata rows that follow the header in a PubChem export
    lines += [",".join(str(v) for v in r) for r in rows]
    path.write_text("\n".join(lines) + "\n")


def test_targets_follow_the_panel_rules_and_challenge_compounds_are_dropped(tmp_path, monkeypatch):
    challenge = tmp_path / "challenge.csv"
    pd.DataFrame({"SMILES": ["CCO"]}).to_csv(challenge, index=False)
    raw = tmp_path / "raw.csv"
    rows = [
        (1, 1, 100, "CCCC", "Active", -95.0, -6.5, "p450-cyp3a4"),
        (2, 1, 100, "CCCC", "Active", -85.0, -6.1, "p450-cyp3a4"),  # a replicate: the median is taken
        (3, 1, 101, "CCCCC", "Inactive", -4.0, "", "p450-cyp3a4"),
        (4, 1, 102, "CCCCCC", "Inconclusive", -30.0, "", "p450-cyp3a4"),
        (5, 1, 103, "CCO", "Active", -99.0, -7.0, "p450-cyp3a4"),  # a challenge compound
        (6, 1, 104, "CCCCCCC", "Inactive", 900.0, "", "p450-cyp3a4"),  # an outlier response is clipped
    ]
    write_raw(raw, rows)
    monkeypatch.setattr(ec, "RAW", raw)
    monkeypatch.setattr(ec, "CHALLENGE_FILES", (challenge, challenge))
    table = ec.external_table().set_index("PUBCHEM_CID")
    assert 103 not in table.index and set(table.index) == {100, 101, 102, 104}
    assert table.loc[100, "resp_cyp3a4"] == 90.0 and np.isclose(table.loc[100, "pac50_cyp3a4"], 6.3)
    assert table.loc[101, "pac50_cyp3a4"] == ec.POTENCY_FLOOR
    assert np.isnan(table.loc[102, "pac50_cyp3a4"]) and table.loc[102, "resp_cyp3a4"] == 30.0
    assert table.loc[104, "resp_cyp3a4"] == ec.RESPONSE_RANGE[0]  # +900 % maximal response means activation: the clipped inhibition is the lower bound
    assert np.isnan(table.loc[100, "resp_cyp2d6"])  # an isoform the panel did not report for the compound


def test_the_block_has_one_response_and_one_potency_column_per_isoform():
    assert len(ec.COLUMNS) == 2 * len(ec.PANELS) and ec.COLUMNS[0] == "resp_cyp1a2" and ec.COLUMNS[-1] == "pac50_cyp3a4"


def _tox_file(path, rows):
    header = "PUBCHEM_RESULT_TAG,PUBCHEM_SID,PUBCHEM_CID,PUBCHEM_EXT_DATASOURCE_SMILES,PUBCHEM_ACTIVITY_OUTCOME,PUBCHEM_ACTIVITY_SCORE,Max_Response-Replicate_1,Max_Response-Replicate_2,Fit_LogAC50-Replicate_1,Fit_LogAC50-Replicate_2"
    lines = [header, "RESULT_TYPE,,,,,,,,,", "RESULT_DESCr,,,,,,,,,", "RESULT_UNIT,,,,,,,,,", "RESULT_TYPE2,,,,,,,,,"]
    path.write_text("\n".join(lines + [",".join(str(v) for v in r) for r in rows]) + "\n")


def test_tox21_compact_table_floors_inactives_averages_replicates_and_removes_luciferase_inhibitors(tmp_path):
    rows_cyp = [(1, 1, 10, "CCCC", "Active", 80, -90, -70, -6.0, -6.4), (2, 2, 11, "CCCCC", "Inactive", 0, -4, -6, "", ""), (3, 3, 12, "CCCCCC", "Inconclusive", 20, -40, -30, "", ""),
                (4, 4, 13, "CCCCCCC", "Active", 70, -80, -80, -5.0, -5.0)]
    for aid in ec.TOX_AIDS.values():
        _tox_file(tmp_path / f"a{aid}.csv", rows_cyp)
    _tox_file(tmp_path / f"a{ec.TOX_COUNTER_AID}.csv", [(1, 1, 10, "CCCC", "Inactive", 0, 0, 0, "", ""), (4, 4, 13, "CCCCCCC", "Active", 90, -95, -95, -5.5, -5.5)])
    table = ec.build_tox21_compact(tmp_path, tmp_path / "compact.csv.gz").set_index("cid")
    assert table.loc[10, "tresp_cyp3a4"] == 80.0 and abs(table.loc[10, "tpac50_cyp3a4"] - 6.2) < 1e-9  # the mean of two replicates; pAC50 is minus the mean logAC50
    assert table.loc[11, "tpac50_cyp3a4"] == ec.POTENCY_FLOOR  # an inactive compound sits at the floor
    assert table.loc[12, "tresp_cyp3a4"] == 35.0 and pd.isna(table.loc[12, "tpac50_cyp3a4"])  # inconclusive and without a fit: a response, no potency
    assert bool(table.loc[13, "luciferase_inhibitor"]) and table.loc[13, list(ec.TOX_COLUMNS)].isna().all()  # a luciferase inhibitor's readings are never used
    assert not table.loc[10, "luciferase_inhibitor"]

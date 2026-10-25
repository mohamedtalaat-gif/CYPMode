# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import json
from unittest.mock import patch

import pandas as pd
import pytest

pytest.importorskip("Bio")  # batch.py imports pipeline.py, which needs Biopython

from cypmode.validation.batch import run_batch, to_dataframe

FAKE_SUMMARY = {
    "compound": "test",
    "motif_nitrogen_distances": [{"atom": "N32", "motif": "imidazole_Nsub", "distance_angstrom": 2.05}],
    "all_ligand_nitrogen_distances": [{"atom": "N32", "distance_angstrom": 2.05}],
    "closest_motif_nitrogen_distance_angstrom": 2.05,
    "coordinated": True,
    "affinity_pred_value": -0.96,
    "affinity_probability_binary": 0.86,
    "confidence_score": 0.92,
}


def test_run_batch_checkpoints_after_each_compound(tmp_path):
    results_path = tmp_path / "results.json"
    calls = []

    def fake_run(name, smiles, **kwargs):
        calls.append(name)
        return {"compound": name}

    with patch("cypmode.validation.batch.run_structural_validation", side_effect=fake_run):
        result = run_batch({"a": "CCO", "b": "CCN"}, results_path)

    assert result == {"a": {"compound": "a"}, "b": {"compound": "b"}}
    assert json.loads(results_path.read_text()) == result
    assert calls == ["a", "b"]


def test_run_batch_records_failure_without_stopping_the_batch(tmp_path):
    results_path = tmp_path / "results.json"

    def fake_run(name, smiles, **kwargs):
        if name == "bad":
            raise RuntimeError("boltz predict failed")
        return {"compound": name}

    with patch("cypmode.validation.batch.run_structural_validation", side_effect=fake_run):
        result = run_batch({"good": "CCO", "bad": "not-a-smiles"}, results_path)

    assert result["good"] == {"compound": "good"}
    assert "error" in result["bad"]


def test_run_batch_resumes_and_skips_already_done_compounds(tmp_path):
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps({"a": {"compound": "a"}}))
    calls = []

    def fake_run(name, smiles, **kwargs):
        calls.append(name)
        return {"compound": name}

    with patch("cypmode.validation.batch.run_structural_validation", side_effect=fake_run):
        result = run_batch({"a": "CCO", "b": "CCN"}, results_path)

    assert calls == ["b"]
    assert result["a"] == {"compound": "a"}
    assert result["b"] == {"compound": "b"}


def test_to_dataframe_flattens_a_successful_result():
    df = to_dataframe({"keto": FAKE_SUMMARY}, compounds={"keto": "CC(=O)N1..."})
    row = df.iloc[0]
    assert row["compound"] == "keto"
    assert row["smiles"] == "CC(=O)N1..."
    assert row["coordinated"] == True
    assert row["closest_motif_nitrogen_distance_angstrom"] == 2.05
    assert row["closest_motif"] == "imidazole_Nsub"
    assert row["closest_motif_atom"] == "N32"
    assert row["error"] is None


def test_to_dataframe_handles_a_failed_compound():
    df = to_dataframe({"bad": {"error": "boltz predict failed"}})
    row = df.iloc[0]
    assert row["compound"] == "bad"
    assert row["error"] == "boltz predict failed"
    assert row["coordinated"] is None


def test_to_dataframe_without_compounds_leaves_smiles_empty():
    df = to_dataframe({"keto": FAKE_SUMMARY})
    assert df.iloc[0]["smiles"] is None


def test_to_dataframe_handles_a_successful_result_with_no_affinity_keys():
    # exactly what summarize_compound(expect_affinity=False) produces --
    # a real batch run with predict_affinity=False crashed on this before
    # to_dataframe used .get() instead of indexing (KeyError: 'affinity_pred_value')
    no_affinity_summary = {k: v for k, v in FAKE_SUMMARY.items() if not k.startswith("affinity_")}
    df = to_dataframe({"keto": no_affinity_summary})
    row = df.iloc[0]
    assert row["coordinated"] == True
    assert row["affinity_pred_value"] is None
    assert row["affinity_probability_binary"] is None


def test_run_batch_survives_a_table_snapshot_write_failure(tmp_path):
    # Real regression: a batch launched without pyarrow/fastparquet
    # installed hit "ImportError: Unable to find a usable engine" on
    # df.to_parquet -- uncaught, it killed the whole remaining batch, not
    # just that one snapshot write, silently losing every compound still
    # queued after the one being processed when it happened. The JSON
    # checkpoint (the real result) must keep accumulating regardless.
    results_path = tmp_path / "results.json"
    table_path = tmp_path / "results"

    def fake_run(name, smiles, **kwargs):
        return {**FAKE_SUMMARY, "compound": name}

    with (
        patch("cypmode.validation.batch.run_structural_validation", side_effect=fake_run),
        patch("pandas.DataFrame.to_parquet", side_effect=ImportError("Unable to find a usable engine")),
    ):
        result = run_batch({"a": "CCO", "b": "CCN"}, results_path, table_path=table_path)

    assert result == {"a": {**FAKE_SUMMARY, "compound": "a"}, "b": {**FAKE_SUMMARY, "compound": "b"}}
    assert json.loads(results_path.read_text()) == result
    # the CSV write (no external engine dependency) still succeeded even
    # though parquet did not -- the failure is isolated to what actually failed
    assert pd.read_csv(table_path.with_suffix(".csv"))["compound"].tolist() == ["a", "b"]


def test_run_batch_writes_csv_and_parquet_when_table_path_given(tmp_path):
    results_path = tmp_path / "results.json"
    table_path = tmp_path / "results"

    with patch("cypmode.validation.batch.run_structural_validation", return_value=FAKE_SUMMARY):
        run_batch({"keto": "CC(=O)N1..."}, results_path, table_path=table_path)

    csv_df = pd.read_csv(table_path.with_suffix(".csv"))
    parquet_df = pd.read_parquet(table_path.with_suffix(".parquet"))
    assert csv_df.iloc[0]["compound"] == "keto"
    assert parquet_df.iloc[0]["compound"] == "keto"


def test_run_batch_skips_a_compound_whose_earlier_result_is_an_error_placeholder(tmp_path):
    """A name already present in the results file is skipped even when its entry is an error: the entry has to be deleted before a retry."""
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps({"a": {"error": "boom"}}))
    calls = []

    def fake_run(name, smiles, **kwargs):
        calls.append(name)
        return {"compound": name}

    with patch("cypmode.validation.batch.run_structural_validation", side_effect=fake_run):
        result = run_batch({"a": "CCO", "b": "CCN"}, results_path)

    assert calls == ["b"] and result["a"] == {"error": "boom"}
    kept = {k: v for k, v in json.loads(results_path.read_text()).items() if "error" not in v}
    results_path.write_text(json.dumps(kept))  # what scripts/clear_batch_errors.py does
    with patch("cypmode.validation.batch.run_structural_validation", side_effect=fake_run):
        run_batch({"a": "CCO", "b": "CCN"}, results_path)
    assert calls == ["b", "a"]

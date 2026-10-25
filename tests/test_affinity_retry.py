# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import json

import pytest

from cypmode.validation import affinity_retry as ar

SKIPPED = {"error": "no affinity_*.json in /somewhere/predictions/cyp3a4_A"}


def build(tmp_path):
    boltz = tmp_path / "out" / "cyp3a4"
    for name in ("A", "B", "C"):
        folder = boltz / f"boltz_results_cyp3a4_{name}" / "predictions" / f"cyp3a4_{name}"
        folder.mkdir(parents=True)
        (folder / f"cyp3a4_{name}_model_0.cif").write_text(f"structure {name}")
        (folder / f"confidence_cyp3a4_{name}_model_0.json").write_text('{"confidence_score": 0.9}')
    results = tmp_path / "structure_track_results.json"
    results.write_text(json.dumps({"A": SKIPPED, "B": {"error": "boltz predict failed"}, "C": {"affinity_probability_binary": 0.4}}))
    return {"results_path": results, "boltz": boltz, "digest_path": tmp_path / "final" / "digests.json", "backup": tmp_path / "final" / "before.json"}


def test_only_the_error_of_a_missing_affinity_file_counts_as_a_skipped_affinity_step(tmp_path):
    paths = build(tmp_path)
    assert ar.affinity_failures(json.loads(paths["results_path"].read_text())) == ["A"]
    assert ar.failed_rows(paths["results_path"]) == ["A", "B"]  # any error leaves a row without its summary
    assert ar.failed_rows(tmp_path / "missing.json") == []


def test_prepare_records_the_structure_digests_and_takes_the_entry_out_of_the_checkpoint(tmp_path):
    paths = build(tmp_path)
    names = ar.prepare(**paths)
    assert names == ["A"]
    assert set(json.loads(paths["results_path"].read_text())) == {"B", "C"}
    assert set(json.loads(paths["backup"].read_text())) == {"A", "B", "C"}  # the original checkpoint is kept
    recorded = json.loads(paths["digest_path"].read_text())
    assert set(recorded["A"]) == {"cyp3a4_A_model_0.cif", "confidence_cyp3a4_A_model_0.json"}


def test_a_second_prepare_changes_nothing(tmp_path):
    paths = build(tmp_path)
    ar.prepare(**paths)
    assert ar.prepare(**paths) == []
    assert set(json.loads(paths["backup"].read_text())) == {"A", "B", "C"}


def test_the_structures_are_reported_unchanged_until_a_file_changes(tmp_path):
    paths = build(tmp_path)
    ar.prepare(**paths)
    assert ar.unchanged(paths["digest_path"], paths["boltz"]) == {"A": True}
    (paths["boltz"] / "boltz_results_cyp3a4_A" / "predictions" / "cyp3a4_A" / "cyp3a4_A_model_0.cif").write_text("another draw")
    assert ar.unchanged(paths["digest_path"], paths["boltz"]) == {"A": False}


def test_prepare_refuses_a_run_without_a_structure(tmp_path):
    paths = build(tmp_path)
    for file in (paths["boltz"] / "boltz_results_cyp3a4_A" / "predictions" / "cyp3a4_A").iterdir():
        file.unlink()
    with pytest.raises(RuntimeError, match="no structure files"):
        ar.prepare(**paths)
    assert "A" in json.loads(paths["results_path"].read_text())  # the checkpoint is untouched


def test_the_batch_runner_passes_no_override_and_the_accelerator_to_the_pipeline(tmp_path):
    pytest.importorskip("Bio")
    import sys
    from unittest.mock import patch

    from cypmode.validation import batch

    seen = {}

    def fake_run_batch(compounds, results_path, **kwargs):
        seen.update(kwargs)
        return {}

    inputs = tmp_path / "inputs.csv"
    inputs.write_text("Molecule_Name,SMILES\nA,CCO\n")
    argv = ["batch", str(inputs), str(tmp_path / "results.json"), "--no-override", "--accelerator", "cpu"]
    with patch.object(sys, "argv", argv), patch.object(batch, "run_batch", fake_run_batch):
        batch._main()
    assert seen["override"] is False and seen["accelerator"] == "cpu"
    with patch.object(sys, "argv", argv[:3]), patch.object(batch, "run_batch", fake_run_batch):
        batch._main()
    assert seen["override"] is True and seen["accelerator"] == "gpu"  # the defaults are the ones of every earlier run

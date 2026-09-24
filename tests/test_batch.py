import json
from unittest.mock import patch

import pytest

pytest.importorskip("Bio")  # batch.py imports pipeline.py, which needs Biopython

from cypmode.validation.batch import run_batch


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

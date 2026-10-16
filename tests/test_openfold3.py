import json
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("yaml")

from cypmode.validation import openfold3

YAML_TEXT = """\
version: 1
sequences:
  - protein:
      id: A
      sequence: MKTAYIAKQR
  - ligand:
      id: B
      smiles: 'CC(=O)N1CCN(CC1)c1ccccc1'
  - ligand:
      id: C
      ccd: HEM
constraints:
  - bond:
      atom1: [A, 442, SG]
      atom2: [C, 1, FE]
"""


def _write_yaml(inputs_dir: Path, name: str = "cmpd1", target: str = "cyp3a4") -> None:
    inputs_dir.mkdir(parents=True, exist_ok=True)
    (inputs_dir / f"{target}_{name}.yaml").write_text(YAML_TEXT)


def _fake_prediction(out_root: Path, name: str, n_samples: int = 1, target: str = "cyp3a4") -> list[Path]:
    seed_dir = out_root / target / name / f"{target}_{name}" / "seed_123"
    seed_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for k in range(1, n_samples + 1):
        cif = seed_dir / f"{target}_{name}_seed_123_sample_{k}_model.cif"
        cif.write_text("data_x\n")
        (seed_dir / f"{target}_{name}_seed_123_sample_{k}_confidences_aggregated.json").write_text(
            json.dumps({"sample_ranking_score": 0.5 + k / 10})
        )
        paths.append(cif)
    return paths


def test_read_boltz_inputs_takes_protein_and_smiles_ligand_but_not_the_heme(tmp_path):
    _write_yaml(tmp_path)
    inputs = openfold3.read_boltz_inputs("cmpd1", inputs_dir=tmp_path)
    assert inputs == {"protein_sequence": "MKTAYIAKQR", "ligand_smiles": "CC(=O)N1CCN(CC1)c1ccccc1"}


def test_read_boltz_inputs_rejects_a_yaml_without_a_smiles_ligand(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "cyp3a4_bad.yaml").write_text("sequences:\n  - protein:\n      id: A\n      sequence: MK\n")
    with pytest.raises(ValueError, match="protein and one SMILES ligand"):
        openfold3.read_boltz_inputs("bad", inputs_dir=tmp_path)


def test_build_query_uses_the_same_a_b_c_chain_layout_as_boltz_with_msas_on():
    query = openfold3.build_query("cyp3a4_x", "MKT", "CCO")
    chains = query["queries"]["cyp3a4_x"]["chains"]
    assert [(c["molecule_type"], c["chain_ids"]) for c in chains] == [
        ("protein", ["A"]),
        ("ligand", ["B"]),
        ("ligand", ["C"]),
    ]
    assert chains[0]["sequence"] == "MKT"
    assert chains[1]["smiles"] == "CCO"
    assert chains[2]["ccd_codes"] == ["HEM"]
    assert query["queries"]["cyp3a4_x"]["use_msas"] is True  # without MSAs the model is useless (pLDDT 33)


def test_model_cif_paths_come_back_in_numeric_sample_order(tmp_path):
    made = _fake_prediction(tmp_path, "cmpd1", n_samples=11)
    found = openfold3.model_cif_paths("cmpd1", out_root=tmp_path)
    assert found == sorted(made, key=lambda p: int(p.stem.split("_sample_")[1].split("_")[0]))
    assert found[1].name.endswith("_sample_2_model.cif") and found[10].name.endswith("_sample_11_model.cif")


def test_ranking_scores_read_the_aggregated_confidences_and_tolerate_a_missing_file(tmp_path):
    cifs = _fake_prediction(tmp_path, "cmpd1", n_samples=2)
    cifs[1].with_name(cifs[1].name.replace("_model.cif", "_confidences_aggregated.json")).unlink()
    assert openfold3.ranking_scores(cifs) == [0.6, None]


def test_run_openfold3_refuses_to_overlap_with_another_prediction(tmp_path, monkeypatch):
    monkeypatch.setattr(openfold3, "_another_prediction_running", lambda: True)
    with pytest.raises(RuntimeError, match="must not overlap"):
        openfold3.run_openfold3("cmpd1", "MKT", "CCO", out_root=tmp_path)


def test_run_openfold3_builds_the_command_writes_the_query_and_returns_the_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(openfold3, "_another_prediction_running", lambda: False)
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["timeout"] = kwargs["timeout"]
        _fake_prediction(tmp_path, "cmpd1")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(openfold3.subprocess, "run", fake_run)
    summary = openfold3.run_openfold3("cmpd1", "MKT", "CCO", num_diffusion_samples=3, out_root=tmp_path, timeout_s=99)

    command = seen["command"]
    assert command[1] == "predict"
    assert "--use-msa-server=true" in command and "--use-templates=false" in command
    assert "--num-diffusion-samples=3" in command and "--num-model-seeds=1" in command
    assert seen["timeout"] == 99
    query = json.loads((tmp_path / "cyp3a4" / "cmpd1" / "query.json").read_text())
    assert "cyp3a4_cmpd1" in query["queries"]
    assert len(summary["model_cifs"]) == 1 and summary["ranking_scores"] == [0.6]
    assert summary["runtime_s"] >= 0


def test_run_openfold3_raises_with_the_stderr_tail_on_a_failed_process(tmp_path, monkeypatch):
    monkeypatch.setattr(openfold3, "_another_prediction_running", lambda: False)
    monkeypatch.setattr(
        openfold3.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="MPS out of memory")
    )
    with pytest.raises(RuntimeError, match="MPS out of memory"):
        openfold3.run_openfold3("cmpd1", "MKT", "CCO", out_root=tmp_path)


def test_run_openfold3_raises_when_the_process_succeeds_but_writes_no_structure(tmp_path, monkeypatch):
    monkeypatch.setattr(openfold3, "_another_prediction_running", lambda: False)
    monkeypatch.setattr(openfold3.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""))
    with pytest.raises(RuntimeError, match="wrote no"):
        openfold3.run_openfold3("cmpd1", "MKT", "CCO", out_root=tmp_path)


def test_batch_is_resumable_skips_successes_and_retries_recorded_errors(tmp_path, monkeypatch):
    inputs_dir, out_root = tmp_path / "inputs", tmp_path / "out"
    for name in ("a", "b"):
        _write_yaml(inputs_dir, name)

    calls = []

    def fake_run(name, protein, smiles, target, samples, root, timeout, memory):
        calls.append(name)
        if name == "b" and calls.count("b") == 1:
            raise RuntimeError("transient failure")
        cifs = _fake_prediction(root, name)
        return {"model_cifs": [str(c) for c in cifs], "ranking_scores": [0.6], "runtime_s": 1.0}

    monkeypatch.setattr(openfold3, "run_openfold3", fake_run)

    first = openfold3.run_openfold3_batch(["a", "b"], out_root=out_root, inputs_dir=inputs_dir)
    assert "error" not in first["a"] and first["b"] == {"error": "transient failure"}
    assert calls == ["a", "b"]

    second = openfold3.run_openfold3_batch(["a", "b"], out_root=out_root, inputs_dir=inputs_dir)
    assert calls == ["a", "b", "b"]  # a skipped (done, CIF on disk), b retried
    assert "error" not in second["b"]
    assert json.loads((out_root / "cyp3a4" / "results.json").read_text()).keys() == {"a", "b"}


def test_batch_reruns_a_compound_whose_recorded_cif_has_vanished(tmp_path, monkeypatch):
    inputs_dir, out_root = tmp_path / "inputs", tmp_path / "out"
    _write_yaml(inputs_dir, "a")
    calls = []

    def fake_run(name, protein, smiles, target, samples, root, timeout, memory):
        calls.append(name)
        cifs = _fake_prediction(root, name)
        return {"model_cifs": [str(c) for c in cifs], "ranking_scores": [None], "runtime_s": 1.0}

    monkeypatch.setattr(openfold3, "run_openfold3", fake_run)
    first = openfold3.run_openfold3_batch(["a"], out_root=out_root, inputs_dir=inputs_dir)
    for cif in first["a"]["model_cifs"]:
        Path(cif).unlink()

    openfold3.run_openfold3_batch(["a"], out_root=out_root, inputs_dir=inputs_dir)
    assert calls == ["a", "a"]


def _capture_command(monkeypatch, tmp_path, memory):
    seen = {}

    def fake_subprocess(command, **kwargs):
        seen["command"] = command
        _fake_prediction(tmp_path / "out", "a")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(openfold3.subprocess, "run", fake_subprocess)
    monkeypatch.setattr(openfold3, "_another_prediction_running", lambda: False)
    openfold3.run_openfold3("a", "MKT", "CCO", out_root=tmp_path / "out", memory=memory)
    return seen["command"]


def test_default_memory_mode_runs_the_openfold_command_with_the_runner_yaml(monkeypatch, tmp_path):
    command = _capture_command(monkeypatch, tmp_path, "config")
    assert command[0].endswith("run_openfold") and command[1] == "predict"
    assert f"--runner-yaml={openfold3.RUNNER_YAML}" in command


def test_lowmem_mode_runs_the_wrapper_script_with_the_same_arguments(monkeypatch, tmp_path):
    command = _capture_command(monkeypatch, tmp_path, "lowmem")
    assert command[0].endswith("python3") and command[1].endswith("run_openfold_lowmem.py") and command[2] == "predict"
    assert f"--runner-yaml={openfold3.RUNNER_YAML}" in command


def test_unknown_memory_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        openfold3.run_openfold3("a", "MKT", "CCO", out_root=tmp_path, memory="fast")


def test_a_silent_failure_reports_the_reason_from_openfold3s_own_log(tmp_path):
    log = tmp_path / "logs"
    log.mkdir()
    (log / "predict_err_rank0.log").write_text("Traceback\n  File x\nRuntimeError: MPS backend out of memory (MPS allocated: 7.42 GiB)\n")
    assert "MPS backend out of memory" in openfold3.failure_reason(tmp_path)
    assert openfold3.failure_reason(tmp_path / "missing") == "no error log"


def test_the_runner_yaml_only_lowers_the_chunk_size():
    import yaml

    settings = yaml.safe_load(openfold3.RUNNER_YAML.read_text())
    assert settings["model_update"]["presets"] == ["predict"]
    assert settings["model_update"]["custom"] == {"settings": {"memory": {"eval": {"chunk_size": 128}}}}


def test_memory_mode_comes_from_the_probe_file_and_defaults_to_the_runner_yaml(tmp_path):
    assert openfold3.default_memory_mode(tmp_path / "absent.txt") == "config"
    (tmp_path / "mode.txt").write_text("lowmem\n")
    assert openfold3.default_memory_mode(tmp_path / "mode.txt") == "lowmem"
    (tmp_path / "bad.txt").write_text("fast")
    with pytest.raises(ValueError):
        openfold3.default_memory_mode(tmp_path / "bad.txt")

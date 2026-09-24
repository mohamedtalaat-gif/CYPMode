"""Tests the orchestration logic in isolation -- mocks the boltz subprocess
call itself (a real run takes tens of minutes and needs a GPU + the
separate .venv-boltz environment; see README.md), so these only check that
this module wires the three real steps together correctly.
"""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("Bio")  # pipeline.py imports structures.py, which needs Biopython

from cypmode.validation.constants import CYP3A4_SEQUENCE
from cypmode.validation.pipeline import run_structural_validation

KETOCONAZOLE_SMILES = "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1"


def _mock_popen(returncode: int, lines: list[str]):
    """Builds a mock matching subprocess.Popen's context-manager protocol,
    with .stdout iterable over `lines` and .wait() returning `returncode`.
    """
    proc = MagicMock()
    proc.__enter__.return_value = proc
    proc.__exit__.return_value = False
    proc.stdout = iter(lines)
    proc.wait.return_value = returncode
    return proc


def test_writes_correct_yaml_and_streams_output_by_default(tmp_path, capsys):
    fake_result = {"compound": "cyp3a4_test", "coordinated": True}

    with (
        patch("cypmode.validation.pipeline.subprocess.Popen") as mock_popen,
        patch("cypmode.validation.pipeline.summarize_compound", return_value=fake_result) as mock_summarize,
    ):
        mock_popen.return_value = _mock_popen(0, ["Running structure prediction...\n", "done\n"])

        result = run_structural_validation(
            "test",
            KETOCONAZOLE_SMILES,
            protein_sequence=CYP3A4_SEQUENCE,
            input_dir=tmp_path,
            out_dir=tmp_path / "out",
        )

    yaml_path = tmp_path / "cyp3a4_test.yaml"
    assert yaml_path.exists()
    assert "atom1: [A, 442, SG]" in yaml_path.read_text()
    assert KETOCONAZOLE_SMILES in yaml_path.read_text()

    command = mock_popen.call_args[0][0]
    assert str(yaml_path) in command
    assert "--use_msa_server" in command
    assert "--override" in command  # default now: confirmed necessary, see pipeline.py's docstring

    assert "Running structure prediction" in capsys.readouterr().out  # actually streamed, not buffered

    mock_summarize.assert_called_once()
    assert result == fake_result


def test_raises_with_boltz_output_on_streaming_failure(tmp_path):
    with patch("cypmode.validation.pipeline.subprocess.Popen") as mock_popen:
        mock_popen.return_value = _mock_popen(1, ["some boltz output\n", "some boltz stderr\n"])

        with pytest.raises(RuntimeError, match="some boltz stderr"):
            run_structural_validation(
                "test",
                KETOCONAZOLE_SMILES,
                input_dir=tmp_path,
                out_dir=tmp_path / "out",
            )


def test_no_override_omits_the_flag(tmp_path):
    with (
        patch("cypmode.validation.pipeline.subprocess.Popen") as mock_popen,
        patch("cypmode.validation.pipeline.summarize_compound", return_value={}),
    ):
        mock_popen.return_value = _mock_popen(0, [])

        run_structural_validation(
            "test", KETOCONAZOLE_SMILES, input_dir=tmp_path, out_dir=tmp_path / "out", override=False
        )

    assert "--override" not in mock_popen.call_args[0][0]


def test_non_streaming_path_still_works(tmp_path):
    fake_result = {"compound": "cyp3a4_test"}

    with (
        patch("cypmode.validation.pipeline.subprocess.run") as mock_run,
        patch("cypmode.validation.pipeline.summarize_compound", return_value=fake_result),
    ):
        mock_run.return_value.returncode = 0

        result = run_structural_validation(
            "test",
            KETOCONAZOLE_SMILES,
            input_dir=tmp_path,
            out_dir=tmp_path / "out",
            stream_output=False,
        )

    assert result == fake_result


def test_non_streaming_raises_with_boltz_output_on_failure(tmp_path):
    with patch("cypmode.validation.pipeline.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = "some boltz stdout"
        mock_run.return_value.stderr = "some boltz stderr"

        with pytest.raises(RuntimeError, match="some boltz stderr"):
            run_structural_validation(
                "test",
                KETOCONAZOLE_SMILES,
                input_dir=tmp_path,
                out_dir=tmp_path / "out",
                stream_output=False,
            )

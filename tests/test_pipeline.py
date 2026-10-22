# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

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


def test_predict_affinity_false_omits_yaml_properties_and_passes_flag_through(tmp_path):
    with (
        patch("cypmode.validation.pipeline.subprocess.Popen") as mock_popen,
        patch("cypmode.validation.pipeline.summarize_compound", return_value={}) as mock_summarize,
    ):
        mock_popen.return_value = _mock_popen(0, [])

        run_structural_validation(
            "test",
            KETOCONAZOLE_SMILES,
            input_dir=tmp_path,
            out_dir=tmp_path / "out",
            predict_affinity=False,
        )

    yaml_path = tmp_path / "cyp3a4_test.yaml"
    assert "properties:" not in yaml_path.read_text()
    mock_summarize.assert_called_once()
    assert mock_summarize.call_args.kwargs.get("expect_affinity") is False


def test_recycling_and_msa_flags_pass_through_when_given(tmp_path):
    with (
        patch("cypmode.validation.pipeline.subprocess.Popen") as mock_popen,
        patch("cypmode.validation.pipeline.summarize_compound", return_value={}),
    ):
        mock_popen.return_value = _mock_popen(0, [])

        run_structural_validation(
            "test",
            KETOCONAZOLE_SMILES,
            input_dir=tmp_path,
            out_dir=tmp_path / "out",
            recycling_steps=1,
            max_msa_seqs=2048,
            num_subsampled_msa=256,
        )

    command = mock_popen.call_args[0][0]
    assert "--recycling_steps" in command and "1" in command
    assert "--max_msa_seqs" in command and "2048" in command
    assert "--num_subsampled_msa" in command and "256" in command


def test_diffusion_samples_flag_passes_through_when_given(tmp_path):
    # Real use: a compound whose default single sample fails PoseBusters
    # (a real steric clash, e.g. minimum_distance_to_protein) can be
    # re-run with multiple diffusion samples, then cypmode.validation.pose_validity.select_valid_model
    # picks the first one that actually passes -- instead of shipping a
    # known-bad pose, see submissions/structure_track/.
    with (
        patch("cypmode.validation.pipeline.subprocess.Popen") as mock_popen,
        patch("cypmode.validation.pipeline.summarize_compound", return_value={}),
    ):
        mock_popen.return_value = _mock_popen(0, [])

        run_structural_validation(
            "test", KETOCONAZOLE_SMILES, input_dir=tmp_path, out_dir=tmp_path / "out", diffusion_samples=5
        )

    command = mock_popen.call_args[0][0]
    assert "--diffusion_samples" in command and "5" in command


def test_diffusion_samples_omitted_by_default(tmp_path):
    with (
        patch("cypmode.validation.pipeline.subprocess.Popen") as mock_popen,
        patch("cypmode.validation.pipeline.summarize_compound", return_value={}),
    ):
        mock_popen.return_value = _mock_popen(0, [])

        run_structural_validation("test", KETOCONAZOLE_SMILES, input_dir=tmp_path, out_dir=tmp_path / "out")

    assert "--diffusion_samples" not in mock_popen.call_args[0][0]


def test_max_parallel_samples_caps_memory_and_is_omitted_by_default(tmp_path):
    # Five samples at once exhausted swap on a 16 GB Mac and kernel-panicked it (watchdog timeout);
    # one at a time keeps the peak at a single sample's footprint.
    commands = []
    for kwargs in ({"max_parallel_samples": 1}, {}):
        with (
            patch("cypmode.validation.pipeline.subprocess.Popen") as mock_popen,
            patch("cypmode.validation.pipeline.summarize_compound", return_value={}),
        ):
            mock_popen.return_value = _mock_popen(0, [])
            run_structural_validation(
                "test", KETOCONAZOLE_SMILES, input_dir=tmp_path, out_dir=tmp_path / "out", diffusion_samples=5, **kwargs
            )
        commands.append(mock_popen.call_args[0][0])

    capped, default = commands
    assert capped[capped.index("--max_parallel_samples") + 1] == "1"
    assert "--max_parallel_samples" not in default


def test_speed_flags_omitted_by_default(tmp_path):
    with (
        patch("cypmode.validation.pipeline.subprocess.Popen") as mock_popen,
        patch("cypmode.validation.pipeline.summarize_compound", return_value={}),
    ):
        mock_popen.return_value = _mock_popen(0, [])

        run_structural_validation("test", KETOCONAZOLE_SMILES, input_dir=tmp_path, out_dir=tmp_path / "out")

    command = mock_popen.call_args[0][0]
    assert "--recycling_steps" not in command
    assert "--max_msa_seqs" not in command
    assert "--num_subsampled_msa" not in command


def test_write_embeddings_flag_passes_through_when_true(tmp_path):
    with (
        patch("cypmode.validation.pipeline.subprocess.Popen") as mock_popen,
        patch("cypmode.validation.pipeline.summarize_compound", return_value={}) as mock_summarize,
    ):
        mock_popen.return_value = _mock_popen(0, [])

        run_structural_validation(
            "test", KETOCONAZOLE_SMILES, input_dir=tmp_path, out_dir=tmp_path / "out", write_embeddings=True
        )

    command = mock_popen.call_args[0][0]
    assert "--write_embeddings" in command
    assert mock_summarize.call_args.kwargs.get("expect_trunk_embedding") is True


def test_target_defaults_to_cyp3a4_naming(tmp_path):
    with (
        patch("cypmode.validation.pipeline.subprocess.Popen") as mock_popen,
        patch("cypmode.validation.pipeline.summarize_compound", return_value={}) as mock_summarize,
    ):
        mock_popen.return_value = _mock_popen(0, [])
        run_structural_validation("test", KETOCONAZOLE_SMILES, input_dir=tmp_path, out_dir=tmp_path / "out")

    assert (tmp_path / "cyp3a4_test.yaml").exists()
    result_dir = mock_summarize.call_args[0][0]
    assert result_dir.name == "boltz_results_cyp3a4_test"


def test_target_cyp2d6_uses_real_distinct_naming_not_hardcoded_cyp3a4(tmp_path):
    # Real regression: target used to be a hardcoded "cyp3a4" literal in
    # both the input YAML filename and the result directory lookup --
    # a CYP2D6 run would have silently written into (or read back from)
    # CYP3A4's own file/directory names instead of its own.
    with (
        patch("cypmode.validation.pipeline.subprocess.Popen") as mock_popen,
        patch("cypmode.validation.pipeline.summarize_compound", return_value={}) as mock_summarize,
    ):
        mock_popen.return_value = _mock_popen(0, [])
        run_structural_validation(
            "test",
            KETOCONAZOLE_SMILES,
            protein_sequence="MSEQUENCEPLACEHOLDERFXXGXXXCXGXX",
            target="cyp2d6",
            input_dir=tmp_path,
            out_dir=tmp_path / "out",
        )

    assert (tmp_path / "cyp2d6_test.yaml").exists()
    assert not (tmp_path / "cyp3a4_test.yaml").exists()
    result_dir = mock_summarize.call_args[0][0]
    assert result_dir.name == "boltz_results_cyp2d6_test"


def test_write_embeddings_flag_omitted_by_default(tmp_path):
    with (
        patch("cypmode.validation.pipeline.subprocess.Popen") as mock_popen,
        patch("cypmode.validation.pipeline.summarize_compound", return_value={}) as mock_summarize,
    ):
        mock_popen.return_value = _mock_popen(0, [])

        run_structural_validation("test", KETOCONAZOLE_SMILES, input_dir=tmp_path, out_dir=tmp_path / "out")

    command = mock_popen.call_args[0][0]
    assert "--write_embeddings" not in command
    assert mock_summarize.call_args.kwargs.get("expect_trunk_embedding") is False

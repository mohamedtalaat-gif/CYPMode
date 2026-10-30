# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

from pathlib import Path

from cypmode.validation import boltz_cpu_fp32, pipeline


class FakeTrainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def trainer(**kwargs):
    patched = type("PatchedTrainer", (FakeTrainer,), {})
    boltz_cpu_fp32.use_float32_on_cpu(patched)
    return patched(**kwargs).kwargs


def test_a_trainer_built_for_the_cpu_runs_in_float32():
    assert trainer(accelerator="cpu", precision="bf16-mixed")["precision"] == 32


def test_a_trainer_built_for_another_accelerator_keeps_the_precision_boltz_chose():
    assert trainer(accelerator="gpu", precision="bf16-mixed")["precision"] == "bf16-mixed"
    assert "precision" not in trainer(devices=1)


def test_the_other_arguments_of_the_trainer_pass_through_unchanged():
    assert trainer(accelerator="cpu", devices=1, strategy="auto") == {"accelerator": "cpu", "devices": 1, "strategy": "auto", "precision": 32}


def test_the_cpu_command_starts_the_wrapper_with_the_python_of_the_boltz_environment():
    binary = Path("/env/bin/boltz")
    assert pipeline.boltz_command(binary, "cpu") == [str(binary.with_name("python")), "-P", str(pipeline.CPU_WRAPPER)]
    assert pipeline.boltz_command(binary, "gpu") == [str(binary)]  # the runs on the GPU are started as before
    assert pipeline.CPU_WRAPPER.name == "boltz_cpu_fp32.py" and pipeline.CPU_WRAPPER.exists()

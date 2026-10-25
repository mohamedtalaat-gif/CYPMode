# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Run ``boltz predict`` in float32 when it runs on the CPU.

Boltz-2 builds the Lightning Trainer of every ``predict`` call with ``precision="bf16-mixed"`` (``third_party/boltz-main/src/boltz/main.py``).
On the CPU that makes PyTorch use its generic bfloat16 matrix product, which runs on one thread
(``at::native::cpublas::cpublas_gemm_impl`` held the main thread in a stack sample of the running process): the
affinity step of one ligand had used 2 h 29 min of a single core without finishing. In float32 the same product
goes to the multi-threaded BLAS.

The file is started by the Python of the Boltz environment, which cannot import this package, as
``python -P boltz_cpu_fp32.py predict ...`` (``-P`` keeps the folder of this file off the import path); the
arguments are the ones of ``boltz predict``. Only the Trainers built for the CPU change; any other accelerator
keeps the precision Boltz chose.
"""

import sys


def use_float32_on_cpu(trainer_class):
    """Make ``trainer_class`` build float32 Trainers whenever its accelerator is the CPU."""
    original = trainer_class.__init__

    def init(self, *args, **kwargs):
        if kwargs.get("accelerator") == "cpu":
            kwargs["precision"] = 32
        original(self, *args, **kwargs)

    trainer_class.__init__ = init
    return trainer_class


def main() -> None:
    import pytorch_lightning

    use_float32_on_cpu(pytorch_lightning.Trainer)
    from boltz.main import cli

    sys.argv[0] = "boltz"
    cli()


if __name__ == "__main__":
    main()

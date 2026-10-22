# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import subprocess
from pathlib import Path

import pytest

PYTHON = Path(".venv-openfold3/bin/python3")
SCRIPT = Path("scripts/run_openfold_lowmem.py")


@pytest.mark.skipif(not PYTHON.exists(), reason="the OpenFold3 environment is not installed")
def test_low_memory_chunking_matches_the_original_chunking_and_the_unchunked_layer():
    """Runs in the OpenFold3 environment (its own process): a real TriangleAttention layer, start and end, on CPU."""
    result = subprocess.run([str(PYTHON), str(SCRIPT), "--self-check"], capture_output=True, text=True, timeout=300, check=False)
    assert result.returncode == 0, result.stderr[-800:]
    assert "self-check OK" in result.stdout


SHARING_CHECK = """
import sys
import torch

n, heads = 20, 4
for samples in (1, 2, 3):
    bias = torch.randn(samples, 1, heads, n, n)
    flat = bias.expand(samples, n, heads, n, n).reshape(-1, heads, n, n)  # what chunk_layer does to the bias of triangle attention
    shared = flat.untyped_storage().data_ptr() == bias.untyped_storage().data_ptr()
    print(samples, shared, flat.untyped_storage().nbytes(), bias.untyped_storage().nbytes())
"""


@pytest.mark.skipif(not PYTHON.exists(), reason="the OpenFold3 environment is not installed")
def test_flattening_the_broadcast_bias_is_free_for_one_sample_and_a_full_copy_for_more():
    """The library of the program (PyTorch) in its own environment: with one sample the flattened bias is a view of the stored one, with two or more it is a copy of S x N times the size (11.9)."""
    result = subprocess.run([str(PYTHON), "-c", SHARING_CHECK], capture_output=True, text=True, timeout=300, check=False)
    assert result.returncode == 0, result.stderr[-800:]
    rows = [line.split() for line in result.stdout.strip().splitlines()]
    sizes = {int(samples): (shared == "True", int(flat), int(stored)) for samples, shared, flat, stored in rows}
    assert sizes[1] == (True, sizes[1][2], sizes[1][2])
    for samples in (2, 3):
        shared, flat, stored = sizes[samples]
        assert not shared and flat == stored * 20  # the stored bias times the 20 rows it is repeated for

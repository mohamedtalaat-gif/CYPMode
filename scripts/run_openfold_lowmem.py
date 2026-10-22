# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Run OpenFold3 with the memory-light chunking path (same arithmetic, much smaller peak memory).

OpenFold3's ``chunk_layer`` first expands every input to the full batch shape and flattens it, which materialises the whole
broadcast bias of a triangle-attention layer ([N, H, N, N] floats) before the first chunk is processed. On the 16 GB Mac this one copy
is 8-9 GiB and the Metal allocator refuses it ("MPS backend out of memory ... Tried to allocate 8.8 GiB"), whatever ``chunk_size`` is.
``chunk_layer`` already has the upstream option ``low_mem=True`` that keeps the inputs as broadcast views and slices every chunk
out of them; none of the call sites sets it. This script makes ``low_mem=True`` the default for every call site, then hands over to
the normal ``run_openfold`` command line. Chunking evaluates the same attention row by row, so predictions are the same model's
predictions; ``--self-check`` verifies that on a real TriangleAttention layer.

Usage:
    python scripts/run_openfold_lowmem.py predict --query-json=... --output-dir=...   (any run_openfold arguments)
    python scripts/run_openfold_lowmem.py --self-check
"""

import sys


def patch_chunk_layer() -> int:
    """Make low_mem=True the default of chunk_layer in every OpenFold3 module that imported it; returns the number of bindings changed."""
    from openfold3.core.utils import chunk_utils

    original = chunk_utils.chunk_layer
    if getattr(original, "_low_mem_default", False):
        return 0

    def chunk_layer(*args, **kwargs):
        if len(args) < 5:  # low_mem is the fifth positional parameter
            kwargs.setdefault("low_mem", True)
        return original(*args, **kwargs)

    chunk_layer._low_mem_default = True
    chunk_layer._original = original
    chunk_utils.chunk_layer = chunk_layer
    changed = 1
    for module in list(sys.modules.values()):
        if getattr(module, "chunk_layer", None) is original:
            module.chunk_layer = chunk_layer
            changed += 1
    return changed


def self_check() -> None:
    """The patched path gives the same output as the original path and as no chunking, and is in effect at every call site."""
    import torch
    from openfold3.core.model.layers.triangular_attention import TriangleAttention
    from openfold3.core.utils import chunk_utils

    original = chunk_utils.chunk_layer
    patch_chunk_layer()
    patched = chunk_utils.chunk_layer
    assert patched is not original and patched._original is original
    import openfold3.core.model.layers.triangular_attention as tri

    assert tri.chunk_layer is patched, "the triangle-attention call site was not patched"

    torch.manual_seed(0)
    n = 24
    for starting in (True, False):
        layer = TriangleAttention(c_in=16, c_hidden=8, no_heads=2, starting=starting).eval()
        x = torch.randn(1, n, n, 16)
        mask = (torch.rand(1, n, n) > 0.1).float()
        with torch.no_grad():
            direct = layer(x, mask=mask)
            via_patched = layer(x, mask=mask, chunk_size=5)
            tri.chunk_layer = original  # the original path: expand and flatten every input, then chunk
            via_original = layer(x, mask=mask, chunk_size=5)
            tri.chunk_layer = patched
        assert torch.allclose(via_patched, direct, atol=1e-5), "low-memory chunking differs from the unchunked layer"
        assert torch.allclose(via_patched, via_original, atol=1e-6), "low-memory chunking differs from the original chunking"
    print("self-check OK: low-memory chunking equals the original chunking and the unchunked layer; call sites patched")


def main() -> None:
    if "--self-check" in sys.argv[1:]:
        self_check()
        return
    patch_chunk_layer()
    from openfold3.run_openfold import cli

    sys.argv[0] = "run_openfold"
    sys.exit(cli())


if __name__ == "__main__":
    main()

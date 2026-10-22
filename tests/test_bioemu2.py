# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import pytest

from cypmode.validation import bioemu2

CONFIG = """\
score_model:
  sequence_embedder:
    trimul_variant: cueq
    opm_variant: vanilla
    tiling_cfg:
      tile_size: null
      chunk_size: null
      transition_chunk_size: null
      pwa_chunk_heads: false
  tiling_cfg:
    chunk_size: null
    transition_chunk_size: null
"""


def test_pure_torch_config_switches_the_kernel_and_chunks_the_layers():
    out = bioemu2.pure_torch_config(CONFIG, chunk_size=32)
    assert "trimul_variant: full" in out
    assert out.count("chunk_size: 32") == 2
    assert "transition_chunk_size: null" in out  # only the global chunk size is set, the per-layer sizes follow it
    assert "pwa_chunk_heads: true" in out
    assert "opm_variant: vanilla" in out


def test_pure_torch_config_refuses_an_unexpected_file():
    with pytest.raises(ValueError, match="trimul_variant"):
        bioemu2.pure_torch_config(CONFIG.replace("cueq", "full"))
    with pytest.raises(ValueError, match="chunk_size"):
        bioemu2.pure_torch_config(CONFIG + "  extra:\n    chunk_size: null\n")


def test_msa_a3m_takes_the_first_rows_with_the_query_first(tmp_path):
    path = tmp_path / "msa.csv"
    path.write_text("key,sequence\n-1,MKT\n0,MK-\n1,M-T\n")
    assert bioemu2.msa_a3m(2, path) == ">0\nMKT\n>1\nMK-\n"
    assert bioemu2.msa_a3m(10, path).count(">") == 3


def test_sample_cifs_are_listed_in_sample_order(tmp_path):
    out = tmp_path / "cyp3a4" / "bench-X"
    out.mkdir(parents=True)
    for i in (10, 2, 1):
        (out / f"sample_{i}.cif").write_text("data_x\n")
    (out / "timings.json").write_text("{}")
    names = [p.name for p in bioemu2.sample_cifs("bench-X", out_root=tmp_path)]
    assert names == ["sample_1.cif", "sample_2.cif", "sample_10.cif"]
    assert bioemu2.sample_cifs("missing", out_root=tmp_path) == []

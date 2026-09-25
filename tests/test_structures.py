import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("Bio")

from cypmode.validation.structures import load_trunk_embeddings, pooled_trunk_embedding, summarize_compound

KETOCONAZOLE_SMILES = "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1"
VARDENAFIL_SMILES = "CCCc1nc(C)c2c(=O)nc(-c3cc(S(=O)(=O)N4CCN(CC)CC4)ccc3OCC)[nH]n12"
OCNT_2395436_SMILES = "CC(C)(C)NS(=O)(=O)C1(CNC2=C(Br)C=NC=C2C#N)CCC1"

RESULT_DIR = Path("data/boltz_test/out/boltz_results_cyp3a4_ketoconazole")
CIF_PATH = RESULT_DIR / "predictions" / "cyp3a4_ketoconazole" / "cyp3a4_ketoconazole_model_0.cif"

pytestmark = pytest.mark.skipif(
    not CIF_PATH.exists(),
    reason="requires a local Boltz-2 run; see README.md's Validation section",
)


def test_summarize_ketoconazole_coordinates():
    # Real, known-correct answer: the imidazole nitrogen (CIF atom N32) sits
    # 2.05 A from the heme iron -- confirmed independently by
    # github.com/PatWalters/cyp-heatmap's ProLIF analysis of the same
    # compound (see README.md's Independent cross-check section).
    result = summarize_compound(RESULT_DIR, KETOCONAZOLE_SMILES)
    assert result["compound"] == "cyp3a4_ketoconazole"
    assert result["coordinated"] is True
    assert result["closest_motif_nitrogen_distance_angstrom"] == pytest.approx(2.05, abs=0.01)
    assert result["motif_nitrogen_distances"][0]["atom"] == "N32"
    assert result["motif_nitrogen_distances"][0]["motif"] == "imidazole_Nsub"
    assert "affinity_pred_value" in result


def test_motif_distance_ignores_closer_non_motif_nitrogen():
    # The regression this module exists to prevent: vardenafil's
    # geometrically closest ligand nitrogen (N50, ~4.84 A) is not part of
    # any matched Type II motif. The motif-restricted distance must not
    # silently fall back to it.
    result_dir = Path("data/boltz_test/out/boltz_results_cyp3a4_vardenafil")
    if not (result_dir / "predictions" / "cyp3a4_vardenafil" / "cyp3a4_vardenafil_model_0.cif").exists():
        pytest.skip("requires a local Boltz-2 run for vardenafil")

    result = summarize_compound(result_dir, VARDENAFIL_SMILES)
    motif_atoms = {d["atom"] for d in result["motif_nitrogen_distances"]}
    assert "N50" not in motif_atoms
    assert result["closest_motif_nitrogen_distance_angstrom"] > 4.84
    assert result["coordinated"] is False

    all_atoms = {d["atom"] for d in result["all_ligand_nitrogen_distances"]}
    assert "N50" in all_atoms  # still visible for manual review, just not driving the call


OCNT_2395436_RESULT_DIR = Path("data/boltz_test/out/boltz_results_cyp3a4_OCNT-2395436")


@pytest.mark.skipif(
    not OCNT_2395436_RESULT_DIR.exists(),
    reason="requires a local Boltz-2 run for OCNT-2395436",
)
def test_summarize_compound_expect_affinity_false_skips_affinity_gracefully(tmp_path):
    # Real completed result dir, copied so the affinity JSON can be removed
    # without touching the actual data -- simulates a predict_affinity=False
    # run (which never produces that file) using real structure output.
    copy_dir = tmp_path / "boltz_results_cyp3a4_OCNT-2395436"
    shutil.copytree(OCNT_2395436_RESULT_DIR, copy_dir)
    predictions_dir = copy_dir / "predictions" / "cyp3a4_OCNT-2395436"
    for affinity_file in predictions_dir.glob("affinity_*.json"):
        affinity_file.unlink()

    result = summarize_compound(copy_dir, OCNT_2395436_SMILES, expect_affinity=False)
    assert "affinity_pred_value" not in result
    assert "affinity_probability_binary" not in result
    assert "confidence_score" in result  # unaffected, still present

    with pytest.raises(FileNotFoundError):
        summarize_compound(copy_dir, OCNT_2395436_SMILES, expect_affinity=True)


def test_pooled_trunk_embedding_gives_a_fixed_size_vector_regardless_of_token_count():
    # s: (n_tokens, d_s), z: (n_tokens, n_tokens, d_z) -- different token
    # counts for two synthetic complexes must still produce comparable,
    # same-length pooled vectors (d_s + d_z each), since that's the whole
    # point of pooling before using this as an ML feature across compounds.
    small = {"s": np.random.default_rng(0).normal(size=(50, 8)), "z": np.random.default_rng(0).normal(size=(50, 50, 4))}
    large = {"s": np.random.default_rng(1).normal(size=(300, 8)), "z": np.random.default_rng(1).normal(size=(300, 300, 4))}
    pooled_small = pooled_trunk_embedding(small)
    pooled_large = pooled_trunk_embedding(large)
    assert pooled_small.shape == (12,)  # 8 (s) + 4 (z)
    assert pooled_large.shape == pooled_small.shape


def test_load_trunk_embeddings_reads_real_npz_shape(tmp_path):
    s = np.random.default_rng(0).normal(size=(20, 8)).astype(np.float32)
    z = np.random.default_rng(0).normal(size=(20, 20, 4)).astype(np.float32)
    np.savez_compressed(tmp_path / "embeddings_test.npz", s=s, z=z)

    embeddings = load_trunk_embeddings(tmp_path)
    assert embeddings["s"].shape == (20, 8)
    assert embeddings["z"].shape == (20, 20, 4)


def test_load_trunk_embeddings_raises_clearly_when_absent(tmp_path):
    with pytest.raises(FileNotFoundError, match="write_embeddings"):
        load_trunk_embeddings(tmp_path)


@pytest.mark.skipif(
    not CIF_PATH.exists(),
    reason="requires a local Boltz-2 run; see README.md's Validation section",
)
def test_summarize_compound_expect_trunk_embedding_true_includes_pooled_vector(tmp_path):
    # Real completed structure output, with a synthetic embeddings_*.npz
    # added -- no real run has used --write_embeddings yet (see
    # module docstring for why the flag+file format are trusted regardless:
    # confirmed directly in third_party/boltz-main's own source).
    copy_dir = tmp_path / "boltz_results_cyp3a4_ketoconazole"
    shutil.copytree(RESULT_DIR, copy_dir)
    predictions_dir = copy_dir / "predictions" / "cyp3a4_ketoconazole"
    s = np.random.default_rng(0).normal(size=(100, 8)).astype(np.float32)
    z = np.random.default_rng(0).normal(size=(100, 100, 4)).astype(np.float32)
    np.savez_compressed(predictions_dir / "embeddings_cyp3a4_ketoconazole.npz", s=s, z=z)

    result = summarize_compound(copy_dir, KETOCONAZOLE_SMILES, expect_trunk_embedding=True)
    assert "pooled_trunk_embedding" in result
    assert isinstance(result["pooled_trunk_embedding"], list)  # JSON-serializable, not a raw ndarray
    assert len(result["pooled_trunk_embedding"]) == 12

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.torch

pytest.importorskip("torch")
pytest.importorskip("transformers")

from cypmode.challenge.chemberta_features import (
    CHEMBERTA_EMBEDDING_DIM,
    _load_model,
    add_chemberta_features,
    compute_chemberta_embedding,
    compute_chemberta_embeddings_batch,
)

KETOCONAZOLE = "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1"
ETHANOL = "CCO"
BENZENE = "c1ccccc1"


@pytest.fixture(scope="module")
def loaded_model():
    return _load_model()


def test_compute_chemberta_embedding_is_a_real_finite_vector(loaded_model):
    tokenizer, model, torch_module = loaded_model
    embedding = compute_chemberta_embedding(KETOCONAZOLE, tokenizer, model, torch_module)
    assert embedding.shape == (CHEMBERTA_EMBEDDING_DIM,)
    assert np.isfinite(embedding).all()
    assert embedding.sum() != 0  # a real model shouldn't collapse to an all-zero vector


def test_compute_chemberta_embedding_distinguishes_two_different_real_molecules(loaded_model):
    tokenizer, model, torch_module = loaded_model
    ethanol_emb = compute_chemberta_embedding(ETHANOL, tokenizer, model, torch_module)
    benzene_emb = compute_chemberta_embedding(BENZENE, tokenizer, model, torch_module)
    assert not np.allclose(ethanol_emb, benzene_emb)


def test_compute_chemberta_embeddings_batch_caches_to_disk_and_resumes(tmp_path):
    cache_path = tmp_path / "cache.csv"
    first = compute_chemberta_embeddings_batch([ETHANOL, BENZENE], cache_path=cache_path)
    assert len(first) == 2
    assert cache_path.exists()
    assert len(pd.read_csv(cache_path)) == 2

    second = compute_chemberta_embeddings_batch([ETHANOL, "C"], cache_path=cache_path)
    assert len(second) == 2
    assert len(pd.read_csv(cache_path)) == 3  # ethanol, benzene, methane


def test_add_chemberta_features_adds_one_fixed_size_array_per_row(tmp_path):
    df = pd.DataFrame({"SMILES": [ETHANOL, BENZENE]})
    out = add_chemberta_features(df, cache_path=tmp_path / "cache.csv")
    assert "chemberta_embedding" in out.columns
    assert all(len(e) == CHEMBERTA_EMBEDDING_DIM for e in out["chemberta_embedding"])

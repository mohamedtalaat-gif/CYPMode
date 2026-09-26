import numpy as np
import pandas as pd
import pytest

from cypmode.challenge.chemberta_features import CHEMBERTA_EMBEDDING_DIM
from cypmode.challenge.chemberta_significance import load_cached_chemberta_embeddings

ETHANOL = "CCO"
BENZENE = "c1ccccc1"


def _fake_cache(path, smiles_list, rng):
    dim_cols = [f"dim_{i}" for i in range(CHEMBERTA_EMBEDDING_DIM)]
    rows = [{"SMILES": s, **dict(zip(dim_cols, rng.normal(size=CHEMBERTA_EMBEDDING_DIM), strict=True))} for s in smiles_list]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_cached_chemberta_embeddings_raises_when_cache_missing(tmp_path):
    df = pd.DataFrame({"SMILES": [ETHANOL]})
    with pytest.raises(FileNotFoundError, match="run `python -m cypmode.challenge.chemberta_features`"):
        load_cached_chemberta_embeddings(df, cache_path=tmp_path / "does_not_exist.csv")


def test_load_cached_chemberta_embeddings_raises_when_a_compound_is_not_cached(tmp_path):
    cache_path = tmp_path / "cache.csv"
    _fake_cache(cache_path, [ETHANOL], np.random.default_rng(0))
    df = pd.DataFrame({"SMILES": [ETHANOL, BENZENE]})  # BENZENE not in the cache
    with pytest.raises(ValueError, match="no cached ChemBERTa embedding"):
        load_cached_chemberta_embeddings(df, cache_path=cache_path)


def test_load_cached_chemberta_embeddings_joins_correctly(tmp_path):
    cache_path = tmp_path / "cache.csv"
    _fake_cache(cache_path, [ETHANOL, BENZENE], np.random.default_rng(0))
    df = pd.DataFrame({"SMILES": [ETHANOL, BENZENE]})
    out = load_cached_chemberta_embeddings(df, cache_path=cache_path)
    assert "chemberta_embedding" in out.columns
    assert all(len(e) == CHEMBERTA_EMBEDDING_DIM for e in out["chemberta_embedding"])

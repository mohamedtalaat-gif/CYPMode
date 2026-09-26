"""Transformer-based SMILES embeddings via ChemBERTa
(seyonec/ChemBERTa-zinc-base-v1, a real, published RoBERTa model
pretrained by masked-language-modeling on ~77M ZINC SMILES strings, via
HuggingFace transformers) -- a third molecular representation alongside
M3's existing two (RDKit descriptors, Morgan fingerprint), the real
extension the MTMM-CYP paper's own multi-representation multi-task
architecture pointed at (see CYPMode_Architecture_Specification.md).

Confirmed directly, real numbers: a 768-dim mean-pooled embedding per
compound, ~35 ms/molecule on CPU (only the one-time ~178s model
download/load is slow) -- ~3.6 minutes for the full ~6,145-compound
training set, no 3D conformer step needed at all (ChemBERTa operates
directly on the tokenized SMILES string, unlike quantum_pharmacophore's
tblite features or pharmacophore_fingerprint's pmapper features, both
of which need a real 3D embedding first) -- so there's essentially no
embedding-failure mode the way those two have; a genuinely unparseable
SMILES is the only real failure case.

CRITICAL: this module must NEVER be imported in the same process as
LightGBM. Confirmed directly, freshly, this session: `import torch`
followed by fitting an LGBMClassifier in the same process crashes with
a real SIGSEGV (exit code 139) -- the same OpenMP-runtime conflict
already documented for cypmode.challenge.multitask (see pyproject.toml's
`torch` marker), now confirmed to also apply here since this module
also needs torch. This module's own `_main()` therefore ONLY extracts
and caches embeddings -- it does not import lightgbm, sklearn, or
statsmodels, and does not run any significance test itself. Run
`cypmode.challenge.chemberta_significance` as a completely separate
process afterward (matching how tests/test_multitask.py must already be
run as its own separate pytest invocation, never mixed with the rest of
the suite) to evaluate the cached embeddings.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CHEMBERTA_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
CHEMBERTA_EMBEDDING_DIM = 768
CACHE_PATH = Path("data/cyp_challenge/tdi_track/chemberta_embedding_cache.csv")


def _load_model():
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL_NAME)
    model = AutoModel.from_pretrained(CHEMBERTA_MODEL_NAME)
    model.eval()
    return tokenizer, model, torch


def compute_chemberta_embedding(smiles: str, tokenizer=None, model=None, torch_module=None) -> np.ndarray:
    """One compound's real 768-dim ChemBERTa embedding (mean-pooled over
    token hidden states -- a standard, simple pooling choice; no [CLS]
    token special-casing needed since RoBERTa-style models don't reserve
    a dedicated pooler for this checkpoint). Loads the model fresh if
    tokenizer/model/torch_module aren't passed in -- pass them explicitly
    when embedding many compounds in a loop, so the ~178s one-time model
    load only happens once (see compute_chemberta_embeddings_batch).

    Returns a NaN-filled vector (not a crash) if tokenization/inference
    fails -- rare for ChemBERTa (no 3D step to fail), but a batch over
    several thousand real compounds should not die on one bad string.
    """
    if tokenizer is None or model is None or torch_module is None:
        tokenizer, model, torch_module = _load_model()

    try:
        inputs = tokenizer(smiles, return_tensors="pt")
        with torch_module.no_grad():
            outputs = model(**inputs)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return pooled.squeeze(0).numpy().astype(np.float64)
    except Exception as exc:
        logger.warning("ChemBERTa embedding failed for %s: %s", smiles, exc)
        return np.full(CHEMBERTA_EMBEDDING_DIM, np.nan)


def compute_chemberta_embeddings_batch(
    smiles_list: list[str], cache_path: str | Path = CACHE_PATH, checkpoint_every: int = 200
) -> pd.DataFrame:
    """Computes (or loads from cache_path) one embedding row per SMILES,
    one column per dimension (`dim_0`..`dim_767`). Resumable and
    checkpointed the same way quantum_pharmacophore's and
    pharmacophore_fingerprint's batch runners are. No multiprocessing
    here (unlike those two): at ~35 ms/molecule this is already fast
    enough single-threaded (~3.6 min for the full training set) that a
    Pool's process-startup overhead (each worker would reload the model)
    isn't worth it.
    """
    dim_cols = [f"dim_{i}" for i in range(CHEMBERTA_EMBEDDING_DIM)]
    all_cols = ["SMILES", *dim_cols]
    cache_path = Path(cache_path)
    if cache_path.exists():
        cached_smiles = set(pd.read_csv(cache_path, usecols=["SMILES"])["SMILES"])
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=all_cols).to_csv(cache_path, index=False)
        cached_smiles = set()

    todo = [s for s in smiles_list if s not in cached_smiles]
    if todo:
        tokenizer, model, torch_module = _load_model()
        pending_rows = []
        for smiles in todo:
            embedding = compute_chemberta_embedding(smiles, tokenizer, model, torch_module)
            pending_rows.append({"SMILES": smiles, **dict(zip(dim_cols, embedding, strict=True))})
            if len(pending_rows) >= checkpoint_every:
                pd.DataFrame(pending_rows, columns=all_cols).to_csv(cache_path, mode="a", header=False, index=False)
                pending_rows = []
        if pending_rows:
            pd.DataFrame(pending_rows, columns=all_cols).to_csv(cache_path, mode="a", header=False, index=False)

    cached = pd.read_csv(cache_path)
    lookup = cached.set_index("SMILES")
    return lookup.loc[smiles_list].reset_index()


def add_chemberta_features(df: pd.DataFrame, cache_path: str | Path = CACHE_PATH) -> pd.DataFrame:
    """Adds a `chemberta_embedding` column (one fixed-size np.array per
    row) to df, keyed off its own SMILES column.
    """
    dim_cols = [f"dim_{i}" for i in range(CHEMBERTA_EMBEDDING_DIM)]
    embeddings_df = compute_chemberta_embeddings_batch(list(df["SMILES"]), cache_path=cache_path)
    out = df.copy()
    out["chemberta_embedding"] = [row[dim_cols].to_numpy(dtype=np.float64) for _, row in embeddings_df.iterrows()]
    return out


def _main() -> None:
    """Extraction only -- see module docstring for why this deliberately
    does not also run a significance test in the same process. Run
    `python -m cypmode.challenge.chemberta_significance` afterward, as a
    separate invocation, to evaluate the cached embeddings.
    """
    from cypmode.challenge.tdi_baseline import load_tdi_training_data

    tdi_df = load_tdi_training_data()
    print(f"computing ChemBERTa embeddings for {len(tdi_df)} compounds...")
    tdi_df = add_chemberta_features(tdi_df)

    n_failed = tdi_df["chemberta_embedding"].apply(lambda e: np.isnan(e).any()).sum()
    print(f"{n_failed}/{len(tdi_df)} compounds had a failed embedding (NaN)")
    print(f"cache written to {CACHE_PATH}")


if __name__ == "__main__":
    _main()

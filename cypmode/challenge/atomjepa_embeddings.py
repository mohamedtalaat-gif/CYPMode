"""Frozen Atom-JEPA embeddings as a label-free feature block.

Atom-JEPA (Hordahl Jordahn and co-workers; code MIT, ``github.com/khelverskovp/atom-jepa``, weights CC BY 4.0, ``huggingface.co/atom-jepa/atom-jepa``) is an equivariant graph network
(EquiformerV3 encoder: 8 blocks, 256 channels, 39.0 million parameters) pretrained on about 19 million 3D conformers of molecules by predicting the representation of one part of a molecule from another.
The authors fine-tune the whole network; a measurement on this machine (2026-10-16, four CPU threads while the queue ran) put one optimiser step at 7.4 s per molecule on the CPU, and the Apple GPU ran out
of memory at 16 molecules per batch, so one fine-tuning of the 3,500 training molecules of a task would take about three days and the nested design needs about 80 of them per isoform. What is
affordable is one forward pass per molecule, about 0.5 s on the Apple GPU (6,895 molecules in under an hour; CPU and GPU embeddings agree to 2e-4).

Every block of the encoder is a different level of description, and the features of a frozen network are usually best in the middle, so every one of the 8 blocks is kept: for every molecule the
mean over its atoms of the invariant (degree 0) channels of each block, 8 x 256 numbers, written to ``caches/atomjepa_embeddings.npz`` keyed by canonical SMILES. The block ``atomjepa`` of the
feature store is the label-free principal-component reduction of the eight 256-vectors (64 components), as for the Monroe and CheMeleon blocks.

The same forward pass also keeps, for blocks 4, 6 and 8, the features of every atom (float16, with the atomic numbers) in ``caches/atomjepa_atoms.npz``. A mean over atoms throws away which atom
carries what; a small readout (attention over the atoms) trained on the cached per-atom features costs seconds per fit instead of days, and is the affordable form of "fine-tune the top of the network".

The extraction runs under ``.venv-atomjepa`` (torch, e3nn, torch-geometric; no LightGBM) in chunks that are saved as they finish, so an interrupted run resumes; the evaluation under ``.venv`` only reads
the cache.

  extract:   PYTHONPATH=. .venv-atomjepa/bin/python -m cypmode.challenge.atomjepa_embeddings --extract [--device mps]
  evaluate:  .venv/bin/python -m cypmode.challenge.aux_experiments --experiment atomjepa --isoform CYP3A4 --repeats 3 --run
GATE (addition_experiment): corrected p < 0.05 at both z0, alone (added to feature sets A and A+fps) and inside the assembled stack, per isoform.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CACHE = Path("data/tdi_track/outputs/caches/atomjepa_embeddings.npz")
CHUNKS = Path("data/tdi_track/outputs/caches/atomjepa_chunks")
N_COMPONENTS = 64
TRAIN_CSV = "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
TEST_CSV = "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv"
CHUNK = 128


ATOM_LAYERS = (3, 5, 7)  # blocks 4, 6 and 8 (0-based): the per-atom features that are kept for a learned readout
ATOMS_CACHE = Path("data/tdi_track/outputs/caches/atomjepa_atoms.npz")


def extract(smiles: list[str], device: str = "cpu", batch_size: int = 4, cache: Path = CACHE, chunks: Path = CHUNKS, atoms_cache: Path = ATOMS_CACHE) -> None:
    """One forward pass per molecule gives both products: the mean over atoms of the invariant channels of all 8 blocks (``cache``) and, for blocks 4, 6 and 8, the features of every atom with
    the atomic numbers (``atoms_cache``, float16), so that a small learned readout can later be trained on cached per-atom features without running the encoder again."""
    import torch
    from atom_jepa import AtomJEPA
    from rdkit import Chem

    keys = list(dict.fromkeys(Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in smiles))
    chunks.mkdir(parents=True, exist_ok=True)
    model = AtomJEPA.from_pretrained("molecules").to(device).eval()
    width = model.embedding_dim

    def run_batch(group: list[str]):
        batch = model.collate(group)
        with torch.no_grad():
            out = model(batch, layers="all", degrees=0, per_atom=True).cpu()  # [atoms, 8, width]
        index = batch["node_graph_index"].cpu()
        pooled = torch.stack([out[index == g].mean(0) for g in range(len(group))]).numpy()
        per_atom = [out[index == g][:, list(ATOM_LAYERS)].numpy().astype(np.float16) for g in range(len(group))]
        numbers = [batch["atomic_numbers"].cpu()[index == g].numpy().astype(np.int16) for g in range(len(group))]
        return pooled, per_atom, numbers

    for start in range(0, len(keys), CHUNK):
        path = chunks / f"chunk_{start:06d}.npz"
        if path.exists():
            continue
        part = keys[start : start + CHUNK]
        values = np.full((len(part), 8, width), np.nan, dtype=np.float32)
        atoms: list[np.ndarray] = [np.zeros((0, len(ATOM_LAYERS), width), dtype=np.float16)] * len(part)
        numbers: list[np.ndarray] = [np.zeros(0, dtype=np.int16)] * len(part)
        failed = []
        for i in range(0, len(part), batch_size):
            group = part[i : i + batch_size]
            try:
                pooled, per_atom, nums = run_batch(group)
                values[i : i + len(group)], atoms[i : i + len(group)], numbers[i : i + len(group)] = pooled, per_atom, nums
            except Exception:
                for j, smi in enumerate(group):
                    try:
                        pooled, per_atom, nums = run_batch([smi])
                        values[i + j], atoms[i + j], numbers[i + j] = pooled[0], per_atom[0], nums[0]
                    except Exception:
                        failed.append(smi)
        offsets = np.concatenate([[0], np.cumsum([len(a) for a in atoms])]).astype(np.int64)
        np.savez_compressed(path, smiles=np.array(part, dtype=object), values=values, failed=np.array(failed, dtype=object), atoms=np.concatenate(atoms), numbers=np.concatenate(numbers), offsets=offsets)
        print(f"{min(start + CHUNK, len(keys))} of {len(keys)} molecules ({len(failed)} failed in this chunk)", flush=True)
    parts = [np.load(p, allow_pickle=True) for p in sorted(chunks.glob("chunk_*.npz"))]
    np.savez_compressed(cache, smiles=np.concatenate([p["smiles"] for p in parts]), values=np.concatenate([p["values"] for p in parts]), failed=np.concatenate([p["failed"] for p in parts]))
    shifts = np.cumsum([0] + [int(p["offsets"][-1]) for p in parts])[:-1]
    np.savez_compressed(atoms_cache, smiles=np.concatenate([p["smiles"] for p in parts]), atoms=np.concatenate([p["atoms"] for p in parts]), numbers=np.concatenate([p["numbers"] for p in parts]),
                        offsets=np.concatenate([p["offsets"][:-1] + s for p, s in zip(parts, shifts, strict=True)] + [[sum(int(p["offsets"][-1]) for p in parts)]]), layers=np.array(ATOM_LAYERS))
    print(f"wrote {cache} and {atoms_cache}: {sum(len(p['smiles']) for p in parts)} molecules, {sum(len(p['failed']) for p in parts)} without an embedding")


def load_atomjepa_pca(smiles: list[str], cache: Path = CACHE, n_components: int = N_COMPONENTS) -> np.ndarray:
    """Label-free PCA of the eight 256-vectors of ALL cached compounds (rows follow ``smiles``); a compound without an embedding gets the mean (zero after centring). Never imports torch."""
    from rdkit import Chem
    from sklearn.decomposition import PCA

    if not Path(cache).exists():
        raise FileNotFoundError(f"{cache} is missing: run `PYTHONPATH=. .venv-atomjepa/bin/python -m cypmode.challenge.atomjepa_embeddings --extract` first")
    z = np.load(cache, allow_pickle=True)
    index = {k: i for i, k in enumerate(z["smiles"])}
    flat = z["values"].reshape(len(z["smiles"]), -1).astype(np.float64)
    flat = np.where(np.isnan(flat), np.nanmean(flat, axis=0, keepdims=True), flat)
    reduced = PCA(n_components=n_components, svd_solver="full", random_state=0).fit_transform((flat - flat.mean(axis=0)) / (flat.std(axis=0) + 1e-6))
    wanted = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in smiles]
    missing = [w for w in wanted if w not in index]
    if missing:
        raise ValueError(f"{len(missing)} compounds have no Atom-JEPA embedding, e.g. {missing[0]!r}")
    return reduced[[index[w] for w in wanted]]


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.extract:
        extract(list(pd.read_csv(TRAIN_CSV).SMILES) + list(pd.read_csv(TEST_CSV).SMILES), args.device, args.batch_size)


if __name__ == "__main__":
    main()

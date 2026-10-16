"""Frozen CheMeleon foundation embeddings as a feature block (checklist item I7, job 050).

CheMeleon (Burns et al., Zenodo 10.5281/zenodo.15426600, weights already on disk at ~/.chemprop/chemeleon_mp.pt) is a
Chemprop directed message-passing encoder pretrained on a large set of computed molecular descriptors. Fine-tuning it
on our few thousand labels was judged infeasible earlier; here it is used FROZEN: each molecule becomes the mean-
aggregated 2048-d message-passing readout, reduced to 64 principal components (fitted on all 6,895 structures, which
needs no label), and added to feature sets A and A+fps as the block ``chemeleon``. A naive 768-d ChemBERTa concatenation
significantly hurt CYP3A4 earlier, hence the reduction.

Two processes, because chemprop's torch and LightGBM must not meet: extraction runs under ``.venv-ml`` (chemprop, no
LightGBM) and writes ``caches/chemeleon_embeddings.npz``; evaluation runs under ``.venv`` and only reads that cache.

  alone:    lgbm_A+chem vs lgbm_A and lgbm_A+fps+chem vs lgbm_A+fps; diagnostic: the block alone;
  stacked:  default stack + the three candidates vs default stack.
GATE: corrected p < 0.05 at both z0 (addition_experiment.py), per isoform.

Usage: .venv-ml/bin/python -m cypmode.challenge.frozen_embeddings --extract
       .venv/bin/python -m cypmode.challenge.frozen_embeddings --isoform CYP3A4 --repeats 3 --run
"""

import argparse
from pathlib import Path

import numpy as np

CACHE = Path("data/tdi_track/outputs/caches/chemeleon_embeddings.npz")
WEIGHTS = Path.home() / ".chemprop" / "chemeleon_mp.pt"
N_COMPONENTS = 64


def extract(smiles: list[str], cache: Path = CACHE, batch_size: int = 128) -> None:
    import torch
    from chemprop import data, featurizers, nn
    from rdkit import Chem

    keys = list(dict.fromkeys(Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in smiles))
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    agg = nn.MeanAggregation()
    checkpoint = torch.load(WEIGHTS, weights_only=True)
    mp = nn.BondMessagePassing(**checkpoint["hyper_parameters"])
    mp.load_state_dict(checkpoint["state_dict"])
    mp.eval()
    points = [data.MoleculeDatapoint.from_smi(s) for s in keys]
    loader = data.build_dataloader(data.MoleculeDataset(points, featurizer), batch_size=batch_size, num_workers=0, shuffle=False)
    out = []
    with torch.no_grad():
        for batch in loader:
            bmg = batch.bmg
            out.append(agg(mp(bmg), bmg.batch).cpu().numpy())
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, smiles=np.array(keys, dtype=object), embeddings=np.vstack(out).astype(np.float32))


def load_chemeleon_pca(smiles: list[str], cache: Path = CACHE, n_components: int = N_COMPONENTS) -> np.ndarray:
    """Label-free PCA of the cached embeddings of ALL cached compounds, rows following ``smiles``; never imports torch."""
    from rdkit import Chem
    from sklearn.decomposition import PCA

    if not Path(cache).exists():
        raise FileNotFoundError(f"{cache} is missing: run `.venv-ml/bin/python -m cypmode.challenge.frozen_embeddings --extract` first")
    z = np.load(cache, allow_pickle=True)
    keys = list(z["smiles"])
    index = {k: i for i, k in enumerate(keys)}
    reduced = PCA(n_components=n_components, svd_solver="full", random_state=0).fit_transform(z["embeddings"].astype(np.float64))
    wanted = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in smiles]
    missing = [w for w in wanted if w not in index]
    if missing:
        raise ValueError(f"{len(missing)} compounds have no CheMeleon embedding, e.g. {missing[0]!r}")
    return reduced[[index[w] for w in wanted]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--isoform", choices=["CYP3A4", "CYP2D6"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if args.extract:
        import pandas as pd

        train = pd.read_csv("data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv")
        test = pd.read_csv("data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv")
        extract(list(train.SMILES) + list(test.SMILES))
        return
    from cypmode.challenge.addition_experiment import evaluate_addition
    from cypmode.challenge.nested_runners import embedding_candidates

    pairs = [("lgbm_A_deep", "lgbm_A+chem_deep"), ("lgbm_A+fps_deep", "lgbm_A+fps+chem_deep")]
    evaluate_addition(args.isoform, embedding_candidates(), args.repeats, args.n_jobs, args.run, pairs, diagnostics=("lgbm_chem_deep",))


if __name__ == "__main__":
    main()

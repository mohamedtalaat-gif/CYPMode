"""Frozen Monroe embeddings as a feature block and as the input of TabICL (job 209).

Monroe (Banaszewski, arXiv 2608.18982; MIT licence, ``third_party/monroe``) is a GRIT graph transformer pretrained on the semi-empirical quantum-chemical properties of
about 81 million molecules (PM6) and on sparse PubChem bioassay labels. It reads the 3D structure of a molecule (an RDKit ETKDG conformer) and returns a 720-dimensional
graph embedding. The published recipe feeds the frozen embeddings of the labelled molecules to an in-context tabular model; this module does both things this project can
use: the embeddings, reduced to 64 principal components, become the block ``monroe`` of the LightGBM candidates, and 128 components feed TabICL (``nested_runners_torch``).

The wrapper is written here and does not use the Monroe inference scripts: the weights are read as plain tensors (``torch.load(weights_only=True)``, no pickled code runs),
the encoder is built from the configuration file with Monroe's ``GritTransformer`` and loaded with ``strict`` key checking (every encoder tensor must be found, none
left over), and the molecules are featurised with Monroe's own ``featurize_smiles`` (ETKDG, MMFF94s with UFF as fallback, a flat 2D layout when no conformer exists; those
molecules are flagged in the cache).

Two processes, as for CheMeleon: extraction runs under ``.venv-monroe`` (torch, torch-geometric, datamol; no LightGBM) and writes ``caches/monroe_embeddings.npz`` keyed by
canonical SMILES; evaluation runs under ``.venv`` and only reads the cache.

  alone:    lgbm_A+mon vs lgbm_A and lgbm_A+fps+mon vs lgbm_A+fps; diagnostic: the block alone;
  stacked:  default stack + the three candidates vs default stack.
GATE: corrected p < 0.05 at both z0 (addition_experiment.py), per isoform.

Usage: PYTHONPATH=.:third_party/monroe .venv-monroe/bin/python -m cypmode.challenge.monroe_embeddings --extract
       .venv/bin/python -m cypmode.challenge.monroe_embeddings --isoform CYP3A4 --repeats 3 --run
"""

import argparse
import json
from pathlib import Path

import numpy as np

CACHE = Path("data/tdi_track/outputs/caches/monroe_embeddings.npz")
CHECKPOINT = Path("data/tdi_track/inputs/monroe")
N_COMPONENTS = 64
TRAIN_CSV = "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
TEST_CSV = "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv"


def load_encoder(checkpoint: Path = CHECKPOINT):
    """The Monroe encoder with the released weights, read without executing pickled code; every encoder tensor must match."""
    import torch
    from monroe.model.constants import EDGE_FEAT_LIST_ONE_HOT, NODE_FEAT_LIST_FLOAT, NODE_FEAT_LIST_ONE_HOT
    from monroe.model.grit import GritTransformer

    config = json.loads((checkpoint / "config.json").read_text())["encoder"]
    config.pop("edge_rbf_dim", None)
    for key in ("node_float_rbf", "node_float_missing"):
        config.pop(key, None)
    encoder = GritTransformer(node_feature_vocab=NODE_FEAT_LIST_ONE_HOT, edge_feature_vocab=EDGE_FEAT_LIST_ONE_HOT, node_float_dim=len(NODE_FEAT_LIST_FLOAT), **config)
    state = torch.load(checkpoint / "weights.pt", weights_only=True, map_location="cpu")
    mine = {k.removeprefix("encoder.").replace("._orig_mod.", "."): v for k, v in state.items() if k.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(mine, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"the Monroe weights do not match the encoder: {len(missing)} missing ({missing[:3]}), {len(unexpected)} unexpected ({unexpected[:3]})")
    return encoder.eval()


def extract(smiles: list[str], cache: Path = CACHE, batch_size: int = 32, n_workers: int = 3) -> None:
    import torch
    from monroe.eval.embed import featurize_smiles, to_pyg
    from rdkit import Chem
    from torch_geometric.data import Batch

    keys = list(dict.fromkeys(Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in smiles))
    encoder = load_encoder()
    built = dict(featurize_smiles(keys, n_workers=n_workers))
    embeddings = np.full((len(keys), 720), np.nan, dtype=np.float32)
    flat = np.zeros(len(keys), dtype=bool)
    order = [i for i, k in enumerate(keys) if k in built]
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            graphs = [to_pyg(built[keys[i]]) for i in rows]
            out, _ = encoder(Batch.from_data_list(graphs))
            embeddings[rows] = out.numpy()
            for i, g in zip(rows, graphs, strict=True):
                flat[i] = bool(np.all(g.pos_in.numpy()[:, 2] == 0))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, smiles=np.array(keys, dtype=object), embeddings=embeddings, flat_2d=flat)
    print(f"{len(keys)} compounds: {len(built)} embedded, {len(keys) - len(built)} failed, {int(flat.sum())} with a flat 2D fallback; wrote {cache}")


def load_monroe_pca(smiles: list[str], cache: Path = CACHE, n_components: int = N_COMPONENTS) -> np.ndarray:
    """Label-free PCA of the cached embeddings of ALL cached compounds, rows following ``smiles``; never imports torch.

    A compound without an embedding (the featuriser failed) receives the mean embedding, which is the zero vector after centring.
    """
    from rdkit import Chem
    from sklearn.decomposition import PCA

    if not Path(cache).exists():
        raise FileNotFoundError(f"{cache} is missing: run `.venv-monroe/bin/python -m cypmode.challenge.monroe_embeddings --extract` first")
    z = np.load(cache, allow_pickle=True)
    index = {k: i for i, k in enumerate(z["smiles"])}
    values = z["embeddings"].astype(np.float64)
    values = np.where(np.isnan(values), np.nanmean(values, axis=0, keepdims=True), values)
    reduced = PCA(n_components=n_components, svd_solver="full", random_state=0).fit_transform(values)
    wanted = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in smiles]
    missing = [w for w in wanted if w not in index]
    if missing:
        raise ValueError(f"{len(missing)} compounds have no Monroe embedding, e.g. {missing[0]!r}")
    return reduced[[index[w] for w in wanted]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--isoform", choices=["CYP3A4", "CYP2D6"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.extract:
        import pandas as pd

        extract(list(pd.read_csv(TRAIN_CSV).SMILES) + list(pd.read_csv(TEST_CSV).SMILES), n_workers=args.workers)
        return
    from cypmode.challenge.addition_experiment import evaluate_addition
    from cypmode.challenge.nested_runners import monroe_candidates

    pairs = [("lgbm_A_deep", "lgbm_A+mon_deep"), ("lgbm_A+fps_deep", "lgbm_A+fps+mon_deep")]
    evaluate_addition(args.isoform, monroe_candidates(), args.repeats, args.n_jobs, args.run, pairs, diagnostics=("lgbm_mon_deep",))


if __name__ == "__main__":
    main()

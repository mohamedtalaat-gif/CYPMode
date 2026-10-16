"""Properties predicted by OpenADMET's surrogate models (logD and mouse brain protein binding) as label-free feature columns.

Two Chemprop directed message-passing networks of the OpenADMET project (Apache-2.0), each started from the ``surrogate_adme`` pretraining weights and fine-tuned on one endpoint of the OpenADMET
ExpansionRx challenge (a different compound library from the CYP challenge's):

  ``openadmet/logd-chemprop_foundation_surrogate-v1``    logD, test R^2 0.83 and Spearman 0.92 on a cluster split (as the repository reports)
  ``openadmet/mpbp-chemprop_foundation_surrogate-v1``    log mouse brain protein binding (LOG_MBPB), test R^2 0.74 and Spearman 0.89

logD is the lipophilicity of a compound at the pH of blood, and protein binding is how much of a compound sticks to the proteins around it instead of staying free: lipophilic, strongly bound compounds
meet cytochrome P450 enzymes more often. The descriptors of the project hold only Crippen logP, a sum of atom contributions that knows nothing about ionisation or binding. Each model reads a structure and
returns one number, so the pair is an ordinary label-free block (two columns): no challenge label is involved and nothing is fitted inside a task.

The extraction runs under ``.venv-ml`` (Chemprop, Lightning) and writes ``caches/openadmet_surrogates.npz`` keyed by canonical SMILES; the evaluation under ``.venv`` only reads it.

  extract:   PYTHONPATH=. .venv-ml/bin/python -m cypmode.challenge.openadmet_surrogates --extract
  evaluate:  .venv/bin/python -m cypmode.challenge.aux_experiments --experiment ext_adme --isoform CYP3A4 --repeats 3 --run
GATE (addition_experiment): corrected p < 0.05 at both z0, alone (added to feature sets A and A+fps) and inside the assembled stack, per isoform.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path("data/tdi_track/inputs/openadmet_surrogate")
CHECKPOINTS = {
    "logd": ROOT / "logd/checkpoints/best-epoch=18-val_loss=0.0544.ckpt",
    "mbpb": ROOT / "mpbp/checkpoints/best-epoch=25-val_loss=0.0665.ckpt",
}
CACHE = Path("data/tdi_track/outputs/caches/openadmet_surrogates.npz")
TRAIN_CSV = "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
TEST_CSV = "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv"


def extract(smiles: list[str], cache: Path = CACHE, batch_size: int = 128) -> None:
    import torch
    from chemprop import data, featurizers
    from chemprop.models import MPNN
    from lightning import pytorch as pl
    from rdkit import Chem

    keys = list(dict.fromkeys(Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in smiles))
    dataset = data.MoleculeDataset([data.MoleculeDatapoint.from_smi(s) for s in keys], featurizers.SimpleMoleculeMolGraphFeaturizer())
    loader = data.build_dataloader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_progress_bar=False, enable_model_summary=False)
    columns = []
    for name, checkpoint in CHECKPOINTS.items():
        model = MPNN.load_from_checkpoint(checkpoint, map_location="cpu")
        columns.append(torch.cat(trainer.predict(model, loader)).numpy().astype(np.float32).reshape(len(keys)))
        print(f"{name}: mean {columns[-1].mean():.2f}, sd {columns[-1].std():.2f}, range {columns[-1].min():.1f} to {columns[-1].max():.1f}", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, smiles=np.array(keys, dtype=object), values=np.stack(columns, axis=1), columns=np.array(list(CHECKPOINTS)))
    print(f"{len(keys)} compounds; wrote {cache}")


def load_openadmet_surrogates(smiles: list[str], cache: Path = CACHE) -> np.ndarray:
    """Read-only lookup of the predicted properties (rows follow ``smiles``, columns logD and MBPB); never imports Chemprop."""
    from cypmode.challenge.external_continuous import load_ext_continuous

    return load_ext_continuous(smiles, cache)


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    if args.extract:
        extract(list(pd.read_csv(TRAIN_CSV).SMILES) + list(pd.read_csv(TEST_CSV).SMILES))


if __name__ == "__main__":
    main()

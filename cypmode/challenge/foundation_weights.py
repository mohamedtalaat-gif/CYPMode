"""Make the OpenADMET Chemprop pretraining weights loadable by ``chemprop train --from-foundation``.

The files of ``openadmet/chemprop-foundation-pretraining-weights`` (Apache-2.0) are in the format of CheMeleon's ``chemeleon_mp.pt``: a dictionary with the hyper-parameters and the state of the
message-passing block only (``BondMessagePassing``). Chemprop accepts a *path* after ``--from-foundation`` only when the file is a complete model checkpoint, so each file is wrapped once into
one: the pretrained message-passing block, a mean aggregation (the one CheMeleon was pretrained with) and an untrained single-output regression head, written with Chemprop's own ``save_model``.
Nothing of the pretrained block changes, and the head is discarded by ``--from-foundation`` (only the message-passing block and the aggregation are taken over).

Usage: PYTHONPATH=. .venv-ml/bin/python -m cypmode.challenge.foundation_weights    (idempotent; run by the jobs that need the converted files)
"""

from __future__ import annotations

from pathlib import Path

SOURCE = Path("data/tdi_track/inputs/openadmet_foundation")
CONVERTED = SOURCE / "converted"
NAMES = ("minimol", "surrogate_adme")


def converted_path(name: str) -> Path:
    return CONVERTED / f"{name}__s42_mpnn.pt"


def convert(name: str) -> Path:
    import torch
    from chemprop import nn
    from chemprop.models import MPNN, save_model

    target = converted_path(name)
    if target.exists():
        return target
    pretrained = torch.load(SOURCE / f"{name}__s42.pt", map_location="cpu", weights_only=True)
    block = nn.BondMessagePassing(**pretrained["hyper_parameters"])
    block.load_state_dict(pretrained["state_dict"])
    model = MPNN(block, nn.MeanAggregation(), nn.RegressionFFN(input_dim=block.output_dim, n_tasks=1))
    CONVERTED.mkdir(parents=True, exist_ok=True)
    save_model(target, model)
    return target


def main() -> None:
    for name in NAMES:
        print(convert(name))


if __name__ == "__main__":
    main()

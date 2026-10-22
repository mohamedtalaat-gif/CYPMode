"""Chemprop (directed message-passing GNN) candidates for a NestedPlan, run through the Chemprop
command line in the isolated torch environment (.venv-ml).

    PYTHONPATH=. .venv-ml/bin/python -m cypmode.challenge.nested_runners_chemprop --isoform CYP3A4

Same contract as the other runners: a task trains on ``task.train`` plus compounds OUTSIDE the plan's
pool and external set (never another pool row, so inner tasks cannot see outer-validation rows),
holds out 10% of those training rows for Chemprop's own early stopping, and predicts
``task.evaluate``. The network is multitask over the eight potency endpoints (direct and TDI-arm
pIC50 of four isoforms; missing values are masked by Chemprop), and the candidate's score is the
isoform's predicted shift t_hat - max(d_hat, 4). ``chemeleon`` starts from the CheMeleon
foundation model instead of random weights.
"""

import argparse
import logging
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cypmode.challenge.nested import NestedPlan, PredictionCache, Task
from cypmode.challenge.tdi_targets import (
    ASSAY_FLOOR,
    ISOFORMS,
    direct_column,
    tdi_column,
)

logger = logging.getLogger(__name__)

TARGETS = [col for iso in ISOFORMS for col in (direct_column(iso), tdi_column(iso))]
FOUNDATION_DIR = Path("data/tdi_track/inputs/openadmet_foundation/converted")  # the OpenADMET pretraining weights (huggingface.co/openadmet/chemprop-foundation-pretraining-weights, Apache-2.0), wrapped by foundation_weights.py
VARIANTS = {
    "chemprop_mt_shift": [],
    "chemeleon_mt_shift": ["--from-foundation", "CheMeleon"],
    # the same network started from other OpenADMET pretraining weights: MiniMol embeddings as pretraining targets and the Novartis ADMET surrogate predictions (OpenADMET blog, "Throwing everything and the kitchen sink at CheMeleon")
    "chemeleon_minimol_mt_shift": ["--from-foundation", str(FOUNDATION_DIR / "minimol__s42_mpnn.pt")],
    "chemeleon_surrogate_mt_shift": ["--from-foundation", str(FOUNDATION_DIR / "surrogate_adme__s42_mpnn.pt")],
}
DEFAULT_VARIANTS = ("chemprop_mt_shift", "chemeleon_mt_shift")  # what the command runs without --variants: the foundation variants are separate, later jobs


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"{command[1]} failed (exit {result.returncode}): {(result.stderr or result.stdout)[-1500:]}")


def _rows(task: Task, outside: np.ndarray) -> np.ndarray:
    return np.union1d(task.train, outside)


def fit_predict(task: Task, smiles: list[str], targets: np.ndarray, outside: np.ndarray, isoform: str, extra_args: list[str], epochs: int, binary: str, seed: int = 0, accelerator: str = "cpu") -> np.ndarray:
    rows = _rows(task, outside)
    position = ISOFORMS.index(isoform)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        train = pd.DataFrame(targets[rows], columns=TARGETS)
        train.insert(0, "smiles", [smiles[i] for i in rows])
        train.to_csv(tmp / "train.csv", index=False)
        pd.DataFrame({"smiles": [smiles[i] for i in task.evaluate]}).to_csv(tmp / "eval.csv", index=False)
        common = ["--num-workers", "0"]
        _run(
            [binary, "train", "-i", str(tmp / "train.csv"), "-s", "smiles", "--target-columns", *TARGETS, "--task-type", "regression", "--epochs", str(epochs), "--split-sizes", "0.9", "0.1", "0.0", "--data-seed", str(seed), "--pytorch-seed", str(seed), "-o", str(tmp / "model"), "--accelerator", accelerator, *common, *extra_args],
        )
        checkpoints = sorted((tmp / "model").rglob("*.pt")) or sorted((tmp / "model").rglob("*.ckpt"))
        best = [p for p in checkpoints if "best" in p.name] or checkpoints
        if not best:
            raise RuntimeError(f"chemprop wrote no checkpoint under {tmp / 'model'}")
        _run(
            [binary, "predict", "-i", str(tmp / "eval.csv"), "-s", "smiles", "--model-paths", str(best[0]), "-o", str(tmp / "pred.csv"), "--accelerator", accelerator, *common],
        )
        predicted = pd.read_csv(tmp / "pred.csv")[TARGETS].to_numpy(dtype=float)
    return predicted[:, 2 * position + 1] - np.maximum(predicted[:, 2 * position], ASSAY_FLOOR)


def run(plan: NestedPlan, cache: PredictionCache, smiles: list[str], targets: np.ndarray, isoform: str, variants: list[str], epochs: int, binary: str, accelerator: str = "cpu", deadline: float | None = None, max_repeat: int | None = None) -> int:
    """Fill the cache; stop after the task that is running when ``deadline`` (a time.time() value) passes. Returns the number of tasks still missing.

    ``max_repeat`` restricts the work to the cross-validation tasks of the repeats 0..max_repeat (no deployment fits): the first stage of a staged design, whose remaining repeats are
    filled later only if a rule written beforehand asks for it. The returned count then covers those tasks only.
    """
    excluded = set(plan.pool.tolist()) | set(plan.external.tolist())
    outside = np.array([i for i in range(len(targets)) if i not in excluded and not np.isnan(targets[i]).all()], dtype=np.int64)
    for name in variants:
        todo = [task for task in cache.missing(name) if max_repeat is None or 0 <= task.repeat <= max_repeat]
        for i, task in enumerate(todo, 1):
            if deadline is not None and time.time() > deadline:
                break
            cache.put(name, task, fit_predict(task, smiles, targets, outside, isoform, VARIANTS[name], epochs, binary, accelerator=accelerator))
            logger.info("%s: %d/%d tasks", name, i, len(todo))
    return sum(len([task for task in cache.missing(name) if max_repeat is None or 0 <= task.repeat <= max_repeat]) for name in variants)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isoform", required=True, choices=["CYP3A4", "CYP2D6"])
    parser.add_argument("--variants", nargs="*", default=list(DEFAULT_VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--binary", default=".venv-ml/bin/chemprop")
    parser.add_argument("--accelerator", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--root", default="data/tdi_track/outputs/nested")
    parser.add_argument("--max-repeat", type=int, default=None, help="only the cross-validation tasks of the repeats 0..N (a staged design)")
    parser.add_argument("--max-minutes", type=float, default=None, help="stop after the task that is running when this many minutes have passed")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = Path(args.root) / args.isoform
    plan = NestedPlan.load(root / "plan.npz")
    train = pd.read_csv("data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv")
    test = pd.read_csv("data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv")
    smiles = list(train["SMILES"]) + list(test["SMILES"])
    targets = np.full((len(smiles), len(TARGETS)), np.nan, dtype=np.float32)
    targets[: len(train)] = train[TARGETS].to_numpy(dtype=np.float32)
    deadline = None if args.max_minutes is None else time.time() + 60 * args.max_minutes
    binary = "scripts/chemprop_mps.sh" if args.accelerator == "mps" and args.binary == ".venv-ml/bin/chemprop" else args.binary  # the GPU has no deterministic scatter: see scripts/chemprop_warn_only.py
    remaining = run(plan, PredictionCache(root / "cache", plan), smiles, targets, args.isoform, args.variants, args.epochs, binary, args.accelerator, deadline, args.max_repeat)
    print(f"TASKS_REMAINING {remaining}")


if __name__ == "__main__":
    _main()

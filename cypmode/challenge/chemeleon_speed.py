# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The speed test and the rule of the decision on the fine-tuned CheMeleon (queue job 202_chemeleon_speed).

The fine-tuned CheMeleon (``chemeleon_mt_shift``, the network of plain Chemprop started from the CheMeleon weights) took 36 to 48 minutes per task on the processor, and stage 1 of its staged design
(``chemeleon_stage``) is the 25 cross-validation tasks of repeat 0 for an enzyme. Whether it is worth running depends on what it can add (little: the plain network added +0.0048 AUC to the default
stack on CYP3A4, a gain below the least difference of 0.0099 that the main design recognises 8 times in 10) and on what it costs. The decision was taken on 2026-10-16, before any measurement:

  1. CYP2D6 is dropped from the CheMeleon design for good: plain Chemprop is at chance on CYP2D6 (outer AUC 0.54 and 0.53) and the frozen CheMeleon block reaches 0.63.
  2. One task of repeat 0 that is already in the cache from the processor is trained again on the Apple GPU (``--accelerator mps``), in a scratch directory (nothing is written to the cache), and
     the seconds and the agreement with the processor's result are recorded. The GPU result is accepted as equivalent when the rank correlation of the two sets of scores is at least
     ``MIN_AGREEMENT`` and the AUC on the labelled rows of the task differs by at most ``MAX_AUC_DIFFERENCE``.
  3. Stage 1 for CYP3A4 runs only if the tasks that are still missing fit in ``STAGE_ONE_HOURS_LIMIT`` hours: on the GPU if the GPU result is equivalent, otherwise on the processor. If they do not,
     the fine-tuned CheMeleon is NOT RUN, for cost; that is recorded in the ledger as "not run: cost" and is not a verdict.

Amendment of 2026-10-22, before any GPU measurement exists: the first run of the speed test (2026-10-17) stopped on a determinism error of the MPS backend and measured no GPU time, so its decision "not run: cost" rests on the
processor's estimate alone. What was dropped for cost is run again on 2026-10-22. The GPU run now goes through ``scripts/chemprop_mps.sh`` (Chemprop with PyTorch's deterministic algorithms in warn-only mode: the seed
stays fixed, only the order of the additions of one scatter operation may vary). The rule is unchanged (equivalence by rank correlation and AUC, at most ``STAGE_ONE_HOURS_LIMIT`` hours for the missing tasks), and the decision
of 2026-10-17 is kept beside the new one in ``data/tdi_track/outputs/comparisons/``.

``python -m cypmode.challenge.chemeleon_speed`` writes ``data/tdi_track/outputs/comparisons/chemeleon_speed_CYP3A4.json`` and the one-line decision
``data/tdi_track/outputs/comparisons/chemeleon_speed_decision.txt`` (``mps``, ``cpu`` or ``not run: cost``), which the jobs 203, 236 and 237 read.
"""

import json
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from cypmode.challenge import nested_runners_chemprop as chemprop_runner
from cypmode.challenge.nested import NestedPlan, PredictionCache

VARIANT = "chemeleon_mt_shift"
ISOFORM = "CYP3A4"
STAGE_ONE_HOURS_LIMIT = 8.0  # the tasks of stage 1 that are still missing must fit in this many hours
MIN_AGREEMENT = 0.8  # Spearman correlation between the GPU and the processor scores of the same task
MAX_AUC_DIFFERENCE = 0.03  # AUC of the two sets of scores on the labelled rows of the task
TIMEOUT_SECONDS = 2700  # a GPU run that needs this long has no speed-up to show
ROOT = Path("data/tdi_track/outputs/nested")
COMPARISONS = Path("data/tdi_track/outputs/comparisons")
LOG = Path("data/structure_track/outputs/cyp3a4/logs/chemprop_CYP3A4.log")


def processor_seconds_per_task(log: Path = LOG) -> float:
    """The median time between two consecutive finished tasks of the same chunk in the log of the processor runs (gaps of more than two hours separate chunks and are left out)."""
    stamps = [datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc) for m in re.finditer(rf"^(\d{{4}}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ {VARIANT}: \d+/\d+ tasks", log.read_text(), re.MULTILINE)]
    gaps = [(b - a).total_seconds() for a, b in pairwise(stamps) if 0 < (b - a).total_seconds() < 7200]
    return float(statistics.median(gaps))


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve as the Mann-Whitney statistic (the project's other code uses scikit-learn; this environment need not have it)."""
    ranks = stats.rankdata(scores)
    positive = labels == 1
    n_positive, n_negative = int(positive.sum()), int((~positive).sum())
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def decide(cpu_seconds: float, gpu_seconds: float | None, equivalent: bool, tasks_left: int, limit_hours: float = STAGE_ONE_HOURS_LIMIT) -> str:
    """``mps`` when the GPU is equivalent and the missing tasks fit the limit on it; ``cpu`` when they fit on the processor; otherwise ``not run: cost``."""
    if gpu_seconds is not None and equivalent and tasks_left * gpu_seconds / 3600 <= limit_hours:
        return "mps"
    if tasks_left * cpu_seconds / 3600 <= limit_hours:
        return "cpu"
    return "not run: cost"


def _timed_run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=TIMEOUT_SECONDS)
    if result.returncode:
        raise RuntimeError(f"{command[1]} failed (exit {result.returncode}): {(result.stderr or result.stdout)[-1500:]}")


def measure() -> dict:
    plan = NestedPlan.load(ROOT / ISOFORM / "plan.npz")
    cache = PredictionCache(ROOT / ISOFORM / "cache", plan)
    cached = [t for t in plan.tasks() if t.repeat == 0 and cache.has(VARIANT, t)]
    if not cached:
        raise SystemExit("no cached CheMeleon task of repeat 0 to compare with")
    task = cached[0]
    tasks_left = len([t for t in cache.missing(VARIANT) if 0 <= t.repeat <= 0])
    train = pd.read_csv("data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv")
    test = pd.read_csv("data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv")
    smiles = list(train["SMILES"]) + list(test["SMILES"])
    targets = np.full((len(smiles), len(chemprop_runner.TARGETS)), np.nan, dtype=np.float32)
    targets[: len(train)] = train[chemprop_runner.TARGETS].to_numpy(dtype=np.float32)
    excluded = set(plan.pool.tolist()) | set(plan.external.tolist())
    outside = np.array([i for i in range(len(targets)) if i not in excluded and not np.isnan(targets[i]).all()], dtype=np.int64)

    chemprop_runner._run = _timed_run
    started = time.time()
    try:
        scores = chemprop_runner.fit_predict(task, smiles, targets, outside, ISOFORM, chemprop_runner.VARIANTS[VARIANT], 30, "scripts/chemprop_mps.sh", accelerator="mps")
        gpu_seconds = time.time() - started
        failure = None
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        gpu_seconds, scores, failure = None, None, str(error)[-400:]
    cpu_seconds = processor_seconds_per_task()
    result = {"task": [task.repeat, task.outer, task.inner], "tasks_left": tasks_left, "cpu_seconds_per_task": cpu_seconds, "gpu_seconds": gpu_seconds, "failure": failure}
    equivalent = False
    if scores is not None:
        reference = cache.get(VARIANT, task)
        rows = np.asarray(task.evaluate)
        labelled = rows < len(train)
        y = train.loc[rows[labelled], f"{ISOFORM}_is_TDI"].to_numpy(dtype=float)
        keep = ~np.isnan(y)
        rho = float(stats.spearmanr(scores, reference)[0])
        auc_gpu = auc(y[keep].astype(int), scores[labelled][keep])
        auc_cpu = auc(y[keep].astype(int), reference[labelled][keep])
        equivalent = bool(rho >= MIN_AGREEMENT and abs(auc_gpu - auc_cpu) <= MAX_AUC_DIFFERENCE)
        result.update({"spearman": rho, "auc_gpu": auc_gpu, "auc_cpu": auc_cpu, "equivalent": equivalent})
    decision = decide(cpu_seconds, gpu_seconds, equivalent, tasks_left)
    result.update({"decision": decision, "hours_left_on_cpu": tasks_left * cpu_seconds / 3600, "hours_left_on_gpu": None if gpu_seconds is None else tasks_left * gpu_seconds / 3600, "limit_hours": STAGE_ONE_HOURS_LIMIT})
    COMPARISONS.mkdir(parents=True, exist_ok=True)
    (COMPARISONS / f"chemeleon_speed_{ISOFORM}.json").write_text(json.dumps(result, indent=1) + "\n")
    (COMPARISONS / "chemeleon_speed_decision.txt").write_text(decision + "\n")
    return result


def main() -> int:
    result = measure()
    print(json.dumps(result, indent=1))
    print(f"CHEMELEON DECISION: {result['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

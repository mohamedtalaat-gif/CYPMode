"""The rule that decides whether the second stage of the CheMeleon design runs.

CheMeleon (about 38 minutes per task on the processor, 80 tasks per enzyme in the full design, 25 of them in stage 1) is run in two stages. Stage 1 fits repeat 0 only (5 outer folds) and compares the stack with and without it
(``stage1_chemeleon_<isoform>.txt``, written by ``aux_experiments --experiment chemeleon --repeats 1``; not a gate log). Stage 2, the other two repeats and the deployment fits, runs for an
enzyme only if the stage 1 comparison at the first confidence cut shows

  * a mean AUC gain of at least ``MIN_GAIN`` (+0.005), and
  * a gain in at least ``MIN_FOLDS`` (4) of the 5 outer folds.

The rule was written on 2026-10-16, before any CheMeleon result existed, for a reason of cost; it can only lower the number of tests that are run, so it cannot raise the share of false
passes of the ordinary 15-fold gate that follows stage 2.

Amendment of 2026-10-16, also before any CheMeleon result: CYP2D6 is dropped from the design, and stage 1 for CYP3A4 runs only if the speed test of ``chemeleon_speed`` finds that the missing tasks fit in 8 hours (on the
Apple GPU or on the processor); stage 1 is the 25 tasks of repeat 0 of an enzyme, stage 2 the other 55. A missing or unreadable stage-1 result is a stop, as above.

Usage: python -m cypmode.challenge.chemeleon_stage --isoform CYP3A4   (prints CONTINUE or STOP)
"""

import argparse
import re
from pathlib import Path

LOGS = Path("data/structure_track/outputs/cyp3a4/logs")
MIN_GAIN, MIN_FOLDS = 0.005, 4
STACKED = re.compile(r"z0=1\.0\s+stacked base\+added vs base: AUC ([+-]\d\.\d+) \(p=[\d.]+, better (\d+)/(\d+)\)")


def decide(path: Path) -> bool:
    """True when stage 1 asks for stage 2; a missing or unreadable stage 1 result is a stop, never a silent continue."""
    if not path.exists():
        return False
    found = STACKED.search(path.read_text(errors="ignore"))
    if found is None:
        return False
    gain, better = float(found.group(1)), int(found.group(2))
    return gain >= MIN_GAIN and better >= MIN_FOLDS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isoform", required=True, choices=["CYP3A4", "CYP2D6"])
    parser.add_argument("--logs", default=str(LOGS))
    args = parser.parse_args()
    print("CONTINUE" if decide(Path(args.logs) / f"stage1_chemeleon_{args.isoform}.txt") else "STOP")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

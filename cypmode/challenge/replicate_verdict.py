# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Verdict of the 10-outer-fold replicate for the physics additions and for the neural pair (the multitask network with plain Chemprop).

The 5-fold design (5 outer folds, 3 repeats) judges an addition by the corrected resampled t of Nadeau and Bengio, whose variance
factor is 1/J + n_test/n_train: with J = 15 folds and n_test/n_train = 1/4 it is 0.317, so effects of +0.005 AUC rarely reach
p < 0.05. The replicate repeats the same comparisons with 10 outer folds (J = 30, n_test/n_train = 1/9, factor 0.144). It is declared for
the physics additions (tier A, compact, tier B) before any 10-fold result exists, as a confirmatory design that is reported next to the
5-fold one; an addition counts as replicated only if

  * the stacked comparison in the 10-fold design has corrected p < 0.05 and a positive mean AUC difference at both z0, AND
  * the same comparison in the 5-fold design has a positive mean difference at both z0 (the sign must agree).

One line per group and isoform is printed in the form the final build reads.

Usage: python -m cypmode.challenge.replicate_verdict --isoform CYP3A4
"""

import argparse
import re
from pathlib import Path

LOGS = Path("data/structure_track/outputs/cyp3a4/logs")
GROUPS = {
    "physics": ("physics_experiment", "physics_experiment_f10"),
    "physics_compact": ("physics_compact", "physics_compact_f10"),
    "physics_b": ("physics_tierb", "physics_tierb_f10"),
    "neural": ("aux_neural", "aux_neural_f10"),
    "pocket_cyp3a4": ("aux_pocket_cyp3a4", "aux_pocket_cyp3a4_f10"),  # declared on 2026-10-23 with the replicate rule, before any result exists
    "pocket_cyp2d6": ("aux_pocket_cyp2d6", "aux_pocket_cyp2d6_f10"),  # declared on 2026-10-16 after the 15-fold discovery, before any 10-fold result exists
}
SCOPE = {"neural": ("CYP3A4",), "pocket_cyp3a4": ("CYP3A4",), "pocket_cyp2d6": ("CYP2D6",)}  # a group is replicated only for the enzymes named here; the others print nothing (not tested is not NOT MET)
STACKED = re.compile(r"z0=(\d\.\d+)\s+stacked base\+added vs base: AUC ([+-]\d\.\d+) \(p=(\d\.\d+)")


def read_stacked(path: Path) -> dict[float, tuple[float, float]]:
    """{z0: (mean AUC difference, corrected p)} of the stacked comparison in one log."""
    if not path.exists():
        return {}
    return {float(z0): (float(diff), float(p)) for z0, diff, p in STACKED.findall(path.read_text(errors="ignore"))}


def verdict(five: dict[float, tuple[float, float]], ten: dict[float, tuple[float, float]]) -> bool:
    return bool(len(ten) == 2 and len(five) == 2 and all(p < 0.05 and d > 0 for d, p in ten.values()) and all(d > 0 for d, _ in five.values()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isoform", required=True, choices=["CYP3A4", "CYP2D6"])
    parser.add_argument("--logs", default=str(LOGS))
    args = parser.parse_args()
    logs = Path(args.logs)
    for group, (five_name, ten_name) in GROUPS.items():
        if args.isoform not in SCOPE.get(group, ("CYP3A4", "CYP2D6")):
            continue
        five = read_stacked(logs / f"{five_name}_{args.isoform}.log")
        ten = read_stacked(logs / f"{ten_name}_{args.isoform}.log")
        fmt = lambda d: "; ".join(f"z0={z}: {diff:+.4f} (p={p:.3f})" for z, (diff, p) in sorted(d.items())) or "no result"
        print(f"{group}: 5-fold stacked {fmt(five)} | 10-fold stacked {fmt(ten)}")
        met = verdict(five, ten)
        print(f"{args.isoform} GATE (replicate {group}, 10 outer folds corrected p < 0.05 at both z0 and the same sign in the 5-fold design): {'MET' if met else 'NOT MET'}")


if __name__ == "__main__":
    main()

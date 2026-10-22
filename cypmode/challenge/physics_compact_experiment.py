# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Compact physics: only the physics columns that matter, chosen inside every task (job 072).

The tier-A block has 133 columns and about a third of them are mostly missing (a role that a compound does not have). Boosting with
column subsampling spends many of its splits on columns that carry nothing. Here a physics-only model is fitted on the TRAINING rows
of each task, its 20 columns with the largest gain are appended to feature set A or A+fps, and the deep LightGBM is fitted on that,
so the evaluation rows never influence which columns are used.

  alone:    lgbm_A+physk20 vs lgbm_A and lgbm_A+fps+physk20 vs lgbm_A+fps (best-single outer-fold AUC / AP);
  stacked:  default stack + the compact candidates vs default stack.
GATE: corrected p < 0.05 at both z0 (addition_experiment.py), per isoform, in the 5-fold design and in the 10-fold replicate.

Usage: python -m cypmode.challenge.physics_compact_experiment --isoform CYP3A4 --repeats 3 --run [--outer 10 --root DIR]
"""

from cypmode.challenge.addition_experiment import evaluate_addition, parse_args
from cypmode.challenge.nested_runners import compact_physics_candidates

ALONE_PAIRS = [("lgbm_A_deep", "lgbm_A+physk20_deep"), ("lgbm_A+fps_deep", "lgbm_A+fps+physk20_deep")]


def main() -> None:
    args = parse_args(__doc__)
    evaluate_addition(args.isoform, compact_physics_candidates(), args.repeats, args.n_jobs, args.run, ALONE_PAIRS, outer=args.outer, root=args.root)


if __name__ == "__main__":
    main()

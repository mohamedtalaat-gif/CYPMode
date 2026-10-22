# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Physics tier B: Boltzmann-weighted conformers and relaxed hydrogen abstraction (job 074).

Tier B (``physics_tierb``) replaces the frozen-geometry hydrogen-abstraction energy by a locally relaxed one for the two best sites and
averages the ionisation potential, solvation and Fukui functions over a conformer ensemble.

  alone:    lgbm_A+physB vs lgbm_A, lgbm_A+fps+physB vs lgbm_A+fps, and lgbm_A+phys+physB vs lgbm_A+phys (tier A alone);
            diagnostic: the tier-B block by itself;
  stacked:  default stack + the four tier-B candidates vs default stack.
GATE: corrected p < 0.05 at both z0 (addition_experiment.py), per isoform, in the 5-fold design and in the 10-fold replicate.

Usage: python -m cypmode.challenge.physics_tierb_experiment --isoform CYP3A4 --repeats 3 --run [--outer 10 --root DIR]
"""

from cypmode.challenge.addition_experiment import evaluate_addition, parse_args
from cypmode.challenge.nested_runners import physics_b_candidates

ALONE_PAIRS = [("lgbm_A_deep", "lgbm_A+physB_deep"), ("lgbm_A+fps_deep", "lgbm_A+fps+physB_deep"), ("lgbm_A+phys_deep", "lgbm_A+phys+physB_deep")]


def main() -> None:
    args = parse_args(__doc__)
    evaluate_addition(args.isoform, physics_b_candidates(), args.repeats, args.n_jobs, args.run, ALONE_PAIRS, diagnostics=("lgbm_physB_deep",), outer=args.outer, root=args.root)


if __name__ == "__main__":
    main()

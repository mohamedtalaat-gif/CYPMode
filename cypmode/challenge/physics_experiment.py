# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Does the atom-, site- and pharmacophore-aware GFN2-xTB block improve the TDI model? (job 070)

Pre-registered before any physics candidate was run. The block (physics_block.py) describes the heme-binding atoms,
the oxidisable C-H sites and the 3D geometry between them. Two questions on the shared yardstick of
addition_experiment.py (similarity x prevalence x confident rows, z0 in {1.0, 1.96}):

  1. alone:    lgbm_A+phys vs lgbm_A and lgbm_A+fps+phys vs lgbm_A+fps (best-single outer-fold AUC / AP);
  2. stacked:  the default stack plus the three physics candidates vs the default stack.

GATE: Nadeau-Bengio corrected resampled t with p < 0.05 at BOTH z0 values, for the same comparison, on the isoform it
is claimed for. The diagnostic lgbm_phys (physics block alone) is reported, not gated.

Process rule: this process loads the cached physics features (never tblite); run ``physics_block --extract`` first.

Usage: python -m cypmode.challenge.physics_experiment --isoform CYP2D6 --repeats 3 --run
"""

from cypmode.challenge.addition_experiment import evaluate_addition, parse_args
from cypmode.challenge.nested_runners import physics_candidates

ALONE_PAIRS = [("lgbm_A_deep", "lgbm_A+phys_deep"), ("lgbm_A+fps_deep", "lgbm_A+fps+phys_deep")]


def main() -> None:
    args = parse_args(__doc__)
    evaluate_addition(args.isoform, physics_candidates(), args.repeats, args.n_jobs, args.run, ALONE_PAIRS, diagnostics=("lgbm_phys_deep",), outer=args.outer, root=args.root)


if __name__ == "__main__":
    main()

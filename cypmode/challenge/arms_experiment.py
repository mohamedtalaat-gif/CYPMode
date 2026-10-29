# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Floor-clipped two-arm regression (job 078_arm_variants).

The label is a function of the two assay arms, t (TDI condition) and d (direct inhibition): positive <=>
t - max(d, 4) > log10 2. Anything below the floor 4 is irrelevant to it, yet the existing two-arm regressors
(lgbm_reg_arms_*) fit raw values down to 1.9. Clipping both arms at 4 before fitting leaves every label unchanged and
removes that irrelevant variance. Measured context: the existing arm regressors rank below the classifiers (outer AUC
0.76-0.77 vs 0.80-0.81 on CYP3A4, 0.62-0.66 vs 0.72 on CYP2D6) and add a small amount to the stack on CYP3A4 only.

  alone:    lgbm_reg_arms_clip_X vs lgbm_reg_arms_X (best-single outer-fold AUC / AP);
  stacked:  default stack + clipped regressors vs default stack.
GATE: corrected p < 0.05 at both z0 (addition_experiment.py).

Usage: python -m cypmode.challenge.arms_experiment --isoform CYP3A4 --repeats 3 --run
"""

from cypmode.challenge.addition_experiment import evaluate_addition, parse_args
from cypmode.challenge.nested_runners import arm_candidates

ALONE_PAIRS = [("lgbm_reg_arms_A", "lgbm_reg_arms_clip_A"), ("lgbm_reg_arms_A+fps", "lgbm_reg_arms_clip_A+fps")]


def main() -> None:
    args = parse_args(__doc__)
    evaluate_addition(args.isoform, arm_candidates(), args.repeats, args.n_jobs, args.run, ALONE_PAIRS)


if __name__ == "__main__":
    main()

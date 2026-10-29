# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Test-likeness-weighted training (job 060).

The blinded test set is analog-heavy: its median nearest-neighbour similarity to the training set is 0.58-0.59 while the
training set's own is 0.44 (diversity panel 0.31). A model fitted with equal weights spends most of its capacity on the
dense, test-irrelevant parts of the training set. Under covariate shift the standard remedy is importance weighting by the
density ratio p_test(x) / p_train(x); here it is approximated from structures alone (see
``nested_runners.testlike_weights``: similarity to the nearest unlabeled test structures over similarity to the nearest
other training compounds, tempered by alpha in {0.5, 1}). Only the 750 test STRUCTURES are used, never labels.

  alone:    lgbm_X_deep_testlikeA vs lgbm_X_deep for X in {A, A+fps}, A in {0.5, 1.0} (best-single outer-fold AUC / AP);
  stacked:  default stack + the four weighted candidates vs default stack.
GATE: corrected p < 0.05 at both z0 (addition_experiment.py), per isoform.

Usage: python -m cypmode.challenge.test_likeness_training --isoform CYP3A4 --repeats 3 --run
"""

from cypmode.challenge.addition_experiment import evaluate_addition, parse_args
from cypmode.challenge.nested_runners import testlike_candidates

ALONE_PAIRS = [(f"lgbm_{fs}_deep", f"lgbm_{fs}_deep_testlike{a}") for fs in ("A", "A+fps") for a in ("0.5", "1.0")]


def main() -> None:
    args = parse_args(__doc__)
    evaluate_addition(args.isoform, testlike_candidates(), args.repeats, args.n_jobs, args.run, ALONE_PAIRS)


if __name__ == "__main__":
    main()

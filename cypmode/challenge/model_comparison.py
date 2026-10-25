# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Inference for "is pipeline A better than pipeline B" -- done on retrained models, not on frozen predictions.

Two shortcuts give p-values that look far better than the evidence:

* **Bootstrapping fixed predictions** (resampling compounds while the models stay as they are) measures
  how much the metric moves with the sample of test compounds. It ignores everything that varies when a
  model is retrained -- the training sample, the seed, the fold -- so two pipelines that differ by chance
  look "significant". Feeding such bootstrap replicates (or per-fold scores treated as independent
  replicates) into Tukey's HSD or a t-test is pseudoreplication: the replicates are not independent
  observations of the quantity being compared. The challenge organizers hit exactly this in their first
  significance analysis.
* **A plain paired t-test across CV folds.** The folds' training sets overlap (each pair shares (K-2)/(K-1)
  of its rows), so their scores are positively correlated and the naive variance of the mean difference is
  too small. Nadeau & Bengio (2003) correct it by inflating the variance factor from 1/J to 1/J + n_test/n_train.

What this module offers instead: the Nadeau-Bengio corrected resampled t-test on differences between two
pipelines evaluated on the SAME nested outer folds (each fold's score comes from a model fitted without that
fold's rows), plus ``compare_candidate_sets`` which runs two candidate sets through the identical selection
procedure. A permutation null (as in committee_uncertainty) is the other legitimate route when a quantity
has to be re-estimated under shuffled inputs. With only a handful of folds the corrected test is
conservative and often cannot reach significance for a small effect; that is the correct answer, not a
defect, and the remedy is more repeats (which reduce partitioning noise) and a larger effect, not a looser
test.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from cypmode.challenge.nested import (
    NestedPlan,
    PredictionCache,
    Selector,
    StackedLogit,
    nested_evaluate,
)


@dataclass(frozen=True)
class Comparison:
    mean_difference: float
    standard_error: float
    t: float
    p_value: float  # two-sided
    n_folds: int
    n_better: int  # folds in which A beats B -- a sanity check on the sign pattern, not an inference

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def corrected_resampled_t(differences: Sequence[float], n_train: int, n_test: int) -> Comparison:
    """Nadeau-Bengio corrected resampled t-test on per-fold score differences (A minus B).

    ``differences`` are the J paired differences over resampling splits (folds x repeats);
    ``n_train`` / ``n_test`` are the sizes of one split's training and test sets.
    """
    d = np.asarray(differences, dtype=float)
    j = len(d)
    if j < 2:
        raise ValueError("need at least two paired differences")
    variance = d.var(ddof=1)
    standard_error = float(np.sqrt((1.0 / j + n_test / n_train) * variance))
    mean = float(d.mean())
    if standard_error == 0:
        t = 0.0 if mean == 0 else float(np.inf * np.sign(mean))
    else:
        t = mean / standard_error
    p = float(2 * stats.t.sf(abs(t), df=j - 1)) if np.isfinite(t) else 0.0
    return Comparison(mean, standard_error, float(t), p, j, int((d > 0).sum()))


def naive_paired_t(differences: Sequence[float]) -> Comparison:
    """The uncorrected paired t-test, kept only so the optimism it adds can be demonstrated and reported."""
    d = np.asarray(differences, dtype=float)
    standard_error = float(d.std(ddof=1) / np.sqrt(len(d)))
    t = float(d.mean() / standard_error) if standard_error > 0 else 0.0
    return Comparison(float(d.mean()), standard_error, t, float(2 * stats.t.sf(abs(t), df=len(d) - 1)), len(d), int((d > 0).sum()))


def compare_candidate_sets(
    cache: PredictionCache,
    candidates_a: Sequence[str],
    candidates_b: Sequence[str],
    selector: Selector | None = None,
    weight_fn=None,
    target_prevalence: float | None = None,
    metric: str = "auc",
) -> Comparison:
    """Is the nested outer score of the selection procedure over ``candidates_a`` better than over
    ``candidates_b``? Both sets go through the same plan, folds and selector; only whole-procedure
    outer scores enter the test.
    """
    selector = selector or StackedLogit()
    a = nested_evaluate(cache, candidates_a, [selector], weight_fn, target_prevalence, metric).set_index(["repeat", "outer"])
    b = nested_evaluate(cache, candidates_b, [selector], weight_fn, target_prevalence, metric).set_index(["repeat", "outer"])
    plan = cache.plan
    n_test = len(plan.pool) // plan.outer_splits
    return corrected_resampled_t((a["outer_score"] - b["outer_score"]).to_numpy(), len(plan.pool) - n_test, n_test)


def plan_sizes(plan: NestedPlan) -> tuple[int, int]:
    """(training rows, test rows) of one outer split."""
    n_test = len(plan.pool) // plan.outer_splits
    return len(plan.pool) - n_test, n_test

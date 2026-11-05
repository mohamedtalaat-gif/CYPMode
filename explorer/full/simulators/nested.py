# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Section 6.3: nested cross-validation, the construction. What is fitted where, and how many fits that makes."""


def plan(n: int, outer: int, inner: int, repeats: int, candidates: int) -> dict:
    """Sizes and counts of a nested plan: every outer fold holds one part back, the choice among ``candidates`` is made by an inner cross-validation on the rest, and the chosen procedure is refitted on the whole
    outer training part and scored on the held-back part."""
    outer_test = n // outer
    outer_train = n - outer_test
    inner_test = outer_train // inner
    inner_train = outer_train - inner_test
    inner_fits = candidates * inner
    per_outer = inner_fits + candidates  # the inner folds, and one refit of each candidate for the stack that is scored
    return {"outer test size": outer_test, "outer training size": outer_train, "inner test size": inner_test, "inner training size": inner_train,
            "fits in the inner loop of one outer fold": inner_fits, "fits per outer fold": per_outer, "outer scores": outer * repeats, "total fits": outer * repeats * per_outer,
            "correction factor of the t-test": 1 / (outer * repeats) + outer_test / outer_train}

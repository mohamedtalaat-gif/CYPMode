# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import numpy as np

from cypmode.validation import ei_sampling as ei


def pools():
    matrices, rmsd = {}, {}
    for name, bad in (("a", 0), ("b", 1)):
        n = 5
        m = np.full((n, n), 0.4)
        good = [i for i in range(n) if i != 4]
        m[np.ix_(good, good)] = 0.9
        np.fill_diagonal(m, 1.0)
        matrices[name] = m
        rmsd[name] = np.array([1.0, 1.1, 0.9, 1.2, 6.0])
    return matrices, rmsd


def test_every_policy_spends_the_same_budget_and_returns_one_pick_per_ligand():
    matrices, rmsd = pools()
    rng = np.random.default_rng(1)
    for policy in ("uniform", "random", "ei"):
        out = ei.simulate(matrices, rmsd, 2, policy, rng)
        assert set(out) == {"a", "b"} and all(v in rmsd["a"] for v in out.values())


def test_ei_gives_extra_samples_to_the_pool_that_disagrees():
    matrices = {"tight": np.full((5, 5), 0.95), "loose": np.full((5, 5), 0.2)}
    for m in matrices.values():
        np.fill_diagonal(m, 1.0)
    rng = np.random.default_rng(0)
    revealed = {"tight": 0, "loose": 0}
    original = ei.pool_disagreement

    def spy(matrix, subset):
        return original(matrix, subset)

    # run the allocation loop directly through simulate with a single extra sample per ligand and count who gets it
    counts = {"tight": 0, "loose": 0}
    for _ in range(40):
        order = {n: rng.permutation(5) for n in matrices}
        rev = {n: ei.K0 for n in matrices}
        n = max(matrices, key=lambda m: ei.pool_disagreement(matrices[m], list(order[m][: rev[m]])))
        counts[n] += 1
    assert counts["loose"] == 40 and counts["tight"] == 0 and revealed == {"tight": 0, "loose": 0}


def test_expected_rmsd_is_deterministic_for_a_seed():
    matrices, rmsd = pools()
    a = ei.expected_rmsd(matrices, rmsd, 1, "ei", 20, seed=3)
    b = ei.expected_rmsd(matrices, rmsd, 1, "ei", 20, seed=3)
    assert a.equals(b)

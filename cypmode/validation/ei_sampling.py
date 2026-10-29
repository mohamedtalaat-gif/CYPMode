# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Expected-improvement sampling: spend extra poses where the pool disagrees most (job 172).

Boltz-2 samples of one ligand come from one trunk pass and mostly agree; the pose that minimum-Bayes-risk selection picks is
the consensus of the pool. A ligand whose pool has already reached a tight consensus gains little from another sample, one whose
pool still disagrees has room for the pick to change. The acquisition value used here is
    u(ligand) = 1 - (best expected LDDT-PLI inside the samples drawn so far)      (1.0 while only one sample exists)
and the policy "ei" gives the next sample to the ligand with the largest u (greedy, sequential). It is compared, at EQUAL
total compute, with "uniform" (every ligand gets the same number of extra samples) and "random" allocation.

Offline simulation on the benchmark pools (no new structure is needed): every ligand starts with ``k0`` samples revealed from
a random permutation of its existing samples, each policy distributes the same extra budget (extra samples per ligand on
average), the pick is MBR over the revealed samples and is scored by its true ligand RMSD. Averaged over ``draws``
permutations per ligand, giving each policy an expected RMSD per ligand; the policies are compared PAIRED OVER LIGANDS.

Gate (fixed in advance): at an extra budget of 2 samples per ligand on average, "ei" must beat "uniform" with a lower mean
expected RMSD, a one-sided paired Wilcoxon p < 0.10 and a positive drop-one mean (same bar as the MBR gate). It is a simulation
on at most five real samples per ligand, so it can only say whether directing samples helps within those pools.

Usage: python -m cypmode.validation.ei_sampling [--table CSV] [--draws 400]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cypmode.validation.pose_pools import mbr_pick, pool_disagreement

K0 = 2
BUDGETS = (1, 2)
PRIMARY_BUDGET = 2


def simulate(matrices: dict[str, np.ndarray], rmsd: dict[str, np.ndarray], extra: int, policy: str, rng: np.random.Generator) -> dict[str, float]:
    """One draw: true RMSD of each ligand's MBR pick after the policy has spent ``extra`` x ligands samples."""
    names = list(matrices)
    order = {n: rng.permutation(len(rmsd[n])) for n in names}
    revealed = {n: min(K0, len(order[n])) for n in names}
    budget = extra * len(names)
    if policy == "uniform":
        for n in names:
            revealed[n] = min(len(order[n]), revealed[n] + extra)
    else:
        while budget > 0:
            open_ligands = [n for n in names if revealed[n] < len(order[n])]
            if not open_ligands:
                break
            if policy == "random":
                n = open_ligands[int(rng.integers(len(open_ligands)))]
            else:  # "ei": the pool that still disagrees most
                n = max(open_ligands, key=lambda m: (pool_disagreement(matrices[m], list(order[m][: revealed[m]])), rng.random()))
            revealed[n] += 1
            budget -= 1
    return {n: float(rmsd[n][mbr_pick(matrices[n], list(order[n][: revealed[n]]))]) for n in names}


def expected_rmsd(matrices, rmsd, extra: int, policy: str, draws: int, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    runs = pd.DataFrame([simulate(matrices, rmsd, extra, policy, rng) for _ in range(draws)])
    return runs.mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default=None)
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--out", default="data/structure_track/outputs/cyp3a4/ei_sampling_outcomes.csv")
    args = parser.parse_args()
    from cypmode.validation.pose_pools import pairwise_lddt, pool_paths
    from cypmode.validation.selection_validation import (
        DEFAULT_TABLES,
        P_VALUE,
        paired_gate,
    )

    path = args.table or next((t for t in DEFAULT_TABLES if Path(t).exists()), None)
    if path is None:
        raise SystemExit("no pose table found (run job 160 first)")
    table = pd.read_csv(path)
    truth = table[table.engine == "boltz"].set_index(["compound", "sample"])["true_rmsd"]
    matrices, rmsd = {}, {}
    for compound in sorted(table["compound"].unique()):
        poses = [(s, p) for e, s, p in pool_paths(compound, ("boltz",)) if p.exists() and not np.isnan(truth.get((compound, s), np.nan))]
        if len(poses) > K0:
            matrices[compound] = pairwise_lddt([p for _, p in poses])
            rmsd[compound] = np.array([truth[(compound, s)] for s, _ in poses])
    print(f"{len(matrices)} ligands with pools of {sorted({len(v) for v in rmsd.values()})} samples; k0 = {K0}, {args.draws} permutations per ligand")
    result = {}
    for extra in BUDGETS:
        cols = {policy: expected_rmsd(matrices, rmsd, extra, policy, args.draws) for policy in ("uniform", "random", "ei")}
        frame = pd.DataFrame(cols)
        result[extra] = frame
        print(f"\nextra budget {extra} samples per ligand on average: mean expected RMSD of the MBR pick  " + "  ".join(f"{p} {frame[p].mean():.3f}" for p in frame))
        g = paired_gate(frame["uniform"], frame["ei"])
        print(f"  ei vs uniform: mean improvement {g['mean_improvement']:+.3f} A, {g['wins']}/{g['n']} ligands better, Wilcoxon p {g['p']:.3f}, worst drop-one mean {g['min_drop_one']:+.3f}")
        if extra == PRIMARY_BUDGET:
            primary = g
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.concat({f"extra{e}": f for e, f in result.items()}, axis=1).to_csv(args.out)
    print(f"STRUCTURE GATE ei_sampling (ei beats uniform at equal compute, paired, drop-one robust, p < {P_VALUE}): {'MET' if primary['met'] else 'NOT MET'}")


if __name__ == "__main__":
    main()

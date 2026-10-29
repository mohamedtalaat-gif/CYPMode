# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""SAR made explicit: matched-molecular-pair transformations (job 047_mmp_transfer).

Structure-activity relationship is usually stated as a transformation: replace one fragment of a molecule by another
at the same site and see how the activity changes. Two compounds form a matched pair when single-bond fragmentation
(RDKit ``rdMMPA``) gives them the same large constant part and different small variable parts (1-10 heavy atoms).
For every directed pair (n -> q) the transformation is the string "variable_n>>variable_q".

For a query compound q the model looks at the labelled training compounds n that form a matched pair with it and asks,
for each, how that same transformation moved the label across ALL other training pairs (a Free-Wilson-like table):
    delta_T = sum over training pairs a -> b with transformation T of (y_b - y_a) / (count_T + 3)      (shrunk toward 0)
and returns five features: number of matched partners, their mean label, mean (y_n + delta_T) clipped to [0, 1],
mean delta_T and the number of partners whose transformation has >= 3 supporting pairs. LightGBM then combines them
with the ordinary descriptors (family ``lgbm_mmp``).

Leak safety: the table and the partner labels come from the TRAINING rows only; for a training row the pairs it takes
part in are removed from the table before its own features are computed (leave-one-compound-out), exactly as the
kNN label features do, so an evaluation row never sees its own label and a training row never sees its own either.

Prior art: matched molecular pair analysis (Kenny and Sadowski; Hussain and Rea, 2010) and Free-Wilson analysis. The
claim here is limited to the gate: base stack + MMP candidates vs base stack with corrected p < 0.05 at both z0.

Usage: python -m cypmode.challenge.mmp_transfer --isoform CYP3A4 --repeats 3 --run
"""

import argparse
import collections
import hashlib
from pathlib import Path

import numpy as np

SHRINK = 3.0
MAX_VARIABLE_HEAVY_ATOMS = 10
CACHE = Path("data/tdi_track/outputs/caches/mmp_pairs.npz")
FEATURE_NAMES = ["mmp_n_partners", "mmp_mean_partner_label", "mmp_mean_predicted_label", "mmp_mean_delta", "mmp_n_supported"]


def _heavy(fragment: str) -> int:
    from rdkit import Chem

    return Chem.MolFromSmiles(fragment.replace("[*:1]", "*")).GetNumHeavyAtoms()  # the attachment dummy is not a heavy atom


def build_pairs(smiles: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Directed matched pairs (source row, target row, transformation id) and the transformation strings."""
    from rdkit import Chem
    from rdkit.Chem import rdMMPA

    buckets: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        try:
            fragmentations = rdMMPA.FragmentMol(mol, maxCuts=1, maxCutBonds=30, resultsAsMols=False)
        except Exception:
            continue
        for _, chains in fragmentations:
            parts = chains.split(".")
            if len(parts) != 2:
                continue
            sizes = [_heavy(p) for p in parts]
            if min(sizes) < 1 or min(sizes) > MAX_VARIABLE_HEAVY_ATOMS:
                continue
            constant, variable = (parts[0], parts[1]) if sizes[0] >= sizes[1] else (parts[1], parts[0])
            buckets[constant].append((i, variable))
    transformations: dict[str, int] = {}
    seen: set[tuple[int, int, int]] = set()
    for items in buckets.values():
        by_variable: dict[str, list[int]] = collections.defaultdict(list)
        for row, variable in items:
            by_variable[variable].append(row)
        for va, rows_a in by_variable.items():
            for vb, rows_b in by_variable.items():
                if va == vb:
                    continue
                tid = transformations.setdefault(f"{va}>>{vb}", len(transformations))
                seen.update((a, b, tid) for a in rows_a for b in rows_b if a != b)
    arr = np.array(sorted(seen), dtype=np.int64).reshape(-1, 3)
    return arr[:, 0], arr[:, 1], arr[:, 2], list(transformations)


def get_pairs(smiles: list[str], cache: Path = CACHE):
    """Label-free, so cached on disk keyed by the compound list."""
    key = hashlib.md5("\n".join(smiles).encode()).hexdigest()
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if str(z["key"]) == key:
            return z["a"], z["b"], z["t"], list(z["names"])
    a, b, t, names = build_pairs(smiles)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, key=key, a=a, b=b, t=t, names=np.array(names, dtype=object))
    return a, b, t, names


def mmp_label_features(pairs, n_all: int, train_rows: np.ndarray, y_train: np.ndarray, query_rows: np.ndarray, leave_one_out: bool) -> np.ndarray:
    """(n_query x 5) features described in the module docstring; NaN where a compound has no matched partner."""
    a, b, t, names = pairs
    n_t = max(len(names), 1)
    y = np.full(n_all, np.nan)
    y[train_rows] = y_train
    in_train = np.zeros(n_all, dtype=bool)
    in_train[train_rows] = True
    both = in_train[a] & in_train[b]
    delta_all = y[b[both]] - y[a[both]]
    total = np.bincount(t[both], weights=delta_all, minlength=n_t)
    count = np.bincount(t[both], minlength=n_t).astype(float)

    position = np.full(n_all, -1, dtype=np.int64)
    position[query_rows] = np.arange(len(query_rows))
    use = in_train[a] & (position[b] >= 0)
    pa, pb, pt = a[use], b[use], t[use]
    out = np.full((len(query_rows), 5), np.nan)
    if not len(pa):
        out[:, 0] = 0.0
        out[:, 4] = 0.0
        return out
    s_adj, c_adj = total[pt].copy(), count[pt].copy()
    if leave_one_out:
        own = in_train[pb]  # the query's own pairs leave the table (its label is part of them)
        key = pb * n_t + pt
        _, inverse = np.unique(key, return_inverse=True)
        d = y[pb] - y[pa]
        sums = np.bincount(inverse, weights=np.where(own, d, 0.0))
        counts = np.bincount(inverse, weights=own.astype(float))
        s_adj = s_adj - sums[inverse]
        c_adj = c_adj - counts[inverse]
    delta = s_adj / (c_adj + SHRINK)
    q = position[pb]
    n_partners = np.bincount(q, minlength=len(query_rows)).astype(float)
    partner = y[pa]
    predicted = np.clip(partner + delta, 0.0, 1.0)
    supported = (c_adj >= 3).astype(float)
    has = n_partners > 0
    for col, values in ((1, partner), (2, predicted), (3, delta)):
        out[has, col] = (np.bincount(q, weights=values, minlength=len(query_rows)) / np.maximum(n_partners, 1))[has]
    out[:, 0] = n_partners
    out[:, 4] = np.bincount(q, weights=supported, minlength=len(query_rows))
    return out


def transformation_report(pairs, train_rows: np.ndarray, y_train: np.ndarray, top: int = 15, min_support: int = 5) -> list[tuple[str, int, float]]:
    """The best-supported transformations with their mean label change (the SAR the model actually learned)."""
    a, b, t, names = pairs
    y = np.full(max(int(a.max(initial=0)), int(b.max(initial=0)), int(train_rows.max(initial=0))) + 1, np.nan)
    y[train_rows] = y_train
    keep = ~np.isnan(y[a]) & ~np.isnan(y[b])
    n_t = max(len(names), 1)
    total = np.bincount(t[keep], weights=y[b[keep]] - y[a[keep]], minlength=n_t)
    count = np.bincount(t[keep], minlength=n_t)
    rows = [(names[i], int(count[i]), float(total[i] / (count[i] + SHRINK))) for i in range(len(names)) if count[i] >= min_support]
    return sorted(rows, key=lambda r: -abs(r[2]) * np.sqrt(r[1]))[:top]


def main() -> None:
    from cypmode.challenge.addition_experiment import (
        STORE,
        TEST_CSV,
        TRAIN_CSV,
        evaluate_addition,
    )
    from cypmode.challenge.nested_runners import build_context_and_plan, mmp_candidates

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isoform", required=True, choices=["CYP3A4", "CYP2D6"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    candidates = mmp_candidates()
    alone = [("lgbm_A_deep", "lgbm_mmp_A"), ("lgbm_A+fps_deep", "lgbm_mmp_A+fps")]
    ctx, plan = build_context_and_plan(TRAIN_CSV, TEST_CSV, args.isoform, STORE, args.repeats, 5, 4, 0, args.n_jobs, tuple(sorted({b for c in candidates for b in c.blocks})))
    pairs = get_pairs(ctx.store.smiles)
    train_idx = plan.pool
    own = mmp_label_features(pairs, len(ctx.store.smiles), train_idx, ctx.labels[train_idx], train_idx, True)
    test = mmp_label_features(pairs, len(ctx.store.smiles), train_idx, ctx.labels[train_idx], plan.external, False)
    print(f"\n{args.isoform}: {len(pairs[0])} directed matched pairs among {len(ctx.store.smiles)} compounds; "
          f"pool compounds with a matched partner in the pool {np.mean(own[:, 0] > 0):.3f}; test compounds with a partner in the pool {np.mean(test[:, 0] > 0):.3f}")
    print("best-supported transformations (shrunk mean label change n -> q, training pairs):")
    for name, n, delta in transformation_report(pairs, train_idx, ctx.labels[train_idx]):
        print(f"  {name:60s} n={n:3d}  delta={delta:+.3f}")
    evaluate_addition(args.isoform, candidates, args.repeats, args.n_jobs, args.run, alone, diagnostics=tuple(c.name for c in candidates))


if __name__ == "__main__":
    main()

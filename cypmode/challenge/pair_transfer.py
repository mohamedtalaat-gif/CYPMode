# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""SAR-aware pair-conditional label transfer (job 046_pair_conditional_transfer).

A nearest-neighbour model assumes that a similar compound has a similar label and cannot tell which differences
matter. SAR is exactly that knowledge: a methyl added next to an amine may flip TDI, a chlorine on a distant ring does
not. This model learns it from the data. For every compound q and each of its K most similar labelled training
compounds n it forms a pair row

    [ descriptors of q , descriptors of q minus descriptors of n , Tanimoto(q, n), label of n, rank of n ]

and fits one LightGBM on the target "label of q". At prediction time the K pair predictions of a new compound are
averaged with similarity weights. When the delta block includes the xTB physics block the differences are physical
(a lower hydrogen-abstraction energy at the best site, a longer binder-to-site distance, a different proton
affinity), so the model can learn which physical change flips the label: physics-aware SAR.

Training pairs use only training rows (neighbours of a training compound exclude itself); an evaluation compound only
ever looks up labels of training compounds, so nothing leaks. Related ideas exist (matched molecular pair analysis,
Free-Wilson, siamese and activity-cliff networks); the claim here is limited to what the gate below measures.

GATE: base stack + pair candidates vs base stack (StackedLogit), corrected p < 0.05 at both z0 (addition_experiment).

Usage: python -m cypmode.challenge.pair_transfer --isoform CYP3A4 --repeats 3 --run [--with-physics]
"""

import numpy as np

PAIR_K = 5


def neighbour_table(similarity: np.ndarray, query: np.ndarray, reference: np.ndarray, k: int, exclude_self: bool) -> tuple[np.ndarray, np.ndarray]:
    """Per query the ``k`` most similar reference rows (positions into ``reference``) and their similarities, best first."""
    sub = similarity[np.ix_(query, reference)].copy()
    if exclude_self:
        np.fill_diagonal(sub, -1.0)
    k = min(k, sub.shape[1] - (1 if exclude_self else 0))
    top = np.argpartition(-sub, k - 1, axis=1)[:, :k]
    sims = np.take_along_axis(sub, top, axis=1)
    order = np.argsort(-sims, axis=1)
    return np.take_along_axis(top, order, axis=1), np.take_along_axis(sims, order, axis=1)


def pair_matrix(x_query: np.ndarray, x_reference: np.ndarray, y_reference: np.ndarray, top: np.ndarray, sims: np.ndarray) -> np.ndarray:
    """Pair rows, ordered query-major: row ``i * k + j`` is query i with its j-th neighbour."""
    n, k = top.shape
    x_q = np.repeat(x_query.astype(np.float32), k, axis=0)
    neighbour = x_reference[top.reshape(-1)].astype(np.float32)
    rank = np.tile(np.arange(k, dtype=np.float32), n)
    return np.column_stack([x_q, x_q - neighbour, sims.reshape(-1), y_reference[top.reshape(-1)], rank]).astype(np.float32)


def fit_predict_pairs(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, similarity_tt: np.ndarray, similarity_et: np.ndarray, k: int, params: dict, n_jobs: int, seed: int) -> np.ndarray:
    """Fit on training pairs, score evaluation compounds by the similarity-weighted mean of their pair predictions."""
    from lightgbm import LGBMClassifier

    n = len(y_train)
    top_t, sims_t = neighbour_table(similarity_tt, np.arange(n), np.arange(n), k, True)
    pairs_t = pair_matrix(x_train, x_train, y_train, top_t, sims_t)
    model = LGBMClassifier(verbose=-1, n_jobs=n_jobs, random_state=seed, **params).fit(pairs_t, np.repeat(y_train, top_t.shape[1]))
    top_e, sims_e = neighbour_table(similarity_et, np.arange(len(x_eval)), np.arange(n), k, False)
    p = model.predict_proba(pair_matrix(x_eval, x_train, y_train, top_e, sims_e))[:, 1].reshape(top_e.shape)
    weights = np.maximum(sims_e, 1e-6)
    return (weights * p).sum(axis=1) / weights.sum(axis=1)


def main() -> None:
    import argparse

    from cypmode.challenge.addition_experiment import evaluate_addition
    from cypmode.challenge.nested_runners import pair_candidates

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isoform", required=True, choices=["CYP3A4", "CYP2D6"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--run", action="store_true", help="fill the prediction cache for the pair candidates first")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--with-physics", action="store_true", help="also run the pair model whose deltas include the xTB physics block")
    args = parser.parse_args()
    evaluate_addition(args.isoform, pair_candidates(args.with_physics), args.repeats, args.n_jobs, args.run, diagnostics=tuple(c.name for c in pair_candidates(True)))


if __name__ == "__main__":
    main()

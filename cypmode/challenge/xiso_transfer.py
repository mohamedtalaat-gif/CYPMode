# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Cross-isoform transfer of the assay arms (job 076_xiso_arm_transfer).

The label of a compound on isoform k is a function of two assay arms, t (TDI condition) and d (direct inhibition). Earlier work looked
only at the correlation of the SHIFT (t - d) between isoforms, which is near zero, and dropped the idea. The arms themselves are
potencies, and potency on one P450 says something about potency on another (shared lipophilicity, basicity, size), so what is
tested here is whether the arms of the other isoforms, predicted from structure, help the classifier of the target isoform.

How a candidate is built (family ``lgbm_xiso``), for one plan task:

  * stage one: for every other isoform j and each arm, a LightGBM regressor on the 2D features (feature set A) predicts the floor-clipped
    pIC50 (clipping at 4 leaves every label unchanged, see arms_experiment). Rows measured on j are cross-fitted (5 folds), so a
    training compound never receives a prediction from a model that saw its own measurement; every other row (unmeasured, or
    evaluation) receives the prediction of the model fitted on all allowed rows.
  * allowed rows: the training rows of the task, plus every compound outside the target's pool (it is never an evaluation row of this
    plan). The other-isoform measurements of an evaluation row, and of a pool row held out of the task, are never read.
  * stage two: the target-isoform classifier (deep LightGBM) on the 2D features plus the six predicted arms.

At deployment the same construction runs with all training compounds as allowed rows; the blinded test structures have no
measurement on any isoform, so they receive only predictions.

  alone:    xiso_X_deep vs lgbm_X_deep (best-single outer-fold AUC / AP);
  stacked:  default stack + the cross-isoform candidates vs default stack.
GATE: corrected p < 0.05 and a positive mean AUC difference at both z0, alone and stacked (addition_experiment.py).

Usage: python -m cypmode.challenge.xiso_transfer --isoform CYP3A4 --repeats 3 --run
"""

import numpy as np
import pandas as pd

from cypmode.challenge.addition_experiment import evaluate_addition, parse_args
from cypmode.challenge.nested import Task
from cypmode.challenge.nested_runners import (
    ISOFORM_ORDER,
    Context,
    _design,
    _labels_for,
    xiso_candidates,
)
from cypmode.challenge.tdi_targets import ASSAY_FLOOR

ARMS = ("tdi", "direct")
STAGE_ONE_BLOCKS = ("morgan_c2", "rdkit_desc")
STAGE_ONE = {"n_estimators": 120, "learning_rate": 0.06, "num_leaves": 15, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.3, "min_child_samples": 10}
MIN_ROWS = 30
FOLDS = 5
ALONE_PAIRS = [("lgbm_A_deep", "xiso_A_deep"), ("lgbm_A+fps_deep", "xiso_A+fps_deep")]

_MEMO: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}  # stage-one features per task, shared by the candidates that read the same blocks


def other_isoforms(isoform: str) -> list[str]:
    return [j for j in ISOFORM_ORDER if j != isoform]


def feature_names(isoform: str) -> list[str]:
    return [f"xiso_{j}_{arm}" for j in other_isoforms(isoform) for arm in ARMS]


def allowed_rows(labels: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Rows whose other-isoform measurements stage one may read: the task's training rows and every compound outside the pool."""
    allowed = labels < 0
    allowed[train] = True
    return allowed


def stage_one_features(ctx: Context, task: Task, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Predicted arms of the other isoforms for the training rows (cross-fitted where measured) and for the evaluation rows."""
    if ctx.xiso_arms is None or ctx.isoform is None:
        raise ValueError("cross-isoform candidates need Context.xiso_arms and Context.isoform")
    key = (task.train.tobytes(), task.evaluate.tobytes(), ctx.isoform, seed)
    if key in _MEMO:
        return _MEMO[key]
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import KFold

    allowed = allowed_rows(ctx.labels, task.train)
    in_train = np.zeros(len(ctx.labels), dtype=bool)
    in_train[task.train] = True
    position = {int(r): p for p, r in enumerate(task.train)}
    train_cols, eval_cols = [], []
    for j in other_isoforms(ctx.isoform):
        for arm in ARMS:
            values = ctx.xiso_arms[f"{j}_{arm}"]
            fit_rows = np.flatnonzero(allowed & ~np.isnan(values))
            if len(fit_rows) < MIN_ROWS:
                fill = float(np.nanmean(values[fit_rows])) if len(fit_rows) else ASSAY_FLOOR
                train_cols.append(np.full(len(task.train), fill))
                eval_cols.append(np.full(len(task.evaluate), fill))
                continue
            target = np.maximum(values[fit_rows], ASSAY_FLOOR)
            x_fit = ctx.store.matrix(STAGE_ONE_BLOCKS, fit_rows)

            def fitted(rows: np.ndarray, x_fit: np.ndarray = x_fit, target: np.ndarray = target) -> LGBMRegressor:
                return LGBMRegressor(verbose=-1, n_jobs=ctx.n_jobs, random_state=seed % 2**31, **STAGE_ONE).fit(x_fit[rows], target[rows])

            everything = fitted(np.arange(len(fit_rows)))
            column_train = everything.predict(ctx.store.matrix(STAGE_ONE_BLOCKS, task.train))
            # a training compound measured on j must not be predicted by a model that saw that measurement
            own = np.flatnonzero(in_train[fit_rows])
            outside = np.flatnonzero(~in_train[fit_rows])
            for a, b in KFold(FOLDS, shuffle=True, random_state=seed % 2**31).split(own):
                model = fitted(np.concatenate([outside, own[a]]))
                held = fit_rows[own[b]]
                column_train[[position[int(r)] for r in held]] = model.predict(ctx.store.matrix(STAGE_ONE_BLOCKS, held))
            train_cols.append(column_train)
            eval_cols.append(everything.predict(ctx.store.matrix(STAGE_ONE_BLOCKS, task.evaluate)))
    out = np.column_stack(train_cols), np.column_stack(eval_cols)
    _MEMO[key] = out
    return out


def fit_predict_xiso(candidate, ctx: Context, task: Task, seed: int) -> np.ndarray:
    from lightgbm import LGBMClassifier

    y = _labels_for(ctx, task)
    train_x, eval_x = _design(candidate, ctx, task, y)
    extra_train, extra_eval = stage_one_features(ctx, task, seed)
    model = LGBMClassifier(verbose=-1, n_jobs=ctx.n_jobs, random_state=seed % 2**31, **candidate.params)
    return model.fit(np.hstack([train_x, extra_train]), y).predict_proba(np.hstack([eval_x, extra_eval]))[:, 1]


def correlation_table(train: pd.DataFrame, isoform: str) -> pd.DataFrame:
    """Spearman correlation of each arm (and of the shift) between the target and every other isoform, on compounds measured on both."""
    rows = []
    for j in other_isoforms(isoform):
        for label, col in (("tdi", "pIC50_TDI_condition"), ("direct", "pIC50_direct_inhibition")):
            pair = train[[f"{isoform}_{col}", f"{j}_{col}"]].dropna()
            rows.append({"other": j, "quantity": label, "n": len(pair), "spearman": pair.corr(method="spearman").iloc[0, 1] if len(pair) > 2 else np.nan})
        both = train[[f"{isoform}_pIC50_TDI_condition", f"{isoform}_pIC50_direct_inhibition", f"{j}_pIC50_TDI_condition", f"{j}_pIC50_direct_inhibition"]].dropna()
        shift = both.iloc[:, 0] - both.iloc[:, 1].clip(lower=ASSAY_FLOOR), both.iloc[:, 2] - both.iloc[:, 3].clip(lower=ASSAY_FLOOR)
        rows.append({"other": j, "quantity": "shift", "n": len(both), "spearman": shift[0].corr(shift[1], method="spearman") if len(both) > 2 else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args(__doc__)
    from cypmode.challenge.addition_experiment import TRAIN_CSV

    table = correlation_table(pd.read_csv(TRAIN_CSV), args.isoform)
    print(f"{args.isoform}: Spearman correlation with the other isoforms on compounds measured on both")
    print(table.round(3).to_string(index=False))
    evaluate_addition(args.isoform, xiso_candidates(), args.repeats, args.n_jobs, args.run, ALONE_PAIRS, outer=args.outer, root=args.root)


if __name__ == "__main__":
    main()

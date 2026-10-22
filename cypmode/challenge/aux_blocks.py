# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Candidates that use the official auxiliary readings (single-concentration screen, Emax) and the two-part form of the TDI label.

The blinded test compounds have no screen or Emax reading, so a reading can never be an input of the final model. It can still teach the model what
the structure says about it: a first-stage regressor is fitted on the compounds that have the reading and predicts it for every compound, and the
second-stage classifier of the TDI label receives those predictions next to the structure features (the same construction as the cross-isoform
transfer of ``xiso_transfer``). Inside a nested task the first stage may read only

  * the training rows of the task, and
  * every compound outside the plan's pool (never an evaluation row of this plan),

and a training row measured on the reading is predicted by a model that did not see it (5-fold cross-fitting); an evaluation row receives the prediction of
the model fitted on all allowed rows. The reading of an evaluation row, or of a pool row held out of the task, is never read.

Families
  ``lgbm_aux``         deep LightGBM on feature set A (or A+fps) plus predicted readings: ``screen`` (log2 fold change of the four enzymes), ``emax``
                       (maximal effect of both arms of the four enzymes) or ``auxall`` (both).
  ``lgbm_gate_shift``  the label is ``(t - d > log10 2) AND (t > 4.301)`` (t = TDI-arm pIC50, d = direct pIC50): the second condition is a statement about
                       the TDI arm alone and can be learned from every compound with a TDI-arm curve, including the compounds without a direct
                       curve that the pool leaves out; the first needs both arms. Parts: ``gate``, ``shift`` and their ``product``.
  ``lgbm_nodirect``    uses the compounds the pool leaves out because they have no direct curve (CYP3A4: 1,249 of them, every one labelled negative by convention).
                       The scored rule needs the direct pIC50 ``d`` (label = ``t - max(d, 4) > log10 2``), which these compounds do not have, but their TDI-arm pIC50
                       ``t`` is measured and the label implies ``t > 4.301``: a compound with ``t <= 4.301`` is a certain negative whatever ``d`` is, and one with a higher
                       ``t`` is positive with a probability that a first-stage model learns on the pool rows of the task (features: structure and ``t``). Mode ``neg`` adds only the
                       certain negatives; mode ``soft`` adds both, each ambiguous compound as a positive row with weight ``p`` and a negative row with weight ``1 - p``,
                       scaled by ``extra_weight``. ``t`` is a feature of the first stage only; the final classifier sees structure alone, as it must for the blinded compounds.
  ``lgbm_screen_two``  CYP2D6 selection-aware candidate whose gate is the real screen: P(screen-active on CYP2D6 | structure), trained on every screened
                       compound allowed in the task, times the TDI model fitted on the hits.
"""

from __future__ import annotations

import numpy as np

from cypmode.challenge.nested import Task
from cypmode.challenge.nested_runners import (
    FEATURE_SETS,
    LGBM_PRESETS,
    Candidate,
    Context,
    _design,
    _labels_for,
)
from cypmode.challenge.official_extra import ENZYMES
from cypmode.challenge.tdi_targets import ASSAY_FLOOR, SHIFT_THRESHOLD

AUX_SETS: dict[str, tuple[str, ...]] = {
    "screen": tuple(f"{e}_screen" for e in ENZYMES),
    "emax": tuple(f"{e}_emax_{arm}" for e in ENZYMES for arm in ("direct", "tdi")),
}
AUX_SETS["auxall"] = AUX_SETS["screen"] + AUX_SETS["emax"]
AUX_SETS["pxr"] = ("PXR_pEC50", "PXR_emax", "PXR_fc_8uM", "PXR_fc_33uM")
TAGS = {"screen": "scrblk", "emax": "emxblk", "auxall": "auxblk", "pxr": "pxrblk"}
STAGE_ONE_BLOCKS = ("morgan_c2", "rdkit_desc")
STAGE_ONE = {"n_estimators": 120, "learning_rate": 0.06, "num_leaves": 15, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.3, "min_child_samples": 10}
MIN_ROWS = 30
FOLDS = 5
GATE_T = ASSAY_FLOOR + SHIFT_THRESHOLD  # 4.301: a TDI-arm pIC50 above this is a potent TDI-arm curve

_MEMO: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}  # one predicted column per (task, reading), shared by every candidate that asks for it


def aux_candidates(keys: tuple[str, ...] = ("screen", "emax")) -> list[Candidate]:
    out = []
    for key in keys:
        tag = TAGS[key]
        for fs in ("A", "A+fps"):
            out.append(Candidate(f"{tag}_{fs}_deep", "lgbm_aux", FEATURE_SETS[fs], params={**LGBM_PRESETS["deep"], "aux": key}))
    return out


def gate_shift_candidates() -> list[Candidate]:
    out = []
    for fs in ("A", "A+fps"):
        for part, tag in (("gate", "tgate"), ("shift", "tshift"), ("product", "gs")):
            out.append(Candidate(f"{tag}_{fs}_deep", "lgbm_gate_shift", FEATURE_SETS[fs], params={**LGBM_PRESETS["deep"], "part": part}))
    return out


def nodirect_candidates() -> list[Candidate]:
    out = []
    for fs in ("A", "A+fps"):
        out.append(Candidate(f"ndneg_{fs}_deep", "lgbm_nodirect", FEATURE_SETS[fs], params={**LGBM_PRESETS["deep"], "mode": "neg", "extra_weight": 1.0}))
        out.append(Candidate(f"ndsoft_{fs}_deep", "lgbm_nodirect", FEATURE_SETS[fs], params={**LGBM_PRESETS["deep"], "mode": "soft", "extra_weight": 0.5}))
    return out


def screen_selection_candidates() -> list[Candidate]:
    """For the selection-aware CYP2D6 plan: the real-screen gate times the TDI-given-hit model, and the end-to-end classifier on the full-population labels with the predicted screen."""
    out = [Candidate(f"scr2_{fs}_deep", "lgbm_screen_two", FEATURE_SETS[fs], params={**LGBM_PRESETS["deep"], "enzyme": "CYP2D6"}) for fs in ("A", "A+fps")]
    out += [Candidate(f"popscr_{fs}_deep", "lgbm_aux", FEATURE_SETS[fs], params={**LGBM_PRESETS["deep"], "aux": "screen"}) for fs in ("A", "A+fps")]
    return out


def allowed_rows(labels: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Rows a first stage may read: the task's training rows and every compound outside the pool (labels < 0 holds exactly the non-pool rows)."""
    allowed = labels < 0
    allowed[train] = True
    return allowed


def predicted_reading(ctx: Context, task: Task, name: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The predicted reading ``name`` for the training rows (cross-fitted where measured) and for the evaluation rows of a task."""
    if ctx.aux is None:
        raise ValueError("auxiliary candidates need Context.aux (the official extra files)")
    key = (task.train.tobytes(), task.evaluate.tobytes(), ctx.isoform, seed, name)
    if key in _MEMO:
        return _MEMO[key]
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import KFold

    values = ctx.aux[name]
    allowed = allowed_rows(ctx.labels, task.train)
    fit_rows = np.flatnonzero(allowed & ~np.isnan(values))
    if len(fit_rows) < MIN_ROWS:
        fill = float(np.nanmean(values[fit_rows])) if len(fit_rows) else 0.0
        out = np.full(len(task.train), fill), np.full(len(task.evaluate), fill)
        _MEMO[key] = out
        return out
    x_fit = ctx.store.matrix(STAGE_ONE_BLOCKS, fit_rows)
    target = values[fit_rows]
    in_train = np.zeros(len(ctx.labels), dtype=bool)
    in_train[task.train] = True
    position = {int(r): p for p, r in enumerate(task.train)}

    def fitted(rows: np.ndarray) -> LGBMRegressor:
        return LGBMRegressor(verbose=-1, n_jobs=ctx.n_jobs, random_state=seed % 2**31, **STAGE_ONE).fit(x_fit[rows], target[rows])

    everything = fitted(np.arange(len(fit_rows)))
    column_train = everything.predict(ctx.store.matrix(STAGE_ONE_BLOCKS, task.train))
    own = np.flatnonzero(in_train[fit_rows])
    outside = np.flatnonzero(~in_train[fit_rows])
    for a, b in KFold(FOLDS, shuffle=True, random_state=seed % 2**31).split(own):
        model = fitted(np.concatenate([outside, own[a]]))
        held = fit_rows[own[b]]
        column_train[[position[int(r)] for r in held]] = model.predict(ctx.store.matrix(STAGE_ONE_BLOCKS, held))
    out = column_train, everything.predict(ctx.store.matrix(STAGE_ONE_BLOCKS, task.evaluate))
    _MEMO[key] = out
    return out


def fit_predict_aux(candidate: Candidate, ctx: Context, task: Task, seed: int) -> np.ndarray:
    from lightgbm import LGBMClassifier

    params = dict(candidate.params)
    names = AUX_SETS[params.pop("aux")]
    y = _labels_for(ctx, task)
    train_x, eval_x = _design(candidate, ctx, task, y)
    columns = [predicted_reading(ctx, task, n, seed) for n in names]
    extra_train, extra_eval = np.column_stack([c[0] for c in columns]), np.column_stack([c[1] for c in columns])
    model = LGBMClassifier(verbose=-1, n_jobs=ctx.n_jobs, random_state=seed % 2**31, **params)
    return model.fit(np.hstack([train_x, extra_train]), y).predict_proba(np.hstack([eval_x, extra_eval]))[:, 1]


def fit_predict_gate_shift(candidate: Candidate, ctx: Context, task: Task, seed: int) -> np.ndarray:
    """P(t > 4.301) x P(t - d > log10 2); the gate part reads every allowed compound with a TDI-arm curve, the shift part the training rows (both arms)."""
    from lightgbm import LGBMClassifier

    params = dict(candidate.params)
    part = params.pop("part")
    t, d = ctx.targets["tdi"], ctx.targets["direct"]
    allowed = allowed_rows(ctx.labels, task.train)

    def classifier(rows: np.ndarray, target: np.ndarray) -> np.ndarray:
        x_rows, x_eval = ctx.store.matrix(candidate.blocks, rows), ctx.store.matrix(candidate.blocks, task.evaluate)
        return LGBMClassifier(verbose=-1, n_jobs=ctx.n_jobs, random_state=seed % 2**31, **params).fit(x_rows, target).predict_proba(x_eval)[:, 1]

    gate = shift = None
    if part in ("gate", "product"):
        rows = np.flatnonzero(allowed & ~np.isnan(t))
        gate = classifier(rows, (t[rows] > GATE_T).astype(int))
    if part in ("shift", "product"):
        rows = task.train[~np.isnan(t[task.train]) & ~np.isnan(d[task.train])]
        shift = classifier(rows, (t[rows] - d[rows] > SHIFT_THRESHOLD).astype(int))
    return gate if part == "gate" else shift if part == "shift" else gate * shift


def nodirect_rows(ctx: Context) -> np.ndarray:
    """Compounds outside the pool with a TDI-arm pIC50 and no direct pIC50 (never a blinded test compound: those have neither)."""
    t, d = ctx.targets["tdi"], ctx.targets["direct"]
    return np.flatnonzero((ctx.labels < 0) & ~np.isnan(t) & np.isnan(d))


def fit_predict_nodirect(candidate: Candidate, ctx: Context, task: Task, seed: int) -> np.ndarray:
    from lightgbm import LGBMClassifier

    params = dict(candidate.params)
    mode, extra_weight = params.pop("mode"), float(params.pop("extra_weight"))
    t = ctx.targets["tdi"]
    y = _labels_for(ctx, task)
    train_x, eval_x = _design(candidate, ctx, task, y)
    extra = nodirect_rows(ctx)
    if len(extra) == 0:
        return LGBMClassifier(verbose=-1, n_jobs=ctx.n_jobs, random_state=seed % 2**31, **params).fit(train_x, y).predict_proba(eval_x)[:, 1]
    extra_x = ctx.store.matrix(candidate.blocks, extra)
    certain = t[extra] <= GATE_T
    blocks_x, blocks_y, blocks_w = [train_x, extra_x[certain]], [y, np.zeros(int(certain.sum()), dtype=int)], [np.ones(len(y)), np.ones(int(certain.sum()))]
    if mode == "soft" and (~certain).any():
        rows = np.flatnonzero(~certain)
        imputer = LGBMClassifier(verbose=-1, n_jobs=ctx.n_jobs, random_state=seed % 2**31, **params)
        imputer.fit(np.hstack([train_x, t[task.train].reshape(-1, 1)]), y)  # fitted on the pool rows of this task only
        p = imputer.predict_proba(np.hstack([extra_x[rows], t[extra[rows]].reshape(-1, 1)]))[:, 1]
        blocks_x += [extra_x[rows], extra_x[rows]]
        blocks_y += [np.ones(len(rows), dtype=int), np.zeros(len(rows), dtype=int)]
        blocks_w += [extra_weight * p, extra_weight * (1.0 - p)]
    model = LGBMClassifier(verbose=-1, n_jobs=ctx.n_jobs, random_state=seed % 2**31, **params)
    return model.fit(np.vstack(blocks_x), np.concatenate(blocks_y), sample_weight=np.concatenate(blocks_w)).predict_proba(eval_x)[:, 1]


def fit_predict_screen_two(candidate: Candidate, ctx: Context, task: Task, seed: int) -> np.ndarray:
    """P(screen-active on the enzyme | structure) x P(TDI label | hit): the gate learns from the real screen instead of from "has a curve"."""
    from lightgbm import LGBMClassifier

    if ctx.aux is None or ctx.hit is None:
        raise ValueError("screen-gated candidates need Context.aux and Context.hit (population plan)")
    params = dict(candidate.params)
    enzyme = params.pop("enzyme")
    active = ctx.aux[f"{enzyme}_screen_active"]
    rows = np.flatnonzero(allowed_rows(ctx.labels, task.train) & ~np.isnan(active))
    y = _labels_for(ctx, task)
    hit = ctx.hit[task.train].astype(bool)
    train_x, eval_x = _design(candidate, ctx, task, y)
    gate = LGBMClassifier(verbose=-1, n_jobs=ctx.n_jobs, random_state=seed % 2**31, **params).fit(ctx.store.matrix(candidate.blocks, rows), active[rows].astype(int))
    tdi = LGBMClassifier(verbose=-1, n_jobs=ctx.n_jobs, random_state=seed % 2**31, **params).fit(train_x[hit], y[hit])
    return gate.predict_proba(eval_x)[:, 1] * tdi.predict_proba(eval_x)[:, 1]

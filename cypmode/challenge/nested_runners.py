# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Turn candidate model specifications into cached predictions for a NestedPlan.

A ``Candidate`` is data (a model family, the feature blocks it reads, its
hyperparameters); ``fit_predict`` executes one plan task for it: train on
``task.train``, return a score per ``task.evaluate`` row. Every runner is
leakage-checked by construction: the label vector a candidate may read is
restricted to the task's training rows before it is touched (see
``_labels_for``), and the tests corrupt the evaluation labels to prove it.

The candidate list is a grid of declarative axes (feature sets x hyperparameter
presets x model families), not a hand-picked handful, and nothing here decides
which candidate wins: cypmode.challenge.nested does that from inner
out-of-fold scores alone.

This module deliberately imports no torch, so it runs in the LightGBM
environment; torch/Chemprop/DeepChem runners write to the same PredictionCache
from their own environments.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cypmode.challenge.feature_store import FeatureStore
from cypmode.challenge.grouping import butina_groups
from cypmode.challenge.label_confidence import confidence_z, training_weights
from cypmode.challenge.nested import NestedPlan, PredictionCache, Task
from cypmode.challenge.tdi_targets import ASSAY_FLOOR, BOTH_ARMS, training_view
from cypmode.challenge.validation_matching import tanimoto_matrix

logger = logging.getLogger(__name__)

# -- the declarative grid ------------------------------------------------------------------------

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "desc": ("rdkit_desc",),
    "A": ("morgan_c2", "rdkit_desc"),
    "A+fps": ("morgan_c2", "rdkit_desc", "motif", "maccs", "avalon", "torsion_c", "fmorgan_c2", "erg"),
    "A+ext": ("morgan_c2", "rdkit_desc", "ext_activity"),
    "A+fps+ext+alerts": ("morgan_c2", "rdkit_desc", "motif", "maccs", "avalon", "torsion_c", "fmorgan_c2", "erg", "ext_activity", "tdi_alerts"),
    "A+fps+ext": ("morgan_c2", "rdkit_desc", "motif", "maccs", "avalon", "torsion_c", "fmorgan_c2", "erg", "ext_activity"),
}

LGBM_PRESETS: dict[str, dict] = {
    "deep": {"n_estimators": 300, "learning_rate": 0.04, "num_leaves": 31, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.3, "min_child_samples": 10},
    "shallow": {
        "n_estimators": 600,
        "learning_rate": 0.02,
        "num_leaves": 8,
        "subsample": 0.7,
        "subsample_freq": 1,
        "colsample_bytree": 0.2,
        "min_child_samples": 20,
        "reg_lambda": 5.0,
    },
}

SVM_C = (0.3, 1.0, 3.0)
KNN_K = (5, 15, 40)
KNN_FEATURE_K = 10


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    blocks: tuple[str, ...] = ()
    knn_label_features: bool = False
    params: dict = field(default_factory=dict, hash=False, compare=False)
    weighting: str = ""  # how training rows are weighted by label confidence, see label_confidence.training_weights
    hyp: str = ""  # "model": the ligand-based pharmacophore hypothesis model of pharmacophore_hypotheses is fitted inside the task and its columns are added; "permuted": the same on permuted labels (the control)


def default_candidates() -> list[Candidate]:
    """Ordered from simplest to most complex (the one-standard-error selector breaks ties by order)."""
    out: list[Candidate] = [Candidate(f"knn_tanimoto_k{k}", "knn_tanimoto", params={"k": k}) for k in KNN_K]
    out += [Candidate(f"svm_tanimoto_C{c:g}", "svm_tanimoto", params={"C": c}) for c in SVM_C]
    for fs_name, blocks in FEATURE_SETS.items():
        for preset, params in LGBM_PRESETS.items():
            if "ext" in fs_name and preset != "deep":
                continue
            out.append(Candidate(f"lgbm_{fs_name}_{preset}", "lgbm_clf", blocks, params=params))
    out += [
        Candidate("extratrees_A", "extratrees", FEATURE_SETS["A"]),
        Candidate("rf_A", "rf", FEATURE_SETS["A"]),
    ]
    for fs_name in ("A", "A+fps"):
        out.append(Candidate(f"lgbm_{fs_name}_deep_knnfeat", "lgbm_clf", FEATURE_SETS[fs_name], True, LGBM_PRESETS["deep"]))
        out.append(Candidate(f"lgbm_reg_shift_{fs_name}", "lgbm_reg_shift", FEATURE_SETS[fs_name], params=LGBM_PRESETS["deep"]))
        out.append(Candidate(f"lgbm_reg_arms_{fs_name}", "lgbm_reg_arms", FEATURE_SETS[fs_name], params=LGBM_PRESETS["deep"]))
    return out


ISOFORM_ORDER = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
WEIGHTINGS = ("drop1.0", "soft")
CONTEXT_SETS: dict[str, tuple[str, ...]] = {"A+ctx": FEATURE_SETS["A"] + ("tdi_alert_context",), "A+fps+ctx": FEATURE_SETS["A+fps"] + ("tdi_alert_context",)}


PHYSICS_SETS: dict[str, tuple[str, ...]] = {"A+phys": FEATURE_SETS["A"] + ("physics",), "A+fps+phys": FEATURE_SETS["A+fps"] + ("physics",)}


SELECT_STAGE = {"n_estimators": 150, "learning_rate": 0.06, "num_leaves": 15, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.5, "min_child_samples": 10}


def compact_physics_candidates(with_tier_b: bool = False, k: int = 20) -> list[Candidate]:
    """Deep LightGBM on A / A+fps plus only the ``k`` physics columns that matter, chosen inside every task from its training rows."""
    pool = ("physics", "physics_b") if with_tier_b else ("physics",)
    tag = "physkb" if with_tier_b else "physk"
    out = []
    for fs in ("A", "A+fps"):
        base = FEATURE_SETS[fs]
        out.append(Candidate(f"lgbm_{fs}+{tag}{k}_deep", "lgbm_clf_select", base + pool, params={"base_blocks": base, "select_blocks": pool, "select_k": k, "lgbm": LGBM_PRESETS["deep"]}))
    return out


BASIC_SETS: dict[str, tuple[str, ...]] = {"A+bc": FEATURE_SETS["A"] + ("basic_centre",), "A+fps+bc": FEATURE_SETS["A+fps"] + ("basic_centre",)}


def basic_centre_candidates() -> list[Candidate]:
    """Deep LightGBM on feature set A / A+fps plus the basic-centre block (protonation states and charge-to-ring geometry)."""
    return [Candidate(f"lgbm_{fs}_deep", "lgbm_clf", blocks, params=LGBM_PRESETS["deep"]) for fs, blocks in BASIC_SETS.items()]


EXTC_SETS: dict[str, tuple[str, ...]] = {"A+ext+xc": FEATURE_SETS["A+ext"] + ("ext_cont",), "A+fps+ext+xc": FEATURE_SETS["A+fps+ext"] + ("ext_cont",)}


AJ_SETS: dict[str, tuple[str, ...]] = {"A+aj": FEATURE_SETS["A"] + ("atomjepa",), "A+fps+aj": FEATURE_SETS["A+fps"] + ("atomjepa",), "aj": ("atomjepa",)}


def atomjepa_candidates() -> list[Candidate]:
    """Deep LightGBM on the frozen Atom-JEPA block alone and added to feature sets A and A+fps."""
    return [Candidate(f"lgbm_{fs}_deep", "lgbm_clf", blocks, params=LGBM_PRESETS["deep"]) for fs, blocks in AJ_SETS.items()]


ADME_SETS: dict[str, tuple[str, ...]] = {"A+ad": FEATURE_SETS["A"] + ("ext_adme",), "A+fps+ad": FEATURE_SETS["A+fps"] + ("ext_adme",)}


def ext_adme_candidates() -> list[Candidate]:
    """Deep LightGBM on feature sets A and A+fps plus the logD and the mouse brain protein binding predicted by OpenADMET's surrogate models (``openadmet_surrogates``)."""
    return [Candidate(f"lgbm_{fs}_deep", "lgbm_clf", blocks, params=LGBM_PRESETS["deep"]) for fs, blocks in ADME_SETS.items()]


PP3A4_SETS: dict[str, tuple[str, ...]] = {"A+pp3a4": FEATURE_SETS["A"] + ("pp3a4",), "A+fps+pp3a4": FEATURE_SETS["A+fps"] + ("pp3a4",)}
PP2D6_SETS: dict[str, tuple[str, ...]] = {"A+pp2d6": FEATURE_SETS["A"] + ("pp2d6",), "A+fps+pp2d6": FEATURE_SETS["A+fps"] + ("pp2d6",)}


def pocket_cyp3a4_candidates() -> list[Candidate]:
    """Deep LightGBM on feature sets A and A+fps plus the compound relative to the CYP3A4 pocket (``pocket_pharmacophore``)."""
    return [Candidate(f"lgbm_{fs}_deep", "lgbm_clf", blocks, params=LGBM_PRESETS["deep"]) for fs, blocks in PP3A4_SETS.items()]


ST3A4_SETS: dict[str, tuple[str, ...]] = {"A+st3a4": FEATURE_SETS["A"] + ("st3a4",), "A+fps+st3a4": FEATURE_SETS["A+fps"] + ("st3a4",)}
ST2D6_SETS: dict[str, tuple[str, ...]] = {"A+st2d6": FEATURE_SETS["A"] + ("st2d6",), "A+fps+st2d6": FEATURE_SETS["A+fps"] + ("st2d6",)}


def site_cyp3a4_candidates() -> list[Candidate]:
    """Deep LightGBM on feature sets A and A+fps plus the structure-based site model of the CYP3A4 pocket (``pocket_pharmacophore``, the six columns ``st_cyp3a4_*``)."""
    return [Candidate(f"lgbm_{fs}_deep", "lgbm_clf", blocks, params=LGBM_PRESETS["deep"]) for fs, blocks in ST3A4_SETS.items()]


def site_cyp2d6_candidates() -> list[Candidate]:
    """The same with the CYP2D6 pocket; on CYP2D6 it is the addition, on CYP3A4 it is the control of the site model."""
    return [Candidate(f"lgbm_{fs}_deep", "lgbm_clf", blocks, params=LGBM_PRESETS["deep"]) for fs, blocks in ST2D6_SETS.items()]


def hypothesis_candidates(permuted: bool = False) -> list[Candidate]:
    """Deep LightGBM on feature sets A and A+fps plus the five columns of the ligand-based hypothesis model (``pharmacophore_hypotheses``), fitted inside every task; ``permuted`` is the control whose
    model is fitted on labels permuted within the training rows."""
    tag = "hypperm" if permuted else "hyp"
    return [Candidate(f"lgbm_{fs}+{tag}_deep", "lgbm_clf", FEATURE_SETS[fs], params=LGBM_PRESETS["deep"], hyp="permuted" if permuted else "model") for fs in ("A", "A+fps")]


def pocket_cyp2d6_candidates() -> list[Candidate]:
    """The same with the CYP2D6 pocket; on CYP2D6 it is the addition, on CYP3A4 it is the control of the pocket block."""
    return [Candidate(f"lgbm_{fs}_deep", "lgbm_clf", blocks, params=LGBM_PRESETS["deep"]) for fs, blocks in PP2D6_SETS.items()]


EXTT_SETS: dict[str, tuple[str, ...]] = {"A+ext+xt": FEATURE_SETS["A+ext"] + ("ext_tox",), "A+fps+ext+xt": FEATURE_SETS["A+fps+ext"] + ("ext_tox",)}


def ext_tox_candidates() -> list[Candidate]:
    """Deep LightGBM on the sets that carry the binary external block, plus the Tox21 CYP-screen readings predicted by models trained on Tox21 compounds only (``external_continuous``)."""
    return [Candidate(f"lgbm_{fs}_deep", "lgbm_clf", blocks, params=LGBM_PRESETS["deep"]) for fs, blocks in EXTT_SETS.items()]


def ext_cont_candidates() -> list[Candidate]:
    """Deep LightGBM on the sets that already carry the binary external-panel block, plus the continuous responses and potencies of the same panel (``external_continuous``)."""
    return [Candidate(f"lgbm_{fs}_deep", "lgbm_clf", blocks, params=LGBM_PRESETS["deep"]) for fs, blocks in EXTC_SETS.items()]


MONROE_SETS: dict[str, tuple[str, ...]] = {"A+mon": FEATURE_SETS["A"] + ("monroe",), "A+fps+mon": FEATURE_SETS["A+fps"] + ("monroe",), "mon": ("monroe",)}


def monroe_candidates() -> list[Candidate]:
    """Deep LightGBM on the frozen Monroe embedding block alone and added to feature sets A and A+fps."""
    return [Candidate(f"lgbm_{fs}_deep", "lgbm_clf", blocks, params=LGBM_PRESETS["deep"]) for fs, blocks in MONROE_SETS.items()]


def physics_b_candidates() -> list[Candidate]:
    """Tier-B physics (Boltzmann conformers, relaxed hydrogen abstraction) alone, with A / A+fps, and with tier A."""
    sets = {"physB": ("physics_b",), "A+physB": FEATURE_SETS["A"] + ("physics_b",), "A+fps+physB": FEATURE_SETS["A+fps"] + ("physics_b",), "A+phys+physB": FEATURE_SETS["A"] + ("physics", "physics_b")}
    return [Candidate(f"lgbm_{name}_deep", "lgbm_clf", blocks, params=LGBM_PRESETS["deep"]) for name, blocks in sets.items()]


def population_physics_candidates() -> list[Candidate]:
    """The four selection-aware families again, with the tier-A physics block added to A and A+fps (panel-block pool)."""
    out = []
    for fs in ("A+phys", "A+fps+phys"):
        for tag, family in (("popgate", "lgbm_gate"), ("pophits", "lgbm_hits_only"), ("poptwo", "lgbm_two_part"), ("pop", "lgbm_clf")):
            out.append(Candidate(f"{tag}_{fs}_deep", family, PHYSICS_SETS[fs], params=LGBM_PRESETS["deep"]))
    return out


def physics_candidates() -> list[Candidate]:
    """Deep LightGBM with the GFN2-xTB physics block added to feature sets A and A+fps (kept out of the default grid)."""
    sets = {"phys": ("physics",), **PHYSICS_SETS}  # physics alone is a diagnostic and a diverse committee member
    return [Candidate(f"lgbm_{name}_deep", "lgbm_clf", blocks, params=LGBM_PRESETS["deep"]) for name, blocks in sets.items()]


def embedding_candidates() -> list[Candidate]:
    """Deep LightGBM with the frozen CheMeleon block added to A and A+fps, plus the block alone as a diagnostic."""
    sets = {"chem": ("chemeleon",), "A+chem": FEATURE_SETS["A"] + ("chemeleon",), "A+fps+chem": FEATURE_SETS["A+fps"] + ("chemeleon",)}
    return [Candidate(f"lgbm_{name}_deep", "lgbm_clf", blocks, params=LGBM_PRESETS["deep"]) for name, blocks in sets.items()]


def testlike_candidates() -> list[Candidate]:
    """Deep LightGBM trained with test-likeness (density-ratio) sample weights, alpha 0.5 and 1 (kept out of the default grid)."""
    return [
        Candidate(f"lgbm_{fs}_deep_testlike{alpha}", "lgbm_clf", FEATURE_SETS[fs], params=LGBM_PRESETS["deep"], weighting=f"testlike{alpha}")
        for fs in ("A", "A+fps")
        for alpha in ("0.5", "1.0")
    ]


def mmp_candidates() -> list[Candidate]:
    """LightGBM with matched-molecular-pair transformation features (mmp_transfer.py)."""
    return [Candidate(f"lgbm_mmp_{fs}", "lgbm_mmp", FEATURE_SETS[fs], params=LGBM_PRESETS["deep"]) for fs in ("A", "A+fps")]


def committee_candidates() -> list[Candidate]:
    """Noise-aware two-arm committee (committee_label_model.py): point estimate, P(label) and expected improvement."""
    light = {"n_estimators": 150, "learning_rate": 0.08, "num_leaves": 15, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.3, "min_child_samples": 15}
    member_sets = [FEATURE_SETS["A"], FEATURE_SETS["A+fps"]]
    return [
        Candidate(f"committee_{output}_A+fps", "arm_committee", FEATURE_SETS["A+fps"], params={"member_sets": member_sets, "n_seeds": 3, "lgbm": light, "output": output})
        for output in ("m", "p", "ei")
    ]


def pair_candidates(with_physics: bool = False) -> list[Candidate]:
    """SAR-aware pair-transfer models (pair_transfer.py); the physics variant needs the xTB cache."""
    shallow = {"n_estimators": 250, "learning_rate": 0.05, "num_leaves": 15, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.3, "min_child_samples": 20}
    out = [Candidate("pair_desc", "pair_lgbm", ("rdkit_desc",), params={"k": 5, "lgbm": shallow})]
    if with_physics:
        out.append(Candidate("pair_desc+phys", "pair_lgbm", ("rdkit_desc", "physics"), params={"k": 5, "lgbm": shallow}))
    return out


def arm_candidates() -> list[Candidate]:
    """Two-arm regressors trained on floor-clipped arms (kept out of the default grid)."""
    return [Candidate(f"lgbm_reg_arms_clip_{fs}", "lgbm_reg_arms_clip", FEATURE_SETS[fs], params=LGBM_PRESETS["deep"]) for fs in ("A", "A+fps")]


def weighting_candidates() -> list[Candidate]:
    """The deep LightGBM candidates again, trained with label-confidence weighting (kept out of the default
    grid so the existing caches and selections stay as they were).
    """
    return [
        Candidate(f"lgbm_{fs}_deep_c{w.replace('.', '')}", "lgbm_clf", FEATURE_SETS[fs], params=LGBM_PRESETS["deep"], weighting=w)
        for fs in ("A", "A+fps")
        for w in WEIGHTINGS
    ]


def context_candidates() -> list[Candidate]:
    """Deep LightGBM with the mechanism-informed alert-context block, trained on every row and with the
    label-confidence drop (kept out of the default grid, like weighting_candidates).
    """
    out = []
    for fs, blocks in CONTEXT_SETS.items():
        out.append(Candidate(f"lgbm_{fs}_deep", "lgbm_clf", blocks, params=LGBM_PRESETS["deep"]))
        out.append(Candidate(f"lgbm_{fs}_deep_cdrop10", "lgbm_clf", blocks, params=LGBM_PRESETS["deep"], weighting="drop1.0"))
    return out


def selection_v2_candidates() -> list[Candidate]:
    """Selection-aware v2 on feature set A (deep LightGBM for the TDI-given-hit part)."""
    return [Candidate(f"sel2{tag}_A", family, FEATURE_SETS["A"], params=LGBM_PRESETS["deep"]) for tag, family in (("gate", "sel2_gate"), ("two", "sel2_two_part"), ("heck", "sel2_heckman"))]


def xiso_candidates() -> list[Candidate]:
    """Deep LightGBM on feature set A / A+fps plus the arms of the other isoforms predicted from structure (xiso_transfer.py)."""
    return [Candidate(f"xiso_{fs}_deep", "lgbm_xiso", FEATURE_SETS[fs], params=LGBM_PRESETS["deep"]) for fs in ("A", "A+fps")]


def xiso_arm_vectors(train: pd.DataFrame, n_all: int) -> dict[str, np.ndarray]:
    out = {}
    for iso in ISOFORM_ORDER:
        for arm, col in (("tdi", "TDI_condition"), ("direct", "direct_inhibition")):
            values = np.full(n_all, np.nan)
            values[: len(train)] = train[f"{iso}_pIC50_{col}"].to_numpy(dtype=float)
            out[f"{iso}_{arm}"] = values
    return out


def population_candidates() -> list[Candidate]:
    """The selection-aware families on feature sets A and A+fps (deep LightGBM): the gate alone, the TDI
    model fitted on hits only (what the hits-only pool trained all along), their product, and the plain
    end-to-end classifier on the full-population labels.
    """
    out = []
    for fs in ("A", "A+fps"):
        for tag, family in (("popgate", "lgbm_gate"), ("pophits", "lgbm_hits_only"), ("poptwo", "lgbm_two_part"), ("pop", "lgbm_clf")):
            out.append(Candidate(f"{tag}_{fs}_deep", family, FEATURE_SETS[fs], params=LGBM_PRESETS["deep"]))
    return out


# -- context ------------------------------------------------------------------------------------


@dataclass
class Context:
    """Everything a candidate may read. ``labels`` holds the 0/1 label of every pool row (and -1
    elsewhere); ``targets`` the continuous quantities (NaN where unmeasured); ``similarity`` the
    full Tanimoto matrix over all compounds.
    """

    store: FeatureStore
    labels: np.ndarray
    targets: dict[str, np.ndarray]
    similarity: np.ndarray
    n_jobs: int = 4
    confidence: np.ndarray | None = None  # |z| of every compound's label (NaN outside the pool), see label_confidence
    hit: np.ndarray | None = None  # 1 where the compound was measured on this isoform (a screening hit), 0 elsewhere; selection-aware families only
    test_rows: np.ndarray | None = None  # indices of the unlabeled blinded test structures (their labels do not exist; structures are usable)
    hit_profile: np.ndarray | None = None  # (n x 4) 1 where the compound was measured on each of ISOFORM_ORDER, the screen's promotion pattern
    isoform: str | None = None
    xiso_arms: dict[str, np.ndarray] | None = None  # "<isoform>_tdi" / "<isoform>_direct": measured pIC50 of every isoform (NaN where unmeasured); read only by the cross-isoform family
    aux: dict[str, np.ndarray] | None = None  # official auxiliary readings (single-concentration screen, Emax) of every compound, NaN where unmeasured and for the blinded test compounds; read only by aux_blocks


def _labels_for(ctx: Context, task: Task) -> np.ndarray:
    """Labels of the TRAINING rows only; evaluation rows are never looked up."""
    y = ctx.labels[task.train]
    if (y < 0).any():
        raise ValueError("training rows without a label")
    return y


TESTLIKE_NEIGHBOURS = 5


def testlike_weights(ctx: Context, train_rows: np.ndarray, alpha: float) -> np.ndarray:
    """Density-ratio proxy for importance weighting under covariate shift, from structures only.

    r_i = (mean similarity of training compound i to its nearest blinded-test structures) / (mean similarity to its
    nearest OTHER training compounds), weight = clip(r_i ** alpha, 0.2, 5) rescaled to mean 1. A compound that sits
    among the test compounds but alone in the training set is up-weighted; a member of a dense training series that
    the test set barely touches is down-weighted. No test label exists or is used.
    """
    if ctx.test_rows is None:
        raise ValueError("testlike weighting needs Context.test_rows")
    k = TESTLIKE_NEIGHBOURS
    to_test = np.sort(ctx.similarity[np.ix_(train_rows, ctx.test_rows)], axis=1)[:, -k:].mean(axis=1)
    within = ctx.similarity[np.ix_(train_rows, train_rows)].copy()
    np.fill_diagonal(within, -1.0)
    to_train = np.sort(within, axis=1)[:, -k:].mean(axis=1)
    w = np.clip((to_test / np.maximum(to_train, 1e-3)) ** alpha, 0.2, 5.0)
    return w / w.mean()


def _seed(task: Task) -> int:
    return (task.repeat + 2) * 1_000_003 + task.outer * 10_007 + (task.inner + 3) * 101


def _knn_label_features(similarity: np.ndarray, train: np.ndarray, y: np.ndarray, query: np.ndarray, k: int, leave_one_out: bool) -> np.ndarray:
    sub = similarity[np.ix_(query, train)].copy()
    if leave_one_out:
        np.fill_diagonal(sub, -1.0)
    kk = min(k, sub.shape[1] - 1)
    top = np.argpartition(-sub, kk - 1, axis=1)[:, :kk]
    sims = np.take_along_axis(sub, top, axis=1)
    labels = y[top]
    weights = np.maximum(sims, 1e-6)
    return np.column_stack([sims.mean(axis=1), sims.max(axis=1), labels.mean(axis=1), (weights * labels).sum(axis=1) / weights.sum(axis=1)])


def _design(candidate: Candidate, ctx: Context, task: Task, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_x = ctx.store.matrix(candidate.blocks, task.train)
    eval_x = ctx.store.matrix(candidate.blocks, task.evaluate)
    if candidate.knn_label_features:
        train_x = np.hstack([train_x, _knn_label_features(ctx.similarity, task.train, y_train, task.train, KNN_FEATURE_K, True)])
        eval_x = np.hstack([eval_x, _knn_label_features(ctx.similarity, task.train, y_train, task.evaluate, KNN_FEATURE_K, False)])
    if candidate.hyp:
        from cypmode.challenge.pharmacophore_hypotheses import hypothesis_columns, load

        hypotheses, _ = load(tuple(ctx.store.smiles))
        train_h, eval_h = hypothesis_columns(hypotheses, task.train, y_train, task.evaluate, _seed(task), permute=candidate.hyp == "permuted")  # out of fold for the training rows, training labels only
        train_x, eval_x = np.hstack([train_x, train_h]), np.hstack([eval_x, eval_h])
    return train_x, eval_x


GATE_STAGE_ONE = {"n_estimators": 120, "learning_rate": 0.06, "num_leaves": 15, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.3, "min_child_samples": 10}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def hit_profile_gate(train_x: np.ndarray, eval_x: np.ndarray, hits: np.ndarray, target: int, n_jobs: int, seed: int, folds: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """P(compound is a screening hit on isoform ``target``) from the screen's whole promotion pattern.

    A first-stage LightGBM per isoform predicts "was promoted on isoform k" (promiscuity is shared between
    isoforms); their out-of-fold logits feed a second-stage logistic regression for the target isoform. The
    out-of-fold scores of the training rows (cross-fitted, so they carry no in-sample optimism) and the full-fit
    scores of the evaluation rows are returned.
    """
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    n, k = hits.shape
    oof, ev = np.zeros((n, k)), np.zeros((len(eval_x), k))
    for j in range(k):
        yj = hits[:, j].astype(int)
        if yj.sum() < 10 or yj.sum() > n - 10:
            oof[:, j] = ev[:, j] = yj.mean()
            continue
        for a, b in StratifiedKFold(folds, shuffle=True, random_state=seed % 2**31).split(train_x, yj):
            oof[b, j] = LGBMClassifier(verbose=-1, n_jobs=n_jobs, random_state=seed % 2**31, **GATE_STAGE_ONE).fit(train_x[a], yj[a]).predict_proba(train_x[b])[:, 1]
        ev[:, j] = LGBMClassifier(verbose=-1, n_jobs=n_jobs, random_state=seed % 2**31, **GATE_STAGE_ONE).fit(train_x, yj).predict_proba(eval_x)[:, 1]
    second = LogisticRegression(C=1.0, max_iter=500).fit(_logit(oof), hits[:, target].astype(int))
    return second.predict_proba(_logit(oof))[:, 1], second.predict_proba(_logit(ev))[:, 1]


# -- execution ----------------------------------------------------------------------------------


def fit_predict(candidate: Candidate, ctx: Context, task: Task) -> np.ndarray:
    seed = _seed(task)
    family = candidate.family
    if family in ("knn_tanimoto", "svm_tanimoto"):
        y = _labels_for(ctx, task)
        kernel_eval = ctx.similarity[np.ix_(task.evaluate, task.train)]
        if family == "knn_tanimoto":
            k = min(candidate.params["k"], len(task.train))
            top = np.argsort(-kernel_eval, axis=1)[:, :k]
            weights = np.take_along_axis(kernel_eval, top, axis=1)
            return (weights * y[top]).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9)
        from sklearn.svm import SVC

        model = SVC(kernel="precomputed", C=candidate.params["C"], class_weight="balanced")
        model.fit(ctx.similarity[np.ix_(task.train, task.train)], y)
        return model.decision_function(kernel_eval)

    if family in ("lgbm_clf", "extratrees", "rf"):
        y = _labels_for(ctx, task)
        train_x, eval_x = _design(candidate, ctx, task, y)
        if family == "lgbm_clf":
            from lightgbm import LGBMClassifier

            model = LGBMClassifier(verbose=-1, n_jobs=ctx.n_jobs, random_state=seed % 2**31, **candidate.params)
        else:
            from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

            cls = ExtraTreesClassifier if family == "extratrees" else RandomForestClassifier
            model = cls(n_estimators=400, max_features=0.1, min_samples_leaf=2, n_jobs=ctx.n_jobs, random_state=seed % 2**31)
        if candidate.weighting.startswith("testlike"):
            return model.fit(train_x, y, sample_weight=testlike_weights(ctx, task.train, float(candidate.weighting.removeprefix("testlike")))).predict_proba(eval_x)[:, 1]
        if candidate.weighting:
            if ctx.confidence is None:
                raise ValueError(f"{candidate.name} needs Context.confidence")
            keep, weight = training_weights(ctx.confidence[task.train], candidate.weighting)  # training rows only
            return model.fit(train_x[keep], y[keep], sample_weight=weight[keep]).predict_proba(eval_x)[:, 1]
        return model.fit(train_x, y).predict_proba(eval_x)[:, 1]

    if family in ("lgbm_gate", "lgbm_hits_only", "lgbm_two_part"):
        # Selection-aware families. TDI labels exist only for compounds promoted to dose-response on this
        # isoform (screening hits); the scored test set also holds compounds that would never have been
        # promoted. P(label) = P(hit) * P(label | hit): a gate fitted on EVERY training compound, and a
        # TDI model fitted on the hits alone, whose product ranks the whole population.
        if ctx.hit is None:
            raise ValueError(f"{candidate.name} needs Context.hit")
        from lightgbm import LGBMClassifier

        y = _labels_for(ctx, task)
        hit = ctx.hit[task.train].astype(bool)
        train_x, eval_x = _design(candidate, ctx, task, y)

        def fit(x, target):
            return LGBMClassifier(verbose=-1, n_jobs=ctx.n_jobs, random_state=seed % 2**31, **candidate.params).fit(x, target)

        gate = fit(train_x, hit.astype(int)).predict_proba(eval_x)[:, 1] if family != "lgbm_hits_only" else None
        tdi = fit(train_x[hit], y[hit]).predict_proba(eval_x)[:, 1] if family != "lgbm_gate" else None
        return gate if family == "lgbm_gate" else tdi if family == "lgbm_hits_only" else gate * tdi

    if family in ("sel2_gate", "sel2_two_part", "sel2_heckman"):
        # Selection-aware v2: the gate sees the screen's whole promotion pattern, and the TDI-given-hit model of the
        # Heckman variant receives the inverse Mills ratio of that gate (the classical selection correction term).
        if ctx.hit_profile is None or ctx.isoform is None:
            raise ValueError(f"{candidate.name} needs Context.hit_profile and Context.isoform")
        from lightgbm import LGBMClassifier
        from scipy.stats import norm

        y = _labels_for(ctx, task)
        hits = ctx.hit_profile[task.train]
        target = ISOFORM_ORDER.index(ctx.isoform)
        train_x, eval_x = _design(candidate, ctx, task, y)
        gate_train, gate_eval = hit_profile_gate(train_x, eval_x, hits, target, ctx.n_jobs, seed)
        if family == "sel2_gate":
            return gate_eval
        hit = hits[:, target].astype(bool)
        if family == "sel2_heckman":
            def mills(p):
                z = norm.ppf(np.clip(p, 0.01, 0.99))
                return np.column_stack([z, norm.pdf(z) / np.clip(norm.cdf(z), 1e-6, None)])

            train_x, eval_x = np.hstack([train_x, mills(gate_train)]), np.hstack([eval_x, mills(gate_eval)])
        tdi = LGBMClassifier(verbose=-1, n_jobs=ctx.n_jobs, random_state=seed % 2**31, **candidate.params).fit(train_x[hit], y[hit]).predict_proba(eval_x)[:, 1]
        return gate_eval * tdi

    if family == "lgbm_clf_select":
        # The physics columns are chosen from the TRAINING rows of this task only (gain of a physics-only model), so the
        # evaluation rows never influence which columns are used.
        from lightgbm import LGBMClassifier

        y = _labels_for(ctx, task)
        p = candidate.params
        sel_train, sel_eval = ctx.store.matrix(p["select_blocks"], task.train), ctx.store.matrix(p["select_blocks"], task.evaluate)
        first = LGBMClassifier(verbose=-1, n_jobs=ctx.n_jobs, random_state=seed % 2**31, importance_type="gain", **SELECT_STAGE).fit(sel_train, y)
        top = np.argsort(-first.feature_importances_, kind="stable")[: p["select_k"]]
        train_x = np.hstack([ctx.store.matrix(p["base_blocks"], task.train), sel_train[:, top]])
        eval_x = np.hstack([ctx.store.matrix(p["base_blocks"], task.evaluate), sel_eval[:, top]])
        return LGBMClassifier(verbose=-1, n_jobs=ctx.n_jobs, random_state=seed % 2**31, **p["lgbm"]).fit(train_x, y).predict_proba(eval_x)[:, 1]

    if family == "lgbm_xiso":
        from cypmode.challenge.xiso_transfer import fit_predict_xiso

        return fit_predict_xiso(candidate, ctx, task, seed)

    if family in ("lgbm_aux", "lgbm_gate_shift", "lgbm_screen_two", "lgbm_nodirect"):
        from cypmode.challenge import aux_blocks

        return {"lgbm_aux": aux_blocks.fit_predict_aux, "lgbm_gate_shift": aux_blocks.fit_predict_gate_shift, "lgbm_screen_two": aux_blocks.fit_predict_screen_two, "lgbm_nodirect": aux_blocks.fit_predict_nodirect}[family](candidate, ctx, task, seed)

    if family == "lgbm_mmp":
        from lightgbm import LGBMClassifier

        from cypmode.challenge.mmp_transfer import get_pairs, mmp_label_features

        y = _labels_for(ctx, task)
        pairs = get_pairs(ctx.store.smiles)
        n_all = len(ctx.store.smiles)
        train_x = np.hstack([ctx.store.matrix(candidate.blocks, task.train), mmp_label_features(pairs, n_all, task.train, y, task.train, True)])
        eval_x = np.hstack([ctx.store.matrix(candidate.blocks, task.evaluate), mmp_label_features(pairs, n_all, task.train, y, task.evaluate, False)])
        return LGBMClassifier(verbose=-1, n_jobs=ctx.n_jobs, random_state=seed % 2**31, **candidate.params).fit(train_x, y).predict_proba(eval_x)[:, 1]

    if family == "arm_committee":
        from cypmode.challenge.committee_label_model import committee_predict

        p = candidate.params
        return committee_predict(ctx, task, p["member_sets"], p["n_seeds"], p["lgbm"], p["output"], seed, p.get("noise_weights", True))

    if family == "pair_lgbm":
        from cypmode.challenge.pair_transfer import fit_predict_pairs

        y = _labels_for(ctx, task)
        return fit_predict_pairs(
            ctx.store.matrix(candidate.blocks, task.train),
            y,
            ctx.store.matrix(candidate.blocks, task.evaluate),
            ctx.similarity[np.ix_(task.train, task.train)],
            ctx.similarity[np.ix_(task.evaluate, task.train)],
            candidate.params["k"],
            candidate.params["lgbm"],
            ctx.n_jobs,
            seed % 2**31,
        )

    if family in ("lgbm_reg_shift", "lgbm_reg_arms", "lgbm_reg_arms_clip"):
        from lightgbm import LGBMRegressor

        train_x, eval_x = _design(candidate, ctx, task, np.zeros(len(task.train), dtype=int))

        def fit(target: str):
            y = ctx.targets[target][task.train]
            if family == "lgbm_reg_arms_clip":
                # Only max(d, 4) and whether t clears 4 matter for the label (t - max(d, 4) > log10 2), so both arms
                # are clipped at the assay floor before fitting: the model no longer spends capacity on noise below it.
                y = np.maximum(y, ASSAY_FLOOR)
            model = LGBMRegressor(verbose=-1, n_jobs=ctx.n_jobs, random_state=seed % 2**31, **candidate.params)
            return model.fit(train_x, y)

        if family == "lgbm_reg_shift":
            return fit("shift").predict(eval_x)
        return fit("tdi").predict(eval_x) - np.maximum(fit("direct").predict(eval_x), ASSAY_FLOOR)

    if family == "cached":
        raise RuntimeError(f"{candidate.name} is computed by the torch runners (nested_runners_torch / nested_runners_chemprop in .venv-ml) and its cache is incomplete")

    raise ValueError(f"unknown candidate family {family!r}")


def run(candidates: list[Candidate], ctx: Context, plan: NestedPlan, cache: PredictionCache, include_deploy: bool = True) -> None:
    for candidate in candidates:
        todo = cache.missing(candidate.name, include_deploy)
        for i, task in enumerate(todo, 1):
            cache.put(candidate.name, task, fit_predict(candidate, ctx, task))
            if i % 10 == 0 or i == len(todo):
                logger.info("%s: %d/%d tasks", candidate.name, i, len(todo))


# -- building a real run ------------------------------------------------------------------------


def build_context_and_plan(
    train_csv: str | Path,
    test_csv: str | Path,
    isoform: str,
    store_path: str | Path,
    repeats: int,
    outer: int,
    inner: int,
    seed: int,
    n_jobs: int = 4,
    block_names: tuple[str, ...] | None = None,
    group_cutoff: float | None = None,
    population: bool = False,
) -> tuple[Context, NestedPlan]:
    """Pool = compounds with BOTH arms measured for ``isoform`` (the population whose labels follow
    the rule the test set is scored by; see cypmode.challenge.tdi_targets).
    """
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    smiles = list(train["SMILES"]) + list(test["SMILES"])
    store = FeatureStore.load_or_build(store_path, smiles, None if block_names is None else sorted(set(block_names) | {"morgan_bits"}))

    view = training_view(train, isoform)
    n_all = len(smiles)
    labels = np.full(n_all, -1, dtype=np.int64)
    pool_mask = (view["population"] == BOTH_ARMS).to_numpy()
    labels[: len(train)][pool_mask] = view["label"][pool_mask].astype(int).to_numpy()
    targets = {name: np.full(n_all, np.nan) for name in ("shift", "tdi", "direct", "tdi_sd", "direct_sd")}
    for name, col in (("shift", "shift"), ("tdi", "tdi"), ("direct", "direct")):
        targets[name][: len(train)] = view[col].to_numpy(dtype=float)
    targets["tdi_sd"][: len(train)] = train[f"{isoform}_pIC50_TDI_condition_std"].to_numpy(dtype=float)
    targets["direct_sd"][: len(train)] = train[f"{isoform}_pIC50_direct_inhibition_std"].to_numpy(dtype=float)

    hit = None
    profile = None
    if population:
        # every training compound is part of the population: measured on the isoform (a hit, labelled by the
        # rule) or not measured at all, which the challenge scores as an "assigned negative" (inactive in both arms)
        from cypmode.challenge.library_blocks import panel_block_mask

        measured = train[f"{isoform}_pIC50_TDI_condition"].notna().to_numpy()
        panel = panel_block_mask(train)  # outside it "not measured" means "never tested", not "inactive"
        labels[: len(train)][~panel] = -1
        labels[: len(train)][panel & ~measured] = 0
        hit = np.zeros(n_all, dtype=np.int64)
        hit[: len(train)][pool_mask & measured & panel] = 1
        profile = np.zeros((n_all, len(ISOFORM_ORDER)), dtype=np.int8)
        for j, name in enumerate(ISOFORM_ORDER):
            profile[: len(train), j] = train[f"{name}_pIC50_TDI_condition"].notna().to_numpy()

    pool = np.where(labels >= 0)[0]
    bits = store.blocks["morgan_bits"]
    groups = None if group_cutoff is None else butina_groups(bits[pool], group_cutoff)
    plan = NestedPlan(pool, labels[pool], repeats, outer, inner, seed, external=np.arange(len(train), n_all), groups=groups)
    confidence = np.full(n_all, np.nan)
    confidence[: len(train)] = confidence_z(train, isoform)
    if population:
        confidence[: len(train)][~train[f"{isoform}_pIC50_TDI_condition"].notna().to_numpy()] = np.inf  # inactive in both arms: unambiguous
    from cypmode.challenge import official_extra

    aux = official_extra.aux_arrays(list(train["Molecule_Name"]) + list(test["Molecule_Name"])) if official_extra.SCREEN_FILE.exists() else None
    ctx = Context(store, labels, targets, tanimoto_matrix(bits), n_jobs, confidence, hit, test_rows=np.arange(len(train), n_all), hit_profile=profile, isoform=isoform, xiso_arms=xiso_arm_vectors(train, n_all), aux=aux)
    return ctx, plan


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fill the nested-CV prediction cache for the default candidate grid.")
    parser.add_argument("--isoform", required=True, choices=["CYP3A4", "CYP2D6"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--outer", type=int, default=5)
    parser.add_argument("--inner", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--candidates", nargs="*", help="restrict to these candidate names")
    parser.add_argument("--weighting-candidates", action="store_true", help="run the label-confidence-weighted candidates instead of the default grid")
    parser.add_argument("--context-candidates", action="store_true", help="run the alert-context candidates instead of the default grid")
    parser.add_argument("--embedding-candidates", action="store_true", help="run the frozen-embedding candidates (needs the CheMeleon cache) instead of the default grid")
    parser.add_argument("--testlike-candidates", action="store_true", help="run the test-likeness-weighted candidates instead of the default grid")
    parser.add_argument("--mmp-candidates", action="store_true", help="run the matched-molecular-pair candidates instead of the default grid")
    parser.add_argument("--committee-candidates", action="store_true", help="run the noise-aware two-arm committee instead of the default grid")
    parser.add_argument("--arm-candidates", action="store_true", help="run the floor-clipped two-arm regressors instead of the default grid")
    parser.add_argument("--pair-candidates", action="store_true", help="run the SAR-aware pair-transfer candidates instead of the default grid")
    parser.add_argument("--xiso-candidates", action="store_true", help="run the cross-isoform arm-transfer candidates instead of the default grid")
    parser.add_argument("--physics-candidates", action="store_true", help="run the physics-block candidates (needs the physics cache) instead of the default grid")
    parser.add_argument("--population", action="store_true", help="pool = every training compound (hits plus non-hits); runs the selection-aware candidates and writes to <isoform>_pop")
    parser.add_argument("--root", default="data/tdi_track/outputs/nested")
    parser.add_argument("--group-cutoff", type=float, help="keep Butina clusters at this Tanimoto similarity whole in every split (series-level validation)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    root = Path(args.root) / (args.isoform if args.group_cutoff is None else f"{args.isoform}_group{args.group_cutoff:g}")
    if args.population:
        root = Path(args.root) / f"{args.isoform}_pop_panel"
    grid = (
        population_candidates()
        if args.population
        else weighting_candidates() if args.weighting_candidates else context_candidates() if args.context_candidates else physics_candidates() if args.physics_candidates else pair_candidates(True) if args.pair_candidates else xiso_candidates() if args.xiso_candidates else arm_candidates() if args.arm_candidates else committee_candidates() if args.committee_candidates else mmp_candidates() if args.mmp_candidates else testlike_candidates() if args.testlike_candidates else embedding_candidates() if args.embedding_candidates else default_candidates()
    )
    candidates = [c for c in grid if not args.candidates or c.name in args.candidates]
    ctx, plan = build_context_and_plan(
        "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv",
        "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv",
        args.isoform,
        "data/tdi_track/outputs/caches/feature_store.npz",
        args.repeats,
        args.outer,
        args.inner,
        args.seed,
        args.n_jobs,
        tuple(sorted({b for c in candidates for b in c.blocks})),
        args.group_cutoff,
        args.population,
    )
    root.mkdir(parents=True, exist_ok=True)
    plan.save(root / "plan.npz")
    cache = PredictionCache(root / "cache", plan)
    logger.info("plan %s: %d pool rows, %d tasks, %d candidates", plan.plan_id, len(plan.pool), len(plan.tasks()), len(candidates))
    run(candidates, ctx, plan, cache)


if __name__ == "__main__":
    from cypmode import provenance

    provenance.announce()
    _main()

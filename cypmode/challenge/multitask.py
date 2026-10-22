# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""A small, plain-PyTorch shared-trunk multitask model -- tests the
"shared representation helps" hypothesis from
CYPMode_Architecture_Specification.md's Decision Log ("Shared trunk:
hypothesis requiring its own benchmark") without installing
openadmet-models' full stack (PyTorch + Lightning + ChemProp/CheMeleon +
CatBoost + XGBoost + TabPFN, conda-only, no pyproject.toml dependency
list at all -- a large, risky addition on a memory-constrained machine
already busy running Boltz-2). See pyproject.toml's `multitask` extra.

One shared trunk, one small head per isoform. Trained with a masked
binary cross-entropy loss: most compounds have a label for only one
isoform (see CYPMode_DATA_AUDIT.md's missingness section), and
masking -- not imputing a fake label -- is how that's actually handled.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import KFold
from torch import nn

ISOFORMS = ["CYP3A4", "CYP2D6"]

# Threshold-sweep logic duplicated from cypmode.challenge.threshold_tuning
# rather than imported: that module imports lightgbm at module level, and
# importing lightgbm and torch into the same process is a real, confirmed
# segfault on this machine (see pyproject.toml's `torch` marker) -- this
# file must stay lightgbm-free.
DEFAULT_THRESHOLD_GRID = np.linspace(0.05, 0.95, 19)


def _find_optimal_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, thresholds: np.ndarray = DEFAULT_THRESHOLD_GRID
) -> tuple[float, float]:
    best_threshold, best_mcc = 0.5, -1.0
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        mcc = matthews_corrcoef(y_true, pred)
        if mcc > best_mcc:
            best_mcc, best_threshold = mcc, float(t)
    return best_threshold, best_mcc


class SharedTrunkModel(nn.Module):
    def __init__(self, input_dim: int, trunk_dim: int = 64, head_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, trunk_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(trunk_dim, trunk_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleDict(
            {cyp: nn.Sequential(nn.Linear(trunk_dim, head_dim), nn.ReLU(), nn.Linear(head_dim, 1)) for cyp in ISOFORMS}
        )

    def forward(self, x: torch.Tensor) -> dict:
        shared = self.trunk(x)
        return {cyp: self.heads[cyp](shared).squeeze(-1) for cyp in ISOFORMS}


def masked_bce_loss(logits: dict, targets: dict, masks: dict, pos_weight: dict | None = None) -> torch.Tensor:
    """Binary cross-entropy per isoform, restricted to that isoform's
    labeled compounds only -- an unlabeled compound contributes zero
    gradient for that head, rather than being imputed as a negative.

    `pos_weight`, if given, is a {isoform: float} multiplier on the
    positive-class term (torch.nn.BCEWithLogitsLoss's own pos_weight
    convention -- see its docs), applied per isoform independently.
    None (the default) reproduces the original unweighted loss exactly.
    """
    total = torch.tensor(0.0)
    count = torch.tensor(0.0)
    for cyp in ISOFORMS:
        if masks[cyp].sum() == 0:
            continue
        weight = torch.tensor(pos_weight[cyp]) if pos_weight and cyp in pos_weight else None
        loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=weight)
        per_example = loss_fn(logits[cyp], targets[cyp])
        total = total + (per_example * masks[cyp]).sum()
        count = count + masks[cyp].sum()
    return total / count.clamp(min=1)


def train_shared_trunk(
    X_train: np.ndarray,
    y_train: dict,
    mask_train: dict,
    epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 42,
    trunk_dim: int = 64,
    head_dim: int = 32,
    dropout: float = 0.2,
    pos_weight: dict | None = None,
) -> SharedTrunkModel:
    """trunk_dim/head_dim/dropout/pos_weight all default to the original
    hardcoded values -- an existing caller passing none of them gets the
    exact same model and loss as before this function grew these knobs.
    """
    torch.manual_seed(seed)
    model = SharedTrunkModel(input_dim=X_train.shape[1], trunk_dim=trunk_dim, head_dim=head_dim, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    X = torch.tensor(X_train, dtype=torch.float32)
    y = {cyp: torch.tensor(y_train[cyp], dtype=torch.float32) for cyp in ISOFORMS}
    mask = {cyp: torch.tensor(mask_train[cyp], dtype=torch.float32) for cyp in ISOFORMS}

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = masked_bce_loss(model(X), y, mask, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
    return model


def predict_proba(model: SharedTrunkModel, X: np.ndarray) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
    return {cyp: torch.sigmoid(logits[cyp]).numpy() for cyp in ISOFORMS}


def _standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def cross_validate_shared_trunk(
    df: pd.DataFrame, feature_col: str = "m3_no_amine_features", n_splits: int = 5, random_state: int = 42, epochs: int = 200
) -> pd.DataFrame:
    """Same spirit as tdi_baseline.cross_validate_fixed_seed, but ONE
    shared model per fold trained on both isoforms' (masked) labels
    jointly, then scored per isoform on that fold's own labeled
    validation compounds -- a genuine M1 (shared trunk) number, directly
    comparable to M0/M3's independent-per-isoform numbers since both use
    df's full row set and the same random_state.
    """
    X_raw = np.stack(df[feature_col]).astype(np.float32)

    y, mask = {}, {}
    for cyp in ISOFORMS:
        col = f"{cyp}_is_TDI"
        mask[cyp] = df[col].notna().to_numpy().astype(np.float32)
        y[cyp] = df[col].fillna(0).astype(int).to_numpy().astype(np.float32)

    rows = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_raw)):
        # mean/std computed from the training fold only -- computing them
        # over the full X_raw before the loop (as this function used to)
        # leaks the validation fold's own feature values into the
        # standardization statistic used to scale it, a real if scale-only
        # leak. cross_validate_shared_trunk_with_threshold_tuning and
        # _inner_oof_threshold_shared_trunk already get this right
        # (train_idx-only); this brings this function in line with them.
        mean, std = X_raw[train_idx].mean(axis=0), X_raw[train_idx].std(axis=0) + 1e-8
        X_train = _standardize(X_raw[train_idx], mean, std)
        X_val = _standardize(X_raw[val_idx], mean, std)

        model = train_shared_trunk(
            X_train,
            {cyp: y[cyp][train_idx] for cyp in ISOFORMS},
            {cyp: mask[cyp][train_idx] for cyp in ISOFORMS},
            epochs=epochs,
            seed=random_state,
        )
        proba = predict_proba(model, X_val)

        for cyp in ISOFORMS:
            val_mask = mask[cyp][val_idx].astype(bool)
            if val_mask.sum() == 0:
                continue
            y_true = y[cyp][val_idx][val_mask].astype(int)
            y_pred = (proba[cyp][val_mask] >= 0.5).astype(int)
            rows.append(
                {
                    "isoform": cyp,
                    "fold": fold,
                    "n_val": int(val_mask.sum()),
                    "mcc": matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _inner_oof_threshold_shared_trunk(
    X_train_raw: np.ndarray, y_train: dict, mask_train: dict, inner_splits: int, epochs: int, random_state: int
) -> dict:
    """Same idea as threshold_tuning._inner_oof_threshold, but for the
    shared-trunk model: an inner K-fold runs entirely inside the outer
    training portion, producing inner-OOF probabilities per isoform (a
    property of the training portion only), and a threshold is swept per
    isoform from those -- zero information from the outer validation fold.
    """
    n = X_train_raw.shape[0]
    inner_oof_proba = {cyp: np.full(n, np.nan, dtype=np.float32) for cyp in ISOFORMS}
    inner_kfold = KFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
    for inner_train_idx, inner_val_idx in inner_kfold.split(X_train_raw):
        mean, std = X_train_raw[inner_train_idx].mean(axis=0), X_train_raw[inner_train_idx].std(axis=0) + 1e-8
        X_inner_train = _standardize(X_train_raw[inner_train_idx], mean, std)
        X_inner_val = _standardize(X_train_raw[inner_val_idx], mean, std)

        model = train_shared_trunk(
            X_inner_train,
            {cyp: y_train[cyp][inner_train_idx] for cyp in ISOFORMS},
            {cyp: mask_train[cyp][inner_train_idx] for cyp in ISOFORMS},
            epochs=epochs,
            seed=random_state,
        )
        proba = predict_proba(model, X_inner_val)
        for cyp in ISOFORMS:
            inner_oof_proba[cyp][inner_val_idx] = proba[cyp]

    thresholds = {}
    for cyp in ISOFORMS:
        valid = mask_train[cyp].astype(bool) & ~np.isnan(inner_oof_proba[cyp])
        threshold, _ = _find_optimal_threshold(y_train[cyp][valid].astype(int), inner_oof_proba[cyp][valid])
        thresholds[cyp] = threshold
    return thresholds


def cross_validate_shared_trunk_with_threshold_tuning(
    df: pd.DataFrame,
    feature_col: str = "m3_no_amine_features",
    n_splits: int = 5,
    inner_splits: int = 3,
    random_state: int = 42,
    epochs: int = 200,
) -> pd.DataFrame:
    """cross_validate_shared_trunk, plus a fold-internal MCC-tuned threshold
    per isoform (never touching the outer validation fold's own labels) --
    the same protocol threshold_tuning.py applies to the independent-head
    (M3) model, so the two are directly comparable.
    """
    X_raw = np.stack(df[feature_col]).astype(np.float32)

    y, mask = {}, {}
    for cyp in ISOFORMS:
        col = f"{cyp}_is_TDI"
        mask[cyp] = df[col].notna().to_numpy().astype(np.float32)
        y[cyp] = df[col].fillna(0).astype(int).to_numpy().astype(np.float32)

    rows = []
    outer_kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(outer_kfold.split(X_raw)):
        y_train = {cyp: y[cyp][train_idx] for cyp in ISOFORMS}
        mask_train = {cyp: mask[cyp][train_idx] for cyp in ISOFORMS}

        tuned_thresholds = _inner_oof_threshold_shared_trunk(
            X_raw[train_idx], y_train, mask_train, inner_splits, epochs, random_state
        )

        mean, std = X_raw[train_idx].mean(axis=0), X_raw[train_idx].std(axis=0) + 1e-8
        X_train = _standardize(X_raw[train_idx], mean, std)
        X_val = _standardize(X_raw[val_idx], mean, std)

        model = train_shared_trunk(X_train, y_train, mask_train, epochs=epochs, seed=random_state)
        proba = predict_proba(model, X_val)

        for cyp in ISOFORMS:
            val_mask = mask[cyp][val_idx].astype(bool)
            if val_mask.sum() == 0:
                continue
            y_true = y[cyp][val_idx][val_mask].astype(int)
            p = proba[cyp][val_mask]
            naive_pred = (p >= 0.5).astype(int)
            tuned_pred = (p >= tuned_thresholds[cyp]).astype(int)
            rows.append(
                {
                    "isoform": cyp,
                    "fold": fold,
                    "n_val": int(val_mask.sum()),
                    "tuned_threshold": tuned_thresholds[cyp],
                    "mcc_naive_0.5": matthews_corrcoef(y_true, naive_pred) if len(set(y_true)) > 1 else float("nan"),
                    "mcc_tuned": matthews_corrcoef(y_true, tuned_pred) if len(set(y_true)) > 1 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


class MultiBranchSharedTrunkModel(nn.Module):
    """Real fix for the naive-concatenation problem confirmed in
    CYPMode_Architecture_Specification.md: bolting ChemBERTa's raw
    768-dim embedding onto M3's feature vector and handing it to
    LightGBM significantly *hurt* CYP3A4 (p=0.0002) -- a tree ensemble
    has no learned projection to denoise a dense, highly-correlated
    embedding the way MTMM-CYP's own real architecture does (each
    representation gets its own branch before fusion). This is that:
    M3 and the ChemBERTa embedding each pass through their own small
    branch (linear -> ReLU -> dropout) BEFORE concatenation, so the
    shared trunk sees two already-denoised, comparably-scaled
    representations instead of one raw 768-dim block dominating a
    smaller M3 vector.
    """

    def __init__(
        self,
        m3_dim: int,
        chemberta_dim: int,
        branch_dim: int = 64,
        trunk_dim: int = 64,
        head_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.m3_branch = nn.Sequential(nn.Linear(m3_dim, branch_dim), nn.ReLU(), nn.Dropout(dropout))
        self.chemberta_branch = nn.Sequential(nn.Linear(chemberta_dim, branch_dim), nn.ReLU(), nn.Dropout(dropout))
        self.trunk = nn.Sequential(
            nn.Linear(branch_dim * 2, trunk_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(trunk_dim, trunk_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleDict(
            {cyp: nn.Sequential(nn.Linear(trunk_dim, head_dim), nn.ReLU(), nn.Linear(head_dim, 1)) for cyp in ISOFORMS}
        )

    def forward(self, x_m3: torch.Tensor, x_chemberta: torch.Tensor) -> dict:
        fused = torch.cat([self.m3_branch(x_m3), self.chemberta_branch(x_chemberta)], dim=-1)
        shared = self.trunk(fused)
        return {cyp: self.heads[cyp](shared).squeeze(-1) for cyp in ISOFORMS}


def train_multi_branch_shared_trunk(
    X_m3_train: np.ndarray,
    X_chemberta_train: np.ndarray,
    y_train: dict,
    mask_train: dict,
    epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 42,
) -> MultiBranchSharedTrunkModel:
    torch.manual_seed(seed)
    model = MultiBranchSharedTrunkModel(m3_dim=X_m3_train.shape[1], chemberta_dim=X_chemberta_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    x_m3 = torch.tensor(X_m3_train, dtype=torch.float32)
    x_chemberta = torch.tensor(X_chemberta_train, dtype=torch.float32)
    y = {cyp: torch.tensor(y_train[cyp], dtype=torch.float32) for cyp in ISOFORMS}
    mask = {cyp: torch.tensor(mask_train[cyp], dtype=torch.float32) for cyp in ISOFORMS}

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = masked_bce_loss(model(x_m3, x_chemberta), y, mask)
        loss.backward()
        optimizer.step()
    return model


def predict_proba_multi_branch(model: MultiBranchSharedTrunkModel, X_m3: np.ndarray, X_chemberta: np.ndarray) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_m3, dtype=torch.float32), torch.tensor(X_chemberta, dtype=torch.float32))
    return {cyp: torch.sigmoid(logits[cyp]).numpy() for cyp in ISOFORMS}


def cross_validate_multi_branch_shared_trunk(
    df: pd.DataFrame,
    m3_col: str = "m3_no_amine_features",
    chemberta_col: str = "chemberta_embedding",
    n_splits: int = 5,
    random_state: int = 42,
    epochs: int = 200,
) -> pd.DataFrame:
    """Same spirit and same fold protocol as cross_validate_shared_trunk,
    but training MultiBranchSharedTrunkModel on both representations
    jointly -- directly comparable to the M3-only shared-trunk number
    (same df, same random_state, same fold assignment) as a fair test of
    whether a proper fusion architecture (not naive concatenation)
    recovers real value from the ChemBERTa embedding.
    """
    X_m3_raw = np.stack(df[m3_col]).astype(np.float32)
    X_chemberta_raw = np.stack(df[chemberta_col]).astype(np.float32)

    y, mask = {}, {}
    for cyp in ISOFORMS:
        col = f"{cyp}_is_TDI"
        mask[cyp] = df[col].notna().to_numpy().astype(np.float32)
        y[cyp] = df[col].fillna(0).astype(int).to_numpy().astype(np.float32)

    rows = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_m3_raw)):
        # Training-fold-only mean/std -- see the identical fix and its
        # rationale in cross_validate_shared_trunk above.
        m3_mean, m3_std = X_m3_raw[train_idx].mean(axis=0), X_m3_raw[train_idx].std(axis=0) + 1e-8
        cb_mean, cb_std = X_chemberta_raw[train_idx].mean(axis=0), X_chemberta_raw[train_idx].std(axis=0) + 1e-8
        X_m3_train = _standardize(X_m3_raw[train_idx], m3_mean, m3_std)
        X_m3_val = _standardize(X_m3_raw[val_idx], m3_mean, m3_std)
        X_cb_train = _standardize(X_chemberta_raw[train_idx], cb_mean, cb_std)
        X_cb_val = _standardize(X_chemberta_raw[val_idx], cb_mean, cb_std)

        model = train_multi_branch_shared_trunk(
            X_m3_train,
            X_cb_train,
            {cyp: y[cyp][train_idx] for cyp in ISOFORMS},
            {cyp: mask[cyp][train_idx] for cyp in ISOFORMS},
            epochs=epochs,
            seed=random_state,
        )
        proba = predict_proba_multi_branch(model, X_m3_val, X_cb_val)

        for cyp in ISOFORMS:
            val_mask = mask[cyp][val_idx].astype(bool)
            if val_mask.sum() == 0:
                continue
            y_true = y[cyp][val_idx][val_mask].astype(int)
            y_pred = (proba[cyp][val_mask] >= 0.5).astype(int)
            rows.append(
                {
                    "isoform": cyp,
                    "fold": fold,
                    "n_val": int(val_mask.sum()),
                    "mcc": matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _main() -> None:
    from pathlib import Path

    # chemberta_features is safe to import here (no lightgbm anywhere in
    # its own import chain) -- see load_cached_chemberta_embeddings' own
    # docstring for why it deliberately lives there and not in
    # chemberta_significance.py, which would pull lightgbm in transitively.
    from cypmode.challenge.chemberta_features import load_cached_chemberta_embeddings
    from cypmode.challenge.tdi_ablation import add_ablation_features
    from cypmode.challenge.tdi_baseline import (
        compute_descriptors,
        load_tdi_training_data,
    )

    df = add_ablation_features(compute_descriptors(load_tdi_training_data()))

    results = cross_validate_shared_trunk(df)
    print("-- untuned (0.5 threshold) --")
    print(results.groupby("isoform")["mcc"].mean().to_string())
    out_path = Path("data/tdi_track/outputs/comparisons/shared_trunk_m1.csv")
    results.to_csv(out_path, index=False)
    print(f"wrote {out_path}\n")

    tuned_results = cross_validate_shared_trunk_with_threshold_tuning(df)
    print("-- fold-internal MCC-tuned threshold --")
    print(tuned_results.groupby("isoform")[["mcc_naive_0.5", "mcc_tuned", "tuned_threshold"]].mean().to_string())
    tuned_out_path = Path("data/tdi_track/outputs/comparisons/shared_trunk_m1_threshold_tuning.csv")
    tuned_results.to_csv(tuned_out_path, index=False)
    print(f"wrote {tuned_out_path}\n")

    # The real point of this run: does a proper fusion architecture (each
    # representation through its own branch, THEN combined) recover value
    # from ChemBERTa that naive LightGBM-feature concatenation lost (see
    # CYPMode_Architecture_Specification.md's real, statistically
    # significant negative finding for that naive approach)? Directly
    # comparable to the M3-only shared-trunk number above: same df, same
    # random_state, same fold assignment.
    df_with_chemberta = load_cached_chemberta_embeddings(df)
    multi_branch_results = cross_validate_multi_branch_shared_trunk(df_with_chemberta)
    print("-- multi-branch fusion (M3 branch + ChemBERTa branch -> shared trunk) --")
    print(multi_branch_results.groupby("isoform")["mcc"].mean().to_string())
    multi_branch_out_path = Path("data/tdi_track/outputs/comparisons/shared_trunk_m1_chemberta_fusion.csv")
    multi_branch_results.to_csv(multi_branch_out_path, index=False)
    print(f"wrote {multi_branch_out_path}")


if __name__ == "__main__":
    _main()

"""A small, plain-PyTorch shared-trunk multitask model -- tests the
"shared representation helps" hypothesis from
CYPMode-TDI_Architecture_Specification.md's Decision Log ("Shared trunk:
hypothesis requiring its own benchmark") without installing
openadmet-models' full stack (PyTorch + Lightning + ChemProp/CheMeleon +
CatBoost + XGBoost + TabPFN, conda-only, no pyproject.toml dependency
list at all -- a large, risky addition on a memory-constrained machine
already busy running Boltz-2). See pyproject.toml's `multitask` extra.

One shared trunk, one small head per isoform. Trained with a masked
binary cross-entropy loss: most compounds have a label for only one
isoform (see CYPMode-TDI_DATA_AUDIT.md's missingness section), and
masking -- not imputing a fake label -- is how that's actually handled.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import KFold

ISOFORMS = ["CYP3A4", "CYP2D6"]


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


def masked_bce_loss(logits: dict, targets: dict, masks: dict) -> torch.Tensor:
    """Binary cross-entropy per isoform, restricted to that isoform's
    labeled compounds only -- an unlabeled compound contributes zero
    gradient for that head, rather than being imputed as a negative.
    """
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    total = torch.tensor(0.0)
    count = torch.tensor(0.0)
    for cyp in ISOFORMS:
        if masks[cyp].sum() == 0:
            continue
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
) -> SharedTrunkModel:
    torch.manual_seed(seed)
    model = SharedTrunkModel(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    X = torch.tensor(X_train, dtype=torch.float32)
    y = {cyp: torch.tensor(y_train[cyp], dtype=torch.float32) for cyp in ISOFORMS}
    mask = {cyp: torch.tensor(mask_train[cyp], dtype=torch.float32) for cyp in ISOFORMS}

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = masked_bce_loss(model(X), y, mask)
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
    df: pd.DataFrame, feature_col: str = "m3_features", n_splits: int = 5, random_state: int = 42, epochs: int = 200
) -> pd.DataFrame:
    """Same spirit as tdi_baseline.cross_validate_fixed_seed, but ONE
    shared model per fold trained on both isoforms' (masked) labels
    jointly, then scored per isoform on that fold's own labeled
    validation compounds -- a genuine M1 (shared trunk) number, directly
    comparable to M0/M3's independent-per-isoform numbers since both use
    df's full row set and the same random_state.
    """
    X_raw = np.stack(df[feature_col]).astype(np.float32)
    mean, std = X_raw.mean(axis=0), X_raw.std(axis=0) + 1e-8

    y, mask = {}, {}
    for cyp in ISOFORMS:
        col = f"{cyp}_is_TDI"
        mask[cyp] = df[col].notna().to_numpy().astype(np.float32)
        y[cyp] = df[col].fillna(0).astype(int).to_numpy().astype(np.float32)

    rows = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_raw)):
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


def _main() -> None:
    from pathlib import Path

    from cypmode.challenge.tdi_ablation import add_ablation_features
    from cypmode.challenge.tdi_baseline import compute_descriptors, load_tdi_training_data

    df = add_ablation_features(compute_descriptors(load_tdi_training_data()))
    results = cross_validate_shared_trunk(df)
    print(results.groupby("isoform")["mcc"].mean().to_string())

    out_path = Path("data/cyp_challenge/tdi_track/shared_trunk_m1.csv")
    results.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    _main()

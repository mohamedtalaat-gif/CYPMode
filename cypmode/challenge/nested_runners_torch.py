# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Torch-side candidates (TabICL, multitask MLP over all potency columns) for a
NestedPlan. Runs in the isolated torch environment (.venv-ml) and writes to the
same PredictionCache the LightGBM runners fill -- the plan is an npz and the
cache plain npy files, so the two never share a process (torch and LightGBM
crash together; see the `torch` marker in pyproject.toml).

Run: PYTHONPATH=. .venv-ml/bin/python -m cypmode.challenge.nested_runners_torch --isoform CYP3A4

The multitask network is the one place where "extra" compounds help: the
training file carries eight continuous endpoints (direct and TDI-arm pIC50 for
four isoforms) for ~6,100 compounds, far more supervision than the ~2,300
pool rows with a boolean label. Using them without leakage means a task may train on
``task.train`` plus compounds OUTSIDE the plan's pool and external set -- and on
nothing else: other pool rows (for an inner task, the outer-validation rows)
must stay invisible, or the nested estimate would be leaky.
"""

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cypmode.challenge.nested import NestedPlan, PredictionCache, Task
from cypmode.challenge.tdi_targets import (
    ASSAY_FLOOR,
    ISOFORMS,
    direct_column,
    tdi_column,
)

logger = logging.getLogger(__name__)

ARM_COLUMNS = [col for iso in ISOFORMS for col in (direct_column(iso), tdi_column(iso))]
AUX_COLUMNS = [f"{iso}_screen" for iso in ISOFORMS] + [f"{iso}_emax_{arm}" for iso in ISOFORMS for arm in ("direct", "tdi")]


@dataclass
class TorchContext:
    log_counts: np.ndarray  # log1p of count/bit fingerprints, one row per compound
    descriptors: np.ndarray
    targets: np.ndarray  # (n_all, 8) continuous endpoints, NaN where unmeasured
    labels: np.ndarray  # 0/1 for pool rows, -1 elsewhere
    isoform_position: int
    outside: np.ndarray  # row ids that are neither pool nor external: always safe to train on
    seed: int = 0
    monroe: np.ndarray | None = None  # label-free PCA (128 components) of the frozen Monroe embeddings of every compound, when the cache exists
    chemeleon: np.ndarray | None = None  # label-free PCA (128 components) of the frozen CheMeleon embeddings of every compound, when the cache exists
    atoms: dict | None = None  # cached per-atom Atom-JEPA features: {"rows": row index of every compound's first atom, "count", "features" (float16, atoms x 3 blocks x 256), "numbers"}, when the cache exists
    aux_targets: np.ndarray | None = None  # (n_all, 12) official auxiliary readings (screen log2 fold change, Emax of both arms), NaN where unmeasured and for the blinded test compounds


def training_rows(ctx: TorchContext, task: Task) -> np.ndarray:
    """Rows a task may learn from: its own training rows plus compounds outside the pool."""
    return np.union1d(task.train, ctx.outside)


def _quantile_features(ctx: TorchContext, rows: np.ndarray) -> tuple[np.ndarray, "object"]:
    from sklearn.preprocessing import QuantileTransformer

    transformer = QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0).fit(ctx.descriptors[rows])
    return transformer.transform(ctx.descriptors).astype(np.float32), transformer


def fit_predict_tabicl(ctx: TorchContext, task: Task) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    from tabicl import TabICLClassifier

    rows = task.train
    desc, _ = _quantile_features(ctx, rows)
    svd = TruncatedSVD(128, random_state=0).fit(ctx.log_counts[rows])
    z = np.hstack([svd.transform(ctx.log_counts), desc]).astype(np.float32)
    model = TabICLClassifier(device="cpu", n_estimators=2, random_state=ctx.seed).fit(z[rows], ctx.labels[rows])
    return model.predict_proba(z[task.evaluate])[:, 1]


def fit_predict_tabicl_monroe(ctx: TorchContext, task: Task) -> np.ndarray:
    """TabICL on the frozen Monroe embedding (128 principal components, label-free) next to the quantile-transformed descriptors."""
    from tabicl import TabICLClassifier

    if ctx.monroe is None:
        raise ValueError("tabicl_monroe needs the Monroe embedding cache (monroe_embeddings --extract)")
    rows = task.train
    desc, _ = _quantile_features(ctx, rows)
    z = np.hstack([ctx.monroe, desc]).astype(np.float32)
    model = TabICLClassifier(device="cpu", n_estimators=2, random_state=ctx.seed).fit(z[rows], ctx.labels[rows])
    return model.predict_proba(z[task.evaluate])[:, 1]


def fit_predict_tabicl_chemeleon(ctx: TorchContext, task: Task) -> np.ndarray:
    """TabICL on the frozen CheMeleon embedding (128 principal components, label-free) next to the quantile-transformed descriptors; the representation is the one of ``tabicl_monroe`` with the other encoder."""
    from tabicl import TabICLClassifier

    if ctx.chemeleon is None:
        raise ValueError("tabicl_chemeleon needs the CheMeleon embedding cache (frozen_embeddings --extract)")
    rows = task.train
    desc, _ = _quantile_features(ctx, rows)
    z = np.hstack([ctx.chemeleon, desc]).astype(np.float32)
    model = TabICLClassifier(device="cpu", n_estimators=2, random_state=ctx.seed).fit(z[rows], ctx.labels[rows])
    return model.predict_proba(z[task.evaluate])[:, 1]


def fit_predict_tabpfn_chemeleon(ctx: TorchContext, task: Task) -> np.ndarray:
    """TabPFN on the same representation as ``tabicl_chemeleon`` (the settings of ``tabpfn_svd``: 4 estimators, batches of about 1,000 rows)."""
    from tabpfn import TabPFNClassifier

    if ctx.chemeleon is None:
        raise ValueError("tabpfn_chemeleon needs the CheMeleon embedding cache (frozen_embeddings --extract)")
    rows = task.train
    desc, _ = _quantile_features(ctx, rows)
    z = np.hstack([ctx.chemeleon, desc]).astype(np.float32)
    model = TabPFNClassifier(device="cpu", n_estimators=4, random_state=ctx.seed, ignore_pretraining_limits=True).fit(z[rows], ctx.labels[rows])
    return np.concatenate([model.predict_proba(z[task.evaluate[i : i + 1000]])[:, 1] for i in range(0, len(task.evaluate), 1000)])


def fit_predict_tabpfn(ctx: TorchContext, task: Task) -> np.ndarray:
    """TabPFN (Prior Labs; Hollmann et al., Nature 2025 and the TabPFN-3 technical report) on the same representation as ``tabicl_svd``: 128 SVD components of the log-count fingerprints next to the quantile-transformed descriptors.

    The weights are released under the Prior Labs non-commercial licences and are downloaded by the package after the licence has been accepted at ux.priorlabs.ai (a token in ``TABPFN_TOKEN``); this
    repository neither contains nor redistributes them. The classifier is a prior-fitted network: ``fit`` stores the training rows and ``predict_proba`` reads them as context in one forward pass.
    """
    from sklearn.decomposition import TruncatedSVD
    from tabpfn import TabPFNClassifier

    rows = task.train
    desc, _ = _quantile_features(ctx, rows)
    svd = TruncatedSVD(128, random_state=0).fit(ctx.log_counts[rows])
    z = np.hstack([svd.transform(ctx.log_counts), desc]).astype(np.float32)
    model = TabPFNClassifier(device="cpu", n_estimators=4, random_state=ctx.seed, ignore_pretraining_limits=True).fit(z[rows], ctx.labels[rows])
    scores = np.concatenate([model.predict_proba(z[task.evaluate[i : i + 1000]])[:, 1] for i in range(0, len(task.evaluate), 1000)])  # batches of about 1,000 rows, as the package advises
    return scores


def fit_predict_atomjepa_head(ctx: TorchContext, task: Task, epochs: int = 25, n_seeds: int = 3) -> np.ndarray:
    """A small learned readout trained on the cached per-atom features of the frozen Atom-JEPA encoder (blocks 4, 6 and 8, 3 x 256 numbers per atom, ``atomjepa_embeddings``).

    Each atom is described by its features and its element (an embedding of the atomic number), projected to 128 numbers; an attention score per atom (a softmax over the atoms of the molecule) weights them
    into one vector, which a two-layer head turns into the logit of the label. Trained with the cross-entropy on the labelled rows of the task, three seeds averaged. The encoder never runs again: this is the
    affordable form of fine-tuning the top of a network whose full fine-tuning costs days per fit.
    """
    import torch
    from torch import nn

    if ctx.atoms is None:
        raise ValueError("atomjepa_head needs the per-atom cache (atomjepa_embeddings --extract)")
    torch.set_num_threads(3)
    first, count, feats, numbers = ctx.atoms["rows"], ctx.atoms["count"], ctx.atoms["features"], ctx.atoms["numbers"]
    rows = task.train[ctx.labels[task.train] >= 0]
    width = feats.shape[1] * feats.shape[2]
    mean = np.concatenate([feats[first[r] : first[r] + count[r]].reshape(-1, width) for r in rows[:800]]).astype(np.float32).mean(axis=0)
    std = np.concatenate([feats[first[r] : first[r] + count[r]].reshape(-1, width) for r in rows[:800]]).astype(np.float32).std(axis=0) + 1e-4
    max_atoms = int(count.max())

    def pad(index: np.ndarray):
        x = np.zeros((len(index), max_atoms, width), dtype=np.float32)
        z = np.zeros((len(index), max_atoms), dtype=np.int64)
        m = np.zeros((len(index), max_atoms), dtype=bool)
        for k, r in enumerate(index):
            n = count[r]
            if n:
                x[k, :n] = (feats[first[r] : first[r] + n].reshape(n, width).astype(np.float32) - mean) / std
                z[k, :n] = np.clip(numbers[first[r] : first[r] + n], 0, 53)
                m[k, :n] = True
        return torch.tensor(x), torch.tensor(z), torch.tensor(m)

    class Readout(nn.Module):
        def __init__(self):
            super().__init__()
            self.element = nn.Embedding(54, 16)
            self.project = nn.Sequential(nn.Linear(width + 16, 128), nn.ReLU(), nn.Dropout(0.3))
            self.score = nn.Linear(128, 1)
            self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

        def forward(self, x, z, mask):
            h = self.project(torch.cat([x, self.element(z)], dim=-1))
            logits = self.score(h).squeeze(-1).masked_fill(~mask, -1e9)
            weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
            return self.head((weights * h).sum(dim=1)).squeeze(-1)

    xt, zt, mt = pad(rows)
    label = torch.tensor(ctx.labels[rows].astype(np.float32))
    xe, ze, me = pad(task.evaluate)
    scores = []
    for s in range(n_seeds):
        torch.manual_seed(ctx.seed * 100 + s)
        net = Readout()
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        for _ in range(epochs):
            net.train()
            for batch in torch.randperm(len(xt)).split(64):
                if len(batch) < 2:
                    continue
                loss = nn.functional.binary_cross_entropy_with_logits(net(xt[batch], zt[batch], mt[batch]), label[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        net.eval()
        with torch.no_grad():
            scores.append(torch.sigmoid(torch.cat([net(xe[i : i + 256], ze[i : i + 256], me[i : i + 256]) for i in range(0, len(xe), 256)])).numpy())
    return np.mean(scores, axis=0)


AUX_HEAD_WEIGHT = 0.5  # the official auxiliary readings shape the representation at half the weight of the eight potency endpoints


def fit_predict_multitask(ctx: TorchContext, task: Task, output: str | None = None, epochs: int = 60, n_seeds: int = 3, aux: bool = False):
    """Multitask MLP over the eight continuous endpoints + a label head. Returns both scores from
    one training run -- {"label": the BCE head, "shift": t_hat - max(d_hat, 4) for this isoform} --
    or just one of them when ``output`` is given. With ``aux`` the regression heads also cover the twelve official
    auxiliary readings (single-concentration screen and Emax of both arms of the four enzymes), which are read only for
    the rows the task may train on and never predicted as scores.
    """
    import torch
    from torch import nn

    torch.set_num_threads(3)
    rows = training_rows(ctx, task)
    desc, _ = _quantile_features(ctx, rows)
    x = np.hstack([ctx.log_counts, desc]).astype(np.float32)
    y = ctx.targets[rows]
    if aux:
        if ctx.aux_targets is None:
            raise ValueError("aux heads need TorchContext.aux_targets (the official extra files)")
        y = np.hstack([y, ctx.aux_targets[rows]])
    n_heads = y.shape[1]
    head_weight = torch.tensor([1.0] * ctx.targets.shape[1] + [AUX_HEAD_WEIGHT] * (n_heads - ctx.targets.shape[1]))
    mu, sd = np.nanmean(y, axis=0), np.nanstd(y, axis=0) + 1e-6
    y_std = ((y - mu) / sd).astype(np.float32)
    label_rows = np.isin(rows, task.train) & (ctx.labels[rows] >= 0)
    label = np.where(label_rows, ctx.labels[rows], 0).astype(np.float32)

    class Net(nn.Module):
        def __init__(self, d_in):
            super().__init__()
            layers, width = [], d_in
            for h in (1024, 512):
                layers += [nn.Linear(width, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.35)]
                width = h
            self.trunk = nn.Sequential(*layers)
            self.regression = nn.Linear(width, n_heads)
            self.classification = nn.Linear(width, 1)

        def forward(self, inputs):
            z = self.trunk(inputs)
            return self.regression(z), self.classification(z).squeeze(-1)

    xt = torch.tensor(x[rows])
    yt = torch.tensor(np.nan_to_num(y_std))
    mask = torch.tensor(~np.isnan(y_std)).float()
    lt, lm = torch.tensor(label), torch.tensor(label_rows.astype(np.float32))
    xe = torch.tensor(x[task.evaluate])
    regs, clss = [], []
    for s in range(n_seeds):
        torch.manual_seed(ctx.seed * 100 + s)
        net = Net(x.shape[1])
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        steps = epochs * ((len(xt) + 127) // 128)
        schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=steps)
        for _ in range(epochs):
            net.train()
            for batch in torch.randperm(len(xt)).split(128):
                if len(batch) < 2:
                    continue
                reg, cls = net(xt[batch])
                loss_reg = ((reg - yt[batch]) ** 2 * mask[batch] * head_weight).sum() / (mask[batch] * head_weight).sum().clamp(min=1)
                loss_cls = (nn.functional.binary_cross_entropy_with_logits(cls, lt[batch], reduction="none") * lm[batch]).sum() / lm[batch].sum().clamp(min=1)
                optimizer.zero_grad()
                (loss_reg + loss_cls).backward()
                optimizer.step()
                schedule.step()
        net.eval()
        with torch.no_grad():
            reg, cls = net(xe)
        regs.append(reg.numpy() * sd + mu)
        clss.append(torch.sigmoid(cls).numpy())
    r = np.mean(regs, axis=0)
    scores = {
        "label": np.mean(clss, axis=0),
        "shift": r[:, 2 * ctx.isoform_position + 1] - np.maximum(r[:, 2 * ctx.isoform_position], ASSAY_FLOOR),
    }
    return scores if output is None else scores[output]


# one fit can feed several cached candidates: group name -> (function returning {candidate: scores})
GROUPS = {
    "tabicl_svd": lambda ctx, task: {"tabicl_svd": fit_predict_tabicl(ctx, task)},
    "tabicl_monroe": lambda ctx, task: {"tabicl_monroe": fit_predict_tabicl_monroe(ctx, task)},
    "tabicl_chemeleon": lambda ctx, task: {"tabicl_chemeleon": fit_predict_tabicl_chemeleon(ctx, task)},
    "tabpfn_chemeleon": lambda ctx, task: {"tabpfn_chemeleon": fit_predict_tabpfn_chemeleon(ctx, task)},
    "tabpfn": lambda ctx, task: {"tabpfn_svd": fit_predict_tabpfn(ctx, task)},
    "atomjepa_head": lambda ctx, task: {"atomjepa_head": fit_predict_atomjepa_head(ctx, task)},
    "mlp_multitask": lambda ctx, task: {f"mlp_multitask_{k}": v for k, v in fit_predict_multitask(ctx, task).items()},
    "mlp_multitask_aux": lambda ctx, task: {f"mlp_multitask_aux_{k}": v for k, v in fit_predict_multitask(ctx, task, aux=True).items()},
}
CANDIDATES = {"tabicl_svd": ["tabicl_svd"], "tabicl_monroe": ["tabicl_monroe"], "tabicl_chemeleon": ["tabicl_chemeleon"], "tabpfn_chemeleon": ["tabpfn_chemeleon"], "tabpfn": ["tabpfn_svd"], "atomjepa_head": ["atomjepa_head"], "mlp_multitask": ["mlp_multitask_label", "mlp_multitask_shift"], "mlp_multitask_aux": ["mlp_multitask_aux_label", "mlp_multitask_aux_shift"]}


def run(ctx: TorchContext, cache: PredictionCache, groups: list[str], deadline: float | None = None) -> int:
    """Fill the cache; stop after the task that is running when ``deadline`` (a time.time() value) passes. Returns the number of tasks still missing."""
    remaining = 0
    for group in groups:
        names = CANDIDATES[group]
        todo = [t for t in cache.plan.tasks() if any(not cache.has(n, t) for n in names)]
        for i, task in enumerate(todo, 1):
            if deadline is not None and time.time() > deadline:
                remaining += len(todo) - i + 1
                break
            for name, scores in GROUPS[group](ctx, task).items():
                cache.put(name, task, scores)
            logger.info("%s: %d/%d tasks", group, i, len(todo))
    return remaining


def build_context(train_csv: str, store_path: str, plan: NestedPlan, isoform: str) -> TorchContext:
    from cypmode.challenge.feature_store import FeatureStore

    train = pd.read_csv(train_csv)
    store = FeatureStore.load_or_build(store_path, _all_smiles(train_csv))
    n_all = len(store.smiles)
    targets = np.full((n_all, len(ARM_COLUMNS)), np.nan, dtype=np.float32)
    targets[: len(train)] = train[ARM_COLUMNS].to_numpy(dtype=np.float32)
    labels = np.full(n_all, -1, dtype=np.int64)
    labels[plan.pool] = plan.y
    excluded = set(plan.pool.tolist()) | set(plan.external.tolist())
    outside = np.array([i for i in range(len(train)) if i not in excluded], dtype=np.int64)
    log_counts = np.log1p(np.hstack([store.blocks["morgan_c2"], store.blocks["maccs"]])).astype(np.float32)
    from cypmode.challenge import monroe_embeddings

    monroe = monroe_embeddings.load_monroe_pca(list(store.smiles), n_components=128) if monroe_embeddings.CACHE.exists() else None
    from cypmode.challenge import frozen_embeddings

    chemeleon = frozen_embeddings.load_chemeleon_pca(list(store.smiles), n_components=128) if frozen_embeddings.CACHE.exists() else None
    from cypmode.challenge import official_extra

    aux_targets = None
    if official_extra.SCREEN_FILE.exists():
        aux = official_extra.aux_arrays(list(train["Molecule_Name"]) + [""] * (n_all - len(train)))
        aux_targets = np.column_stack([aux[name] for name in AUX_COLUMNS]).astype(np.float32)
    atoms = None
    from cypmode.challenge import atomjepa_embeddings

    if atomjepa_embeddings.ATOMS_CACHE.exists():
        from rdkit import Chem

        cached = np.load(atomjepa_embeddings.ATOMS_CACHE, allow_pickle=True)
        position = {k: i for i, k in enumerate(cached["smiles"])}
        order = np.array([position[Chem.MolToSmiles(Chem.MolFromSmiles(s))] for s in store.smiles])
        offsets = cached["offsets"]
        atoms = {"rows": offsets[order], "count": (offsets[1:] - offsets[:-1])[order], "features": cached["atoms"], "numbers": cached["numbers"]}
    return TorchContext(log_counts, store.blocks["rdkit_desc"], targets, labels, ISOFORMS.index(isoform), outside, monroe=monroe, chemeleon=chemeleon, atoms=atoms, aux_targets=aux_targets)


def _all_smiles(train_csv: str) -> list[str]:
    train = pd.read_csv(train_csv)
    test = pd.read_csv(str(train_csv).replace("TRAIN_TDI", "TEST-BLINDED"))
    return list(train["SMILES"]) + list(test["SMILES"])


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isoform", required=True, choices=["CYP3A4", "CYP2D6"])
    parser.add_argument("--groups", nargs="*", default=list(CANDIDATES), choices=list(CANDIDATES))
    parser.add_argument("--root", default="data/tdi_track/outputs/nested")
    parser.add_argument("--max-minutes", type=float, default=None, help="stop after the task that is running when this many minutes have passed")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = Path(args.root) / args.isoform
    plan = NestedPlan.load(root / "plan.npz")
    ctx = build_context("data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv", "data/tdi_track/outputs/caches/feature_store.npz", plan, args.isoform)
    deadline = None if args.max_minutes is None else time.time() + 60 * args.max_minutes
    print(f"TASKS_REMAINING {run(ctx, PredictionCache(root / 'cache', plan), args.groups, deadline)}")


if __name__ == "__main__":
    from cypmode import provenance

    provenance.announce()
    _main()

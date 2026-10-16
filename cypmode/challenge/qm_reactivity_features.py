"""Continuous QM-surrogate reactivity features via qmdesc (a real,
published graph neural network -- Guan et al., "Regio-Selectivity
Prediction with a Machine-Learned Reaction Representation and On-the-Fly
Quantum Mechanical Descriptors" -- github.com/yanfeiguan/qmdesc),
predicting real B3LYP/def2svp-level QM descriptors (Hirshfeld partial
charge, nucleophilic/electrophilic Fukui indices, NMR shielding, bond
order/length) per atom/bond from a SMILES string, no DFT calculation
needed.

Motivation: a real competitor's public lab notebook (leaderboard
research, 2026-10-01) identified a continuous QM-reactivity readout at
the same basic-nitrogen atoms this project's own binary amine_flag
already targets (cypmode.metrics.amines.has_trialkylamine) as their
single best CYP2D6 signal -- a genuinely different KIND of feature than
anything else tried for CYP2D6 this session (not another external
bioactivity dataset, not another architecture change): real quantum-
chemical reactivity instead of a yes/no structural flag.

Verified directly, real numbers on verapamil (a trialkylamine-bearing
TDI-positive drug from this project's own 6-drug reference panel):
qmdesc's per-atom output arrays are indexed over ALL atoms including
explicit hydrogens (RDKit's Chem.AddHs appends H atoms after the
original heavy atoms, preserving their indices), so a SMARTS match's
atom index on the plain (no-Hs) mol indexes correctly into qmdesc's
output without any remapping -- confirmed: atom 28/7 (the two real
basic nitrogens) came back as real Nitrogen with real, chemically
sensible values (fukui_neu notably exceeding fukui_elec, consistent
with a lone-pair-bearing nucleophilic tertiary amine).

Real, two-part install fix needed, both patched directly in the
installed package (not vendored/gitignored like third_party/TDC-main,
but the same kind of real third-party incompatibility, not a bug in our
own code): (1) qmdesc/handler.py's own `import pkg_resources` -- removed
by modern setuptools, same issue fixed earlier this session for
third_party/TDC-main -- replaced with importlib.resources. (2) PyTorch
2.6+ changed torch.load's default weights_only to True, and qmdesc's own
checkpoint stores its model args as a pickled argparse.Namespace, not on
the new default-safe allowlist -- fixed narrowly via
torch.serialization.add_safe_globals([argparse.Namespace]), not by
disabling the safety check entirely (qmdesc's own officially-distributed
checkpoint is a trusted source for that one addition).

CRITICAL: this module must NEVER be imported in the same process as
LightGBM -- qmdesc is torch-based, same real, confirmed segfault as
every other torch module in this project (see
cypmode.challenge.multitask's own docstring). This module's own
`_main()` therefore ONLY extracts and caches features -- it does not
import lightgbm, sklearn, or statsmodels. Run
cypmode.challenge.qm_reactivity_significance as a completely separate
process afterward to evaluate the cached features.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cypmode.metrics.amines import _TRIALKYLAMINE_PATTERNS

logger = logging.getLogger(__name__)

CACHE_PATH = Path("data/tdi_track/outputs/caches/qm_reactivity_cache.csv")
FEATURE_NAMES = [
    "qm_basicN_fukui_elec",
    "qm_basicN_fukui_neu",
    "qm_basicN_partial_charge",
    "qm_has_basicN",
    "qm_mol_fukui_elec_mean",
    "qm_mol_fukui_elec_max",
    "qm_mol_fukui_neu_mean",
    "qm_mol_fukui_neu_max",
]


def _load_handler():
    from qmdesc import ReactivityDescriptorHandler

    return ReactivityDescriptorHandler()


def compute_qm_reactivity_features(smiles: str, handler=None) -> np.ndarray:
    """One compound's real QM-reactivity feature vector (see
    FEATURE_NAMES for the 8 columns). The four qm_basicN_* columns are
    computed over the SAME trialkylamine-nitrogen atoms
    cypmode.metrics.amines.has_trialkylamine already screens for (max
    |value| across every match, i.e. the most reactive basic nitrogen if
    more than one is present); qm_has_basicN=0 and the three qm_basicN_*
    values are NaN (not zero -- a real zero would be a false claim of
    "measured, non-reactive," not "nothing to measure") when the
    compound has no trialkylamine nitrogen at all, matching this
    project's own established NaN-for-missingness convention (e.g.
    cypmode.challenge.tier3_features). The four qm_mol_* columns are
    whole-molecule Fukui summary stats, always present even when there's
    no basic nitrogen -- general reactivity context no feature here
    depended on before.

    Returns an all-NaN vector (not a crash) if parsing/inference fails,
    matching cypmode.challenge.chemberta_features's own resilience
    convention -- a batch over several thousand real compounds should
    not die on one bad string.
    """
    from rdkit import Chem

    if handler is None:
        handler = _load_handler()

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"could not parse SMILES: {smiles!r}")

        basicN_idx: set[int] = set()
        for pattern in _TRIALKYLAMINE_PATTERNS.values():
            for match in mol.GetSubstructMatches(pattern):
                basicN_idx.update(match)

        results = handler.predict(smiles)
        fukui_elec, fukui_neu, partial_charge = results["fukui_elec"], results["fukui_neu"], results["partial_charge"]

        if basicN_idx:
            best = max(basicN_idx, key=lambda i: abs(fukui_elec[i]))
            basicN_fukui_elec = float(fukui_elec[best])
            basicN_fukui_neu = float(fukui_neu[best])
            basicN_partial_charge = float(partial_charge[best])
            has_basicN = 1.0
        else:
            basicN_fukui_elec = basicN_fukui_neu = basicN_partial_charge = np.nan
            has_basicN = 0.0

        n_heavy = mol.GetNumAtoms()
        heavy_fukui_elec = fukui_elec[:n_heavy]
        heavy_fukui_neu = fukui_neu[:n_heavy]

        return np.array(
            [
                basicN_fukui_elec,
                basicN_fukui_neu,
                basicN_partial_charge,
                has_basicN,
                float(heavy_fukui_elec.mean()),
                float(heavy_fukui_elec.max()),
                float(heavy_fukui_neu.mean()),
                float(heavy_fukui_neu.max()),
            ]
        )
    except Exception as exc:
        logger.warning("QM reactivity features failed for %s: %s", smiles, exc)
        return np.full(len(FEATURE_NAMES), np.nan)


def compute_qm_reactivity_features_batch(
    smiles_list: list[str], cache_path: str | Path = CACHE_PATH, checkpoint_every: int = 200
) -> pd.DataFrame:
    """Computes (or loads from cache_path) one feature row per SMILES --
    same resumable, checkpointed pattern as
    cypmode.challenge.chemberta_features.compute_chemberta_embeddings_batch.
    """
    all_cols = ["SMILES", *FEATURE_NAMES]
    cache_path = Path(cache_path)
    if cache_path.exists():
        cached_smiles = set(pd.read_csv(cache_path, usecols=["SMILES"])["SMILES"])
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=all_cols).to_csv(cache_path, index=False)
        cached_smiles = set()

    todo = [s for s in smiles_list if s not in cached_smiles]
    if todo:
        handler = _load_handler()
        pending_rows = []
        for smiles in todo:
            features = compute_qm_reactivity_features(smiles, handler)
            pending_rows.append({"SMILES": smiles, **dict(zip(FEATURE_NAMES, features, strict=True))})
            if len(pending_rows) >= checkpoint_every:
                pd.DataFrame(pending_rows, columns=all_cols).to_csv(cache_path, mode="a", header=False, index=False)
                pending_rows = []
        if pending_rows:
            pd.DataFrame(pending_rows, columns=all_cols).to_csv(cache_path, mode="a", header=False, index=False)

    cached = pd.read_csv(cache_path)
    lookup = cached.set_index("SMILES")
    return lookup.loc[smiles_list].reset_index()


def load_cached_qm_reactivity_features(df: pd.DataFrame, cache_path: str | Path = CACHE_PATH) -> pd.DataFrame:
    """Read-only counterpart to add_qm_reactivity_features -- see
    cypmode.challenge.chemberta_features.load_cached_chemberta_embeddings
    for why this lives here rather than in the significance module.
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} doesn't exist -- run `python -m cypmode.challenge.qm_reactivity_features` "
            "first to populate it (a separate, torch-only process)."
        )
    cached = pd.read_csv(cache_path)
    missing = set(df["SMILES"]) - set(cached["SMILES"])
    if missing:
        raise ValueError(
            f"{len(missing)} compound(s) in df have no cached QM reactivity features -- "
            "re-run qm_reactivity_features.py's extraction to cover them."
        )
    lookup = cached.set_index("SMILES")
    out = df.copy()
    out["qm_reactivity_features"] = [lookup.loc[s, FEATURE_NAMES].to_numpy(dtype=np.float64) for s in df["SMILES"]]
    return out


def add_qm_reactivity_features(df: pd.DataFrame, cache_path: str | Path = CACHE_PATH) -> pd.DataFrame:
    """Adds a `qm_reactivity_features` column (one fixed-size np.array
    per row, 8 dims) to df, keyed off its own SMILES column."""
    features_df = compute_qm_reactivity_features_batch(list(df["SMILES"]), cache_path=cache_path)
    out = df.copy()
    out["qm_reactivity_features"] = [row[FEATURE_NAMES].to_numpy(dtype=np.float64) for _, row in features_df.iterrows()]
    return out


def _main() -> None:
    """Extraction only -- see module docstring for why this deliberately
    never imports lightgbm."""
    from cypmode.challenge.submission import TEST_SET_PATH
    from cypmode.challenge.tdi_baseline import load_tdi_training_data

    train_df = load_tdi_training_data()
    test_df = pd.read_csv(TEST_SET_PATH)
    all_smiles = sorted(set(train_df["SMILES"]) | set(test_df["SMILES"]))

    print(f"computing QM reactivity features for {len(all_smiles)} unique compounds...")
    result = compute_qm_reactivity_features_batch(all_smiles)
    n_with_basicN = int(result["qm_has_basicN"].sum())
    print(f"done: {n_with_basicN}/{len(result)} compounds have a real trialkylamine nitrogen")
    print(f"wrote {CACHE_PATH}")


if __name__ == "__main__":
    _main()

"""3-point pharmacophore fingerprints via pmapper
(github.com/DrrDom/pmapper) -- real, actively maintained, and lightweight
(only rdkit + networkx, no heavy compiled backend, unlike tblite --
confirmed directly no OpenMP-conflict risk with LightGBM the way
quantum_pharmacophore.py's tblite dependency has).

Real motivation: this project's existing feature stack has Morgan
fingerprints (2D topological substructure bits) and, separately,
tblite-derived scalar electronic-structure properties (HOMO/LUMO,
dipole, charges -- 0D, no spatial arrangement). Neither captures the
*3D spatial arrangement* of pharmacophoric features (hydrogen-bond
donors/acceptors, aromatic rings, hydrophobic regions, ionizable
groups) relative to each other -- the actual geometric pattern that
determines whether a molecule can present a given set of interactions
to a binding site in 3D. pmapper's fingerprint encodes exactly that:
every combination of 2-3 pharmacophoric features is hashed together
with the binned distances between them, into a fixed-size bit vector
(the same "3-point pharmacophore fingerprint" concept used in
established virtual-screening tools).

Real per-molecule cost, confirmed directly on a real complex drug
molecule (ketoconazole): ~350 ms (RDKit ETKDG embedding + MMFF
optimization + pmapper's own feature detection and fingerprint
hashing) -- ~36 minutes single-threaded for the full ~6,145-compound
TDI training set, trivially parallelizable via multiprocessing the same
way quantum_pharmacophore.py's batch runner is.

A compound whose 3D embedding fails gets an all-zero fingerprint (not a
crash) -- a real, if uninformative, representation for a binary bit
vector, unlike quantum_pharmacophore.py's NaN-filled fallback for
scalar QM properties, where zero would be a misleading fake value
rather than an honest "no bits set."
"""

import functools
import logging
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

PMAPPER_NBITS = 2048
PMAPPER_MIN_FEATURES = 2
PMAPPER_MAX_FEATURES = 3
CACHE_PATH = Path("data/tdi_track/outputs/caches/pharmacophore_fingerprint_cache.csv")


def _embed_3d_conformer(smiles: str, random_seed: int = 42):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=random_seed) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def compute_pharmacophore_fingerprint(
    smiles: str,
    nbits: int = PMAPPER_NBITS,
    min_features: int = PMAPPER_MIN_FEATURES,
    max_features: int = PMAPPER_MAX_FEATURES,
) -> np.ndarray:
    """One compound's dense binary pharmacophore fingerprint (nbits,).
    Any real failure (embedding, pmapper's own feature detection) is
    caught and logged by SMILES, returning an all-zero vector rather
    than propagating -- a batch over several thousand real compounds
    should not die on one difficult structure.
    """
    from pmapper.pharmacophore import Pharmacophore

    fp = np.zeros(nbits, dtype=np.float64)
    try:
        mol = _embed_3d_conformer(smiles)
        if mol is None:
            logger.warning("3D embedding failed for %s", smiles)
            return fp

        pharmacophore = Pharmacophore()
        pharmacophore.load_from_mol(mol)
        active_bits = pharmacophore.get_fp(min_features=min_features, max_features=max_features, nbits=nbits)
        if active_bits:
            fp[list(active_bits)] = 1.0
        return fp
    except Exception as exc:
        logger.warning("pmapper fingerprint failed for %s: %s", smiles, exc)
        return fp


def compute_pharmacophore_fingerprints_batch(
    smiles_list: list[str],
    cache_path: str | Path = CACHE_PATH,
    n_workers: int = 1,
    checkpoint_every: int = 100,
    nbits: int = PMAPPER_NBITS,
) -> pd.DataFrame:
    """Computes (or loads from cache_path) one fingerprint row per SMILES,
    one column per bit (`bit_0`..`bit_{nbits-1}`). Resumable the same way
    quantum_pharmacophore.compute_quantum_features_batch is: SMILES
    already cached aren't recomputed, and results are flushed to disk
    every checkpoint_every compounds, not only once at the end.

    Unlike quantum_pharmacophore's cache (8 scalar columns), this one is
    nbits+1 columns wide (2049 by default) -- rewriting the whole,
    growing file from scratch at every checkpoint (read all + concat +
    write all) would get expensive well before the full ~6,145-compound
    set finishes. New rows are appended directly to the CSV instead, so
    each checkpoint's cost depends only on checkpoint_every, not on how
    much of the cache already exists.
    """
    bit_cols = [f"bit_{i}" for i in range(nbits)]
    all_cols = ["SMILES", *bit_cols]
    cache_path = Path(cache_path)
    if cache_path.exists():
        cached_smiles = set(pd.read_csv(cache_path, usecols=["SMILES"])["SMILES"])
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=all_cols).to_csv(cache_path, index=False)
        cached_smiles = set()

    todo = [s for s in smiles_list if s not in cached_smiles]
    if todo:
        pool = multiprocessing.Pool(n_workers) if n_workers > 1 else None
        # a plain module-level function bound via functools.partial, not a
        # local lambda -- multiprocessing.Pool pickles the callable to send
        # to worker processes, and a lambda closed over local variables
        # isn't picklable (confirmed directly: this crashed the first real
        # n_workers=8 batch run with AttributeError, even though it worked
        # fine in tests, which only ever exercised the n_workers=1 path).
        compute = functools.partial(compute_pharmacophore_fingerprint, nbits=nbits)
        results_iter = pool.imap(compute, todo) if pool else map(compute, todo)

        pending_rows = []
        try:
            for smiles, fp in zip(todo, results_iter, strict=True):
                pending_rows.append({"SMILES": smiles, **dict(zip(bit_cols, fp, strict=True))})
                if len(pending_rows) >= checkpoint_every:
                    pd.DataFrame(pending_rows, columns=all_cols).to_csv(cache_path, mode="a", header=False, index=False)
                    pending_rows = []
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        if pending_rows:
            pd.DataFrame(pending_rows, columns=all_cols).to_csv(cache_path, mode="a", header=False, index=False)

    cached = pd.read_csv(cache_path)
    lookup = cached.set_index("SMILES")
    return lookup.loc[smiles_list].reset_index()


def add_pharmacophore_fingerprints(
    df: pd.DataFrame, cache_path: str | Path = CACHE_PATH, n_workers: int = 1
) -> pd.DataFrame:
    """Adds a `pharmacophore_fp` column (one fixed-size np.array per row)
    to df, keyed off its own SMILES column.
    """
    bit_cols = [f"bit_{i}" for i in range(PMAPPER_NBITS)]
    fps_df = compute_pharmacophore_fingerprints_batch(list(df["SMILES"]), cache_path=cache_path, n_workers=n_workers)
    out = df.copy()
    out["pharmacophore_fp"] = [row[bit_cols].to_numpy(dtype=np.float64) for _, row in fps_df.iterrows()]
    return out


def _main() -> None:
    import argparse

    from cypmode.challenge.knn_features import _morgan_fp
    from cypmode.challenge.stacking_significance import (
        repeated_cv_stacking_comparison,
        paired_comparison_summary,
    )
    from cypmode.challenge.tdi_ablation import add_ablation_features
    from cypmode.challenge.tdi_baseline import (
        TDI_ISOFORMS,
        compute_descriptors,
        load_tdi_training_data,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args()

    tdi_df = add_ablation_features(compute_descriptors(load_tdi_training_data()))
    print(f"computing pharmacophore fingerprints for {len(tdi_df)} compounds ({args.n_workers} worker(s))...")
    tdi_df = add_pharmacophore_fingerprints(tdi_df, n_workers=args.n_workers)

    n_empty = tdi_df["pharmacophore_fp"].apply(lambda f: f.sum() == 0).sum()
    print(f"{n_empty}/{len(tdi_df)} compounds got an all-zero fingerprint (embedding or feature-detection failure)")

    # Going straight to the real repeated-CV + Tukey HSD rigor this
    # project's own standing rule requires before trusting a stacking
    # result -- not a single-split number first, per the lesson learned
    # from quantum_pharmacophore.py's own single-split-then-recheck
    # detour.
    all_results = []
    all_summaries = []
    for cyp in TDI_ISOFORMS:
        col = f"{cyp}_is_TDI"
        cyp_df = tdi_df.dropna(subset=[col]).reset_index(drop=True)
        # An all-zero fingerprint isn't a neutral value -- it's a specific
        # false claim ("no pharmacophoric features detected"), the real
        # signature of a failed embedding/feature-detection, not a valid
        # observation. Excluded here the same way quantum_pharmacophore.py
        # and chemberta_significance.py exclude their own per-compound
        # failure sentinels before scoring -- n_empty above is a global
        # count across all compounds, not yet isoform-filtered.
        valid = cyp_df["pharmacophore_fp"].apply(lambda f: f.sum() != 0)
        cyp_df = cyp_df[valid].reset_index(drop=True)
        y = cyp_df[col].astype(int).to_numpy()
        m3_features = np.stack(cyp_df["m3_no_amine_features"])
        fps = [_morgan_fp(s) for s in cyp_df["SMILES"]]
        aux_features = {"Pharmacophore": np.stack(cyp_df["pharmacophore_fp"])}

        print(f"\nRunning 5x5=25-fold repeated comparison for {cyp}...")
        results = repeated_cv_stacking_comparison(m3_features, y, fps, aux_features, n_repeats=5, n_splits=5)
        results.insert(0, "isoform", cyp)
        all_results.append(results)

        print(f"-- {cyp}: mean MCC per variant (25 folds each) --")
        print(results.groupby("variant")["mcc"].agg(["mean", "std"]).to_string())

        summary = paired_comparison_summary(results)
        summary.insert(0, "isoform", cyp)
        all_summaries.append(summary)
        print(f"-- {cyp}: Tukey HSD pairwise comparison --")
        print(summary.to_string(index=False))

    raw_path = Path("data/tdi_track/outputs/comparisons/pharmacophore_fingerprint_significance_raw.csv")
    pd.concat(all_results, ignore_index=True).to_csv(raw_path, index=False)
    print(f"\nwrote {raw_path}")

    tukey_path = Path("data/tdi_track/outputs/comparisons/pharmacophore_fingerprint_significance_tukey.csv")
    pd.concat(all_summaries, ignore_index=True).to_csv(tukey_path, index=False)
    print(f"wrote {tukey_path}")


if __name__ == "__main__":
    _main()

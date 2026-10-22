# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Quantum-derived pharmacophore features via tblite's own Python
bindings to GFN2-xTB, a real, published semi-empirical quantum method --
NOT xtb-python (confirmed unmaintained) and NOT full DFT (confirmed
infeasible: minutes-to-hours per molecule on this hardware). Confirmed
directly tractable at ~1 second per molecule, CPU-only, no GPU needed
(this machine's one GPU stays free for Boltz-2's structural runs) --
the full TDI training set (~6,100 compounds) does not need scoping down
to any subset.

Testing note: this module's own tests (tests/test_quantum_pharmacophore.py)
must run in their own separate pytest invocation, never mixed into the
main suite -- see pyproject.toml's `tblite` marker for why: tblite's
own bundled OpenMP runtime conflicts with LightGBM's once both are
loaded into one process, the same real conflict class already
documented for torch+LightGBM (test_multitask.py), except this one
hangs (a real deadlock) rather than segfaulting. Confirmed directly by
reproducing it twice with the full suite and once more with a minimal
two-file repro.

Real motivation: RDKit descriptors and Morgan fingerprints are purely
2D-topology-derived. CYP450-mediated oxidation and TDI both depend on a
compound's actual electronic structure -- how electron-rich a site is,
how readily the molecule donates electrons to the heme iron -- which no
graph descriptor captures directly. GFN2-xTB gives a real electronic-
structure single-point calculation: molecular orbital energies, per-atom
partial charges, and the dipole moment, from an actual (if approximate)
quantum calculation rather than a topological proxy for one.

Feature vector (8 values, one row per compound), from a single GFN2-xTB
single-point on one MMFF-optimized 3D conformer:
- homo_energy, lumo_energy, homo_lumo_gap (Hartree). HOMO = the
  highest-energy orbital with occupation > 0; LUMO = the next orbital
  up (occupation == 0), read off tblite's own orbital-energies /
  orbital-occupations arrays, which are already energy-sorted.
- dipole_magnitude (atomic units, i.e. e*a0; confirmed against a real
  reference value -- ethanol's GFN2-xTB dipole here works out to ~2.0 D,
  the right ballpark for its known experimental gas-phase dipole of
  ~1.69 D from a single semi-empirical single-point).
- total_energy (Hartree; not physically comparable across molecules of
  different size/composition on its own, but usable as one input among
  several to a classifier the same way any other scale-free-in-context
  descriptor is).
- charge_min, charge_max, charge_std (per-atom GFN2-xTB partial
  charges' summary statistics -- a real QM-derived alternative to
  RDKit's Gasteiger charges; charge_mean is omitted, since GFN2-xTB
  partial charges sum to the molecule's net formal charge and so
  charge_mean is just that constant divided by atom count, redundant
  with descriptors already in M3).

A compound whose 3D embedding fails (RDKit's ETKDG occasionally can't
embed a valid conformer) or whose xTB single-point doesn't converge
gets a NaN-filled feature row, logged by SMILES -- not a hard crash
that kills a several-thousand-compound batch.
"""

import logging
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

BOHR_PER_ANGSTROM = 1.0 / 0.52917721067
FEATURE_NAMES = [
    "qm_homo_energy",
    "qm_lumo_energy",
    "qm_homo_lumo_gap",
    "qm_dipole_magnitude",
    "qm_total_energy",
    "qm_charge_min",
    "qm_charge_max",
    "qm_charge_std",
]
CACHE_PATH = Path("data/tdi_track/outputs/caches/quantum_pharmacophore_cache.csv")


def embed_3d_conformer(smiles: str, random_seed: int = 42):
    """RDKit ETKDG embedding + MMFF optimization. Returns None (not
    raised) if embedding fails, since a handful of difficult SMILES
    shouldn't crash a batch run -- see module docstring.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=random_seed) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def compute_gfn2_xtb_properties(mol) -> dict:
    """Runs one real GFN2-xTB single-point via tblite and returns its raw
    dict (energy, charges, dipole, orbital-energies, orbital-occupations,
    etc.) -- imports tblite lazily so the rest of this module (and any
    caller that only needs FEATURE_NAMES or cached results) doesn't need
    tblite installed.
    """
    from tblite.interface import Calculator

    conf = mol.GetConformer()
    positions_bohr = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    ) * BOHR_PER_ANGSTROM
    numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    charge = float(Chem.GetFormalCharge(mol))

    calc = Calculator("GFN2-xTB", numbers, positions_bohr, charge=charge)
    calc.set("verbosity", 0)
    result = calc.singlepoint()
    return result.dict()


def _nan_features() -> dict:
    return {name: np.nan for name in FEATURE_NAMES}


def extract_quantum_features(smiles: str) -> dict:
    """One compound's full pipeline: embed -> GFN2-xTB single-point ->
    fixed-size feature dict (FEATURE_NAMES). Any failure (embedding,
    SCF non-convergence, or any other real error tblite/RDKit raises)
    is caught and logged by SMILES, returning a NaN-filled dict rather
    than propagating -- a batch over several thousand real compounds
    should not die on one difficult structure.
    """
    try:
        mol = embed_3d_conformer(smiles)
        if mol is None:
            logger.warning("3D embedding failed for %s", smiles)
            return _nan_features()

        props = compute_gfn2_xtb_properties(mol)
        occ = np.asarray(props["orbital-occupations"])
        energies = np.asarray(props["orbital-energies"])
        occupied = np.where(occ > 0.5)[0]
        if occupied.size == 0 or occupied.max() + 1 >= len(energies):
            logger.warning("could not locate a HOMO/LUMO pair for %s", smiles)
            return _nan_features()
        homo_idx = occupied.max()
        lumo_idx = homo_idx + 1
        homo = float(energies[homo_idx])
        lumo = float(energies[lumo_idx])

        charges = np.asarray(props["charges"])
        dipole = np.asarray(props["dipole"])

        return {
            "qm_homo_energy": homo,
            "qm_lumo_energy": lumo,
            "qm_homo_lumo_gap": lumo - homo,
            "qm_dipole_magnitude": float(np.linalg.norm(dipole)),
            "qm_total_energy": float(props["energy"]),
            "qm_charge_min": float(charges.min()),
            "qm_charge_max": float(charges.max()),
            "qm_charge_std": float(charges.std()),
        }
    except Exception as exc:
        logger.warning("GFN2-xTB single-point failed for %s: %s", smiles, exc)
        return _nan_features()


def compute_quantum_features_batch(
    smiles_list: list[str],
    cache_path: str | Path = CACHE_PATH,
    n_workers: int = 1,
    checkpoint_every: int = 50,
) -> pd.DataFrame:
    """Computes (or loads from cache_path) one feature row per SMILES.
    Resumable: SMILES already present in cache_path are not recomputed,
    so a batch interrupted partway through (real risk at ~1 sec/compound
    over several thousand compounds) picks up where it left off -- not
    only across separate invocations, but within a single long run too:
    results are flushed to cache_path every checkpoint_every compounds,
    not only once at the very end, so a crash mid-batch loses at most
    checkpoint_every compounds' worth of work, matching the same
    checkpoint-as-you-go discipline cypmode.validation.batch.run_batch
    uses for its own multi-hour Boltz-2 batches.

    n_workers > 1 uses multiprocessing.Pool.imap (order-preserving, so
    results can still be zipped against todo directly) -- safe here
    since extract_quantum_features has no shared state and each call is
    fully independent.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
    else:
        cached = pd.DataFrame(columns=["SMILES", *FEATURE_NAMES])

    todo = [s for s in smiles_list if s not in set(cached["SMILES"])]
    if todo:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pool = multiprocessing.Pool(n_workers) if n_workers > 1 else None
        results_iter = pool.imap(extract_quantum_features, todo) if pool else map(extract_quantum_features, todo)

        pending_rows = []
        try:
            for smiles, features in zip(todo, results_iter, strict=True):
                pending_rows.append({"SMILES": smiles, **features})
                if len(pending_rows) >= checkpoint_every:
                    cached = pd.concat([cached, pd.DataFrame(pending_rows)], ignore_index=True)
                    cached.to_csv(cache_path, index=False)
                    pending_rows = []
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        if pending_rows:
            cached = pd.concat([cached, pd.DataFrame(pending_rows)], ignore_index=True)
            cached.to_csv(cache_path, index=False)

    lookup = cached.set_index("SMILES")
    return lookup.loc[smiles_list].reset_index()


def add_quantum_features(
    df: pd.DataFrame, cache_path: str | Path = CACHE_PATH, n_workers: int = 1
) -> pd.DataFrame:
    """Adds a `quantum_features` column (one fixed-size np.array per row,
    order matching FEATURE_NAMES) to df, keyed off its own SMILES column.
    Rows whose features couldn't be computed carry a NaN-filled array --
    a caller stacking this into a classifier should drop/impute those
    rather than silently letting LightGBM (which does tolerate NaN) mask
    a real computation failure as a missing-at-random value.
    """
    features_df = compute_quantum_features_batch(list(df["SMILES"]), cache_path=cache_path, n_workers=n_workers)
    out = df.copy()
    out["quantum_features"] = [row[FEATURE_NAMES].to_numpy(dtype=np.float64) for _, row in features_df.iterrows()]
    return out


def load_cached_quantum_features(df: pd.DataFrame, cache_path: str | Path = CACHE_PATH) -> pd.DataFrame:
    """Joins df against the real quantum-feature cache by SMILES, without
    computing anything (and therefore never importing tblite) -- the
    read-only counterpart to add_quantum_features, for a caller that
    must not risk importing tblite into a process that also uses
    LightGBM (see module docstring). Raises a clear error if the cache
    doesn't cover every compound in df, rather than a silent, all-NaN
    join. Lives here, not in quantum_pharmacophore_significance.py,
    specifically so a lightgbm-free caller could use it too, matching
    why chemberta_features.load_cached_chemberta_embeddings lives in
    the torch-only module rather than its own significance module.
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} doesn't exist -- run `python -m cypmode.challenge.quantum_pharmacophore` "
            "first to populate it (a separate, tblite-only process)."
        )
    cached = pd.read_csv(cache_path)
    missing = set(df["SMILES"]) - set(cached["SMILES"])
    if missing:
        raise ValueError(
            f"{len(missing)} compound(s) in df have no cached quantum features -- "
            "run the extraction step above against the full compound set first."
        )
    lookup = cached.set_index("SMILES")
    out = df.copy()
    out["quantum_features"] = [
        lookup.loc[s, FEATURE_NAMES].to_numpy(dtype=np.float64) for s in df["SMILES"]
    ]
    return out


def _main() -> None:
    """Extraction only -- computes and caches quantum features, nothing
    else. Deliberately never imports lightgbm: see module docstring for
    the real, confirmed tblite+LightGBM same-process conflict. Run
    `python -m cypmode.challenge.quantum_pharmacophore_significance`
    (a separate, LightGBM-only process) afterward for the ablation/
    significance testing against M3 -- the same two-process split
    chemberta_features.py/chemberta_significance.py already use, for
    the analogous torch+LightGBM conflict.
    """
    import argparse

    from cypmode.challenge.tdi_ablation import add_ablation_features
    from cypmode.challenge.tdi_baseline import (
        compute_descriptors,
        load_tdi_training_data,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args()

    tdi_df = add_ablation_features(compute_descriptors(load_tdi_training_data()))
    print(f"computing GFN2-xTB features for {len(tdi_df)} compounds ({args.n_workers} worker(s))...")
    tdi_df = add_quantum_features(tdi_df, n_workers=args.n_workers)

    n_failed = tdi_df["quantum_features"].apply(lambda f: np.isnan(f).any()).sum()
    print(f"{n_failed}/{len(tdi_df)} compounds had a failed QM calculation (NaN features)")
    print(f"cached to {CACHE_PATH}")


if __name__ == "__main__":
    _main()

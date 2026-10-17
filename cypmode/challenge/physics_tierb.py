"""Physics block, tier B: Boltzmann-weighted conformers and relaxed hydrogen-abstraction energies for the two best sites.

Tier A (``physics_block``) uses one MMFF conformer and vertical energies: removing a hydrogen from a frozen geometry leaves the
radical unrelaxed, which overstates the abstraction energy by tens of kcal/mol (toluene benzylic: about 128 against 90
experimentally) and adds site-dependent noise. Tier B keeps the quantities that carried the most split gain in tier A and makes them
more physical, for a compact set of about twenty columns:

  * conformer ensemble: up to ``MAX_CONFORMERS`` lowest-MMFF conformers (inside the ensemble window), each scored by a GFN2-xTB
    single point in implicit water (ALPB); Boltzmann weights at 298 K; ionisation potential, solvation free energy, HOMO-LUMO gap and
    the condensed Fukui f- of binder and aromatic atoms are Boltzmann averages over the conformers;
  * relaxed hydrogen abstraction for the two sites with the lowest vertical energy in tier A (read from the tier-A atom table): the
    parent and the radical are relaxed locally (the radical centre, its bonded neighbours and their hydrogens move, the rest is frozen; at most
    ``RELAX_ITERATIONS`` L-BFGS steps on GFN2 gradients; the hydrogen taken is the one with the lowest vertical energy), and the energy is E(radical) + E(H atom) - E(parent), both relaxed in the
    same region so that the frozen remainder cancels. The relaxation gain over the vertical energy is a column of its own.

tblite and LightGBM must not meet in one process: extraction here, evaluation (which only reads the CSV cache) elsewhere.

Usage: python -m cypmode.challenge.physics_tierb --extract [--n-workers 4] [--limit N]
"""

import argparse
import logging
import multiprocessing
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from cypmode.challenge import physics_block as pb

logger = logging.getLogger(__name__)

CACHE_PATH = Path("data/tdi_track/outputs/caches/physics_b_cache.csv")
MAX_CONFORMERS = 3
N_SITES = 2
RELAX_ITERATIONS = 8
RELAX_SHELL_BONDS = 1
SITE_ROLES = ("alpha_N", "alpha_O", "benzylic", "allylic", "aliphatic")
FEATURE_NAMES = [
    "phys_b_n_conf", "phys_b_conf_spread_kcal", "phys_b_conf_entropy", "phys_b_ip_ev", "phys_b_dg_solv_kcal", "phys_b_gap_ev",
    "phys_b_fm_top1", "phys_b_fm_azole", "phys_b_fm_pyridine", "phys_b_fm_amine", "phys_b_fm_aromatic", "phys_b_fm_alpha_n",
    "phys_b_hat_relaxed_1", "phys_b_hat_relaxed_2", "phys_b_hat_relaxed_gap", "phys_b_hat_relax_gain_1", "phys_b_hat_relax_gain_2",
    *[f"phys_b_site1_is_{r}" for r in SITE_ROLES],
]


def boltzmann_weights(energies_kcal: np.ndarray) -> np.ndarray:
    e = np.asarray(energies_kcal, dtype=float)
    w = np.exp(-(e - e.min()) / pb.KT_KCAL)
    return w / w.sum()


def mobile_atoms(mol_h: Chem.Mol, centre: int, shell: int = RELAX_SHELL_BONDS) -> list[int]:
    """The radical centre, every atom within ``shell`` bonds of it, and the hydrogens of those atoms."""
    topo = Chem.GetDistanceMatrix(mol_h)
    near = {i for i in range(mol_h.GetNumAtoms()) if topo[centre, i] <= shell}
    near |= {n.GetIdx() for i in list(near) for n in mol_h.GetAtomWithIdx(i).GetNeighbors() if n.GetAtomicNum() == 1}
    return sorted(near)


def relax_local(numbers: np.ndarray, positions: np.ndarray, charge: int, mobile: list[int], max_iter: int = RELAX_ITERATIONS) -> tuple[float, np.ndarray]:
    """GFN2-xTB energy (Hartree) after a bounded L-BFGS relaxation of the ``mobile`` atoms; the other atoms stay frozen."""
    from scipy.optimize import minimize

    mobile = np.asarray(mobile)

    def fun(x: np.ndarray):
        pos = positions.copy()
        pos[mobile] = x.reshape(-1, 3)
        result = pb._single_point(numbers, pos, charge)
        gradient = np.asarray(result.get("gradient"))[mobile] * pb.BOHR_PER_ANGSTROM  # Hartree/bohr -> Hartree/A
        return float(result.get("energy")), gradient.ravel()

    fit = minimize(fun, positions[mobile].ravel(), jac=True, method="L-BFGS-B", options={"maxiter": max_iter, "maxfun": 2 * max_iter + 2})
    relaxed = positions.copy()
    relaxed[mobile] = fit.x.reshape(-1, 3)
    return float(fit.fun), relaxed


def relaxed_hat(mol_h: Chem.Mol, numbers: np.ndarray, positions: np.ndarray, charge: int, centre: int, max_iter: int = RELAX_ITERATIONS) -> float | None:
    """Relaxed hydrogen-abstraction energy (kcal/mol, free-hydrogen reference) at ``centre``; the best of its hydrogens."""
    mobile = mobile_atoms(mol_h, centre)
    parent_energy, _ = relax_local(numbers, positions, charge, mobile, max_iter)
    best_h, best_vertical = None, None
    for h in pb._hydrogens(mol_h, centre):  # the hydrogen the enzyme takes is the one whose removal is cheapest
        keep = np.array([k for k in range(len(numbers)) if k != h])
        vertical = pb._safe(pb._single_point, numbers[keep], positions[keep], charge)
        if vertical is not None and (best_vertical is None or float(vertical.get("energy")) < best_vertical):
            best_h, best_vertical = h, float(vertical.get("energy"))
    if best_h is None:
        return None
    keep = np.array([k for k in range(len(numbers)) if k != best_h])
    remap = {old: new for new, old in enumerate(keep)}
    energy, _ = relax_local(numbers[keep], positions[keep], charge, [remap[m] for m in mobile if m != best_h], max_iter)
    return (energy + pb.hydrogen_atom_energy() - parent_energy) * pb.HARTREE_KCAL


def compute_molecule_b(smiles: str, sites: list[tuple[int, float, str]]) -> dict:
    """Tier-B features of one compound; ``sites`` = [(atom index, tier-A vertical HAT, C-H role)] best first."""
    out = {n: np.nan for n in FEATURE_NAMES}
    mol = pb.largest_fragment(smiles)
    if mol is None:
        return out
    mol_h = Chem.AddHs(mol)
    ids, energies = pb.conformer_ensemble(mol_h)
    if not ids:
        flat = Chem.Mol(mol)
        Chem.RemoveStereochemistry(flat)
        mol, mol_h = flat, Chem.AddHs(flat)
        ids, energies = pb.conformer_ensemble(mol_h)
    if not ids:
        return out
    n_heavy = mol.GetNumAtoms()
    numbers = np.array([a.GetAtomicNum() for a in mol_h.GetAtoms()])
    charge = Chem.GetFormalCharge(mol)
    ch_role, binder_role = pb.assign_roles(mol_h)
    window = [i for i in np.argsort(energies) if energies[i] - energies.min() <= pb.ENSEMBLE_WINDOW_KCAL][:MAX_CONFORMERS]

    rows = []
    for i in window:
        positions = mol_h.GetConformer(ids[i]).GetPositions()
        water = pb._safe(pb._single_point, numbers, positions, charge, "water")
        gas = pb._safe(pb._single_point, numbers, positions, charge)
        cation = pb._safe(pb._single_point, numbers, positions, charge + 1)
        if water is None or gas is None or cation is None:
            continue
        q0 = pb._heavy_charges(gas.get("charges"), mol_h, n_heavy)
        fm = pb._heavy_charges(cation.get("charges"), mol_h, n_heavy) - q0
        occ = np.asarray(gas.get("orbital-occupations"))
        homo = int(np.where(occ > 0.5)[0].max())
        eps = np.asarray(gas.get("orbital-energies"))
        rows.append({
            "e_water": float(water.get("energy")) * pb.HARTREE_KCAL, "ip": (float(cation.get("energy")) - float(gas.get("energy"))) * pb.HARTREE_EV,
            "dg": (float(water.get("energy")) - float(gas.get("energy"))) * pb.HARTREE_KCAL, "gap": (eps[homo + 1] - eps[homo]) * pb.HARTREE_EV, "fm": fm,
        })
    if not rows:
        return out
    w = boltzmann_weights(np.array([r["e_water"] for r in rows]))
    avg = lambda key: float(sum(wi * r[key] for wi, r in zip(w, rows, strict=True)))
    fm_avg = sum(wi * r["fm"] for wi, r in zip(w, rows, strict=True))

    def group_max(roles: list[str], names: tuple[str, ...]) -> float:
        atoms = [i for i in range(n_heavy) if roles[i] in names]
        return float(np.max(fm_avg[atoms])) if atoms else np.nan

    out.update({
        "phys_b_n_conf": float(len(rows)), "phys_b_conf_spread_kcal": float(max(r["e_water"] for r in rows) - min(r["e_water"] for r in rows)),
        "phys_b_conf_entropy": float(-(w * np.log(np.clip(w, 1e-12, None))).sum()), "phys_b_ip_ev": avg("ip"), "phys_b_dg_solv_kcal": avg("dg"),
        "phys_b_gap_ev": avg("gap"), "phys_b_fm_top1": float(np.max(fm_avg)),
        "phys_b_fm_azole": group_max(binder_role, ("azole_N",)), "phys_b_fm_pyridine": group_max(binder_role, ("pyridine_N",)),
        "phys_b_fm_amine": group_max(binder_role, ("amine_N",)), "phys_b_fm_aromatic": group_max(ch_role, ("aromatic_ch",)),
        "phys_b_fm_alpha_n": group_max(ch_role, ("alpha_N",)),
    })
    lowest = window[int(np.argmin([r["e_water"] for r in rows]))] if len(rows) == len(window) else window[0]
    positions = mol_h.GetConformer(ids[lowest]).GetPositions()
    relaxed = []
    for atom, vertical, _ in sites[:N_SITES]:
        value = pb._safe(relaxed_hat, mol_h, numbers, positions, charge, int(atom))
        relaxed.append((value, vertical))
    for k, (value, vertical) in enumerate(relaxed, 1):
        if value is not None:
            out[f"phys_b_hat_relaxed_{k}"] = float(value)
            out[f"phys_b_hat_relax_gain_{k}"] = float(vertical - value)
    if len(relaxed) == 2 and relaxed[0][0] is not None and relaxed[1][0] is not None:
        out["phys_b_hat_relaxed_gap"] = float(relaxed[1][0] - relaxed[0][0])
    if sites:
        for role in SITE_ROLES:
            out[f"phys_b_site1_is_{role}"] = float(sites[0][2] == role)
    return out


def _worker(job: tuple[str, list]):
    smiles, sites = job
    try:
        return compute_molecule_b(smiles, sites)
    except Exception as exc:
        logger.warning("tier-B extraction failed for %s: %s", smiles, exc)
        return {n: np.nan for n in FEATURE_NAMES}


def top_sites(atoms: pd.DataFrame, n: int = N_SITES) -> dict[str, list[tuple[int, float, str]]]:
    """{canonical SMILES: [(atom, vertical HAT, role)] best first} from the tier-A atom table."""
    table = atoms.dropna(subset=["hat_kcal"]).sort_values(["SMILES", "hat_kcal", "atom"])
    return {s: list(zip(g["atom"].astype(int), g["hat_kcal"].astype(float), g["ch_role"].astype(str), strict=True))[:n] for s, g in table.groupby("SMILES")}


def extract(smiles_list: list[str], cache_path: Path = CACHE_PATH, atoms_path: Path = pb.ATOMS_PATH, n_workers: int = 1, checkpoint_every: int = 50) -> None:
    keys = list(dict.fromkeys(pb.canonical(s) for s in smiles_list))
    done = set(pd.read_csv(cache_path, usecols=["SMILES"])["SMILES"]) if cache_path.exists() else set()
    sites = top_sites(pd.read_csv(atoms_path)) if Path(atoms_path).exists() else {}
    todo = [k for k in keys if k not in done]
    logger.info("tier-B extraction: %d compounds, %d cached, %d to do", len(keys), len(done), len(todo))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    jobs = [(k, sites.get(k, [])) for k in todo]
    pool = multiprocessing.Pool(n_workers, initializer=pb._init_worker) if n_workers > 1 else None
    results = pool.imap(_worker, jobs, chunksize=2) if pool else map(_worker, jobs)
    pending: list[dict] = []

    def flush():
        if pending:
            pd.DataFrame(pending, columns=["SMILES", *FEATURE_NAMES]).to_csv(cache_path, mode="a", header=not cache_path.exists(), index=False)
            pending.clear()

    try:
        for n, (smiles, features) in enumerate(zip(todo, results, strict=True), 1):
            pending.append({"SMILES": smiles, **features})
            if n % checkpoint_every == 0:
                flush()
                logger.info("tier-B extraction: %d / %d", n, len(todo))
    finally:
        flush()
        if pool is not None:
            pool.close()
            pool.join()


def load_tierb_features(smiles: list[str], cache_path: Path = CACHE_PATH) -> np.ndarray:
    """Feature matrix (rows follow ``smiles``) from the cache; never imports tblite."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"{cache_path} is missing: run `python -m cypmode.challenge.physics_tierb --extract` first (a tblite-only process)")
    table = pd.read_csv(cache_path).drop_duplicates("SMILES").set_index("SMILES")
    keys = [pb.canonical(s) for s in smiles]
    missing = [k for k in keys if k not in table.index]
    if missing:
        raise ValueError(f"{len(missing)} compounds have no tier-B physics features, e.g. {missing[0]!r}")
    return table.loc[keys, FEATURE_NAMES].to_numpy(dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cache", default=str(CACHE_PATH))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.extract:
        train = pd.read_csv("data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv")
        test = pd.read_csv("data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv")
        smiles = list(train.SMILES) + list(test.SMILES)
        extract(smiles[: args.limit] if args.limit else smiles, Path(args.cache), n_workers=args.n_workers)


if __name__ == "__main__":
    main()

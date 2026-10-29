# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Atom-, site- and pharmacophore-aware GFN2-xTB physics block (job 070).

The two earlier quantum blocks (global GFN2 descriptors, an ML surrogate of Fukui indices at the basic nitrogen)
were null. Both described the molecule as a whole or looked at one atom type. TDI is a statement about specific
atoms: a heme-binding atom that coordinates the iron, and an oxidisable C-H or heteroatom that the enzyme turns
into a reactive intermediate. This block therefore measures the physics of those atoms and of the geometry between
them, in four levels:

  L1 molecule   conformer-aware frontier energies, vertical ionisation potential and electron affinity, conceptual-DFT
                hardness / electrophilicity, ALPB(water) solvation, dipole, shape.
  L2 atom       GFN2 charges and condensed Fukui functions (finite differences of N, N+1, N-1 electron charges,
                hydrogens folded into their heavy atom), the vertical hydrogen-abstraction energy of every
                non-aromatic C-H (the step Compound I performs), the vertical proton affinity of every nitrogen that
                could coordinate the heme iron (gas phase and ALPB water), solvent-accessible surface of the atom.
  L3 site       the atoms grouped by chemical role (benzylic / alpha-N / alpha-O / allylic / methylenedioxy ... C-H;
                azole / pyridine / amine / aniline / nitrile N; thioether S; furan/thiophene), with per-role
                statistics, plus how the best oxidation sites compete with each other (sorted energies, count
                within 3 kcal/mol, Boltzmann site entropy).
  L4 pharmacophore  3D distances over an MMFF conformer ensemble between the heme-binding atoms and the
                energetically best oxidation sites (mean / min / max over conformers), their solvent exposure, how
                many good sites sit in the 4-8 A shell around a binder, the reach of the molecule from the binder.

The hydrogen-abstraction and proton-affinity energies are vertical (frozen geometry, no relaxation): they rank sites
inside one molecule well but carry a role-dependent offset, which the tree model learns from the role statistics.
Nothing here is validated by being physical: the evaluation (physics_experiment.py) decides, alone and stacked.

Process rule (same as quantum_pharmacophore.py): tblite and LightGBM deadlock in one process. This module imports
tblite lazily and only in extraction; ``load_physics_features`` reads the CSV cache and is safe next to LightGBM.

Usage: python -m cypmode.challenge.physics_block --extract [--n-workers 3] [--limit N]
"""

import argparse
import logging
import multiprocessing
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdFreeSASA, rdMolDescriptors

logger = logging.getLogger(__name__)

HARTREE_KCAL = 627.5095
HARTREE_EV = 27.211386
BOHR_PER_ANGSTROM = 1.0 / 0.52917721067
DEBYE_PER_AU = 2.541746
CACHE_PATH = Path("data/tdi_track/outputs/caches/physics_cache.csv")
ATOMS_PATH = Path("data/tdi_track/outputs/caches/physics_atoms.csv.gz")
N_CONFORMERS = 8
ENSEMBLE_WINDOW_KCAL = 5.0
N_H_BOND_ANGSTROM = 1.02
KT_KCAL = 0.5925  # 298 K

# Disjoint roles (first match wins). C-H roles carry a hydrogen-abstraction energy, binder roles a proton affinity.
CH_ROLES = (
    ("methylenedioxy", "[CH2;R]([OX2]c)[OX2]c"),
    ("alpha_N", "[CX4;!H0;$(C-[#7])]"),
    ("alpha_O", "[CX4;!H0;$(C-[#8])]"),
    ("alpha_S", "[CX4;!H0;$(C-[#16])]"),
    ("benzylic", "[CX4;!H0;$(C-a)]"),
    ("allylic", "[CX4;!H0;$(C-[CX3]=[CX3])]"),
    ("aliphatic", "[CX4;!H0]"),
    ("alkene_ch", "[CX3;!H0;!$(C=[O,S,N])]"),
    ("alkyne_ch", "[CX2;H1]#C"),
    ("aromatic_ch", "[cH]"),
)
BINDER_ROLES = (
    ("azole_N", "[nX2;r5]"),
    ("pyridine_N", "[nX2;r6]"),
    ("amine_N", "[NX3;!$(N-[C,S,P]=[O,S,N]);!$(N-a);!$(N-[SX4]);!$(N-[N,O])]"),
    ("aniline_N", "[NX3;$(N-a);!$(N-[C,S]=[O,S,N])]"),
    ("nitrile_N", "[NX1]#[CX2]"),
    ("thioether_S", "[SX2;!$(S-[#1])]"),
    ("thiophene_S", "[sX2]"),
    ("furan_O", "[o;r5]"),
)
NITROGEN_BINDERS = ("azole_N", "pyridine_N", "amine_N", "aniline_N", "nitrile_N")
HAT_ROLES = tuple(name for name, _ in CH_ROLES if name != "aromatic_ch")
_CH_PATTERNS = [(name, Chem.MolFromSmarts(s)) for name, s in CH_ROLES]
_BINDER_PATTERNS = [(name, Chem.MolFromSmarts(s)) for name, s in BINDER_ROLES]

GLOBAL_FEATURES = [
    "homo_ev", "lumo_ev", "gap_ev", "dipole_d", "quadrupole_norm", "dg_solv_kcal", "ip_ev", "ea_ev", "hardness_ev",
    "chem_potential_ev", "electrophilicity_ev", "q_heavy_min", "q_heavy_max", "q_heavy_std", "sasa_total",
    "sasa_polar_fraction", "npr1", "npr2", "rg", "rg_std_ensemble", "n_ensemble",
]
COMPETITION_FEATURES = [
    "hat_1", "hat_2", "hat_3", "hat_gap12", "hat_n_within_3", "hat_site_entropy", "n_hat_sites",
    "fm_top1", "fm_top2", "fm_top3", "fm_top1_minus_top2", "fm_top1_is_hetero", "fp_top1",
]
PHARMACOPHORE_FEATURES = [
    "n_binders", "d_binder_best_mean", "d_binder_best_min", "d_binder_best_max", "d_binder_best_std", "best_is_adjacent",
    "d_binder_remote_mean", "d_binder_remote_min", "d_binder_remote_max", "hat_remote_minus_best",
    "d_amine_remote_mean", "d_aromN_remote_mean", "n_good_sites_4_8A", "binder_sasa_max", "best_site_sasa",
    "reach_from_binder", "d_ring_to_binder_min", "n_rings_within_7A",
]


def _role_feature_names() -> list[str]:
    names = []
    for role in HAT_ROLES:
        names += [f"{role}_n", f"{role}_fm_max", f"{role}_hat_min", f"{role}_hat_mean"]
    names += ["aromatic_ch_n", "aromatic_ch_fm_max", "aromatic_ch_q_min"]
    for role in NITROGEN_BINDERS:
        names += [f"{role}_n", f"{role}_fm_max", f"{role}_q_min", f"{role}_pa_gas_max", f"{role}_pa_water_max", f"{role}_sasa_max"]
    for role in ("thioether_S", "thiophene_S", "furan_O"):
        names += [f"{role}_n", f"{role}_fm_max", f"{role}_q_min", f"{role}_sasa_max"]
    return names


FEATURE_NAMES = ["phys_" + n for n in GLOBAL_FEATURES + COMPETITION_FEATURES + _role_feature_names() + PHARMACOPHORE_FEATURES]
ATOM_COLUMNS = ["SMILES", "atom", "symbol", "ch_role", "binder_role", "q", "f_minus", "f_plus", "valence", "sasa", "hat_kcal", "pa_gas_kcal", "pa_water_kcal"]


# ---------------------------------------------------------------- roles and geometry (no tblite)

def assign_roles(mol: Chem.Mol) -> tuple[list[str], list[str]]:
    """Chemical role of every heavy atom: (C-H role or '', binder role or ''), first match wins."""
    ch = [""] * mol.GetNumAtoms()
    binder = [""] * mol.GetNumAtoms()
    for roles, out in ((_CH_PATTERNS, ch), (_BINDER_PATTERNS, binder)):
        for name, pattern in roles:
            for match in mol.GetSubstructMatches(pattern):
                if not out[match[0]]:
                    out[match[0]] = name
    return ch, binder


def largest_fragment(smiles: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return max(Chem.GetMolFrags(mol, asMols=True), key=lambda f: f.GetNumHeavyAtoms())


def conformer_ensemble(mol_h: Chem.Mol, n: int = N_CONFORMERS, seed: int = 42) -> tuple[list[int], np.ndarray]:
    """ETKDG conformers with force-field energies (kcal/mol); empty if embedding fails."""
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.5
    params.numThreads = 1
    ids = list(AllChem.EmbedMultipleConfs(mol_h, n, params))
    if not ids:
        return [], np.array([])
    if AllChem.MMFFHasAllMoleculeParams(mol_h):
        result = AllChem.MMFFOptimizeMoleculeConfs(mol_h, maxIters=2000, numThreads=1)
    else:
        result = AllChem.UFFOptimizeMoleculeConfs(mol_h, maxIters=2000, numThreads=1)
    return ids, np.array([e for _, e in result], dtype=float)


def proton_position(positions: np.ndarray, atom: int, neighbours: list[int]) -> np.ndarray:
    """Where a proton would sit on a lone pair: opposite to the summed bond directions, N-H length away."""
    vectors = positions[neighbours] - positions[atom]
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    direction = -unit.sum(axis=0)
    norm = np.linalg.norm(direction)
    if norm < 1e-3:  # symmetric planar/linear case: any perpendicular to the first bond
        v = unit[0]
        direction = np.cross(v, [1.0, 0.0, 0.0] if abs(v[0]) < 0.9 else [0.0, 1.0, 0.0])
        norm = np.linalg.norm(direction)
    return positions[atom] + N_H_BOND_ANGSTROM * direction / norm


def atom_sasa(mol_h: Chem.Mol, conf_id: int) -> np.ndarray:
    radii = rdFreeSASA.classifyAtoms(mol_h)
    rdFreeSASA.CalcSASA(mol_h, radii, confIdx=conf_id)
    return np.array([a.GetDoubleProp("SASA") for a in mol_h.GetAtoms()])


def _hydrogens(mol_h: Chem.Mol, atom: int) -> list[int]:
    return [n.GetIdx() for n in mol_h.GetAtomWithIdx(atom).GetNeighbors() if n.GetAtomicNum() == 1]


def pharmacophore_geometry(mol_h: Chem.Mol, conf_ids: list[int], energies: np.ndarray, roles: tuple[list[str], list[str]], hat: dict[int, float], best_conf: int) -> dict:
    """L4: binder-to-oxidation-site distances over the conformer ensemble and the exposure of both."""
    _, binder_role = roles
    binders = [i for i, r in enumerate(binder_role) if r in NITROGEN_BINDERS + ("thioether_S", "thiophene_S")]
    out = {n: np.nan for n in PHARMACOPHORE_FEATURES}
    out["n_binders"] = float(len(binders))
    if not binders or not hat:
        return out
    ordered = sorted(hat, key=hat.get)
    best = ordered[0]
    topo = Chem.GetDistanceMatrix(mol_h)
    remote = [t for t in ordered if all(topo[b, t] >= 3 for b in binders)]
    out["best_is_adjacent"] = float(best not in remote)
    ensemble = [c for c, e in zip(conf_ids, energies, strict=True) if e - energies.min() <= ENSEMBLE_WINDOW_KCAL] or conf_ids
    amine = [b for b in binders if binder_role[b] == "amine_N"]
    arom = [b for b in binders if binder_role[b] in ("azole_N", "pyridine_N")]

    def distances(site: int) -> np.ndarray:
        rows = []
        for c in ensemble:
            xyz = mol_h.GetConformer(c).GetPositions()
            rows.append((
                np.linalg.norm(xyz[binders] - xyz[site], axis=1).min(),
                np.linalg.norm(xyz[amine] - xyz[site], axis=1).min() if amine else np.nan,
                np.linalg.norm(xyz[arom] - xyz[site], axis=1).min() if arom else np.nan,
            ))
        return np.array(rows, dtype=float)

    arr = distances(best)
    out.update(d_binder_best_mean=arr[:, 0].mean(), d_binder_best_min=arr[:, 0].min(), d_binder_best_max=arr[:, 0].max(), d_binder_best_std=arr[:, 0].std())
    if remote:
        arr = distances(remote[0])
        out.update(d_binder_remote_mean=arr[:, 0].mean(), d_binder_remote_min=arr[:, 0].min(), d_binder_remote_max=arr[:, 0].max(), hat_remote_minus_best=hat[remote[0]] - hat[best])
        if not np.isnan(arr[:, 1]).all():
            out["d_amine_remote_mean"] = float(np.nanmean(arr[:, 1]))
        if not np.isnan(arr[:, 2]).all():
            out["d_aromN_remote_mean"] = float(np.nanmean(arr[:, 2]))
    xyz = mol_h.GetConformer(best_conf).GetPositions()
    top = (remote or ordered)[:5]
    shell = [np.linalg.norm(xyz[binders] - xyz[t], axis=1).min() for t in top]
    out["n_good_sites_4_8A"] = float(sum(4.0 <= s <= 8.0 for s in shell))
    sasa = atom_sasa(mol_h, best_conf)
    focus_site = remote[0] if remote else best
    out["binder_sasa_max"] = float(sasa[binders].max())
    out["best_site_sasa"] = float(sasa[focus_site] + sum(sasa[h] for h in _hydrogens(mol_h, focus_site)))
    focus = remote[0] if remote else best
    nearest = min(binders, key=lambda b: np.linalg.norm(xyz[b] - xyz[focus]))
    heavy = [a.GetIdx() for a in mol_h.GetAtoms() if a.GetAtomicNum() > 1]
    out["reach_from_binder"] = float(np.linalg.norm(xyz[heavy] - xyz[nearest], axis=1).max())
    centroids = []
    for ring in mol_h.GetRingInfo().AtomRings():
        if all(mol_h.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            centroids.append(xyz[list(ring)].mean(axis=0))
    if centroids:
        dist = np.array([np.linalg.norm(c - xyz[nearest]) for c in centroids])
        out["d_ring_to_binder_min"] = float(dist.min())
        out["n_rings_within_7A"] = float((dist <= 7.0).sum())
    else:
        out["n_rings_within_7A"] = 0.0
    return out


# ---------------------------------------------------------------- quantum part (tblite, extraction process only)

def _single_point(numbers: np.ndarray, positions_angstrom: np.ndarray, charge: int, solvent: str | None = None):
    """GFN2-xTB single point; odd electron counts run unrestricted with one unpaired electron."""
    from tblite.interface import Calculator

    uhf = int((int(numbers.sum()) - charge) % 2)
    calc = Calculator("GFN2-xTB", numbers, positions_angstrom * BOHR_PER_ANGSTROM, charge=float(charge), uhf=uhf)
    calc.set("verbosity", 0)
    if solvent:
        calc.add("alpb-solvation", solvent)
    return calc.singlepoint()


_H_ATOM_ENERGY: list[float] = []


def hydrogen_atom_energy() -> float:
    """GFN2 energy of a free hydrogen atom, so that hydrogen-abstraction energies are bond-dissociation-like."""
    if not _H_ATOM_ENERGY:
        _H_ATOM_ENERGY.append(float(_single_point(np.array([1]), np.zeros((1, 3)), 0).get("energy")))
    return _H_ATOM_ENERGY[0]


def _heavy_charges(charges: np.ndarray, mol_h: Chem.Mol, n_heavy: int) -> np.ndarray:
    q = np.array(charges[:n_heavy], dtype=float)
    for atom in mol_h.GetAtoms():
        if atom.GetAtomicNum() == 1 and atom.GetIdx() >= n_heavy:
            q[atom.GetNeighbors()[0].GetIdx()] += charges[atom.GetIdx()]
    return q


def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # SCF failures are data, not crashes
        logger.debug("single point failed: %s", exc)
        return None


def compute_molecule(smiles: str) -> tuple[dict, list[dict]]:
    """All four levels for one compound: (feature dict over FEATURE_NAMES, per-atom rows)."""
    features = {n: np.nan for n in FEATURE_NAMES}
    mol = largest_fragment(smiles)
    if mol is None:
        return features, []
    mol_h = Chem.AddHs(mol)
    n_heavy = mol.GetNumAtoms()
    conf_ids, energies = conformer_ensemble(mol_h)
    if not conf_ids:  # impossible bridged-ring stereo: the electronic structure hardly depends on it, so drop it and retry
        flat = Chem.Mol(mol)
        Chem.RemoveStereochemistry(flat)
        mol, mol_h = flat, Chem.AddHs(flat)
        conf_ids, energies = conformer_ensemble(mol_h)
    if not conf_ids:
        logger.warning("no conformer for %s", smiles)
        return features, []
    best_conf = conf_ids[int(np.argmin(energies))]
    positions = mol_h.GetConformer(best_conf).GetPositions()
    numbers = np.array([a.GetAtomicNum() for a in mol_h.GetAtoms()])
    charge = Chem.GetFormalCharge(mol)
    ch_role, binder_role = assign_roles(mol_h)

    neutral = _safe(_single_point, numbers, positions, charge)
    if neutral is None:
        return features, []
    e0 = float(neutral.get("energy"))
    q0 = _heavy_charges(neutral.get("charges"), mol_h, n_heavy)
    cation = _safe(_single_point, numbers, positions, charge + 1)
    anion = _safe(_single_point, numbers, positions, charge - 1)
    water = _safe(_single_point, numbers, positions, charge, "water")
    f_minus = _heavy_charges(cation.get("charges"), mol_h, n_heavy) - q0 if cation is not None else np.full(n_heavy, np.nan)
    f_plus = q0 - _heavy_charges(anion.get("charges"), mol_h, n_heavy) if anion is not None else np.full(n_heavy, np.nan)
    ip = (float(cation.get("energy")) - e0) * HARTREE_EV if cation is not None else np.nan
    ea = (e0 - float(anion.get("energy"))) * HARTREE_EV if anion is not None else np.nan

    orbital_energies = np.asarray(neutral.get("orbital-energies"))
    occupations = np.asarray(neutral.get("orbital-occupations"))
    occupied = np.where(occupations > 0.5)[0]
    homo = float(orbital_energies[occupied.max()]) * HARTREE_EV if occupied.size else np.nan
    lumo = float(orbital_energies[occupied.max() + 1]) * HARTREE_EV if occupied.size and occupied.max() + 1 < len(orbital_energies) else np.nan
    bond_orders = np.asarray(neutral.get("bond-orders"))[..., 0]
    valence = bond_orders.sum(axis=1)[:n_heavy]

    sasa = atom_sasa(mol_h, best_conf)
    sasa_heavy = np.array([sasa[i] + sum(sasa[h] for h in _hydrogens(mol_h, i)) for i in range(n_heavy)])
    polar = sum(sasa_heavy[i] for i in range(n_heavy) if mol.GetAtomWithIdx(i).GetAtomicNum() in (7, 8, 16))
    radii_of_gyration = []
    for c in conf_ids:
        xyz = mol_h.GetConformer(c).GetPositions()
        radii_of_gyration.append(float(np.sqrt(((xyz - xyz.mean(axis=0)) ** 2).sum(axis=1).mean())))
    hardness = (ip - ea) / 2 if cation is not None and anion is not None else np.nan
    mu = -(ip + ea) / 2 if cation is not None and anion is not None else np.nan
    g = {
        "homo_ev": homo, "lumo_ev": lumo, "gap_ev": lumo - homo,
        "dipole_d": float(np.linalg.norm(neutral.get("dipole"))) * DEBYE_PER_AU,
        "quadrupole_norm": float(np.linalg.norm(neutral.get("quadrupole"))),
        "dg_solv_kcal": (float(water.get("energy")) - e0) * HARTREE_KCAL if water is not None else np.nan,
        "ip_ev": ip, "ea_ev": ea, "hardness_ev": hardness, "chem_potential_ev": mu,
        "electrophilicity_ev": mu**2 / (2 * hardness) if hardness and hardness > 0 else np.nan,
        "q_heavy_min": float(q0.min()), "q_heavy_max": float(q0.max()), "q_heavy_std": float(q0.std()),
        "sasa_total": float(sasa_heavy.sum()), "sasa_polar_fraction": float(polar / max(sasa_heavy.sum(), 1e-9)),
        "npr1": rdMolDescriptors.CalcNPR1(mol_h, confId=best_conf), "npr2": rdMolDescriptors.CalcNPR2(mol_h, confId=best_conf),
        "rg": radii_of_gyration[int(np.argmin(energies))], "rg_std_ensemble": float(np.std(radii_of_gyration)),
        "n_ensemble": float(int(np.sum(energies - energies.min() <= ENSEMBLE_WINDOW_KCAL))),
    }

    # vertical hydrogen abstraction of every non-aromatic C-H (the best hydrogen of each carbon)
    hat: dict[int, float] = {}
    for i in range(n_heavy):
        if ch_role[i] in HAT_ROLES:
            best_rad = None
            for h in _hydrogens(mol_h, i):  # the stereoelectronically best hydrogen is the one the enzyme abstracts
                keep = np.array([k for k in range(len(numbers)) if k != h])
                result = _safe(_single_point, numbers[keep], positions[keep], charge)
                if result is not None:
                    energy = float(result.get("energy"))
                    best_rad = energy if best_rad is None else min(best_rad, energy)
            if best_rad is not None:
                hat[i] = (best_rad + hydrogen_atom_energy() - e0) * HARTREE_KCAL

    # vertical proton affinity of every nitrogen that could coordinate the heme iron
    pa_gas: dict[int, float] = {}
    pa_water: dict[int, float] = {}
    for i in range(n_heavy):
        if binder_role[i] in NITROGEN_BINDERS:
            neighbours = [n.GetIdx() for n in mol_h.GetAtomWithIdx(i).GetNeighbors()]
            numbers_h = np.append(numbers, 1)
            positions_h = np.vstack([positions, proton_position(positions, i, neighbours)])
            protonated = _safe(_single_point, numbers_h, positions_h, charge + 1)
            if protonated is not None:
                pa_gas[i] = -(float(protonated.get("energy")) - e0) * HARTREE_KCAL
            protonated_w = _safe(_single_point, numbers_h, positions_h, charge + 1, "water")
            if protonated_w is not None and water is not None:
                pa_water[i] = -(float(protonated_w.get("energy")) - float(water.get("energy"))) * HARTREE_KCAL

    features.update({"phys_" + k: v for k, v in g.items()})
    sites = sorted(hat.values())
    comp = {n: np.nan for n in COMPETITION_FEATURES}
    comp["n_hat_sites"] = float(len(sites))
    for k, v in enumerate(sites[:3], 1):
        comp[f"hat_{k}"] = v
    if len(sites) >= 2:
        comp["hat_gap12"] = sites[1] - sites[0]
    if sites:
        weights = np.exp(-(np.array(sites) - sites[0]) / KT_KCAL)
        p = weights / weights.sum()
        comp["hat_n_within_3"] = float(sum(s - sites[0] <= 3.0 for s in sites))
        comp["hat_site_entropy"] = float(-(p * np.log(p)).sum())
    valid = np.where(~np.isnan(f_minus))[0]
    if valid.size:
        order = valid[np.argsort(-f_minus[valid])]
        for k, idx in enumerate(order[:3], 1):
            comp[f"fm_top{k}"] = float(f_minus[idx])
        if len(order) >= 2:
            comp["fm_top1_minus_top2"] = float(f_minus[order[0]] - f_minus[order[1]])
        comp["fm_top1_is_hetero"] = float(mol.GetAtomWithIdx(int(order[0])).GetAtomicNum() != 6)
    valid_p = np.where(~np.isnan(f_plus))[0]
    if valid_p.size:
        comp["fp_top1"] = float(f_plus[valid_p].max())
    features.update({"phys_" + k: v for k, v in comp.items()})

    def role_atoms(role_list, role):
        return [i for i in range(n_heavy) if role_list[i] == role]

    for role in HAT_ROLES:
        atoms = role_atoms(ch_role, role)
        features[f"phys_{role}_n"] = float(len(atoms))
        if atoms:
            fm = f_minus[atoms]
            features[f"phys_{role}_fm_max"] = float(np.nanmax(fm)) if not np.isnan(fm).all() else np.nan
            energies_role = [hat[i] for i in atoms if i in hat]
            if energies_role:
                features[f"phys_{role}_hat_min"], features[f"phys_{role}_hat_mean"] = float(min(energies_role)), float(np.mean(energies_role))
    atoms = role_atoms(ch_role, "aromatic_ch")
    features["phys_aromatic_ch_n"] = float(len(atoms))
    if atoms:
        features["phys_aromatic_ch_fm_max"] = float(np.nanmax(f_minus[atoms])) if not np.isnan(f_minus[atoms]).all() else np.nan
        features["phys_aromatic_ch_q_min"] = float(q0[atoms].min())
    for role in NITROGEN_BINDERS + ("thioether_S", "thiophene_S", "furan_O"):
        atoms = role_atoms(binder_role, role)
        features[f"phys_{role}_n"] = float(len(atoms))
        if atoms:
            features[f"phys_{role}_fm_max"] = float(np.nanmax(f_minus[atoms])) if not np.isnan(f_minus[atoms]).all() else np.nan
            features[f"phys_{role}_q_min"] = float(q0[atoms].min())
            features[f"phys_{role}_sasa_max"] = float(sasa_heavy[atoms].max())
            if role in NITROGEN_BINDERS:
                gas = [pa_gas[i] for i in atoms if i in pa_gas]
                aq = [pa_water[i] for i in atoms if i in pa_water]
                if gas:
                    features[f"phys_{role}_pa_gas_max"] = float(max(gas))
                if aq:
                    features[f"phys_{role}_pa_water_max"] = float(max(aq))
    geometry = pharmacophore_geometry(mol_h, conf_ids, energies, (ch_role, binder_role), hat, best_conf)
    features.update({"phys_" + k: v for k, v in geometry.items()})

    rows = []
    for i in range(n_heavy):
        rows.append({
            "SMILES": smiles, "atom": i, "symbol": mol.GetAtomWithIdx(i).GetSymbol(), "ch_role": ch_role[i], "binder_role": binder_role[i],
            "q": q0[i], "f_minus": f_minus[i], "f_plus": f_plus[i], "valence": valence[i], "sasa": sasa_heavy[i],
            "hat_kcal": hat.get(i, np.nan), "pa_gas_kcal": pa_gas.get(i, np.nan), "pa_water_kcal": pa_water.get(i, np.nan),
        })
    return features, rows


def _worker(smiles: str):
    try:
        return compute_molecule(smiles)
    except Exception as exc:
        logger.warning("physics extraction failed for %s: %s", smiles, exc)
        return {n: np.nan for n in FEATURE_NAMES}, []


def _init_worker() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"


def canonical(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def extract(smiles_list: list[str], cache_path: Path = CACHE_PATH, atoms_path: Path = ATOMS_PATH, n_workers: int = 1, checkpoint_every: int = 100) -> None:
    """Resumable extraction: rows already in the cache are skipped, results are appended every ``checkpoint_every``."""
    keys = list(dict.fromkeys(canonical(s) for s in smiles_list))
    done = set(pd.read_csv(cache_path, usecols=["SMILES"])["SMILES"]) if cache_path.exists() else set()
    todo = [k for k in keys if k not in done]
    logger.info("physics extraction: %d compounds, %d cached, %d to do", len(keys), len(done), len(todo))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    pool = multiprocessing.Pool(n_workers, initializer=_init_worker) if n_workers > 1 else None
    results = pool.imap(_worker, todo, chunksize=4) if pool else map(_worker, todo)
    feature_rows, atom_rows = [], []

    def flush():
        if feature_rows:
            pd.DataFrame(feature_rows, columns=["SMILES", *FEATURE_NAMES]).to_csv(cache_path, mode="a", header=not cache_path.exists(), index=False)
        if atom_rows:
            pd.DataFrame(atom_rows, columns=ATOM_COLUMNS).to_csv(atoms_path, mode="a", header=not atoms_path.exists(), index=False, compression="gzip")
        feature_rows.clear()
        atom_rows.clear()

    try:
        for n, (smiles, (features, rows)) in enumerate(zip(todo, results, strict=True), 1):
            feature_rows.append({"SMILES": smiles, **features})
            atom_rows.extend(rows)
            if n % checkpoint_every == 0:
                flush()
                logger.info("physics extraction: %d / %d", n, len(todo))
    finally:
        flush()
        if pool is not None:
            pool.close()
            pool.join()


def load_physics_features(smiles: list[str], cache_path: Path = CACHE_PATH) -> np.ndarray:
    """Feature matrix (rows follow ``smiles``) from the cache; never imports tblite."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"{cache_path} is missing: run `python -m cypmode.challenge.physics_block --extract` first (a tblite-only process)")
    table = pd.read_csv(cache_path).drop_duplicates("SMILES").set_index("SMILES")
    keys = [canonical(s) for s in smiles]
    missing = [k for k in keys if k not in table.index]
    if missing:
        raise ValueError(f"{len(missing)} compounds have no physics features, e.g. {missing[0]!r}")
    return table.loc[keys, FEATURE_NAMES].to_numpy(dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--n-workers", type=int, default=3)
    parser.add_argument("--limit", type=int, help="only the first N compounds (timing runs)")
    parser.add_argument("--cache", default=str(CACHE_PATH))
    parser.add_argument("--atoms", default=str(ATOMS_PATH))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.extract:
        train = pd.read_csv("data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv")
        test = pd.read_csv("data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv")
        smiles = list(train.SMILES) + list(test.SMILES)
        extract(smiles[: args.limit] if args.limit else smiles, Path(args.cache), Path(args.atoms), args.n_workers)


if __name__ == "__main__":
    from cypmode import provenance

    provenance.announce()
    main()

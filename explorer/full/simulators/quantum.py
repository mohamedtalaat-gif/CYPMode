# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part X, the logic: the Schrodinger equation on a grid (10.1), a self-consistent field on a two-site model (10.2), the exchange of a one-electron density in the local-density approximation (10.3) and a Hueckel calculation on the
conjugated atoms of a molecule (10.4). Atomic units (hbar = m = e = 1) throughout."""

import numpy as np
from rdkit import Chem
from scipy.integrate import quad

C_X = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)  # the constant of the local-density exchange energy density: e_x = -C_X n^(4/3)


def schrodinger_1d(potential: str, parameter: float, states: int = 4, points: int = 600) -> dict:
    """The lowest states of -1/2 psi'' + V psi = E psi by finite differences, and the exact energies where they are known."""
    if potential == "particle in a box of width a":
        x = np.linspace(0, parameter, points + 2)[1:-1]
        v = np.zeros_like(x)
        exact = [(n * np.pi) ** 2 / (2 * parameter**2) for n in range(1, states + 1)]
    elif potential == "harmonic oscillator of frequency w":
        x = np.linspace(-8 / np.sqrt(parameter), 8 / np.sqrt(parameter), points)
        v = 0.5 * parameter**2 * x**2
        exact = [parameter * (n + 0.5) for n in range(states)]
    elif potential == "Morse well of depth D":
        a = 1.0
        x = np.linspace(-1.5, 12.0, points)
        v = parameter * (1 - np.exp(-a * x)) ** 2
        w0 = a * np.sqrt(2 * parameter)
        exact = [w0 * (n + 0.5) - (w0 * (n + 0.5)) ** 2 / (4 * parameter) for n in range(states)]
    else:  # a double well
        x = np.linspace(-3.0, 3.0, points)
        v = parameter * (x**2 - 1.0) ** 2
        exact = [float("nan")] * states
    h = x[1] - x[0]
    kinetic = (np.diag(np.full(len(x), 1.0 / h**2)) - 0.5 * np.diag(np.full(len(x) - 1, 1.0 / h**2), 1) - 0.5 * np.diag(np.full(len(x) - 1, 1.0 / h**2), -1))
    energies, vectors = np.linalg.eigh(kinetic + np.diag(v))
    return {"x": x, "v": v, "energies": energies[:states], "states": vectors[:, :states] / np.sqrt(h), "exact": np.array(exact)}


def hubbard_dimer_scf(t: float, u: float, asymmetry: float, mixing: float, max_iter: int = 60) -> dict:
    """The unrestricted Hartree-Fock energy of two electrons (one up, one down) on two sites with hopping t and repulsion u on a site, iterated to self-consistency from a starting density that favours site 1 by ``asymmetry``, and the exact energy."""
    n_up, n_down = np.array([0.5 + asymmetry, 0.5 - asymmetry]), np.array([0.5 - asymmetry, 0.5 + asymmetry])
    history = []
    for _ in range(max_iter):
        h_up, h_down = np.array([[u * n_down[0], -t], [-t, u * n_down[1]]]), np.array([[u * n_up[0], -t], [-t, u * n_up[1]]])
        (e_up, v_up), (e_down, v_down) = np.linalg.eigh(h_up), np.linalg.eigh(h_down)
        new_up, new_down = v_up[:, 0] ** 2, v_down[:, 0] ** 2
        energy = e_up[0] + e_down[0] - u * float(n_up @ n_down)
        history.append(float(energy))
        change = float(np.abs(new_up - n_up).max() + np.abs(new_down - n_down).max())
        n_up, n_down = (1 - mixing) * n_up + mixing * new_up, (1 - mixing) * n_down + mixing * new_down
        if change < 1e-10:
            break
    exact = u / 2 - np.sqrt((u / 2) ** 2 + 4 * t**2)
    return {"history": history, "energy": history[-1], "exact": float(exact), "density up": n_up, "density down": n_down, "iterations": len(history), "converged": change < 1e-10}


def hydrogenic(zeta: float) -> dict:
    """A 1s density with the exponent zeta (nuclear charge 1): the kinetic, nuclear and Hartree energies, the exact exchange (minus the Hartree energy for one electron) and the exchange in the local-spin-density approximation, integrated numerically (the factor 2^(1/3) of a fully polarised density)."""
    density = lambda r: zeta**3 / np.pi * np.exp(-2 * zeta * r)
    n_electrons = quad(lambda r: 4 * np.pi * r**2 * density(r), 0, 50 / zeta)[0]
    lda = -(2.0 ** (1.0 / 3.0)) * C_X * quad(lambda r: 4 * np.pi * r**2 * density(r) ** (4 / 3), 0, 50 / zeta)[0]  # the lone electron has all its spin up: the spin-polarised form of the functional
    kinetic, nuclear, hartree = zeta**2 / 2, -zeta, 5 * zeta / 16
    return {"electrons": n_electrons, "kinetic": kinetic, "nuclear": nuclear, "Hartree": hartree, "exact exchange": -hartree, "LDA exchange": lda, "E exact": kinetic + nuclear, "E LDA": kinetic + nuclear + hartree + lda}


def huckel(smiles: str) -> dict | None:
    """The pi levels of the conjugated atoms of a molecule: the adjacency of the conjugated bonds, a shift of the on-site energy for N (0.5 beta) and O (1.0 beta), and one pi electron per atom (two for an atom that donates a lone pair)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atoms = sorted({a for b in mol.GetBonds() if b.GetIsConjugated() for a in (b.GetBeginAtomIdx(), b.GetEndAtomIdx())})
    if not atoms:
        return {"atoms": 0}
    index = {a: i for i, a in enumerate(atoms)}
    matrix = np.zeros((len(atoms), len(atoms)))
    for b in mol.GetBonds():
        if b.GetIsConjugated() and b.GetBeginAtomIdx() in index and b.GetEndAtomIdx() in index:
            matrix[index[b.GetBeginAtomIdx()], index[b.GetEndAtomIdx()]] = matrix[index[b.GetEndAtomIdx()], index[b.GetBeginAtomIdx()]] = 1.0
    shift = {"N": 0.5, "O": 1.0}
    electrons = 0
    for a in atoms:
        atom = mol.GetAtomWithIdx(a)
        matrix[index[a], index[a]] = shift.get(atom.GetSymbol(), 0.0)
        donor = (atom.GetSymbol() == "N" and atom.GetTotalDegree() == 3) or (atom.GetSymbol() in ("O", "S") and atom.GetTotalDegree() == 2)
        electrons += 2 if donor else 1
    levels = np.sort(np.linalg.eigvalsh(matrix))[::-1]  # in units of |beta|, the most bonding first (E = alpha + x beta with beta < 0)
    occupied = electrons // 2
    gap = float(levels[occupied - 1] - levels[occupied]) if 0 < occupied < len(levels) else float("nan")
    return {"atoms": len(atoms), "electrons": electrons, "levels": levels, "occupied": occupied, "gap": gap}

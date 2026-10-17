"""Part X (first part): the ladder of quantum chemistry, from the Schroedinger equation to Hartree-Fock (10.1, 10.2)."""

from notebooks.notebook_builder import code, md

QC_TOOLS = '''
# Quantum-chemistry programs are run in a fresh interpreter of their own, as the project's extraction jobs are: a library for electronic structure (PySCF or tblite) and LightGBM must not meet in one process, because the
# two bring their own copies of the OpenMP runtime and the program can end with a crash or a deadlock (the same rule as for the torch libraries, 4.8). The helper hands a short script to the interpreter of the notebook,
# lets it print one line of JSON as its last output, and reads that line back.
import json
import os
import subprocess
import sys
import textwrap

HARTREE_TO_EV, HARTREE_TO_KCAL, BOHR_TO_ANGSTROM = 27.211386, 627.5095, 0.529177210903
def run_isolated(source, timeout=900):
    done = subprocess.run([sys.executable, "-c", textwrap.dedent(source)], capture_output=True, text=True, timeout=timeout, cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)})
    if done.returncode != 0:
        raise RuntimeError(done.stderr[-3000:])
    return json.loads(done.stdout.strip().splitlines()[-1])
def run_isolated_text(source, timeout=900):
    """The same, returning everything the program printed (used where a program writes its own report to the screen)."""
    done = subprocess.run([sys.executable, "-c", textwrap.dedent(source)], capture_output=True, text=True, timeout=timeout, cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)})
    if done.returncode != 0:
        raise RuntimeError(done.stderr[-3000:])
    return done.stdout
print(f"1 hartree = {HARTREE_TO_EV} eV = {HARTREE_TO_KCAL} kcal/mol; 1 bohr = {BOHR_TO_ANGSTROM} angstrom")
'''

BOX_HYDROGEN = '''
# The Schroedinger equation as a matrix eigenvalue problem. In atomic units (hbar = m = e = 1) the operator is H = -1/2 d^2/dx^2 + V(x). On a grid of n interior points with spacing h the second derivative becomes the matrix with
# -2 on the diagonal and 1 beside it, divided by h^2, so that H is a symmetric matrix and its eigenvalues are the energies, as the eigenvalues of the covariance matrix were the variances in 4.9.
# (1) a particle in a box of length L = 1 (V = 0 inside, infinite outside): exact energies E_n = n^2 pi^2 / (2 L^2).
# (2) the hydrogen atom in its s states: with psi(r) = u(r) / r (the angular part is a constant) the equation for u(r) is -1/2 u'' - u / r = E u with u(0) = 0, a grid in r with the potential V = -1/r;
#     exact energies E_n = -1 / (2 n^2) hartree.
import numpy as np

def second_difference(n, h):
    return (np.diag(np.full(n, 2.0)) - np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1)) / (2 * h * h)

n_box, box_length = 800, 1.0
h_box = box_length / (n_box + 1)
box_levels = np.linalg.eigvalsh(second_difference(n_box, h_box))[:4]
box_exact = np.array([(k * np.pi) ** 2 / (2 * box_length**2) for k in (1, 2, 3, 4)])

n_radial, r_max = 4000, 60.0
h_radial = r_max / (n_radial + 1)
radius = h_radial * np.arange(1, n_radial + 1)
hydrogen_levels = np.linalg.eigvalsh(second_difference(n_radial, h_radial) - np.diag(1 / radius))[:3]
hydrogen_exact = np.array([-0.5 / n**2 for n in (1, 2, 3)])

box_table = pd.DataFrame({"level n": [1, 2, 3, 4], "grid eigenvalue (hartree)": box_levels, "exact n^2 pi^2 / 2 (hartree)": box_exact, "difference": box_levels - box_exact})
hydrogen_table = pd.DataFrame({"level n": [1, 2, 3], "grid eigenvalue (hartree)": hydrogen_levels, "exact -1/(2 n^2) (hartree)": hydrogen_exact, "grid eigenvalue (eV)": hydrogen_levels * 27.211386})
show_table(box_table, floatfmt=".6f")
show_table(hydrogen_table, floatfmt=".6f")
box_error = float(np.abs(box_levels - box_exact).max())
box_relative = float((np.abs(box_levels - box_exact) / box_exact).max())
hydrogen_error = float(np.abs(hydrogen_levels - hydrogen_exact).max())
'''

VARIATIONAL = '''
# The variational principle on the hydrogen atom. For a normalised trial function the energy E[psi] = <psi|H|psi> can only be at or above the true ground-state energy E_0 (proof in the text). Two one-parameter families of s-type trial
# functions: the exponential exp(-a r) and the Gaussian exp(-b r^2). With the angular part integrated out, E[psi] = (T + V) / N with N = integral of psi^2 r^2 dr, T = 1/2 integral of psi'^2 r^2 dr and V = -integral of psi^2 r dr.
# The integrals are done numerically, the energy is minimised over the parameter, and the result is set against the exact -0.5 hartree.
from scipy import integrate, optimize

def trial_energy(shape, parameter):
    psi = (lambda r: np.exp(-parameter * r)) if shape == "exponential" else (lambda r: np.exp(-parameter * r**2))
    slope = (lambda r: -parameter * np.exp(-parameter * r)) if shape == "exponential" else (lambda r: -2 * parameter * r * np.exp(-parameter * r**2))
    norm = integrate.quad(lambda r: psi(r) ** 2 * r**2, 0, np.inf)[0]
    kinetic = 0.5 * integrate.quad(lambda r: slope(r) ** 2 * r**2, 0, np.inf)[0]
    potential = -integrate.quad(lambda r: psi(r) ** 2 * r, 0, np.inf)[0]
    return (kinetic + potential) / norm

variational_rows = []
for shape, symbol in (("exponential", "a"), ("gaussian", "b")):
    best = optimize.minimize_scalar(lambda x: trial_energy(shape, x), bounds=(0.05, 3.0), method="bounded")
    variational_rows.append({"trial function": f"exp(-{symbol} r)" if shape == "exponential" else f"exp(-{symbol} r^2)", "best parameter": float(best.x), "energy at the best parameter (hartree)": float(best.fun), "exact (hartree)": -0.5, "above the exact by (hartree)": float(best.fun + 0.5)})
variational_table = pd.DataFrame(variational_rows)
show_table(variational_table, floatfmt=".5f")
scan_values = [0.5, 0.75, 1.0, 1.25, 1.5]
scan_table = pd.DataFrame({"parameter a": scan_values, "E(a) of the exponential trial (hartree)": [trial_energy("exponential", a) for a in scan_values]})
show_table(scan_table, floatfmt=".5f")
exponential_best, gaussian_best = variational_rows[0], variational_rows[1]
'''

HF_HAND = '''
# Hartree-Fock written out for the smallest molecules, so that every matrix of the method can be seen. Basis: one 1s function on each nucleus, itself a sum of three Gaussians (the STO-3G basis: the shape of a Slater orbital exp(-zeta r),
# approximated by three Gaussians exp(-alpha r^2) with fixed exponents and weights scaled by zeta). For s-type Gaussians every integral has a closed form (Szabo and Ostlund, chapter 3 and appendix A), built from the Boys function
# F0(t) = 1/2 sqrt(pi / t) erf(sqrt(t)). The matrices: overlap S, kinetic T, nuclear attraction V and the two-electron integrals (mu nu | lambda sigma); h = T + V. The self-consistent field loop follows the text: guess the density matrix
# from h alone, build the Fock matrix, diagonalise it in an orthogonal basis (S^(-1/2)), build the new density matrix from the lowest orbital (two electrons), and repeat until the energy stops changing.
from scipy.special import erf

BASE_ALPHA, BASE_COEFFICIENT = np.array([0.109818, 0.405771, 2.22766]), np.array([0.444635, 0.535328, 0.154329])   # STO-3G for zeta = 1
def boys0(t):
    return 1.0 if t < 1e-12 else 0.5 * np.sqrt(np.pi / t) * erf(np.sqrt(t))

def two_centre_integrals(distance, zeta):
    """S, T, V and (mu nu | lambda sigma) of two 1s STO-3G functions, one on each of two nuclei a distance apart (bohr). zeta: Slater exponents of the two functions; the nuclear charges enter later (V is built for unit charges)."""
    centres = [np.zeros(3), np.array([0.0, 0.0, distance])]
    primitives = [[(BASE_COEFFICIENT[k] * (2 * (z**2 * BASE_ALPHA[k]) / np.pi) ** 0.75, z**2 * BASE_ALPHA[k]) for k in range(3)] for z in zeta]
    sq = lambda a, b: float(np.dot(a - b, a - b))
    S, T = np.zeros((2, 2)), np.zeros((2, 2))
    V = np.zeros((2, 2, 2))      # V[c] is the attraction to a unit charge on centre c
    eri = np.zeros((2, 2, 2, 2))
    for a in range(2):
        for b in range(2):
            for ca, ea in primitives[a]:
                for cb, eb in primitives[b]:
                    p, q = ea + eb, ea * eb / (ea + eb)
                    centre_p = (ea * centres[a] + eb * centres[b]) / p
                    overlap = ca * cb * (np.pi / p) ** 1.5 * np.exp(-q * sq(centres[a], centres[b]))
                    S[a, b] += overlap
                    T[a, b] += overlap * q * (3 - 2 * q * sq(centres[a], centres[b]))
                    for c in range(2):
                        V[c, a, b] += -overlap * (2 * np.sqrt(p / np.pi)) * boys0(p * sq(centre_p, centres[c]))
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    for ca, ea in primitives[a]:
                        for cb, eb in primitives[b]:
                            for cc, ec in primitives[c]:
                                for cd, ed in primitives[d]:
                                    p, r = ea + eb, ec + ed
                                    centre_p, centre_r = (ea * centres[a] + eb * centres[b]) / p, (ec * centres[c] + ed * centres[d]) / r
                                    k1, k2 = np.exp(-ea * eb / p * sq(centres[a], centres[b])), np.exp(-ec * ed / r * sq(centres[c], centres[d]))
                                    eri[a, b, c, d] += ca * cb * cc * cd * 2 * np.pi**2.5 / (p * r * np.sqrt(p + r)) * k1 * k2 * boys0(p * r / (p + r) * sq(centre_p, centre_r))
    return S, T, V, eri

def scf_two_electrons(distance, charges, zeta, tolerance=1e-10):
    """Closed-shell Hartree-Fock for two electrons in two 1s functions. Returns the iteration history, the converged energy and the quantities of the last iteration."""
    S, T, V, eri = two_centre_integrals(distance, zeta)
    h = T + charges[0] * V[0] + charges[1] * V[1]
    nuclear = charges[0] * charges[1] / distance
    values, vectors = np.linalg.eigh(S)
    X = vectors @ np.diag(values**-0.5) @ vectors.T                       # S^(-1/2): orthogonalises the basis
    orbital_energies, Cp = np.linalg.eigh(X.T @ h @ X)                    # first guess from the core Hamiltonian
    C = X @ Cp
    P = 2 * np.outer(C[:, 0], C[:, 0])
    history, previous = [], None
    for iteration in range(1, 100):
        J = np.einsum("cd,abcd->ab", P, eri)
        K = np.einsum("cd,acbd->ab", P, eri)
        F = h + J - 0.5 * K
        electronic = 0.5 * np.sum(P * (h + F))
        orbital_energies, Cp = np.linalg.eigh(X.T @ F @ X)
        C = X @ Cp
        P = 2 * np.outer(C[:, 0], C[:, 0])
        history.append({"iteration": iteration, "total energy (hartree)": electronic + nuclear, "lowest orbital energy (hartree)": orbital_energies[0], "coefficient on nucleus A": abs(C[0, 0]), "coefficient on nucleus B": abs(C[1, 0])})
        if previous is not None and abs(electronic - previous) < tolerance:
            break
        previous = electronic
    return pd.DataFrame(history), {"S": S, "h": h, "eri": eri, "C": C, "nuclear": nuclear, "energy": electronic + nuclear}

# HeH+ near its equilibrium distance (0.7751 angstrom): helium and hydrogen, Slater exponents of the STO-3G basis 1.69 and 1.24
heh_distance = 0.7751 / BOHR_TO_ANGSTROM
heh_history, heh_state = scf_two_electrons(heh_distance, (2, 1), (1.69, 1.24))
show_table(heh_history.head(8), floatfmt=".6f")
heh_by_hand = float(heh_history["total energy (hartree)"].iloc[-1])
heh_reference = run_isolated(f"""
    import json
    from pyscf import gto, scf
    molecule = gto.M(atom="He 0 0 0; H 0 0 0.7751", basis="sto-3g", charge=1, unit="Angstrom", verbose=0)
    print(json.dumps(scf.RHF(molecule).run().e_tot))
""")
n_scf_cycles = len(heh_history)
'''

H2_CURVE = '''
# The same machinery for H2 at several distances, and the step beyond Hartree-Fock. With two electrons in the minimal basis there are only two orbitals, the bonding sigma_g (lower) and the antibonding sigma_u. The exact solution
# within this basis (full configuration interaction) mixes the two configurations "both electrons in sigma_g" and "both in sigma_u": a 2 x 2 matrix with H00 = 2 h_gg + (gg|gg), H11 = 2 h_uu + (uu|uu) and H01 = (gu|gu),
# where h_pq are the one-electron matrix elements in the orbitals and (pq|rs) the two-electron integrals in the orbitals. The difference between this exact energy and the Hartree-Fock energy is the correlation energy.
def h2_energies(distance_angstrom):
    distance = distance_angstrom / BOHR_TO_ANGSTROM
    history, state = scf_two_electrons(distance, (1, 1), (1.24, 1.24))
    C, h, eri = state["C"], state["h"], state["eri"]
    h_mo = C.T @ h @ C
    mo = lambda p, q, r, s: np.einsum("abcd,a,b,c,d->", eri, C[:, p], C[:, q], C[:, r], C[:, s])
    matrix = np.array([[2 * h_mo[0, 0] + mo(0, 0, 0, 0), mo(0, 1, 0, 1)], [mo(0, 1, 0, 1), 2 * h_mo[1, 1] + mo(1, 1, 1, 1)]])
    return state["energy"], float(np.linalg.eigvalsh(matrix)[0] + state["nuclear"])

distances = [0.5, 0.74, 1.0, 1.5, 2.0, 3.0, 4.0]
h2_rows = []
for distance in distances:
    hartree_fock, exact_in_basis = h2_energies(distance)
    h2_rows.append({"distance (angstrom)": distance, "Hartree-Fock (hartree)": hartree_fock, "exact in the basis (hartree)": exact_in_basis, "correlation energy (kcal/mol)": (exact_in_basis - hartree_fock) * HARTREE_TO_KCAL})
h2_table = pd.DataFrame(h2_rows)
h2_reference = run_isolated(f"""
    import json
    from pyscf import gto, scf, fci
    out = {{}}
    for r in {distances}:
        mol = gto.M(atom=f"H 0 0 0; H 0 0 {{r}}", basis="sto-3g", unit="Angstrom", verbose=0)
        mf = scf.RHF(mol).run()
        out[str(r)] = [mf.e_tot, fci.FCI(mf).kernel()[0]]
    atom = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1, verbose=0)
    out["atom"] = scf.UHF(atom).run().e_tot
    print(json.dumps(out))
""")
h2_table["Hartree-Fock, PySCF"] = [h2_reference[str(d)][0] for d in distances]
h2_table["exact, PySCF"] = [h2_reference[str(d)][1] for d in distances]
show_table(h2_table, floatfmt=".5f")
h2_agreement = float(max(np.abs(h2_table["Hartree-Fock (hartree)"] - h2_table["Hartree-Fock, PySCF"]).max(), np.abs(h2_table["exact in the basis (hartree)"] - h2_table["exact, PySCF"]).max()))
atom_energy = float(h2_reference["atom"])
far = h2_table.iloc[-1]
dissociation_error_hf = (far["Hartree-Fock (hartree)"] - 2 * atom_energy) * HARTREE_TO_KCAL
dissociation_error_exact = (far["exact in the basis (hartree)"] - 2 * atom_energy) * HARTREE_TO_KCAL
equilibrium = h2_table.iloc[1]
'''

CELLS = [
    md("""
# Part X. Quantum chemistry and the physics features

Parts III to IX gave the models descriptions of a molecule that come from its drawing: which atoms are bonded, which fragments occur, how large and how polar it is. Mechanism-based inactivation (section 5.1) is a statement about something else. The enzyme has to turn the compound into a reactive intermediate, and that happens at particular atoms: a carbon-hydrogen bond that the enzyme's oxidising species pulls a hydrogen atom from, a nitrogen whose lone pair binds the iron of the heme. Which bond gives way first, which atom gives up an electron most easily, whether a nitrogen is likely to hold a proton or an iron: these are questions about the electrons of the molecule, and a drawing answers them only through the statistics of many other molecules. Quantum chemistry computes them directly.

This part builds the means in the order in which a physicist would, from the exact equation to the approximation that is cheap enough to run on 6,895 compounds, and then uses it. Sections 10.1 to 10.4 are a ladder. The top rung is the Schrödinger equation, which cannot be solved for a molecule. Each rung down gives up accuracy for speed: the approximation in which each electron sees only the average of the others (10.2), density functional theory (10.3), and the semi-empirical tight-binding method that the project runs (10.4). Section 10.5 defines the quantities that are computed with it, and 10.6 puts them through the gates of Part VI and reports the verdict, which is negative. The earlier attempts at quantum features, a set of global descriptors and a network that imitates a calculation (7.8), were null as well; the block of this part describes the molecule atom by atom, which was the reason to try again.
"""),
    md("""
## 10.1 The Schrödinger equation, the wavefunction and the Born-Oppenheimer approximation

> **Field:** Quantum chemistry · Mathematics

*Where this section sits:* it is the first rung of the ladder. It states what has to be computed and why it cannot be computed exactly, and it settles the one simplification on which all of molecular quantum chemistry stands. The mathematics needed is the eigenvalue problem that 4.9 already used for the principal components.

**The wavefunction.** The state of an electron is not a position and a velocity. It is a **wavefunction** $\\psi(\\mathbf r)$, a number (in general complex) attached to every point $\\mathbf r$ of space, with one rule of interpretation: $|\\psi(\\mathbf r)|^2$ is the probability density of finding the electron at $\\mathbf r$, so that $\\int |\\psi|^2\\,d\\mathbf r = 1$ (the wavefunction is **normalised**). A quantity that can be measured, such as the energy, corresponds to an **operator**, a recipe that turns one function into another. The energy operator is the **Hamiltonian** $\\hat H = \\hat T + \\hat V$, the sum of the kinetic-energy operator $\\hat T = -\\tfrac12\\nabla^2$ (in the units introduced next, $\\nabla^2$ being the sum of the second derivatives in the three directions) and the potential energy $\\hat V$. The **time-independent Schrödinger equation** says that a state of definite energy $E$ satisfies

$$\\hat H\\,\\psi \\;=\\; E\\,\\psi .$$

Only special functions $\\psi$, the **eigenfunctions**, and special numbers $E$, the **eigenvalues** (the allowed energies), solve it. It is the equation $Cw = \\lambda w$ of 4.9 with the matrix $C$ replaced by an operator and the vector $w$ by a function, and when space is cut into a grid of points the operator becomes a matrix and the equation becomes literally the eigenvalue problem of linear algebra. That is how the first two cells solve it.

**Atomic units.** The equations of molecules are tidy in units in which four constants are 1: the electron's mass, its charge, the reduced Planck constant $\\hbar$ and $1/(4\\pi\\varepsilon_0)$ (the constant that turns charges into the Coulomb force). Energy is then in **hartree** (1 hartree = 27.2114 eV = 627.5 kcal/mol) and length in **bohr** (1 bohr = 0.529177 Å). The hydrogen atom's ground-state energy is $-\\tfrac12$ hartree, which is the 13.6 eV needed to ionise it.

**Two cases that can be solved.** A particle confined to a box of length $L$ (potential zero inside and infinite outside) has the energies $E_n = n^2\\pi^2/(2L^2)$ with $n = 1, 2, \\dots$: confinement makes the energies discrete, and the energy of the lowest state rises as the box shrinks. The hydrogen atom, one electron in the potential $-1/r$ of a proton at distance $r$, has $E_n = -1/(2n^2)$. For the s states, in which the wavefunction depends on $r$ only, writing $\\psi(r) = u(r)/r$ turns the equation into $-\\tfrac12 u'' - u/r = E\\,u$ with $u(0) = 0$. The cell discretises both on a grid and asks a matrix routine for the lowest eigenvalues.
"""),
    code(QC_TOOLS),
    code(BOX_HYDROGEN),
    md("""
The grid reproduces the exact energies: the largest difference is {{=box_error:.1e}} hartree for the box (on its highest level, a relative error of {{=box_relative:.0e}}) and {{=hydrogen_error:.1e}} hartree for the hydrogen atom, whose lowest level comes out as {{=hydrogen_levels[0] * 27.211386:.3f}} eV. For two particles that attract or repel each other the same method is out of reach, as the next paragraph shows.

**Many electrons.** A molecule with $N$ electrons and nuclei $A$ of charge $Z_A$ has the Hamiltonian (in atomic units, with $i, j$ running over the electrons, $A, B$ over the nuclei, $M_A$ the mass of nucleus $A$, $r_{iA}$, $r_{ij}$ and $R_{AB}$ the distances)

$$\\hat H \\;=\\; -\\sum_i \\tfrac12\\nabla_i^2 \\;-\\; \\sum_A \\frac{1}{2M_A}\\nabla_A^2 \\;-\\; \\sum_{i,A}\\frac{Z_A}{r_{iA}} \\;+\\; \\sum_{i<j}\\frac{1}{r_{ij}} \\;+\\; \\sum_{A<B}\\frac{Z_A Z_B}{R_{AB}} .$$

The terms are the kinetic energy of the electrons and of the nuclei, the attraction of electrons to nuclei, the repulsion between electrons and the repulsion between nuclei. The wavefunction is a function of $3N$ coordinates, three for each electron (and a spin each, which is explained below). The repulsion $1/r_{ij}$ ties every electron to every other, so that the equation does not separate into one equation per electron. A water molecule has 10 electrons; a grid with 100 points in each coordinate would hold $100^{30}$ numbers. The exact equation is therefore not solved for any molecule larger than a few electrons, and everything that follows is an approximation whose quality can be judged on the cases that can be solved.

**Spin and the Pauli principle.** An electron carries a two-valued internal property, its **spin**. The wavefunction of electrons must change sign when two electrons are exchanged (it is **antisymmetric**), which is the **Pauli principle**. Its consequence for chemistry is that two electrons cannot occupy the same orbital with the same spin: orbitals fill two at a time, with opposite spins, and the shell structure of atoms and the pattern of bonding follow.

**The variational principle.** The tool for approximate solutions is a bound. Let $\\phi_k$ be the exact eigenfunctions with energies $E_0 \\le E_1 \\le \\dots$. Any normalised trial function can be written $\\psi = \\sum_k c_k\\phi_k$ with $\\sum_k |c_k|^2 = 1$, and its energy $E[\\psi] = \\langle\\psi|\\hat H|\\psi\\rangle = \\sum_k |c_k|^2 E_k$ cannot be below $E_0\\sum_k |c_k|^2 = E_0$. **No trial function has an energy below the true ground state**, so the lower the energy, the better the function, and one can improve a trial function by minimising $E[\\psi]$ over parameters. The cell applies it to the hydrogen atom with two families of trial functions, $e^{-ar}$ and $e^{-br^2}$.
"""),
    code(VARIATIONAL),
    md("""
The exponential family contains the exact answer ($a = 1$), and the minimisation finds it: the best parameter is {{=exponential_best['best parameter']:.3f}} and the energy {{=exponential_best['energy at the best parameter (hartree)']:.5f}} hartree. The Gaussian family does not contain it, and the best Gaussian ends {{=gaussian_best['above the exact by (hartree)']:.4f}} hartree above the exact energy, never below, as the principle requires. This is also the reason that Gaussians appear in 10.2 although orbitals are not Gaussian-shaped: a sum of several Gaussians approximates the right shape, and the variational principle guarantees that adding functions can only lower the energy toward the truth.

**The Born-Oppenheimer approximation.** The nuclei are at least 1,836 times heavier than an electron, so that they move slowly while the electrons rearrange almost instantly. The approximation of **Born and Oppenheimer** (1927) treats the nuclei as fixed points at positions $\\mathbf R$ and solves for the electrons alone, $\\hat H_{\\text{el}}(\\mathbf R)\\,\\psi_{\\text{el}} = E_{\\text{el}}(\\mathbf R)\\,\\psi_{\\text{el}}$, with the kinetic energy of the nuclei dropped from $\\hat H$. Adding the repulsion of the nuclei, which is a constant for fixed positions, gives an energy $E(\\mathbf R)$ that depends on where the nuclei are: the **potential energy surface**. The nuclei then move on this surface as classical particles do on a landscape. Three facts of chemistry are read from it. A stable geometry is a minimum, found by following the forces $-\\nabla E$ downhill (a **geometry optimisation**). The energy of a reaction is the difference of $E$ between the minima of the products and the reactants. A **single point** is the calculation of $E$ for one fixed arrangement of the nuclei, the unit of work in everything below. The approximation fails where two surfaces come close, which does not concern this project.

*On what actually ran.* The cells above are the notebook's own: nothing is read from a cache. The helper `run_isolated` starts the programs of 10.2 and 10.3 in a separate interpreter, for the reason given in its comment. The project's calculations on the 6,895 compounds use the tight-binding method of 10.4 through `cypmode/challenge/physics_block.py`, and are described in 10.5 and 10.6.
"""),
    md("""
## 10.2 Hartree-Fock, the self-consistent field and basis sets

> **Field:** Quantum chemistry · Mathematics (linear algebra)

*Where this section sits:* second rung. 10.1 left us with an equation that cannot be solved and a bound that tells us how to rank approximations. Hartree-Fock is the simplest approximation that is good enough to be useful: it replaces the wavefunction of $N$ electrons by one built from $N$ one-electron functions. The section builds it for the smallest molecules, so that the machinery can be seen, and measures what it leaves out.

**Orbitals and the Slater determinant.** An **orbital** $\\varphi_i(\\mathbf r)$ is a one-electron wavefunction. If the electrons did not repel each other, the $N$-electron wavefunction would be a product of orbitals. A plain product does not change sign when two electrons are exchanged, so it violates the Pauli principle, and the cure is to antisymmetrise: the **Slater determinant** is the determinant of the $N \\times N$ matrix whose entry in row $i$ and column $j$ is orbital $i$ (with its spin) evaluated at the coordinates of electron $j$; exchanging two electrons exchanges two columns, and the determinant changes sign. For a closed shell (every orbital holds two electrons of opposite spin, $N/2$ spatial orbitals) the energy of the determinant is

$$E_{\\text{HF}} \\;=\\; 2\\sum_i h_{ii} \\;+\\; \\sum_{i,j}\\big(2J_{ij} - K_{ij}\\big) \\;+\\; V_{nn},$$

with $i, j$ over the occupied orbitals. Here $h_{ii}$ is the kinetic energy of an electron in orbital $i$ plus its attraction to the nuclei; $J_{ij}$ is the **Coulomb** repulsion between the charge clouds $|\\varphi_i|^2$ and $|\\varphi_j|^2$; $K_{ij}$ is the **exchange** energy, which has no classical counterpart and exists because electrons of the same spin avoid each other; and $V_{nn}$ is the repulsion of the nuclei. The **Hartree-Fock** method finds the orbitals that minimise $E_{\\text{HF}}$ under the condition that they stay orthonormal. Setting the variation to zero gives one equation per orbital, $\\hat f\\,\\varphi_i = \\varepsilon_i\\,\\varphi_i$, where the **Fock operator** $\\hat f$ holds the kinetic energy, the attraction to the nuclei, and the average field of the other electrons, which depends on the orbitals themselves.

**Iteration.** The equation for the orbitals needs the orbitals to build its operator. The way out is iteration: guess the orbitals, build $\\hat f$ from them, solve for new orbitals, and repeat until they stop changing. This loop is the **self-consistent field** method (SCF), and its result is self-consistent: the field produced by the orbitals is the field that produces them. The energies $\\varepsilon_i$ are the **orbital energies**.

**A basis, and the matrices.** To solve for functions on a computer, each orbital is written as a sum of fixed **basis functions** $\\chi_\\mu$ centred on the atoms, $\\varphi_i = \\sum_\\mu C_{\\mu i}\\chi_\\mu$, so that the unknowns are the numbers $C_{\\mu i}$ (the **LCAO** expansion, linear combination of atomic orbitals). The equations become the matrix equation of **Roothaan and Hall** (1951), $FC = SC\\varepsilon$, with the **overlap matrix** $S_{\\mu\\nu} = \\int\\chi_\\mu\\chi_\\nu\\,d\\mathbf r$ (the basis functions are not orthogonal), the **core Hamiltonian** $h_{\\mu\\nu}$ (kinetic energy and nuclear attraction between $\\chi_\\mu$ and $\\chi_\\nu$), and the **Fock matrix**

$$F_{\\mu\\nu} \\;=\\; h_{\\mu\\nu} + \\sum_{\\lambda\\sigma} P_{\\lambda\\sigma}\\Big[(\\mu\\nu|\\lambda\\sigma) - \\tfrac12(\\mu\\lambda|\\nu\\sigma)\\Big], \\qquad P_{\\lambda\\sigma} = 2\\sum_{i\\in\\text{occ}} C_{\\lambda i}C_{\\sigma i}, \\qquad E = \\tfrac12\\sum_{\\mu\\nu}P_{\\mu\\nu}(h_{\\mu\\nu} + F_{\\mu\\nu}) + V_{nn}.$$

$P$ is the **density matrix** (it holds the electron density), and $(\\mu\\nu|\\lambda\\sigma)$ is the **two-electron integral**, the repulsion between the charge distribution $\\chi_\\mu\\chi_\\nu$ and the distribution $\\chi_\\lambda\\chi_\\sigma$. The second term in the bracket is the exchange. The number of such integrals grows as the fourth power of the number of basis functions, which is what makes the method expensive.

**Why Gaussians.** The best basis functions would be the shape of a hydrogen orbital, $e^{-\\zeta r}$ (a **Slater-type orbital**). Integrals of those are hard. The product of two Gaussians $e^{-\\alpha r^2}$ on different centres is again a Gaussian, so that every integral has a closed form, with the **Boys function** $F_0(t) = \\tfrac12\\sqrt{\\pi/t}\\,\\operatorname{erf}\\sqrt t$ for the Coulomb ones. A Slater function is therefore approximated by a fixed sum of Gaussians: in the **STO-3G** basis, three. Larger bases (a function for each valence orbital in several sizes, plus polarisation functions of higher angular momentum) approach the **basis-set limit**, the energy that an infinitely flexible basis would give, as the variational principle of 10.1 promises: more functions, lower energy.

The cell writes all of this for two nuclei, each with one 1s function. The molecule **HeH⁺** (helium and hydrogen sharing two electrons) is the example, because its two atoms differ and the orbital is not fixed by symmetry: the self-consistent iteration really has work to do. The final energy is checked against the program PySCF.
"""),
    code(HF_HAND),
    md("""
The loop needed {{=n_scf_cycles}} cycles. In the first, the occupied orbital sits mostly on the nucleus with the larger charge (coefficients, in absolute value, {{=heh_history['coefficient on nucleus A'].iloc[0]:.3f}} on helium and {{=heh_history['coefficient on nucleus B'].iloc[0]:.3f}} on hydrogen); as the other electron's field is taken into account the orbital shifts toward the hydrogen ({{=heh_history['coefficient on nucleus A'].iloc[-1]:.3f}} and {{=heh_history['coefficient on nucleus B'].iloc[-1]:.3f}} in the last cycle), and the energy settles at {{=heh_by_hand:.6f}} hartree. PySCF, a program written by specialists, gives {{=heh_reference:.6f}}: the two agree to {{=abs(heh_by_hand - heh_reference):.0e}} hartree, the difference being the rounded exponents of the table. Every matrix of the method is in the cell, and a method that is run with a library afterwards is run with its internals known.

**What Hartree-Fock leaves out.** Each electron feels only the *average* field of the others. In reality the electrons avoid each other instantaneously, and the energy that this adds is the **correlation energy**, defined as the exact energy minus the Hartree-Fock energy (it is negative: correlation lowers the energy). For H₂ in the minimal basis the exact answer is available, since the basis holds only two orbitals, the bonding orbital $\\sigma_g$ and the antibonding $\\sigma_u$, and the exact state (**configuration interaction**, mixing every arrangement of the electrons in the orbitals) is a mix of "both electrons in $\\sigma_g$" with a little of "both in $\\sigma_u$". The second cell computes both energies as the two nuclei are pulled apart, and checks them against PySCF.
"""),
    code(H2_CURVE),
    md("""
Hand-coded and library values agree to {{=h2_agreement:.0e}} hartree for both methods at all {{=len(distances)}} distances. At the equilibrium distance of 0.74 Å the correlation energy is {{=equilibrium['correlation energy (kcal/mol)']:.1f}} kcal/mol, a small fraction of the total energy and several times the energy of a hydrogen bond (a few kcal/mol), which is the scale at which reactivity is decided. When the two atoms are pulled far apart, the picture breaks: at {{=far['distance (angstrom)']:.0f}} Å the Hartree-Fock energy lies {{=dissociation_error_hf:.0f}} kcal/mol above the energy of two separate hydrogen atoms, while the exact one is within {{=abs(dissociation_error_exact):.1f}} kcal/mol of it. The reason is that the restricted method puts both electrons into the same spatial orbital, which at a long distance describes a mixture of two neutral atoms and of a proton and a hydride ion. Correlated methods repair this and cost far more: the cost of Hartree-Fock grows as the fourth power of the number of basis functions, that of the standard correlated method **coupled cluster** with single, double and perturbative triple excitations, CCSD(T), as the seventh. CCSD(T) is the reference used to judge cheaper methods, and it cannot be run on a drug-like molecule in a basis of useful size, let alone on thousands.

*On what actually ran.* The cells above (the integrals, the loop, the 2 × 2 configuration interaction), and PySCF in an isolated interpreter for the comparison. The HeH⁺ exponents are 1.69 for helium and 1.24 for hydrogen and the weights of the three Gaussians are the tabulated ones for $\\zeta = 1$ (Szabo and Ostlund, *Modern Quantum Chemistry*, 1989, chapter 3 and appendix A).
"""),
]

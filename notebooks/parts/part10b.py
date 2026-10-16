"""Part X (second part): density functional theory (10.3) and the semi-empirical and tight-binding methods, GFN2-xTB (10.4)."""

from notebooks.notebook_builder import code, figure, md

DFT_WATER = '''
# Density functional theory against Hartree-Fock on the water molecule, with the program PySCF in an isolated interpreter (10.1): the geometry is the experimental one (O-H 0.957 angstrom, angle 104.5 degrees), the basis def2-SVP
# (a split-valence basis with polarisation functions, larger than the minimal one of 10.2). Four methods: Hartree-Fock; the local density approximation (LDA, SVWN); a generalised-gradient functional (PBE); and the hybrid B3LYP.
# For each: the total energy, the energy of the highest occupied orbital, the dipole moment, and the first ionisation potential as the energy of the cation minus the energy of the neutral molecule (both at the geometry of the neutral:
# a "vertical" ionisation, with the electrons of the cation allowed to relax). The experimental values (NIST Computational Chemistry Comparison and Benchmark Database) are an ionisation potential of 12.62 eV and a dipole of 1.85 debye.
water_rows = run_isolated("""
    import json, time
    from pyscf import gto, scf, dft
    atom = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"
    out = []
    for name, xc in (("Hartree-Fock", None), ("LDA (SVWN)", "lda,vwn"), ("GGA (PBE)", "pbe"), ("hybrid (B3LYP)", "b3lyp")):
        start = time.time()
        neutral = gto.M(atom=atom, basis="def2-svp", verbose=0)
        mf = scf.RHF(neutral) if xc is None else dft.RKS(neutral).set(xc=xc)
        mf.run()
        cation = gto.M(atom=atom, basis="def2-svp", charge=1, spin=1, verbose=0)
        mc = scf.UHF(cation) if xc is None else dft.UKS(cation).set(xc=xc)
        mc.run()
        dipole = float((mf.dip_moment(verbose=0) ** 2).sum() ** 0.5)
        out.append([name, mf.e_tot, float(mf.mo_energy[neutral.nelectron // 2 - 1]) * 27.211386, (mc.e_tot - mf.e_tot) * 27.211386, dipole, time.time() - start])
    print(json.dumps(out))
""")
water_table = pd.DataFrame(water_rows, columns=["method", "total energy (hartree)", "energy of the highest occupied orbital (eV)", "ionisation potential from energies (eV)", "dipole (debye)", "seconds"])
show_table(water_table, floatfmt=".3f")
water_hf, water_lda, water_pbe, water_b3 = (water_table.iloc[i] for i in range(4))
'''

GFN2_TERMS = '''
# One GFN2-xTB single point on toluene with the program's own report switched on (the library tblite, verbosity 2, in an isolated interpreter). The report lists the energy in parts: the repulsion, the dispersion, the electronic
# energy (which holds the extended Hueckel, electrostatic, exchange-correlation and Fermi terms of the expression in the text, converged together by the loop below), and the total. It also prints the loop that makes the charges
# self-consistent: for each cycle the total energy and the "density error", the change of the charges from one cycle to the next. The geometry is the lowest-energy conformer of RDKit's force field.
import re

from rdkit import Chem
from rdkit.Chem import AllChem

def geometry_of(smiles):
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(molecule, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(molecule)
    return [atom.GetSymbol() for atom in molecule.GetAtoms()], molecule.GetConformer().GetPositions().tolist()

ATOMIC_NUMBER = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16}
toluene_symbols, toluene_positions = geometry_of("Cc1ccccc1")
report = run_isolated_text(f"""
    import numpy as np
    from tblite.interface import Calculator
    numbers = np.array({[ATOMIC_NUMBER[s] for s in toluene_symbols]})
    positions = np.array({toluene_positions}) / 0.52917721067
    calculator = Calculator("GFN2-xTB", numbers, positions)
    calculator.set("verbosity", 2)
    result = calculator.singlepoint()
    print("NORB", len(result.get("orbital-energies")))
""")
def report_value(label):
    return float(re.search(rf"^{label}\\s+(-?[\\d.]+E[+-]\\d+) Eh", report, re.M).group(1))
repulsion, dispersion, electronic, total_energy = (report_value(name) for name in ("repulsion energy", "dispersion energy", "electronic energy", "total energy"))
cycles = pd.DataFrame([{"cycle": int(c), "total energy (hartree)": float(e), "density error": float(d)} for c, e, _, d in re.findall(r"^\\s+(\\d+)\\s+(-?\\d+\\.\\d+)\\s+(-?[\\d.]+E[+-]\\d+)\\s+([\\d.]+E[+-]\\d+)\\s*$", report, re.M)])
terms_table = pd.DataFrame([{"term": "repulsion", "hartree": repulsion, "kcal/mol": repulsion * HARTREE_TO_KCAL}, {"term": "dispersion line of the report", "hartree": dispersion, "kcal/mol": dispersion * HARTREE_TO_KCAL},
                            {"term": "electronic (the loop: extended Hueckel, electrostatics, exchange-correlation, Fermi, and the charge-dependent dispersion)", "hartree": electronic, "kcal/mol": electronic * HARTREE_TO_KCAL}, {"term": "total", "hartree": total_energy, "kcal/mol": total_energy * HARTREE_TO_KCAL}])
show_table(terms_table, floatfmt=".4f")
show_table(cycles, floatfmt=".8f")
terms_sum_error = abs(repulsion + dispersion + electronic - total_energy)
# where does a noncovalent attraction appear in the report? Two benzene molecules, stacked 3.8 angstrom apart and far apart (30 angstrom), the change of each printed line
benzene_symbols, benzene_positions = geometry_of("c1ccccc1")
def printed_lines(separation):
    text = run_isolated_text(f"""
        import numpy as np
        from tblite.interface import Calculator
        numbers = np.array({[ATOMIC_NUMBER[s] for s in benzene_symbols] * 2})
        first = np.array({benzene_positions})
        positions = np.vstack([first, first + np.array([0.0, 0.0, {separation}])]) / 0.52917721067
        calculator = Calculator("GFN2-xTB", numbers, positions)
        calculator.set("verbosity", 2)
        calculator.singlepoint()
    """)
    return [float(re.search(rf"^{label}\\s+(-?[\\d.]+E[+-]\\d+) Eh", text, re.M).group(1)) for label in ("repulsion energy", "dispersion energy", "electronic energy", "total energy")]
near, far_apart = printed_lines(3.8), printed_lines(30.0)
dimer_change = pd.DataFrame({"line of the report": ["repulsion", "dispersion line", "electronic", "total (the binding of the pair)"], "change on bringing the molecules to 3.8 angstrom (kcal/mol)": [(a - b) * HARTREE_TO_KCAL for a, b in zip(near, far_apart)]})
show_table(dimer_change, floatfmt=".2f")
dimer_binding, dimer_electronic, dimer_dispersion_line = float(dimer_change.iloc[3, 1]), float(dimer_change.iloc[2, 1]), float(dimer_change.iloc[1, 1])
n_atoms_toluene, n_cycles_toluene = len(toluene_symbols), len(cycles)
n_orbitals_toluene = int(re.search(r"^NORB (\\d+)", report, re.M).group(1))
'''

GFN2_VS_DFT = '''
# What the approximation costs and what it keeps. Twelve small molecules (their geometries from RDKit's force field, the same for both methods), the vertical ionisation potential as the energy of the cation minus the energy of the
# neutral molecule at the same geometry, by GFN2-xTB (tblite) and by the hybrid functional B3LYP in the def2-SVP basis (PySCF), each in an isolated interpreter. Reported: the two values for each molecule, the constant offset between
# the methods, the correlation of the two sets, the scatter around the offset, and the time.
from scipy import stats

ip_molecules = {"water": "O", "methanol": "CO", "ammonia": "N", "trimethylamine": "CN(C)C", "pyridine": "c1ccncc1", "benzene": "c1ccccc1", "toluene": "Cc1ccccc1", "aniline": "Nc1ccccc1", "thiophene": "c1ccsc1", "furan": "c1ccoc1", "imidazole": "c1c[nH]cn1", "acetonitrile": "CC#N"}
ip_geometries = {name: geometry_of(smiles) for name, smiles in ip_molecules.items()}
ip_xtb = run_isolated(f"""
    import json, time
    import numpy as np
    from tblite.interface import Calculator
    geometries = {ip_geometries!r}
    numbers_of = {ATOMIC_NUMBER!r}
    out, start = {{}}, time.time()
    for name, (symbols, positions) in geometries.items():
        numbers, xyz = np.array([numbers_of[s] for s in symbols]), np.array(positions) / 0.52917721067
        def energy(charge):
            calculator = Calculator("GFN2-xTB", numbers, xyz, charge=float(charge), uhf=int((numbers.sum() - charge) % 2))
            calculator.set("verbosity", 0)
            return float(calculator.singlepoint().get("energy"))
        out[name] = (energy(1) - energy(0)) * 27.211386
    out["_seconds"] = time.time() - start
    print(json.dumps(out))
""")
ip_dft = run_isolated(f"""
    import json, time
    from pyscf import gto, dft
    geometries = {ip_geometries!r}
    out, start = {{}}, time.time()
    for name, (symbols, positions) in geometries.items():
        atom = "; ".join(f"{{s}} {{p[0]}} {{p[1]}} {{p[2]}}" for s, p in zip(symbols, positions))
        neutral = dft.RKS(gto.M(atom=atom, basis="def2-svp", verbose=0)).set(xc="b3lyp").run()
        cation = dft.UKS(gto.M(atom=atom, basis="def2-svp", charge=1, spin=1, verbose=0)).set(xc="b3lyp").run()
        out[name] = (cation.e_tot - neutral.e_tot) * 27.211386
    out["_seconds"] = time.time() - start
    print(json.dumps(out))
""")
ip_names = list(ip_molecules)
ip_table = pd.DataFrame({"molecule": ip_names, "GFN2-xTB (eV)": [ip_xtb[n] for n in ip_names], "B3LYP / def2-SVP (eV)": [ip_dft[n] for n in ip_names]})
ip_table["difference (eV)"] = ip_table["GFN2-xTB (eV)"] - ip_table["B3LYP / def2-SVP (eV)"]
show_table(ip_table, floatfmt=".2f")
ip_offset = float(ip_table["difference (eV)"].mean())
ip_scatter = float(ip_table["difference (eV)"].std(ddof=1))
ip_pearson = float(np.corrcoef(ip_table["GFN2-xTB (eV)"], ip_table["B3LYP / def2-SVP (eV)"])[0, 1])
ip_spearman = float(stats.spearmanr(ip_table["GFN2-xTB (eV)"], ip_table["B3LYP / def2-SVP (eV)"])[0])
ip_speedup = float(ip_dft["_seconds"] / ip_xtb["_seconds"])
ip_seconds_xtb, ip_seconds_dft = float(ip_xtb["_seconds"]), float(ip_dft["_seconds"])
'''

CELLS = [
    md("""
## 10.3 Density functional theory

> **Field:** Quantum chemistry · Physics

*Where this section sits:* third rung. Hartree-Fock (10.2) treats each electron in the average field of the others and misses the correlation of their motions; the methods that recover it from the wavefunction (coupled cluster and its relatives) cost too much for molecules of the size of a drug. Density functional theory takes another road: it gives up the wavefunction altogether and works with a quantity that has three variables however many electrons there are.

**The electron density.** The **electron density** $\\rho(\\mathbf r)$ is the number of electrons per unit volume at the point $\\mathbf r$: $N$ times the probability density of finding any one electron there. Water has 10 electrons and a wavefunction of 30 coordinates; its density is a function of 3. **Hohenberg and Kohn** (1964) proved two statements. First, the density of the ground state determines the external potential $v(\\mathbf r)$ in which the electrons move (here, the field of the nuclei) up to a constant, and with it the Hamiltonian and therefore every property of the system: the ground-state energy is a **functional** of the density, $E[\\rho]$ (a functional is a function of a function). Second, a variational principle like that of 10.1 holds for densities: $E[\\rho] \\ge E_0$ for every trial density that is non-negative and integrates to $N$, with equality at the true density. The theorems say that the functional exists. They do not say what it is.

The **Kohn-Sham** scheme. **Kohn and Sham** (1965) found a way to use the theorems. Replace the interacting electrons by fictitious electrons that do not interact, chosen so that they have the same density. Their orbitals $\\varphi_i$, with orbital energies $\\varepsilon_i$, obey one-electron equations,

$$\\Big[-\\tfrac12\\nabla^2 + v(\\mathbf r) + \\int\\frac{\\rho(\\mathbf r')}{|\\mathbf r - \\mathbf r'|}\\,d\\mathbf r' + v_{xc}(\\mathbf r)\\Big]\\varphi_i = \\varepsilon_i\\,\\varphi_i, \\qquad \\rho(\\mathbf r) = 2\\sum_{i\\in\\text{occ}}|\\varphi_i(\\mathbf r)|^2 ,$$

where $v(\\mathbf r)$ is the attraction of the nuclei, the integral is the classical (Hartree) repulsion of the electron cloud with itself, and $v_{xc}$ is the **exchange-correlation potential**, the derivative of the **exchange-correlation energy** $E_{xc}[\\rho]$ with respect to the density. The total energy is $E = T_s[\\rho] + \\int v\\rho + \\tfrac12\\iint\\rho\\rho'/|\\mathbf r - \\mathbf r'| + E_{xc}[\\rho]$, with $T_s$ the kinetic energy of the fictitious electrons. Everything that is hard is in $E_{xc}$: the exchange, the correlation, and the small difference between the true kinetic energy and $T_s$. The equations are solved by the same self-consistent loop as in 10.2, since the potential depends on the density that the orbitals make. In principle the scheme is exact; in practice $E_{xc}$ is approximated.

**The approximations.** They form a ladder of increasing information (Perdew's "Jacob's ladder"). The **local density approximation** (LDA) takes $E_{xc}$ at each point from the exchange-correlation energy per electron of a uniform gas of electrons at the density found there. A **generalised gradient approximation** (GGA; PBE, Perdew, Burke and Ernzerhof, 1996, is the standard one) adds the gradient of the density, so that the functional knows whether the density is changing. A **hybrid** functional replaces a fraction of the approximate exchange by the exact exchange of Hartree-Fock; **B3LYP** (Becke, 1993, with the correlation of Lee, Yang and Parr, 1988) mixes in 20 percent and is, with its relatives, the functional most used in chemistry. The cost grows with the third to fourth power of the size of the basis, below the fifth to seventh power of the correlated wavefunction methods.

The cell runs water with Hartree-Fock and with one functional of each kind. Two properties are compared with experiment: the first **ionisation potential**, the energy needed to remove one electron, computed as the energy of the cation minus the energy of the neutral molecule (a difference of two separate calculations, each allowed to relax its electrons), and the energy of the highest occupied orbital.
"""),
    code(DFT_WATER),
    md("""
Two lessons for the sections that follow. First, **energy differences are reliable and orbital energies are not.** The ionisation potential obtained as a difference is {{=water_lda['ionisation potential from energies (eV)']:.2f}} eV (LDA), {{=water_pbe['ionisation potential from energies (eV)']:.2f}} (PBE) and {{=water_b3['ionisation potential from energies (eV)']:.2f}} (B3LYP) against 12.62 measured, while the energy of the highest occupied orbital, which in an exact theory would equal minus the ionisation potential, is only {{=-water_lda['energy of the highest occupied orbital (eV)']:.1f}} eV (LDA), {{=-water_pbe['energy of the highest occupied orbital (eV)']:.1f}} (PBE) and {{=-water_b3['energy of the highest occupied orbital (eV)']:.1f}} (B3LYP). The reason is the **self-interaction error** of approximate functionals: an electron partly repels itself, which pushes the orbital energies up. Hartree-Fock has no such error for one electron, and its orbital energy ({{=-water_hf['energy of the highest occupied orbital (eV)']:.1f}} eV) is near the measured value because two neglected effects cancel (the relaxation of the other electrons when one is removed, and the correlation), while its energy difference ({{=water_hf['ionisation potential from energies (eV)']:.2f}} eV) is {{=12.62 - water_hf['ionisation potential from energies (eV)']:.1f}} eV too low because the correlation energy of the cation and of the neutral molecule differ. Second, the dipole moments of the four methods lie between {{=water_table['dipole (debye)'].min():.2f}} and {{=water_table['dipole (debye)'].max():.2f}} debye against 1.85 measured; the remaining difference is a matter of the basis set, not of the method. The numbers of this cell are the reason that the project computes ionisation potentials, electron affinities and hydrogen-abstraction energies as differences of total energies, and does not use orbital energies except as descriptors of shape (10.5).

*On what actually ran.* PySCF in an isolated interpreter (`run_isolated`, 10.1), restricted Hartree-Fock and Kohn-Sham for the neutral molecule and the unrestricted versions for the cation. The experimental values are from the NIST Computational Chemistry Comparison and Benchmark Database.
"""),
    md("""
## 10.4 Semi-empirical and tight-binding methods: the energy of GFN2-xTB

> **Field:** Quantum chemistry · Chemistry

*Where this section sits:* fourth and last rung, and the one the project runs. A hybrid-functional calculation in a useful basis is a matter of minutes for a small molecule and hours for a drug, and the physics block (10.5) needs about forty single points for each of 6,895 compounds. The sum is years of processor time. The cheaper rungs exist for this reason, and the price of each is stated here so that the reader knows what the features that came out of it can and cannot mean.

**Semi-empirical methods.** A **semi-empirical** method keeps the structure of Hartree-Fock and removes most of its cost by four steps: only the valence electrons are treated (the core electrons are folded into the charge of the nucleus), the basis is minimal (one function for each valence orbital), most of the two-electron integrals that make the cost grow as the fourth power are neglected or replaced by simple functions of the distance, and the integrals that remain are replaced by **parameters** fitted to experimental or high-level results. The older families of this kind (AM1, PM3, and PM6 of Stewart, 2007) were fitted mainly to heats of formation and molecular geometries. The result is about a thousand times faster than a density functional calculation, with errors that are not random but depend on the chemistry the parameters were fitted to.

**Tight-binding from density functional theory.** The family used here, **GFN-xTB** of Grimme and co-workers, descends from density-functional tight binding. Write the true density as $\\rho = \\rho_0 + \\delta\\rho$, with $\\rho_0$ the sum of the densities of the neutral atoms and $\\delta\\rho$ the change that bonding makes, which is small, and expand the density-functional energy in powers of $\\delta\\rho$: $E = E[\\rho_0] + E^{(1)} + E^{(2)} + E^{(3)} + \\dots$. The zeroth-order term is the energy of the atoms and the short-range repulsion between them; the first-order term is the energy of the electrons moving in the field of the neutral atoms, which has the form of an old and simple model of bonding, the **extended Hückel** model, in which the coupling of two orbitals is proportional to their overlap and to the average of their on-site energies; the second-order term is the response to the charges that bonding moves between atoms, which makes the method self-consistent in the charges; the third-order term is the dependence of that response on how charged each atom already is. Each term is then given a parametrised form and the parameters are fitted to reproduce the geometries, vibrational frequencies and noncovalent interactions of molecules across the periodic table (this is what GFN stands for, geometry, frequency, noncovalent; the "x" is for extended). The valence basis is minimal.

**The energy of GFN2-xTB.** The second version (Bannwarth, Ehlert and Grimme, 2019) writes the energy as the sum

$$E \\;=\\; E_{\\text{rep}} + E_{\\text{disp}} + E_{\\text{EHT}} + E_{\\text{IES}} + E_{\\text{IXC}} + E_{\\text{AES}} + E_{\\text{AXC}} + G_{\\text{Fermi}} .$$

$E_{\\text{rep}}$ is the classical short-range repulsion between pairs of atoms. $E_{\\text{disp}}$ is the **dispersion** energy, the weak attraction between any two groups of atoms that comes from correlated fluctuations of their electrons (and decides noncovalent contacts, such as a ligand sitting in a pocket); GFN2 uses the D4 model, in which the dispersion coefficients of an atom depend on its charge. $E_{\\text{EHT}}$ is the extended Hückel energy of the first order. $E_{\\text{IES}}$ and $E_{\\text{IXC}}$ are the **isotropic** electrostatic and exchange-correlation energies of the second and third orders, which treat the charge of each atom as a point (in GFN2 the second-order terms resolve it by shell). $E_{\\text{AES}}$ and $E_{\\text{AXC}}$ are the **anisotropic** corrections: they let each atom carry a dipole and a quadrupole as well as a charge, so that a lone pair, a $\\pi$ cloud or a polar bond has a direction. $G_{\\text{Fermi}}$ is the energy that comes from letting the occupation of the orbitals be smeared over levels near the highest occupied one, as at a finite **electronic temperature** (300 K by default); it makes the iterations converge for awkward molecules and is almost zero for ordinary ones. The cell below runs one calculation on toluene with the program's own report switched on, and shows the parts of the energy that the program prints and the self-consistent loop in the charges.
"""),
    code(GFN2_TERMS),
    md("""
The parts add up to the total to {{=terms_sum_error:.0e}} hartree. For the {{=n_atoms_toluene}} atoms of toluene the repulsion is {{=terms_table.iloc[0]['kcal/mol']:+,.0f}} kcal/mol, the dispersion {{=terms_table.iloc[1]['kcal/mol']:+.2f}} kcal/mol, and the electronic energy, in which the other terms of the expression are converged together, {{=terms_table.iloc[2]['kcal/mol']:+,.0f}} kcal/mol. The absolute values mean nothing by themselves: the total is on a scale of the method's own, not comparable with the total energies of 10.2 and 10.3. Only differences between arrangements or charge states of the same atoms are physical, which is what 10.5 will use. One line of the report needs a remark. The dispersion line is tiny and, for a single molecule, positive; it is not the D4 attraction. The second table shows where that attraction is counted: two benzene molecules brought from 30 Å to a stacked distance of 3.8 Å change the total energy by {{=dimer_binding:.1f}} kcal/mol, and the change sits in the electronic line ({{=dimer_electronic:.1f}} kcal/mol), not in the dispersion line ({{=dimer_dispersion_line:+.2f}}). This is consistent with the charge dependence of the D4 dispersion of GFN2: the charges change in the loop, and the attraction that binds the pair is counted there, in the electronic energy. The loop took {{=n_cycles_toluene}} cycles, and the density error, the change of the charges between one cycle and the next, fell from {{=cycles['density error'].iloc[0]:.1e}} to {{=cycles['density error'].iloc[-1]:.1e}}. Each cycle builds the Hamiltonian from the current charges, diagonalises a matrix with one row for each valence basis function (here {{=n_orbitals_toluene}} for toluene), and takes new charges from the occupied orbitals. That is the loop of 10.2 with a very small matrix, which is the source of the speed.

**What the approximation keeps.** The question that decides whether the method can be used is whether it preserves the ordering that the density functional method gives. The next cell takes twelve small molecules with the same geometry for both methods, computes the vertical ionisation potential as in 10.3 by GFN2-xTB and by B3LYP in the def2-SVP basis, and compares.
"""),
    code(GFN2_VS_DFT),
    md("""
The two methods order the molecules in nearly the same way (a Pearson correlation of {{=ip_pearson:.2f}} and a Spearman rank correlation of {{=ip_spearman:.2f}}), but GFN2-xTB gives values that are higher than B3LYP by {{=ip_offset:.1f}} eV on average, with a scatter of {{=ip_scatter:.1f}} eV around that offset. The offset is large and is not a defect of the code: the method was fitted to structures, frequencies and noncovalent interactions and not to the energies of charged molecules, and an ionisation potential taken from it is not an ionisation potential in the experimental sense. It is a number that increases when a molecule is harder to oxidise, which is what a tree model can use, since it splits on the ordering and a constant offset moves no split. The same holds for the other energies of 10.5. The price is that these features carry a role-dependent offset, as the docstring of the module says, which the tree learns from the other columns. The time: GFN2-xTB took {{=ip_seconds_xtb:.1f}} seconds for the twelve molecules (both charge states) and B3LYP {{=ip_seconds_dft:.0f}}, a ratio of about {{=ip_speedup:.0f}} for molecules of at most 15 atoms; the ratio grows with size, because the cost of the density functional method rises more steeply.

*On what actually ran.* `tblite` (the library that implements the GFN-xTB family, version 0.7.0, called from `cypmode/challenge/physics_block.py`) and PySCF, each in an isolated interpreter. The decomposition of the energy is the program's own report (verbosity 2); the sum is checked by the cell.
"""),
]

# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part X (third part): the quantities of the physics block (10.5) and the experiments with them (10.6)."""

from notebooks.notebook_builder import code, md

QUANTITIES = '''
# The quantities of the physics block (cypmode/challenge/physics_block.py) computed on three small molecules with the project's own functions, run in an isolated interpreter (10.1). Toluene has a benzylic C-H, the
# methylene group next to a ring; N-methylpiperidine has a tertiary amine and the C-H bonds next to its nitrogen; propane has plain aliphatic C-H bonds. For each molecule the routine returns the energy-based quantities of the
# whole molecule (ionisation potential, electron affinity, hardness, chemical potential, electrophilicity, solvation free energy in water), for each atom the partial charge, the two condensed Fukui functions and, for the nitrogens,
# the proton affinity in the gas phase and in water, and for each C-H site the hydrogen-abstraction energy, vertical (the geometry frozen, physics_block) and relaxed (the radical centre, its neighbours and their hydrogens
# allowed to move, physics_tierb.relaxed_hat). The experimental C-H bond dissociation enthalpies at 298 K (toluene 89.7 kcal/mol, the secondary C-H of propane 98.6) are those compiled by Blanksby and Ellison (Acc. Chem. Res. 2003).
quantities = run_isolated("""
    import json
    import numpy as np
    from rdkit import Chem
    from cypmode.challenge import physics_block as pb
    from cypmode.challenge import physics_tierb as tb
    out = {}
    for name, smiles in (("toluene", "Cc1ccccc1"), ("N-methylpiperidine", "CN1CCCCC1"), ("propane", "CCC")):
        features, rows = pb.compute_molecule(smiles)
        mol_h = Chem.AddHs(pb.largest_fragment(smiles))
        ids, energies = pb.conformer_ensemble(mol_h)
        positions = mol_h.GetConformer(ids[int(np.argmin(energies))]).GetPositions()
        numbers = np.array([a.GetAtomicNum() for a in mol_h.GetAtoms()])
        sites = []
        for row in rows:
            if row["hat_kcal"] == row["hat_kcal"]:
                relaxed = tb.relaxed_hat(mol_h, numbers, positions, 0, int(row["atom"]))
                sites.append({"atom": int(row["atom"]), "role": row["ch_role"], "vertical": float(row["hat_kcal"]), "relaxed": None if relaxed is None else float(relaxed)})
        out[name] = {"molecule": {k: float(v) for k, v in features.items() if k in ("phys_ip_ev", "phys_ea_ev", "phys_hardness_ev", "phys_chem_potential_ev", "phys_electrophilicity_ev", "phys_dg_solv_kcal", "phys_homo_ev", "phys_lumo_ev", "phys_dipole_d")},
                     "atoms": [{k: (None if (isinstance(v, float) and v != v) else (float(v) if isinstance(v, (int, float, np.floating)) else v)) for k, v in row.items() if k != "SMILES"} for row in rows], "sites": sites}
    print(json.dumps(out))
""")
experimental_bde = {"toluene": 89.7, "propane": 98.6}
molecule_table = pd.DataFrame({name: data["molecule"] for name, data in quantities.items()}).T.rename(columns={"phys_ip_ev": "ionisation potential (eV)", "phys_ea_ev": "electron affinity (eV)", "phys_hardness_ev": "hardness (eV)", "phys_chem_potential_ev": "chemical potential (eV)",
                       "phys_electrophilicity_ev": "electrophilicity (eV)", "phys_dg_solv_kcal": "solvation free energy in water (kcal/mol)", "phys_homo_ev": "HOMO (eV)", "phys_lumo_ev": "LUMO (eV)", "phys_dipole_d": "dipole (debye)"})
show_table(molecule_table.reset_index().rename(columns={"index": "molecule"}), floatfmt=".2f")
site_rows = []
for name, data in quantities.items():
    for site in data["sites"]:
        site_rows.append({"molecule": name, "atom": site["atom"], "C-H role": site["role"], "vertical (kcal/mol)": site["vertical"], "relaxed (kcal/mol)": site["relaxed"], "measured (kcal/mol)": experimental_bde.get(name) if site["role"] in ("benzylic",) or (name == "propane" and site["atom"] == 1) else np.nan})
site_table = pd.DataFrame(site_rows)
show_table(site_table, floatfmt=".1f")
atom_rows = []
for name in ("N-methylpiperidine",):
    for row in quantities[name]["atoms"]:
        atom_rows.append({"atom": row["atom"], "element": row["symbol"], "charge q": row["q"], "f-": row["f_minus"], "f+": row["f_plus"], "proton affinity, gas (kcal/mol)": row["pa_gas_kcal"], "proton affinity, water (kcal/mol)": row["pa_water_kcal"]})
atom_table = pd.DataFrame(atom_rows)
show_table(atom_table, floatfmt=".3f")
fukui_sum_minus, fukui_sum_plus = float(atom_table["f-"].sum()), float(atom_table["f+"].sum())
best = {name: sorted((s for s in data["sites"] if s["relaxed"] is not None), key=lambda s: s["vertical"]) for name, data in quantities.items()}
toluene_site, piperidine_site, propane_site = best["toluene"][0], best["N-methylpiperidine"][0], best["propane"][0]
vertical_order = [name for name, site in sorted(((n, d[0]) for n, d in best.items()), key=lambda x: x[1]["vertical"])]
relaxed_order = [name for name, site in sorted(((n, d[0]) for n, d in best.items()), key=lambda x: x[1]["relaxed"])]
'''

CONFORMERS = '''
# A flexible molecule is not one shape. Propranolol (a CYP2D6 substrate: a basic nitrogen on a short chain attached to a naphthalene) is embedded 8 times by RDKit's ETKDG method and each shape is relaxed with the force field MMFF.
# The project then scores every shape with GFN2-xTB in implicit water (ALPB) and weights them by the Boltzmann factor exp(-E / kT) of the quantum-chemical energy, kT = 0.5925 kcal/mol at 298 K. The table compares the order
# of the shapes by the force field and by GFN2/ALPB, gives the weights, the vertical ionisation potential of each shape, and the distance from the basic nitrogen to the centres of the two aromatic rings of the naphthalene
# (the window of the CYP2D6 pharmacophore is 4.5 to 7.5 angstrom, 10.5).
conformers = run_isolated("""
    import json
    import numpy as np
    from rdkit import Chem
    from cypmode.challenge import physics_block as pb
    from cypmode.challenge import physics_tierb as tb
    mol = pb.largest_fragment("CC(C)NCC(O)COc1cccc2ccccc12")
    mol_h = Chem.AddHs(mol)
    ids, mmff = pb.conformer_ensemble(mol_h)
    numbers = np.array([a.GetAtomicNum() for a in mol_h.GetAtoms()])
    rings = [list(r) for r in mol.GetRingInfo().AtomRings() if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    nitrogen = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N"][0]
    rows = []
    for cid, e_mmff in zip(ids, mmff):
        xyz = mol_h.GetConformer(cid).GetPositions()
        gas, cation, water = pb._single_point(numbers, xyz, 0), pb._single_point(numbers, xyz, 1), pb._single_point(numbers, xyz, 0, "water")
        distances = sorted(float(np.linalg.norm(xyz[r].mean(axis=0) - xyz[nitrogen])) for r in rings)
        rows.append({"mmff": float(e_mmff), "gfn2_water": float(water.get("energy")) * pb.HARTREE_KCAL, "ip": (float(cation.get("energy")) - float(gas.get("energy"))) * pb.HARTREE_EV, "nearest": distances[0], "farthest": distances[-1]})
    weights = tb.boltzmann_weights(np.array([r["gfn2_water"] for r in rows]))
    for r, w in zip(rows, weights):
        r["weight"] = float(w)
    print(json.dumps(rows))
""")
conformer_table = pd.DataFrame(conformers)
conformer_table["MMFF, relative (kcal/mol)"] = conformer_table["mmff"] - conformer_table["mmff"].min()
conformer_table["GFN2/ALPB, relative (kcal/mol)"] = conformer_table["gfn2_water"] - conformer_table["gfn2_water"].min()
conformer_table = conformer_table.sort_values("GFN2/ALPB, relative (kcal/mol)").reset_index(drop=True)
display_table = conformer_table[["MMFF, relative (kcal/mol)", "GFN2/ALPB, relative (kcal/mol)", "weight", "ip", "nearest", "farthest"]].rename(columns={"weight": "Boltzmann weight", "ip": "vertical ionisation potential (eV)", "nearest": "N to nearest ring (angstrom)", "farthest": "N to farthest ring (angstrom)"})
show_table(display_table, floatfmt=".2f")
mmff_best = int(conformer_table["MMFF, relative (kcal/mol)"].idxmin())
weight_of_mmff_best = float(conformer_table.loc[mmff_best, "weight"])
weighted_ip = float((conformer_table["weight"] * conformer_table["ip"]).sum())
ip_spread = float(conformer_table["ip"].max() - conformer_table["ip"].min())
lowest_ip = float(conformer_table.loc[0, "ip"])
n_conformers_9 = len(conformer_table)
within_window = conformer_table[(conformer_table["nearest"] >= 4.5) & (conformer_table["nearest"] <= 7.5)]
share_in_window = float(within_window["weight"].sum())
'''

PHYSICS_FACTS = '''
# What the physics block is made of and what it cost, read from the code and the caches. The block (cypmode/challenge/physics_block.py) has four levels, molecule, atom, site and pharmacophore; the tables of column names of the
# module give the number of columns of each; the cache holds one row of the columns for each of the 6,895 compounds, and a column that describes a role that a compound does not have (for example the proton affinity of an azole
# nitrogen in a compound without an azole) is missing for it. The tier-B block (physics_tierb.py) is the compact block of Boltzmann-weighted and relaxed quantities.
from cypmode.challenge import physics_block as physics_a
from cypmode.challenge import physics_tierb as physics_b

n_tier_a = len(physics_a.FEATURE_NAMES)
n_tier_b = len(physics_b.FEATURE_NAMES)
n_ch_roles, n_binder_roles = len(physics_a.CH_ROLES), len(physics_a.BINDER_ROLES)
cache_a = pd.read_csv(ROOT / "data/tdi_track/outputs/caches/physics_cache.csv")
columns_a = cache_a.drop(columns=cache_a.columns[0])
missing_share = columns_a.isna().mean()
n_rows_cache = len(cache_a)
n_mostly_missing, n_nearly_empty = int((missing_share > 0.5).sum()), int((missing_share > 0.9).sum())
extract_hours_a = (pd.Timestamp("2026-10-08 17:05:20") - pd.Timestamp("2026-10-08 09:59:46")).total_seconds() / 3600
extract_hours_b = (pd.Timestamp("2026-10-10 02:39:23") - pd.Timestamp("2026-10-10 00:45:25")).total_seconds() / 3600
print(f"tier A: {n_tier_a} columns, {n_mostly_missing} of them missing for more than half of the {n_rows_cache:,} compounds; tier B: {n_tier_b} columns; extraction {extract_hours_a:.1f} h and {extract_hours_b:.1f} h")
'''

PHYSICS_VERDICTS = '''
# The verdicts of the physics experiments, read from the logs: the block in the stack (tier A, the compact selection of 20 columns chosen inside each task, tier B), the pair model with the physics columns, and the physics inside
# the selection-aware CYP2D6 model; each in the 5-outer-fold design, and the three blocks again in the 10-outer-fold replicate (6.7). The change is the paired change of the outer-fold AUC when the addition joins the stack, with the
# corrected resampled t. The last lines take the block alone: its own AUC (a model on the physics columns only) and the change that it makes to the plain descriptor model (feature set A) and to the descriptor-plus-fingerprint model.
POP_PHYSICS_LINE = re.compile(r"z0=(\\d\\.\\d+)\\s+stacked with physics vs without: AUC ([+-][\\d.]+) \\(p=([\\d.]+)")
def physics_row(label, stem, iso, design):
    result = read_gate_log(stem, iso)
    if result is not None and not result["stacked"].empty:
        table = result["stacked"].set_index("z0")
        return {"experiment": label, "design": design, "enzyme": iso, "AUC change at z0 1.0": float(table.loc[1.0, "gain"]), "p at z0 1.0": float(table.loc[1.0, "p"]), "AUC change at z0 1.96": float(table.loc[1.96, "gain"]), "p at z0 1.96": float(table.loc[1.96, "p"]),
                "gate": "MET" if result["verdicts"].get("stacked") else "NOT MET"}
    path = LOGS9 / f"{stem}_{iso}.log"
    if path.exists():
        text = path.read_text(errors="ignore")
        found = {float(z): (float(g), float(p)) for z, g, p in POP_PHYSICS_LINE.findall(text)}
        if len(found) == 2:
            return {"experiment": label, "design": design, "enzyme": iso, "AUC change at z0 1.0": found[1.0][0], "p at z0 1.0": found[1.0][1], "AUC change at z0 1.96": found[1.96][0], "p at z0 1.96": found[1.96][1], "gate": "NOT MET" if "NOT MET" in text else "MET"}
    return {"experiment": label, "design": design, "enzyme": iso, "AUC change at z0 1.0": np.nan, "p at z0 1.0": np.nan, "AUC change at z0 1.96": np.nan, "p at z0 1.96": np.nan, "gate": "not run yet"}

physics_plan = [("tier A (133 columns)", "physics_experiment", "5 folds"), ("compact selection (20 columns)", "physics_compact", "5 folds"), ("tier B (relaxed and averaged)", "physics_tierb", "5 folds"),
                ("pair model with physics", "pair_transfer_physics", "5 folds"), ("physics in the selection-aware model", "population_physics", "5 folds"),
                ("tier A (133 columns)", "physics_experiment_f10", "10 folds"), ("compact selection (20 columns)", "physics_compact_f10", "10 folds"), ("tier B (relaxed and averaged)", "physics_tierb_f10", "10 folds")]
physics_rows = []
for label, stem, design in physics_plan:
    for iso in ("CYP3A4", "CYP2D6"):
        if stem == "population_physics" and iso == "CYP3A4":
            continue
        physics_rows.append(physics_row(label, stem, iso, design))
physics_table = pd.DataFrame(physics_rows)
show_table(physics_table, floatfmt=".4f")

alone_rows = []
for iso in ("CYP3A4", "CYP2D6"):
    result = read_gate_log("physics_experiment", iso)
    diagnostic = result["diagnostic"].set_index("z0")["outer AUC alone"]
    alone = result["alone"].set_index(["z0", "against"])
    alone_rows.append({"enzyme": iso, "physics columns alone, AUC at z0 1.0": float(diagnostic.loc[1.0]), "added to feature set A": float(alone.loc[(1.0, "lgbm_A_deep"), "gain"]), "p (A)": float(alone.loc[(1.0, "lgbm_A_deep"), "p"]),
                       "added to A+fps": float(alone.loc[(1.0, "lgbm_A+fps_deep"), "gain"]), "p (A+fps)": float(alone.loc[(1.0, "lgbm_A+fps_deep"), "p"])})
alone_table = pd.DataFrame(alone_rows)
show_table(alone_table, floatfmt=".4f")
def physics_cell(label, design, iso):
    r = physics_table[(physics_table["experiment"] == label) & (physics_table["design"] == design) & (physics_table["enzyme"] == iso)]
    return r.iloc[0]
tier_a_3a4, compact_3a4, tierb_3a4 = (physics_cell(l, "5 folds", "CYP3A4") for l in ("tier A (133 columns)", "compact selection (20 columns)", "tier B (relaxed and averaged)"))
compact_ten_3a4 = physics_cell("compact selection (20 columns)", "10 folds", "CYP3A4")
n_positive_physics = int((physics_table["AUC change at z0 1.0"] > 0).sum())
n_physics_tests = int(physics_table["AUC change at z0 1.0"].notna().sum())
n_physics_met = int((physics_table["gate"] == "MET").sum())
largest_physics = float(physics_table["AUC change at z0 1.0"].max())
'''

CELLS = [
    md("""
## 10.5 The quantities of the physics block

> **Field:** Quantum chemistry · Biochemistry

*Where this section sits:* 10.4 gave the engine; this section says what is asked of it, quantity by quantity, and why each quantity is connected with the mechanism of time-dependent inhibition of 1.3, 1.9 and 1.10: Compound I takes a hydrogen atom from a carbon, a nitrogen binds the iron, a molecule is oxidised. Each quantity is a difference of the energies of single points, defined here, and each is computed in the cell below with the project's own functions (`cypmode/challenge/physics_block.py` and `cypmode/challenge/physics_tierb.py`) on three small molecules, so that every number can be followed by hand.

**Ionisation potential, electron affinity and what they build.** Let $E(N)$ be the energy of the molecule with its $N$ electrons and $E(N \\pm 1)$ that of the cation and the anion at the same geometry (a **vertical** change: the nuclei do not move). The **ionisation potential** is $\\mathrm{IP} = E(N-1) - E(N)$, the cost of taking an electron away, and the **electron affinity** is $\\mathrm{EA} = E(N) - E(N+1)$, the gain from adding one. From them come three quantities of **conceptual density functional theory** (Parr and co-workers): the **chemical potential** $\\mu = -(\\mathrm{IP} + \\mathrm{EA})/2$ (the tendency of electrons to leave; minus the Mulliken electronegativity), the **hardness** $\\eta = (\\mathrm{IP} - \\mathrm{EA})/2$ (the resistance to rearranging the electrons: a hard molecule has a large gap between the energy to remove an electron and the energy gained by adding one) and the **electrophilicity** $\\omega = \\mu^2/(2\\eta)$ (the energy lowering when the molecule takes up electrons from its surroundings, large for a molecule that is both eager and soft). The orbital energies of the highest occupied and lowest unoccupied orbitals, HOMO and LUMO (7.8), are kept too, and the dipole moment.

**Solvation.** Water is a dielectric continuum around the molecule. In the **ALPB** model (analytical linearised Poisson-Boltzmann; Ehlert, Stahn, Spicher and Grimme, 2021) the electric field of the solute's charges polarises the continuum, which acts back on them, and the cavity also costs a surface-dependent energy; the solvation free energy is $\\Delta G_{\\text{solv}} = E_{\\text{water}} - E_{\\text{gas}}$ at the same geometry. It matters because the enzyme's pocket is not a vacuum, and because the charge states of 10.5 live in water.

**Charges and Fukui functions.** The **partial charge** $q_k$ of atom $k$ is the Mulliken-type charge that the self-consistent loop of 10.4 produces. The **Fukui function** $f(\\mathbf r) = (\\partial\\rho/\\partial N)_v$ (Parr and Yang, 1984) says where the electron density changes when the number of electrons changes, and it has two branches because the derivative is different on the two sides of an integer: $f^-$, where an electron is removed from, and $f^+$, where an added electron goes. Condensed to atoms and computed by finite differences, in terms of the charges of the cation, the neutral molecule and the anion at the same geometry,

$$f^-_k \\;=\\; q_k(N-1) - q_k(N), \\qquad f^+_k \\;=\\; q_k(N) - q_k(N+1),$$

each sums to 1 over the atoms (an electron is removed or added in total), with the hydrogens counted with the heavy atom they are bonded to. The atom with the largest $f^-$ is where the molecule is oxidised first, which makes it a candidate site for the enzyme's oxidant.

**Hydrogen abstraction.** The step that Compound I performs is the removal of a hydrogen atom from a carbon, $\\text{R-H} \\to \\text{R}^\\bullet + \\text{H}^\\bullet$. Its energy, $\\Delta E_{\\text{HAT}} = E(\\text{R}^\\bullet) + E(\\text{H}^\\bullet) - E(\\text{R-H})$ with $E(\\text{H}^\\bullet)$ the energy of a free hydrogen atom, is the bond dissociation energy of that C-H bond if the radical is relaxed. Two versions are computed. The **vertical** one keeps the atoms of the radical where they were in the parent: it is cheap and consistent, but it leaves the radical unrelaxed, which overstates the energy by tens of kcal/mol, and by an amount that depends on the kind of radical, because a benzylic radical gains its stability only when the carbon flattens and its electron spreads over the ring. The **relaxed** one lets the radical centre, the atoms bonded to it and their hydrogens move to the nearest minimum of the energy (at most 8 steps of a quasi-Newton minimiser on the analytic gradients, the rest of the molecule frozen; the relaxation gain is a column of its own). In the project only the two best sites of a compound are relaxed, for the cost; the cell relaxes every site of its small molecules.

**Proton affinity.** The nitrogens that can bind the iron (1.9) are the ones whose lone pair is available. The **proton affinity** $\\mathrm{PA} = E(\\text{B}) - E(\\text{BH}^+)$ at a frozen geometry, with a proton added along the lone pair, measures how strongly a nitrogen holds a proton, in the gas phase and in water. It is a ranking quantity, not an absolute one, for the reasons of the paragraph after the cell.

**Conformers.** A flexible molecule is a mixture of shapes, and the properties above differ from shape to shape. If the shapes $i$ have energies $E_i$, the fraction of molecules in shape $i$ at temperature $T$ (with $k$ the Boltzmann constant and $j$ running over all the shapes) is its **Boltzmann weight**

$$w_i \\;=\\; \\frac{e^{-E_i/kT}}{\\sum_j e^{-E_j/kT}}, \\qquad kT = 0.5925\\ \\text{kcal/mol at } 298\\ \\text{K},$$

and an observed property is the average $\\sum_i w_i x_i$. A shape 1 kcal/mol above the lowest has weight 0.18 relative to it, one 3 kcal/mol above has 0.006. Shapes are generated by RDKit (ETKDG embedding, MMFF optimisation) and, because a force field is not reliable for the relative energies of shapes, re-weighted by their GFN2-xTB energies in water.

The first cell computes the molecule-level quantities, the hydrogen-abstraction energies and the atom-level quantities.
"""),
    code(QUANTITIES),
    md("""
**What the numbers show.** For toluene the ionisation potential is {{=molecule_table.loc['toluene', 'ionisation potential (eV)']:.1f}} eV and the electron affinity {{=molecule_table.loc['toluene', 'electron affinity (eV)']:.1f}}, so that $\\eta$ = {{=molecule_table.loc['toluene', 'hardness (eV)']:.1f}} and $\\omega$ = {{=molecule_table.loc['toluene', 'electrophilicity (eV)']:.1f}} eV. For N-methylpiperidine the ionisation potential is lower ({{=molecule_table.loc['N-methylpiperidine', 'ionisation potential (eV)']:.1f}} eV: the lone pair of the amine leaves more easily than an electron of a ring) and the electron affinity is negative ({{=molecule_table.loc['N-methylpiperidine', 'electron affinity (eV)']:.1f}} eV, meaning that the anion is higher in energy than the neutral molecule); it is a hard, poorly electrophilic molecule ($\\eta$ = {{=molecule_table.loc['N-methylpiperidine', 'hardness (eV)']:.1f}}, $\\omega$ = {{=molecule_table.loc['N-methylpiperidine', 'electrophilicity (eV)']:.1f}} eV). The condensed Fukui functions of N-methylpiperidine sum to {{=fukui_sum_minus:.2f}} ($f^-$) and {{=fukui_sum_plus:.2f}} ($f^+$) over its heavy atoms, as they must. The nitrogen carries the largest $f^-$ among the atoms of the table ({{=atom_table['f-'].max():.2f}}), which says that the first electron to leave is the nitrogen's. Its proton affinity is {{=atom_table['proton affinity, gas (kcal/mol)'].dropna().iloc[0]:.0f}} kcal/mol in the gas phase and {{=atom_table['proton affinity, water (kcal/mol)'].dropna().iloc[0]:.0f}} in water, higher in water because the water continuum stabilises the cation that is formed.

The hydrogen-abstraction energies are the instructive part. The best site of toluene is the benzylic C-H, and the energies are {{=toluene_site['vertical']:.1f}} kcal/mol vertical and {{=toluene_site['relaxed']:.1f}} relaxed, against a measured bond dissociation enthalpy of about 90. The best site of N-methylpiperidine, a C-H next to the nitrogen, gives {{=piperidine_site['vertical']:.1f}} and {{=piperidine_site['relaxed']:.1f}}; the secondary C-H of propane {{=propane_site['vertical']:.1f}} and {{=propane_site['relaxed']:.1f}}, against 98.6 measured. Three things follow. First, **the absolute values are not bond energies**: the vertical numbers are {{=toluene_site['vertical'] - 89.7:.0f}} (toluene) and {{=propane_site['vertical'] - 98.6:.0f}} (propane) kcal/mol too high and the relaxed ones {{=toluene_site['relaxed'] - 89.7:.0f}} and {{=propane_site['relaxed'] - 98.6:.0f}}. That is an error of the method and of the bounded relaxation, which this notebook does not separate. Second, **the vertical ordering is wrong and the relaxed ordering is right.** Measured, the benzylic C-H of toluene is the weakest of the three kinds (about 90 against about 99 for the secondary C-H of propane); vertically the order of the three molecules' best sites is {{=' < '.join(vertical_order)}}, which puts the amine C-H first and the benzylic C-H second, while relaxed it is {{=' < '.join(relaxed_order)}}, with the benzylic C-H first and a gap to propane of {{=propane_site['relaxed'] - toluene_site['relaxed']:.1f}} kcal/mol against a measured 8.9. Third, relaxation is what a benzylic radical needs: its gain over the vertical value is {{=toluene_site['vertical'] - toluene_site['relaxed']:.1f}} kcal/mol for toluene against {{=propane_site['vertical'] - propane_site['relaxed']:.1f}} for propane. This is why the project has two tiers (10.6): the vertical energies of tier A for every C-H of every molecule, the relaxed energies of tier B for the two best sites.

**The same holds for the other energies.** The ionisation potentials of GFN2-xTB are {{=ip_offset:.0f}} eV above those of the density functional method (10.4), the proton affinities are not proton affinities (the energy of the free proton is not in the difference, and the geometry is frozen), and the solvation free energies are small ({{=molecule_table.loc['toluene', 'solvation free energy in water (kcal/mol)']:.1f}} kcal/mol for toluene and {{=molecule_table.loc['N-methylpiperidine', 'solvation free energy in water (kcal/mol)']:.1f}} for N-methylpiperidine; the measured hydration free energy of benzene is about −1 kcal/mol and that of methanol about −5, so that the sign and the polar/non-polar ordering are right and the size of the polar values is probably not). What can be used is the ranking of a quantity across sites and across molecules. A tree-based model, which splits on the ordering, can use a quantity with an offset; this is the working assumption of the whole block, and 10.6 tests it.

The second cell takes a molecule of the kind the challenge is about, propranolol, and shows the conformers.
"""),
    code(CONFORMERS),
    md("""
The force field and the quantum-chemical energy disagree about the shapes. The lowest shape by MMFF, whose weight after re-weighting is {{=weight_of_mmff_best:.3f}}, is not the lowest by GFN2/ALPB, and the shape with the largest weight, {{=conformer_table['weight'].max():.2f}}, is a shape that MMFF puts {{=conformer_table.loc[conformer_table['weight'].idxmax(), 'MMFF, relative (kcal/mol)']:.1f}} kcal/mol above its lowest. The Boltzmann-weighted ionisation potential is {{=weighted_ip:.2f}} eV, against {{=lowest_ip:.2f}} for the shape of lowest GFN2/ALPB energy; the {{=n_conformers_9}} shapes span {{=ip_spread:.2f}} eV, little for this molecule, so that the choice of the shape matters less for the energies than for the geometry. The geometry varies more: the distance from the basic nitrogen to the nearest aromatic ring runs from {{=conformer_table['nearest'].min():.1f}} to {{=conformer_table['nearest'].max():.1f}} Å among the shapes, the upper end at the edge of the window of 4.5 to 7.5 Å in which the pharmacophore of CYP2D6 places the cation and the ring ({{=share_in_window:.0%}} of the Boltzmann weight lies inside it). A geometric feature computed from one shape would take a different value from another shape of the same molecule. The geometric features of the block (10.6, level 4) are therefore computed over the ensemble.

*On what actually ran.* The functions `compute_molecule` (`physics_block.py`), `relaxed_hat`, `boltzmann_weights` and `conformer_ensemble` of the project, called in an isolated interpreter by the cells above. The 6,895 compounds were processed by the same functions through the extraction job `070_physics_xtb_block` (tier A) and `073_physics_tierb_extract` (tier B), which write the caches `physics_cache.csv`, `physics_atoms.csv.gz` and `physics_b_cache.csv`; the notebook does not repeat them. The measured bond dissociation enthalpies are from the compilation of Blanksby and Ellison (2003); the hydration free energies quoted in the text, benzene and methanol, are from the FreeSolv database (Mobley and Guthrie, 2014).
"""),
    code(PHYSICS_FACTS),
    md("""
## 10.6 The physics block in the models: four levels, two refinements, and why none passed

> **Field:** Machine learning · Quantum chemistry

*Where this section sits:* 10.1 to 10.5 built an instrument and defined what it measures. This section points it at the models and applies the rule of 6.7, which was written before any of the results below existed. It is a section about an idea with a mechanism behind it that did not pass, and about how far the numbers allow one to conclude that it does not work.

**The idea and the block.** The two quantum features that were tried first (7.8) described the molecule as a whole or looked at one kind of atom; time-dependent inhibition is a statement about specific atoms and the geometry between them. The physics block (`cypmode/challenge/physics_block.py`) therefore measures four levels. *Level 1, the molecule:* the conformer-aware frontier orbital energies, the ionisation potential and electron affinity, the hardness and electrophilicity, the dipole, the solvation free energy in water, and the shape. *Level 2, the atom:* the partial charge, the two condensed Fukui functions, the vertical hydrogen-abstraction energy of every non-aromatic C-H, the vertical proton affinity of every nitrogen that could bind the iron (gas phase and water), and the accessible surface of the atom. *Level 3, the site:* the atoms are grouped by the chemical **role** that a pattern assigns them ({{=n_ch_roles}} kinds of C-H, among them benzylic, next to a nitrogen, next to an oxygen, allylic, methylenedioxy, and {{=n_binder_roles}} kinds of atom that can bind the iron or be oxidised, such as azole, pyridine and amine nitrogens and thioether sulfur), with statistics per role and with how the best oxidation sites compete with each other (the sorted energies, how many lie within 3 kcal/mol of the lowest, the entropy of their Boltzmann weights). *Level 4, the pharmacophore:* over an ensemble of conformers, the distances between the iron-binding atoms and the best oxidation sites, their exposure to solvent, and how many good sites lie in the shell 4 to 8 Å around a binder. Together these are {{=n_tier_a}} columns for each compound, computed by the extraction job `070_physics_xtb_block` in {{=extract_hours_a:.1f}} hours for the {{=n_rows_cache:,}} compounds. {{=n_mostly_missing}} of the columns are missing for more than half of the compounds, {{=n_nearly_empty}} for more than nine in ten, because they describe a role that a compound lacks (the proton affinity of an azole nitrogen in a compound with no azole); a tree handles a missing value, but each such column dilutes the splits.

The block is label-free (it is computed from the structure alone), so that no label can leak through it. **Tier A** is the block as described. **The compact selection** (`cypmode/challenge/physics_compact_experiment.py`, job `072_physics_compact`) keeps the 20 columns with the largest gain, chosen by a physics-only model fitted on the training rows of each task, so that the evaluation rows never influence the choice. **Tier B** (`cypmode/challenge/physics_tierb.py`, jobs `073_physics_tierb_extract` and `074_physics_tierb_experiment`) replaces the vertical hydrogen-abstraction energy of the two best sites by the relaxed one of 10.5 and averages the ionisation potential, solvation and Fukui functions over up to three Boltzmann-weighted conformers: {{=n_tier_b}} columns, {{=extract_hours_b:.1f}} hours of extraction. Two further uses were tried: the block in the pair model of 9.2 (`cypmode/challenge/pair_transfer.py`, job `075_pair_transfer_physics`), and the block in the selection-aware CYP2D6 model of 9.1, in all four of its families (`cypmode/challenge/population_physics.py`, job `071_population_physics`), because whether the screen promoted a compound depends on how reactive and how promiscuous it is, which the block measures.

**The rule.** For each use, the gate of 6.7: the change of the outer-fold AUC when the addition joins the default stack, with the corrected resampled t, and p below 0.05 at both confidence cut-offs. For tier A, the compact selection and tier B the 10-outer-fold replicate of 6.7 was declared before any ten-fold result existed (jobs `081_replicate_grid_f10`, `082_replicate_physics_f10` and `083_replicate_verdict`). The cell reads every log.
"""),
    code(PHYSICS_VERDICTS),
    md("""
{{=n_physics_met}} of the {{=n_physics_tests}} comparisons meet the gate. Tier A, in the stack, changes the AUC of CYP3A4 by {{=tier_a_3a4['AUC change at z0 1.0']:+.4f}} (p = {{=tier_a_3a4['p at z0 1.0']:.3f}}) at $z_0 = 1.0$ and {{=tier_a_3a4['AUC change at z0 1.96']:+.4f}} (p = {{=tier_a_3a4['p at z0 1.96']:.3f}}) at 1.96; the compact selection by {{=compact_3a4['AUC change at z0 1.0']:+.4f}} (p = {{=compact_3a4['p at z0 1.0']:.3f}}) and {{=compact_3a4['AUC change at z0 1.96']:+.4f}} (p = {{=compact_3a4['p at z0 1.96']:.3f}}); tier B by {{=tierb_3a4['AUC change at z0 1.0']:+.4f}} (p = {{=tierb_3a4['p at z0 1.0']:.2f}}) and {{=tierb_3a4['AUC change at z0 1.96']:+.4f}} (p = {{=tierb_3a4['p at z0 1.96']:.2f}}). The block itself carries information. A model on the physics columns alone reaches an AUC of {{=alone_table.iloc[0]['physics columns alone, AUC at z0 1.0']:.3f}} for CYP3A4 and {{=alone_table.iloc[1]['physics columns alone, AUC at z0 1.0']:.3f}} for CYP2D6 at $z_0 = 1.0$. Added to the plain descriptor model (feature set A) it raises the CYP3A4 AUC by {{=alone_table.iloc[0]['added to feature set A']:+.4f}} (p = {{=alone_table.iloc[0]['p (A)']:.2f}}); added to the descriptor-and-fingerprint model it raises it by {{=alone_table.iloc[0]['added to A+fps']:+.4f}} (p = {{=alone_table.iloc[0]['p (A+fps)']:.2f}}). For CYP2D6 the changes are larger and as uncertain ({{=alone_table.iloc[1]['added to feature set A']:+.4f}}, p = {{=alone_table.iloc[1]['p (A)']:.2f}}, and {{=alone_table.iloc[1]['added to A+fps']:+.4f}}, p = {{=alone_table.iloc[1]['p (A+fps)']:.2f}}). A reading, which the experiment does not test: the fingerprints already encode much of what the physics describes, since the substructure that defines a role is what the fingerprint records, and the stack of 23 members spans the rest.

The nearest approach to the gate is the ten-fold replicate of the compact selection on CYP3A4, with {{=compact_ten_3a4['AUC change at z0 1.0']:+.4f}} (p = {{=compact_ten_3a4['p at z0 1.0']:.3f}}) at $z_0 = 1.0$ and {{=compact_ten_3a4['AUC change at z0 1.96']:+.4f}} (p = {{=compact_ten_3a4['p at z0 1.96']:.3f}}) at 1.96: significant at one cut-off and not quite at the other, and the gate asks for both. It is not met, and the rule was not softened afterwards. The physics inside the selection-aware model and in the pair model do not come close.

**Why none passed.** The changes are small and nearly all positive: {{=n_positive_physics}} of the {{=n_physics_tests}} changes at $z_0 = 1.0$ are above zero, the largest being {{=largest_physics:+.4f}}. The main design recognises a difference 8 times in 10 only above {{=floor_main:.4f}} AUC, and the replicate above {{=floor_replicate:.4f}} (6.7), so that effects of the size seen here are below what either design can establish. A verdict "not met" therefore means that the block was not shown to help; it does not mean that it was shown not to help. A true gain of 0.004 AUC is compatible with every number of the table, and so is no gain. What the numbers do establish is a bound, with the usual qualification: whatever the block adds to the stack on these compounds is probably less than about a hundredth of an AUC, since a gain of that size would be recognised in 8 of 10 repetitions of the experiment. The rule of the project for such a case was fixed before the results: an idea that fails its gate becomes a measured negative result in the ledger and is not tuned until it passes. After tier B no further physics variant was built. The block stays in the repository, with its caches and its code; it is not in the final stack.

*On what actually ran.* `cypmode/challenge/physics_block.py` (levels 1 to 4, `extract`, `load_physics_features`), `physics_tierb.py`, `physics_experiment.py`, `physics_compact_experiment.py`, `physics_tierb_experiment.py`, `population_physics.py`, `pair_transfer.py` (the physics variant), each run through `addition_experiment.py`; the logs `physics_experiment_<enzyme>.log`, `physics_compact_<enzyme>.log`, `physics_tierb_<enzyme>.log`, `pair_transfer_physics_<enzyme>.log`, `population_physics_CYP2D6.log` and their `_f10` replicates; the queue jobs named in the text. The notebook reads the logs and the cache of columns; it does not repeat the extraction. The two earlier quantum features (global GFN2 descriptors in `cypmode/challenge/quantum_pharmacophore.py` and `quantum_pharmacophore_significance.py`, and the surrogate of 7.8 in `qm_reactivity_features.py`) were put through the same kind of test and were null.
"""),
]

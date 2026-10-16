"""Part V (third part): the iron-nitrogen distance in predicted structures (5.5) and the pocket geometry (5.6)."""

from notebooks.notebook_builder import code, figure, md

STRUCTURE_FILE = '''
# What a predicted structure file contains, read from the real file of the ketoconazole prediction (Boltz-2, cypmode/validation): the names of the columns of the atom table, three of its lines, and the
# parts of the complex as the Biopython reader sees them.
import re

import numpy as np
import pandas as pd
from rdkit import Chem

from cypmode.metrics.motifs import TYPE_II_MOTIFS
from cypmode.validation.openfold3 import read_boltz_inputs
from cypmode.validation.structures import _ligand_atoms, find_heme_iron, load_structure

OUT_DIR = ROOT / "data/structure_track/outputs/out/cyp3a4"
REFERENCE_NAMES = ["ketoconazole", "ritonavir", "azamulin", "vardenafil"]

def cif_of(name):
    return OUT_DIR / f"boltz_results_cyp3a4_{name}" / "predictions" / f"cyp3a4_{name}" / f"cyp3a4_{name}_model_0.cif"

lines = cif_of("ketoconazole").read_text().splitlines()
columns = [l.removeprefix("_atom_site.") for l in lines if l.startswith("_atom_site.")]
first_atom = next(l for l in lines if l.startswith("ATOM"))
iron_line = next(l for l in lines if l.startswith("HETATM") and " FE " in l)
ligand_line = next(l for l in lines if l.startswith("HETATM") and " LIG1 " in l)
print("columns of the atom table:", ", ".join(columns))
for label, line in (("a protein atom", first_atom), ("the heme iron", iron_line), ("a ligand atom", ligand_line)):
    print(f"{label:15s}", line)

structure = load_structure(cif_of("ketoconazole"))
chain_rows = [{"chain": chain.id, "residues": len(list(chain.get_residues())), "atoms": len(list(chain.get_atoms())), "what it is": {"A": "the protein, CYP3A4", "B": "the predicted molecule (residue LIG1)", "C": "the heme (residue HEM)"}.get(chain.id, "")} for chain in structure[0]]
show_table(pd.DataFrame(chain_rows))
iron = find_heme_iron(structure)
n_atoms_total = len(list(structure.get_atoms()))
n_protein_residues = chain_rows[0]["residues"]
ligand_heavy_atoms = len(_ligand_atoms(structure))
iron_xyz = [float(v) for v in iron.coord]
'''

DISTANCES = '''
# The distance by hand. For every nitrogen of the ligand: the Euclidean distance to the iron from the coordinates (numpy), then which of the nitrogens belong to a motif of section 5.1 (the project's
# function matches the patterns on the molecule rebuilt from the predicted coordinates). Vardenafil is the case that taught the project to look at the matched nitrogen and not at the nearest one.
from cypmode.validation.structures import all_ligand_nitrogen_distances, motif_nitrogen_distances
from cypmode.validation.constants import COORDINATION_DISTANCE_CUTOFF_A

def nitrogen_table(name):
    structure = load_structure(cif_of(name))
    iron = find_heme_iron(structure)
    smiles = read_boltz_inputs(name)["ligand_smiles"]
    by_hand = {atom.get_name(): float(np.sqrt(((atom.coord - iron.coord) ** 2).sum())) for atom in _ligand_atoms(structure) if atom.element == "N"}
    library = {r["atom"]: r["distance_angstrom"] for r in all_ligand_nitrogen_distances(structure, iron)}
    motif = {r["atom"]: r["motif"] for r in motif_nitrogen_distances(structure, smiles, iron)}
    rows = [{"nitrogen": atom, "distance to the iron (A)": distance, "matches a motif": motif.get(atom, "no"), "library value": library[atom]} for atom, distance in sorted(by_hand.items(), key=lambda kv: kv[1])]
    return pd.DataFrame(rows), by_hand, motif

keto_table, keto_by_hand, keto_motif = nitrogen_table("ketoconazole")
show_table(keto_table, floatfmt=".2f")
varde_table, varde_by_hand, varde_motif = nitrogen_table("vardenafil")
show_table(varde_table, floatfmt=".2f")
nearest_varde = min(varde_by_hand.items(), key=lambda kv: kv[1])
nearest_motif_varde = min((kv for kv in varde_by_hand.items() if kv[0] in varde_motif), key=lambda kv: kv[1])
nearest_motif_keto = min((kv for kv in keto_by_hand.items() if kv[0] in keto_motif), key=lambda kv: kv[1])
by_hand_matches_library = all(abs(r["distance to the iron (A)"] - r["library value"]) < 0.006 for t in (keto_table, varde_table) for _, r in t.iterrows())
print(f"the by-hand distances agree with the library's (to the rounding of 0.01 A): {by_hand_matches_library}; cutoff {COORDINATION_DISTANCE_CUTOFF_A} A")
'''

BONDS = '''
# Why the molecule has to be rebuilt. The structure file gives the elements and the coordinates of the ligand atoms but no bonds. The project first joins atoms that sit close enough to be bonded (proximity
# bonding), then asks RDKit to lay the molecule's own SMILES over that graph and to assign bond orders and aromaticity (AssignBondOrdersFromTemplate); the atom numbers of the SMILES then carry the coordinates.
from rdkit.Chem import AllChem

structure_k = load_structure(cif_of("ketoconazole"))
ligand_k = _ligand_atoms(structure_k)
block = "\\n".join(f"HETATM{i:>5} {a.get_name():<4} LIG A   1    {a.coord[0]:8.3f}{a.coord[1]:8.3f}{a.coord[2]:8.3f}  1.00  0.00          {a.element:>2}" for i, a in enumerate(ligand_k, start=1)) + "\\nEND"
raw = Chem.MolFromPDBBlock(block, sanitize=False, proximityBonding=True)
template = Chem.MolFromSmiles(read_boltz_inputs("ketoconazole")["ligand_smiles"])
rebuilt = AllChem.AssignBondOrdersFromTemplate(template, raw)
bond_orders = pd.Series([str(b.GetBondType()) for b in rebuilt.GetBonds()]).value_counts()
n_proximity_bonds, n_template_bonds = raw.GetNumBonds(), template.GetNumBonds()
n_aromatic_atoms = sum(a.GetIsAromatic() for a in rebuilt.GetAtoms())
print(f"ligand atoms {len(ligand_k)}; bonds found by proximity {n_proximity_bonds}; bonds of the SMILES {n_template_bonds}; after the template: {rebuilt.GetNumBonds()} bonds, {n_aromatic_atoms} aromatic atoms")
print("bond types after the template:", bond_orders.to_dict())

# The motif nitrogen is now an atom of a molecule, so the SMARTS match of section 5.1 can be applied to it and each matched nitrogen has a position.
conformer = rebuilt.GetConformer()
matched_positions = {}
for name_, smarts in TYPE_II_MOTIFS.items():
    for match in rebuilt.GetSubstructMatches(Chem.MolFromSmarts(smarts)):
        for index in match:
            if rebuilt.GetAtomWithIdx(index).GetSymbol() == "N":
                matched_positions[index] = name_
def has_free_lone_pair(atom):
    return atom.GetIsAromatic() and atom.GetDegree() == 2 and atom.GetTotalNumHs() == 0

iron_k = find_heme_iron(structure_k).coord
matched_rows = []
for index in sorted(matched_positions):
    position = conformer.GetAtomPosition(index)
    matched_rows.append({"nitrogen": ligand_k[index].get_name(), "pattern": matched_positions[index], "free lone pair": has_free_lone_pair(rebuilt.GetAtomWithIdx(index)),
                         "distance to the iron (A)": float(np.linalg.norm(np.array([position.x, position.y, position.z]) - iron_k))})
show_table(pd.DataFrame(matched_rows), floatfmt=".2f")
free_row = next(r for r in matched_rows if r["free lone pair"])
other_row = next(r for r in matched_rows if not r["free lone pair"])
'''

SUMMARY = '''
# The four reference compounds, from the project's own function (cypmode/validation/structures.py: summarize_compound), checked against the file that the project wrote when it first ran them; the figure is the
# distance of the closest motif nitrogen with the cutoff.
import json

from cypmode.validation.structures import summarize_compound

stored_summary = json.loads((ROOT / "data/structure_track/outputs/cyp3a4/validation_summary.json").read_text())
summary_rows = []
recomputed_matches_stored = True
for name in REFERENCE_NAMES:
    smiles = read_boltz_inputs(name)["ligand_smiles"]
    result = summarize_compound(OUT_DIR / f"boltz_results_cyp3a4_{name}", smiles)
    recomputed_matches_stored &= result["closest_motif_nitrogen_distance_angstrom"] == stored_summary[name]["closest_motif_nitrogen_distance_angstrom"]
    summary_rows.append({"compound": name, "motif found": ", ".join(sorted({r["motif"] for r in result["motif_nitrogen_distances"]})), "closest motif nitrogen (A)": result["closest_motif_nitrogen_distance_angstrom"],
                         "coordinated (<= 2.6 A)": "yes" if result["coordinated"] else "no", "Boltz-2 confidence": result["confidence_score"], "Boltz-2 binding probability": result["affinity_probability_binary"]})
show_table(pd.DataFrame(summary_rows), floatfmt=".2f")
fig, axis = plt.subplots(figsize=(6.6, 3.5))
distances = [r["closest motif nitrogen (A)"] for r in summary_rows]
colours = [theme["green"] if r["coordinated (<= 2.6 A)"] == "yes" else theme["red"] for r in summary_rows]
axis.bar([r["compound"] for r in summary_rows], distances, color=colours)
axis.axhline(COORDINATION_DISTANCE_CUTOFF_A, color=theme["muted"], ls="--", lw=1.2)
axis.text(3.45, COORDINATION_DISTANCE_CUTOFF_A + 0.3, f"cutoff {COORDINATION_DISTANCE_CUTOFF_A} A", ha="right", color=theme["muted"], fontsize=9)
axis.set(ylabel="iron to closest motif nitrogen (A)")
tidy(axis, theme)
fig.tight_layout()
rows_by_name = {r["compound"]: r for r in summary_rows}
n_coordinated = sum(r["coordinated (<= 2.6 A)"] == "yes" for r in summary_rows)
'''

SASA = '''
# The solvent-accessible surface by the Shrake-Rupley method, written out. Each atom is a sphere (van der Waals radius, Bondi 1964) grown by the radius of a water molecule, 1.4 A; n points are spread evenly over
# the grown sphere (the golden-spiral formula); a point counts as accessible when it lies outside the grown sphere of every other atom; the area of the atom is that of its grown sphere times the share of its
# points that are accessible. First two checks against geometry (one sphere, and two spheres whose overlap can be computed exactly), then the four reference complexes against Biopython's implementation.
import logging

from Bio.PDB.SASA import ATOMIC_RADII
from scipy.spatial import cKDTree

logging.getLogger().setLevel(logging.ERROR)                      # Biopython warns once per atom of a residue it has no radii for; the radii of the elements are used all the same
PROBE = 1.4

def golden_spiral(n):
    """n points spread evenly over the unit sphere: z runs from near 1 to near -1 in equal steps and the longitude advances by the golden angle."""
    k = np.arange(n)
    z = 1 - (2 * k + 1) / n
    longitude = k * np.pi * (3 - np.sqrt(5))
    r = np.sqrt(1 - z**2)
    return np.column_stack([r * np.cos(longitude), r * np.sin(longitude), z])

def atom_areas(coords, radii, n=100):
    """The accessible area of every atom (A^2): coords an (m, 3) array, radii the van der Waals radii."""
    grown = radii + PROBE
    points = golden_spiral(n)
    tree = cKDTree(coords)
    areas = np.zeros(len(coords))
    for i in range(len(coords)):
        test = coords[i] + points * grown[i]
        free = np.ones(n, dtype=bool)
        for j in tree.query_ball_point(coords[i], grown[i] + grown.max()):
            if j != i:
                free &= ((test - coords[j]) ** 2).sum(axis=1) >= grown[j] ** 2
        areas[i] = 4 * np.pi * grown[i] ** 2 * free.mean()
    return areas

# check 1: one carbon atom has the area of its grown sphere; check 2: two carbons 1.5 A apart: each loses a spherical cap of height h = R - D/2, whose area is 2 pi R h
R_carbon = ATOMIC_RADII["C"] + PROBE
single = atom_areas(np.zeros((1, 3)), np.array([ATOMIC_RADII["C"]]), n=2000)[0]
pair_distance = 1.5
pair = atom_areas(np.array([[0.0, 0, 0], [pair_distance, 0, 0]]), np.array([ATOMIC_RADII["C"]] * 2), n=20000)
exact_single = 4 * np.pi * R_carbon**2
exact_pair = 4 * np.pi * R_carbon**2 - 2 * np.pi * R_carbon * (R_carbon - pair_distance / 2)
print(f"one atom: {single:.2f} A^2 against 4 pi R^2 = {exact_single:.2f}; two atoms 1.5 A apart: {pair[0]:.2f} each against the exact {exact_pair:.2f}")

def element_radii(atoms):
    return np.array([ATOMIC_RADII[a.element.upper()] for a in atoms])

def burial(name):
    structure = load_structure(cif_of(name))
    ligand = _ligand_atoms(structure)
    everything = list(structure.get_atoms())
    positions = {id(a): i for i, a in enumerate(everything)}
    coords, radii = np.array([a.coord for a in everything]), element_radii(everything)
    bound = atom_areas(coords, radii)[[positions[id(a)] for a in ligand]].sum()
    free = atom_areas(np.array([a.coord for a in ligand]), element_radii(ligand)).sum()
    return structure, bound, free

from cypmode.validation.pocket_geometry import ligand_burial_sasa

burial_rows, library_difference = [], 0.0
for name in REFERENCE_NAMES:
    structure, bound, free = burial(name)
    library = ligand_burial_sasa(structure)
    library_difference = max(library_difference, abs(bound - library["bound_sasa_A2"]), abs(free - library["unbound_sasa_A2"]))
    burial_rows.append({"compound": name, "ligand atoms": len(_ligand_atoms(structure)), "area with the protein and heme around it (A^2)": bound, "area of the molecule alone (A^2)": free, "buried share": 1 - bound / free})
show_table(pd.DataFrame(burial_rows), floatfmt=".2f")
burial_table = pd.DataFrame(burial_rows).set_index("compound")

# the settings of the cavity program that section 5.6 describes, read from the program itself
import inspect

import pyKVFinder

defaults = {k: v.default for k, v in inspect.signature(pyKVFinder.run_workflow).parameters.items() if k in ("step", "probe_in", "probe_out", "removal_distance", "volume_cutoff", "ligand_cutoff")}
'''

POCKET_TOY = '''
# The idea of cavity detection on a toy in two dimensions (the real program works on a three-dimensional grid in the same way). Atoms are discs; a grid of points covers the picture. A small probe (radius 1) can
# stand at a grid point when no atom is closer than its radius plus 1; a large probe (radius 3) does the same with a wider margin. The large probe is reached from the outside by flooding from the border
# through the points where it fits. A cavity is a place the small probe can be and the large probe, coming from the outside, cannot reach: a space that is enclosed or has too narrow an opening.
from scipy import ndimage

step = 0.25
xs = np.arange(0, 28 + step, step)
grid_x, grid_y = np.meshgrid(xs, xs)
protein_atoms = [(14 + 6.5 * np.cos(t), 14 + 6.5 * np.sin(t)) for t in np.linspace(0.45, 2 * np.pi - 0.45, 20)]       # a C-shaped wall of atoms with an opening on the right
ligand_atoms_2d = [(14.5, 14.0), (15.8, 14.0)]
atoms_2d = protein_atoms + ligand_atoms_2d
atom_radius = 1.5
distance_to_atom_surface = np.min([np.hypot(grid_x - x, grid_y - y) - atom_radius for x, y in atoms_2d], axis=0)

small_fits = distance_to_atom_surface >= 1.0                       # the centre of the small probe (radius 1) can stand here
large_fits = distance_to_atom_surface >= 3.0                       # the centre of the large probe (radius 3) can stand here
labels, _ = ndimage.label(large_fits)
touching_border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))) - {0}
reached = np.isin(labels, list(touching_border))                   # where the large probe's centre can get to, coming from the outside
covered = ndimage.distance_transform_edt(~reached) * step <= 3.0   # every place that some reachable large probe covers with its body
cavity = small_fits & ~covered
cavity_cells = int(cavity.sum())
toy_area = cavity_cells * step**2
fig, axis = plt.subplots(figsize=(5.2, 5.0))
axis.imshow(np.where(cavity, 1.0, np.where(small_fits, 0.35, 0.0)), origin="lower", extent=(0, 28, 0, 28), cmap="viridis", vmin=0, vmax=1, alpha=0.9)
for x, y in protein_atoms:
    axis.add_patch(plt.Circle((x, y), atom_radius, fill=False, color=theme["blue"], lw=1.4))
for x, y in ligand_atoms_2d:
    axis.add_patch(plt.Circle((x, y), atom_radius, fill=False, color=theme["orange"], lw=2.0))
axis.set(xlim=(0, 28), ylim=(0, 28), xticks=[], yticks=[], title="circles: wall (blue) and ligand (orange); yellow: cavity; teal: open solvent")
fig.tight_layout()
print(f"grid step {step} units; cells where the small probe fits: {int(small_fits.sum())}; cells of the cavity: {cavity_cells} (area {toy_area:.1f} square units)")
'''

POCKET_REAL = '''
# The real thing, with the project's function (cypmode/validation/pocket_geometry.py: pocket_volume, which calls pyKVFinder). The program is run on the complex with the cavity search restricted to the region
# within 5 A of the ligand; the largest cavity found there is taken as "the pocket". Its volume is the empty space left around the ligand.
from cypmode.validation.pocket_geometry import pocket_volume
pocket_rows = []
for name in REFERENCE_NAMES:
    result = pocket_volume(load_structure(cif_of(name)))
    pocket_rows.append({"compound": name, "empty cavity next to the ligand (A^3)": result["pocket_volume_A3"], "its surface (A^2)": result["pocket_area_A2"], "cavities found within 5 A": result["n_cavities_detected"],
                        "buried share of the ligand": float(burial_table.loc[name, "buried share"])})
show_table(pd.DataFrame(pocket_rows), floatfmt=".2f")
pocket_table = pd.DataFrame(pocket_rows).set_index("compound")
'''

CELLS = [
    md("""
## 5.5 Checking the screen against predicted structures: the distance from the iron to the nitrogen

> **Field:** Structural biology · Computational chemistry

*Where this section sits:* so far the rule has been applied to lists of molecules, and a list cannot say whether a molecule will *use* its nitrogen. That depends on where the molecule sits in the pocket, and for that a three-dimensional model of the molecule inside the enzyme is needed. A co-folding model such as Boltz-2 (section 0.2; Part XI explains how it works) produces one from the sequence of the protein and the formula of
the molecule. This section reads such a prediction, measures one distance in it, and compares the result with what the chemistry of section 1.9 and the screen of 5.1 lead one to expect. Part XI makes the same measurement on every pose it handles, and later uses it as a restraint.

**What a structure file contains.** A structure of a protein with a molecule bound to it is, in the end, a table of atoms. Each row names an atom and says where it is: three numbers $x$, $y$ and $z$, its coordinates in ångström, measured from an arbitrary origin; and to which residue (an amino acid of the protein, or a separate molecule such as the heme or the bound compound) and to which **chain** (one continuous piece of the complex) it belongs. The
format that Boltz-2 writes, **mmCIF**, is a text file in which a header names the columns of that table and every following line is one atom. The first cell prints the names of the columns of the real file of the ketoconazole prediction, three of its lines, and the parts of the complex as the Biopython reader sees them (a *structure* holds *models*, a model holds chains, a chain holds residues and a residue holds atoms). Two of the columns need a word. The
occupancy is the share of the copies of the molecule in which the atom is found at this place; in an experiment it can be below 1, and in a prediction, which is a single model, it is always 1. The B-factor measures, in an experiment, how much the atom is smeared by motion and disorder; in a Boltz-2 prediction the same column carries the model's own confidence in the position of the atom, the **pLDDT**, a number from 0 to 100 (section 11.4).
"""),
    code(STRUCTURE_FILE),
    md("""
The file holds {{=n_atoms_total:,}} atoms in three chains: the protein, with {{=n_protein_residues}} residues, the predicted molecule, with {{=ligand_heavy_atoms}} atoms (heavy atoms only: hydrogens are not predicted), and the heme, in which the iron is the atom named `FE`, at $x, y, z$ = ({{=iron_xyz[0]:.2f}}, {{=iron_xyz[1]:.2f}}, {{=iron_xyz[2]:.2f}}) ångström. Nothing in the input places the molecule near the iron: the input is the protein sequence, the heme bound to the cysteine and the formula of the
molecule, so where the molecule ends up is a prediction and not something built into the question.

**The distance and the rule.** For two atoms $P$ and $Q$ with coordinates $(x_P, y_P, z_P)$ and $(x_Q, y_Q, z_Q)$, the distance is the length of the straight line between them (the **Euclidean distance**, from Pythagoras's theorem applied in three dimensions):

$$d(P, Q) \\;=\\; \\sqrt{(x_P - x_Q)^2 + (y_P - y_Q)^2 + (z_P - z_Q)^2}.$$

In solved structures of Type II complexes the nitrogen is 2.0 to 2.3 ångström from the iron (Poulos and Johnson, 2005; section 1.9). The project calls a compound **coordinated** in a prediction when the closest nitrogen that matches a motif pattern of section 5.1 is within $r = 2.6$ ångström of the iron, a little above the solved range to allow for the imprecision of a prediction. With $d_{\\min}$ the smallest of the distances from the iron to the matched nitrogens, the rule is $c = \\mathbb{1}[d_{\\min} \\le r]$, the indicator of 5.1.
That is the whole decision; everything difficult in this section is in the words *the matched nitrogens*.

**Which nitrogen?** A molecule has several nitrogens, and the closest to the iron is not necessarily the one the screen flagged. The next cell computes, from the coordinates and with numpy, the distance from the iron to every nitrogen of two molecules, and marks the nitrogens that match a pattern. The first is ketoconazole, whose imidazole is the textbook Type II group. The second is vardenafil, the case that taught the project to measure the matched nitrogen and not the nearest one: an earlier version of the code
reported the nearest nitrogen of any kind, and for vardenafil that is a different nitrogen from the one the screen had flagged.
"""),
    code(DISTANCES),
    md("""
For ketoconazole the closest nitrogen to the iron, {{=nearest_motif_keto[0]}} at {{=nearest_motif_keto[1]:.2f}} ångström, is also a matched one, and the other nitrogens of the molecule (those of the piperazine) are more than 14 ångström away. For vardenafil the closest nitrogen of any kind is {{=nearest_varde[0]}}, at {{=nearest_varde[1]:.2f}} ångström, and it is not part of any matched pattern; the closest *matched* one is {{=nearest_motif_varde[0]}}, at {{=nearest_motif_varde[1]:.2f}}. (The project's summary says 10.54; the difference of 0.01 is the rounding of the coordinates to three decimals when the molecule is
rebuilt, below.) Measuring the nearest nitrogen would have said "4.84 ångström, nearly coordinated" for a compound whose flagged nitrogens are more than ten ångström from the iron, which is why the function of the project measures the matched nitrogens. The agreement of the by-hand distances with the project's function is {{=by_hand_matches_library}}.

**Matching a pattern to coordinates.** A pattern of 5.1 describes atoms and bonds, and the structure file gives atoms and coordinates but no bonds, so a bridge is needed. The project joins atoms of the ligand that sit close enough to be bonded (proximity bonding), then asks RDKit to lay the SMILES of the molecule over that graph and to assign bond orders and aromaticity to it (`AssignBondOrdersFromTemplate`, the technique that Walters's `cyp-heatmap` also uses); the atoms of the rebuilt
molecule then carry the coordinates, and a pattern match gives atoms whose positions are known. Distance-based bonding can go wrong: when the predicted conformation of a molecule puts two atoms that are not bonded closer together than two that are, the heuristic finds a bond that the SMILES does not have, and the assignment fails. The project met this on one compound of a later batch (Part XI), and `cypmode/validation/structures.py` has a second route for it: if the atoms of the file are in the order of the SMILES and every bond of the SMILES has a plausible length,
the coordinates are assigned to the atoms of the SMILES directly, by position in the list. The next cell checks the main route on ketoconazole and then asks which of the nitrogens that the screen matches is the one near the iron.
"""),
    code(BONDS),
    md("""
Proximity bonding finds {{=n_proximity_bonds}} bonds in the ligand of ketoconazole, the SMILES has {{=n_template_bonds}}, and the assignment succeeds, giving single, double and aromatic bonds ({{=n_aromatic_atoms}} aromatic atoms). The screen matches two nitrogens, the two nitrogens of the imidazole (5.1: the pattern does not distinguish them), and the table shows that they behave as the chemistry says: the one with a free lone pair, {{=free_row['nitrogen']}}, is
{{=free_row['distance to the iron (A)']:.2f}} ångström from the iron, and the one that carries a substituent and has used its lone pair in the ring, {{=other_row['nitrogen']}}, is {{=other_row['distance to the iron (A)']:.2f}}. Of the nitrogens the pattern cannot tell apart, the prediction chooses the right one.

**The four reference compounds.** The project's original structural check covered four drugs for which cryo-EM structures of CYP3A4 had been solved (Walters and co-workers, 2026; preprint \"Structural Plasticity and Ligand Promiscuity of CYP3A4 Revealed by Cryo-EM\", bioRxiv): ketoconazole, ritonavir, azamulin and vardenafil, each predicted once by Boltz-2 with the heme explicitly present and bonded to its cysteine. The next cell reads the four predictions with the function `summarize_compound`, checks the result against the summary file the project wrote when it first
ran them, and draws the distance of the closest matched nitrogen, with the cutoff. The confidence is the model's own score for the whole structure, from 0 to 1, and the binding probability is the output of its affinity module (section 11.2); neither is used for the decision.
"""),
    code(SUMMARY),
    md("""
The recomputed distances equal the stored summary: {{=recomputed_matches_stored}}. Ketoconazole ({{=rows_by_name['ketoconazole']['closest motif nitrogen (A)']:.2f}} ångström) and ritonavir ({{=rows_by_name['ritonavir']['closest motif nitrogen (A)']:.2f}}) are coordinated, inside the range of solved structures and with nothing in the input to force it; azamulin ({{=rows_by_name['azamulin']['closest motif nitrogen (A)']:.2f}}) and vardenafil ({{=rows_by_name['vardenafil']['closest motif nitrogen (A)']:.2f}}) are not. {{=n_coordinated}} of the four flagged compounds
are coordinated, and the other two are informative in different ways and not simple failures of the rule.

*Azamulin* matches the triazole pattern and is not near the iron. It is a known mechanism-based inactivator of CYP3A4: the enzyme turns it into a reactive species that modifies the enzyme (Lim and co-workers, 2005, doi 10.1124/dmd.104.003475; Sevrioukova, 2019, doi 10.3390/ijms20174245), a process that depends on the turnover of the enzyme and that a single static prediction cannot represent. The screen's call, Type II, is wrong about the mechanism in a specific, documented way, for a compound that inhibits for real. *Vardenafil*
matches the imidazole pattern inside the fused ring system of its core, which does not behave like a free imidazole; the structure confirms that the match was a coincidence of connectivity of the kind that 5.1 warned about. *Ritonavir* is coordinated, as the thiazole pattern was added to the screen to say, but the binding probability of Boltz-2 for it is lower than for ketoconazole although the two coordinate the iron at similar distances and confidence: the geometric signal and the learned
affinity signal do not fully agree, and the project has no explanation for that beyond noting it.

A caution about what this check can and cannot show. It is a check of four molecules, each predicted once; it is not a measurement of how often the screen is right. The announcement of the cryo-EM structures states that current co-folding models handle CYP3A4 badly because of the plasticity of its pocket, which the structures they were trained on do not show (Walters and co-workers, 2026), so a large distance can be a failure of the prediction as well as a fact about the molecule; for azamulin the chemistry explains the result, but the project cannot rule out the second reason from one static prediction. An independent
analysis supports the binary calls: Walters's `cyp-heatmap` (https://github.com/PatWalters/cyp-heatmap) fingerprints the interactions of 122 CYP3A4 structures with ProLIF, a different method on different structures, and its metal-coordination contact with the heme iron agrees with the calls above for the same four compounds. That is agreement on a yes-or-no call between two unrelated methods, and no more: a contact fingerprint is not a distance.

*On what actually ran.* The four predictions were produced once, on the local machine, by `boltz predict` with the heme and its covalent bond to the axial cysteine written into the input by `cypmode/validation/build_input.py` (about 40 minutes per compound); `cypmode/validation/structures.py` reads them and writes the summary (`build_summary`, `data/structure_track/outputs/cyp3a4/validation_summary.json`). The cells above re-read the stored structures and do not run Boltz-2 again. The batch runner `cypmode/validation/batch.py` and the one-call
wrapper `cypmode/validation/pipeline.py` run the same measurement for any new SMILES, which is how the structural part of the project grew into Part XI.
"""),
    md("""
## 5.6 Pocket geometry from a predicted structure: how buried the molecule is and how much room is left

> **Field:** Structural biology · Computational chemistry

*Where this section sits:* section 5.5 measured one distance in a predicted structure. Two other properties of a pose can be computed from the same file with no further prediction: how much of the molecule is covered by the protein, its *burial*, and how much empty space is left in the pocket around it. Both are cheap, both are independent of the iron-nitrogen distance, and both are the
structural evidence that section 7.10 offers to the models. This section builds them from first principles and measures them on the four reference complexes.

**The surface of a molecule.** A molecule has no surface in the sense that a table has: its atoms are clouds of electrons. Two conventions make the idea precise. Each atom is a sphere whose radius, the **van der Waals radius**, is the distance at which two non-bonded atoms of that element begin to repel each other (Bondi, 1964: 1.70 ångström for carbon, 1.55 for nitrogen, 1.52 for oxygen, 1.75 for chlorine).
And the surface that matters for a molecule in water is the one that a water molecule can reach: the **solvent-accessible surface**, traced by the centre of a probe sphere the size of water, with radius $r_w = 1.4$ ångström, as it rolls over the atoms. Its area, the **SASA**, is the part of the molecule that solvent can touch. Shrake and Rupley (1973) gave the simple algorithm that is still used. Let $r_i$ be the radius of atom $i$ and
$R_i = r_i + r_w$ the radius of the sphere on which the centre of the probe moves when it touches the atom. Spread $n$ test points evenly over that sphere. For a point to be evenly spread, the golden-spiral construction puts the $k$-th point, for $k = 0, \\dots, n-1$, at height $z_k$ and longitude $\\varphi_k$,

$$z_k \\;=\\; 1 - \\frac{2k+1}{n}, \\qquad \\varphi_k \\;=\\; k\\,\\pi\\,(3-\\sqrt{5}),$$

on the unit sphere (the point is $(\\sqrt{1-z_k^2}\\cos\\varphi_k,\\ \\sqrt{1-z_k^2}\\sin\\varphi_k,\\ z_k)$): the heights are equally spaced, and the longitude advances by the golden angle, about 137.5 degrees, so that no two points ever line up. The points are then scaled by $R_i$ and moved to the centre of the atom. A test point is *accessible* if it lies outside the sphere of radius $R_j$ of every other atom $j$,
that is, if the probe could stand there without overlapping another atom. If $a_i$ of the $n$ points of atom $i$ are accessible, its accessible area is the area of its sphere times that share,

$$A_i \\;=\\; 4\\pi R_i^2\\,\\frac{a_i}{n},$$

and the SASA of a group of atoms is the sum of their $A_i$. The cell writes this out and checks it against geometry before using it. A single atom has the area of its sphere, $4\\pi R^2$. Two equal spheres whose centres are a distance $D$ apart each hide a spherical cap of the other, of height $h = R - D/2$, and the area of a cap is $2\\pi R h$ (Archimedes), so each of the two atoms has the exact area $4\\pi R^2 - 2\\pi R h$.
Then the burial of the ligand in each of the four complexes is computed. The **buried share** of the ligand is

$$B \\;=\\; 1 - \\frac{A_{\\text{bound}}}{A_{\\text{free}}},$$

where $A_{\\text{bound}}$ is the SASA of the ligand's atoms in the complex, with the protein and the heme around them as neighbours, and $A_{\\text{free}}$ that of the same atoms alone, in the same conformation, as if the molecule were not bound: $B=0$ for a molecule that solvent reaches everywhere and $B$ near 1 for one that is almost entirely covered. The result is compared with Biopython's implementation of the same method, which the project uses.
"""),
    code(SASA),
    md("""
Both geometric checks pass: one atom gives {{=single:.2f}} against the exact {{=exact_single:.2f}} square ångström, and the two overlapping atoms {{=pair[0]:.2f}} each against {{=exact_pair:.2f}}. The by-hand areas of the four complexes agree with Biopython's: the largest difference is {{=library_difference:.1e}} square ångström. All four molecules are deeply buried: {{=burial_table['buried share'].min():.0%}} to {{=burial_table['buried share'].max():.0%}} of the surface of each is covered by the protein and
the heme, ketoconazole the most, with {{=burial_table.loc['ketoconazole', 'area with the protein and heme around it (A^2)']:.0f}} square ångström of its {{=burial_table.loc['ketoconazole', 'area of the molecule alone (A^2)']:.0f}} still open to solvent. The numbers are close for the four, and that is a finding: azamulin, which sits {{=rows_by_name['azamulin']['closest motif nitrogen (A)']:.0f}} ångström from the iron, is almost as deeply buried ({{=burial_table.loc['azamulin', 'buried share']:.0%}}) as ketoconazole ({{=burial_table.loc['ketoconazole', 'buried share']:.0%}}), which is bonded to it. A pose can be deep in the pocket and still be far from the iron, so burial and the iron-nitrogen distance
measure different things.

**Empty space.** The second measurement asks how much room is left. A cavity is a region of empty space in a protein that solvent can reach only through a narrow opening or not at all, and detecting one is a problem of geometry on a grid. The program the project uses, pyKVFinder (a Python version of KVFinder; Oliveira and co-workers, 2014; Guerra and co-workers, 2021), covers the protein with a grid of points a fixed step apart ({{=defaults['step']}} ångström by default) and uses two probes: a small one ({{=defaults['probe_in']}} ångström, water) and a
large one ({{=defaults['probe_out']}} ångström). The grid points at which the small probe can stand without touching an atom are the space that solvent could occupy. The large probe is rolled in from the outside: the points it can reach, and everything its body covers on the way, are the *outside*. A **cavity** is the part of the small probe's space that the large probe never covers, that is, space the small probe could occupy that is enclosed or has too narrow an opening for the large one. Its volume is the number of grid points times the volume of a grid cell, the step cubed.
The cell shows the idea on a toy in two dimensions, with a C-shaped wall of atoms, two ligand atoms inside and the opening too narrow for the large probe, and counts the cells of the cavity; the same program, in three dimensions, is then run on the four complexes through the project's function, with the search restricted to within {{=defaults['ligand_cutoff']}} ångström of the ligand (otherwise it reports dozens of cavities all over the protein) and the largest cavity of that region taken as the pocket.
"""),
    code(POCKET_TOY),
    code(POCKET_REAL),
    md("""
In the toy the small probe fits in {{=int(small_fits.sum()):,}} cells and {{=cavity_cells}} of them belong to the cavity, a crescent of free space left between the ligand and the wall that the large probe cannot enter. For the four complexes the volume of that leftover space, the empty cavity that the program finds next to the ligand with the ligand's own atoms in place, is {{=pocket_table.loc['ketoconazole', 'empty cavity next to the ligand (A^3)']:.0f}} cubic ångström for ketoconazole,
{{=pocket_table.loc['vardenafil', 'empty cavity next to the ligand (A^3)']:.0f}} for vardenafil, {{=pocket_table.loc['ritonavir', 'empty cavity next to the ligand (A^3)']:.0f}} for ritonavir and {{=pocket_table.loc['azamulin', 'empty cavity next to the ligand (A^3)']:.0f}} for azamulin. It is therefore smallest for the compound that is bonded to the iron and largest for the one that sits far from it, which is the direction one would expect if a molecule that finds its place fills the pocket and one that does not leaves room.
That reading is tentative, and the reason is as important as the numbers: four compounds, each predicted once, of very different sizes (the molecule has {{=burial_table.loc['ketoconazole', 'ligand atoms']}} to {{=burial_table['ligand atoms'].max()}} atoms); a large molecule leaves a different amount of free space than a small one for reasons that have nothing to do with the pose; and the pocket of CYP3A4 is known to change shape with its ligand, which a single static prediction cannot show. The two measurements are what they are, cheap descriptions of a
pose; section 7.10 asks, with the procedures of Part VI and for many compounds, whether they carry any information that a model can use.

*On what actually ran.* `ligand_burial_sasa` in `cypmode/validation/pocket_geometry.py` calls Biopython's `ShrakeRupley` twice, on the complex and on an isolated copy of the ligand residue, and returns the two areas and the buried share. `pocket_volume` writes the complex to a temporary PDB file with the ligand residue renamed (Boltz-2 names it `LIG1`, four characters, which overflows the three-character field of the PDB format and corrupted the columns that follow, a bug the project found by running the function), writes the ligand alone, and calls pyKVFinder with
the ligand and the cutoff above; the largest cavity of the region is reported, a simplification that the module states openly, since the largest cavity is not guaranteed to be the one the ligand occupies.
"""),
]

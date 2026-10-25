# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part XI (second part): how a pose is scored (11.4)."""

from notebooks.notebook_builder import code, figure, md

SCORING_TOYS = '''
# The three ideas behind the scores, on small examples that can be checked by hand. (1) The Kabsch algorithm finds the rotation that best superposes two sets of points (the singular value decomposition of the matrix of their
# products of coordinates); with it, a ligand's distance from its true pose is measured after the protein pocket is superposed, not after the ligand is. (2) A symmetric molecule can be drawn in two ways that are the same: when a
# phenyl ring turns over, its equivalent atoms swap places, and a distance that matches atoms by their number reports an error that is not there. (3) LDDT-PLI looks at the distances between the ligand and the protein atoms around it
# in the true complex and asks how many of them the model keeps within 0.5, 1, 2 and 4 angstrom; it needs no superposition. The last function is the project's own (cypmode/validation/pose_selection.py).
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from cypmode.validation.pose_selection import lddt_pli_arrays

def kabsch(moving, fixed):
    """The rotation and translation that best map the points `moving` onto `fixed` (least squares), as a function."""
    mc, fc = moving.mean(0), fixed.mean(0)
    u, _, vt = np.linalg.svd((moving - mc).T @ (fixed - fc))
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1, 1, d]) @ vt
    return lambda x: (x - mc) @ rotation + fc

rng_score = np.random.default_rng(5)
pocket = rng_score.uniform(-12, 12, size=(160, 3))
pocket = pocket[np.linalg.norm(pocket, axis=1) < 12]
ligand = np.array([[0, 0, 0], [1.4, 0, 0], [2.1, 1.2, 0], [3.5, 1.2, 0.2], [4.2, 0, 0.5], [3.5, -1.2, 0.2], [2.1, -1.2, 0], [5.6, 0, 0.7]]) - 2.4

def rmsd(a, b):
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()))

# (1) a ligand with the right shape but 6 angstrom from its true place relative to the pocket
shifted = ligand + np.array([6.0, 0.0, 0.0])
ligand_only = rmsd(kabsch(shifted, ligand)(shifted), ligand)      # superpose the ligand on the true ligand: the error disappears
pocket_based = rmsd(shifted, ligand)                               # the pocket is already superposed: the error stays
# (2) the same ring drawn the other way: the two ortho carbons (atoms 2 and 7) and the two meta carbons (atoms 3 and 6) of 4-fluorotoluene exchange places
molecule = Chem.AddHs(Chem.MolFromSmiles("Cc1ccc(F)cc1"))
AllChem.EmbedMolecule(molecule, randomSeed=11)
heavy = Chem.RemoveHs(molecule)
positions = heavy.GetConformer().GetPositions()
flipped_positions = positions.copy()
flipped_positions[[2, 7]] = positions[[7, 2]]
flipped_positions[[3, 6]] = positions[[6, 3]]
flipped = Chem.Mol(heavy)
for i, p in enumerate(flipped_positions):
    flipped.GetConformer().SetAtomPosition(i, p.tolist())
by_number = rmsd(positions, flipped_positions)
symmetric = float(rdMolAlign.CalcRMS(flipped, heavy))
# (3) LDDT-PLI against displacement of the ligand, and against a rigid move of the whole complex
def complex_dicts(lig, prot):
    return {f"L{i}": p for i, p in enumerate(lig)}, {(i, "CA"): p for i, p in enumerate(prot)}
reference = complex_dicts(ligand, pocket)
rows = []
for shift in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
    direction = rng_score.standard_normal(3)
    moved_ligand = ligand + shift * direction / np.linalg.norm(direction)
    moved = complex_dicts(moved_ligand, pocket)
    rows.append({"ligand displaced by (angstrom)": shift, "RMSD (angstrom)": rmsd(moved_ligand, ligand), "LDDT-PLI": lddt_pli_arrays(*moved, *reference)})
rotation_matrix = np.linalg.qr(rng_score.standard_normal((3, 3)))[0]
turned = complex_dicts(ligand @ rotation_matrix, pocket @ rotation_matrix)
lddt_after_rotation = lddt_pli_arrays(*turned, *reference)
lddt_table = pd.DataFrame(rows)
show_table(lddt_table, floatfmt=".3f")
print(f"ligand-only superposition {ligand_only:.2f} A, pocket-based {pocket_based:.2f} A; ring flipped: by atom number {by_number:.2f} A, symmetry-aware {symmetric:.2f} A; LDDT-PLI after turning the whole complex {lddt_after_rotation:.3f}")
'''

OST_AGREEMENT = '''
# How close the project's LDDT-PLI is to the one the organisers compute with OpenStructure (cypmode/validation/ost_official.py, queue job 207_ost_vs_ours). The project's measure is a re-implementation, and every gate of the structure
# track rests on it, so the agreement was registered as a gate before it was measured: Spearman rank correlation at least 0.9, mean absolute difference at most 0.05, and the order of the poses of one ligand the same in at least 90 percent
# of the pairs. The table of 80 poses (20 ligands, restrained and not, two samples each) is read from the file the job wrote, and the three numbers are recomputed.
from scipy import stats

from cypmode.validation.ost_official import MAX_MEAN_ABSOLUTE_DIFFERENCE, MIN_PAIR_AGREEMENT, MIN_RANK_CORRELATION, pairs_agreement

ost_table = pd.read_csv(ROOT / "data/structure_track/outputs/cyp3a4/ost_vs_ours.csv").dropna(subset=["ours", "ost"])
ost_rank = float(stats.spearmanr(ost_table["ours"], ost_table["ost"])[0])
ost_difference = float((ost_table["ours"] - ost_table["ost"]).abs().mean())
ost_signed = float((ost_table["ours"] - ost_table["ost"]).mean())
ost_pairs = float(pairs_agreement(ost_table))
agreement_table = pd.DataFrame([
    {"measure": "Spearman rank correlation over all poses", "value": ost_rank, "limit": f">= {MIN_RANK_CORRELATION}", "met": ost_rank >= MIN_RANK_CORRELATION},
    {"measure": "mean absolute difference", "value": ost_difference, "limit": f"<= {MAX_MEAN_ABSOLUTE_DIFFERENCE}", "met": ost_difference <= MAX_MEAN_ABSOLUTE_DIFFERENCE},
    {"measure": "pairs of poses of a ligand ordered the same way", "value": ost_pairs, "limit": f">= {MIN_PAIR_AGREEMENT}", "met": ost_pairs >= MIN_PAIR_AGREEMENT}])
show_table(agreement_table, floatfmt=".3f")
n_ost_poses, n_ost_ligands = len(ost_table), int(ost_table["ligand"].nunique())
n_ost_met = int(agreement_table["met"].sum())
bisy_median, bisy_max = float(ost_table["ost_bisyrmsd"].median()), float(ost_table["ost_bisyrmsd"].max())
'''

POSEBUSTERS_TABLE = '''
# The physical-validity checks, as the organisers' scorer calls them (cypmode/validation/official_posebusters.py): PoseBusters in its docking configuration, 22 yes/no checks, on the ligand read from the submitted file with bond
# orders taken from the SMILES. Two tables are read: the 20 poses of the final submission, and the 200 poses of the pools from which they were chosen (every run of every official ligand), with the "distance margin" of each, the
# smallest ratio between the distance of a ligand atom to a receptor atom and the sum of their van der Waals radii; the threshold of the check is 0.75 and the project calls a pose safe above 0.80.
validity = pd.read_csv(ROOT / "data/structure_track/outputs/cyp3a4/posebusters_validity_official20.csv")
check_columns = [c for c in validity.columns if c not in ("compound", "all_passed", "model_index")]
pool_poses = pd.read_csv(ROOT / "data/structure_track/outputs/cyp3a4/final/official_posebusters_pools.csv")
n_checks_pb = len(check_columns)
checks_by_group = {"file and chemistry": check_columns[:6], "geometry of the ligand": check_columns[6:13], "against the receptor": check_columns[13:]}
n_final_pass = int(validity["all_passed"].sum())
n_pool, n_pool_pass = len(pool_poses), int((pool_poses["failed"].fillna("").astype(str).str.len() == 0).sum())
pool_valid = pool_poses[pool_poses["failed"].fillna("").astype(str).str.len() == 0]
n_pool_safe = int((pool_valid["distance_margin"] >= 0.80).sum())
smallest_margin_pool = float(pool_valid["distance_margin"].min())
group_table = pd.DataFrame([{"group of checks": g, "checks": len(cols), "examples": ", ".join(c.replace("_", " ") for c in cols[:3])} for g, cols in checks_by_group.items()])
show_table(group_table)
'''

CELLS = [
    md("""
## 11.4 How a pose is scored: RMSD, LDDT-PLI, BiSyRMSD and the PoseBusters checks

> **Field:** Structural biology · Mathematics (linear algebra)

*Where this section sits:* the levers of 11.5 to 11.9 are judged by how close their poses come to crystal poses, and the 20 submitted poses are judged by the organisers with their own scorer. This section defines the four quantities in use, builds a small example of each, and measures how well the project's re-implementation of the main one agrees with the organisers' program.

**The distance of a pose from the truth.** The simplest measure is the **root-mean-square deviation**, RMSD: for $n$ ligand atoms with predicted positions $x_i$ and true positions $y_i$,

$$\\mathrm{RMSD} \\;=\\; \\sqrt{\\frac{1}{n}\\sum_{i=1}^{n}\\lVert x_i - y_i\\rVert^2},$$

in ångströms. It is meaningful only if the two sets of coordinates are in the same frame, and a predicted complex can sit anywhere in space, so the frames are brought together first. Two choices exist and they answer different questions. If the *ligand* of the model is superposed on the ligand of the reference, the RMSD says whether the ligand has the right shape and not whether it is in the right place. If the *pocket* is superposed and the ligand is left where it is, the RMSD says how far the ligand is from where it should be in the pocket, which is the question. The best superposition of two point sets in the least-squares sense is given by the **Kabsch algorithm**: subtract the centres, form the matrix $H = \\sum_i (x_i - \\bar x)(y_i - \\bar y)^{\\top}$, take its singular value decomposition $H = U\\Sigma V^{\\top}$, and the optimal rotation is $R = V\\,\\mathrm{diag}(1,1,d)\\,U^{\\top}$ with $d = \\pm 1$ chosen so that $R$ is a rotation and not a reflection. The second source of error is **symmetry**: a molecule can be drawn in several ways that are the same molecule (a ring that turns over, a carboxylate whose two oxygens are equivalent), and an RMSD that pairs atoms by number reports an error that is not there. The correct RMSD is the smallest one over the symmetry-equivalent ways of pairing the atoms.

**LDDT-PLI.** The score the challenge is built on is the **local distance difference test** (Mariani, Biasini, Barbato and Schwede, 2013), adapted to a ligand and the protein around it. It needs no superposition. Take every pair of a ligand atom $a$ and a protein atom $b$ whose distance in the *reference* complex is below the inclusion radius $R_0 = 6$ Å, and compare the distance in the model with that in the reference:

$$\\text{LDDT-PLI} \\;=\\; \\frac14\\sum_{t\\in\\{0.5,\\,1,\\,2,\\,4\\}}\\frac{1}{|P|}\\sum_{(a,b)\\in P}\\mathbf 1\\big[\\,|d^{\\text{model}}_{ab} - d^{\\text{ref}}_{ab}| < t\\,\\big],$$

where $P$ is the set of such pairs and $\\mathbf 1[\\cdot]$ is 1 when the condition holds and 0 otherwise (distances in ångströms). It is the share of the contacts that the model keeps, averaged over four tolerances $t$ (0.5, 1, 2 and 4 Å); 1 is perfect, and a ligand in a different place keeps few of its contacts. It does not care in which frame the complex is written, and it is symmetric in the sense of the atoms (the organisers' program maps equivalent atoms). Because it asks about contacts, it is the right target for the use of the structure (a ligand that has the right contacts is in the right pocket), and it is harder than it looks: a displacement of the ligand by 1 Å already costs a seventh of the score.

**BiSyRMSD.** The secondary score, the **binding-site superposed, symmetry-corrected RMSD**, is the RMSD of the first paragraph made precise: the binding site is every protein residue with a heavy atom within 4 Å of the reference ligand, the model is superposed on the reference by a Kabsch fit over the C-alpha atoms of those residues, and the RMSD of the ligand atoms is then computed in place, minimised over the symmetry-equivalent pairings of its atoms.

**Validity.** A pose can have the right contacts and be chemically impossible. **PoseBusters** (Buttenschoen, Morris and Deane, 2024) is a set of checks on the chemistry and physics of a pose, 22 in the configuration the organisers use for docking: that the file loads and the molecule can be built from it, that the bond lengths and angles lie in the range seen in real molecules, that no two atoms of the ligand are closer than their radii allow, that aromatic rings and double bonds are flat, that the internal energy is not far above that of a relaxed conformer, and that the ligand does not collide with the protein, the heme or the waters (its atoms are not closer than a fraction of the sum of the radii, and the volumes do not overlap beyond a limit). The organisers score a pose that fails *any* of them as if it had no correct contacts at all: LDDT-PLI 0 and the worst BiSyRMSD. A valid pose is therefore a condition of every score. The first cell works the geometry out on examples.
"""),
    code(SCORING_TOYS),
    md("""
The cell shows each idea in a line. A ligand of the right shape sitting {{=pocket_based:.0f}} Å from its true place relative to the pocket has an RMSD of {{=ligand_only:.2f}} Å if the ligand alone is superposed (the shape is right) and {{=pocket_based:.2f}} Å if the pocket is the frame, which is why the pocket is used. A ring turned over has an RMSD of {{=by_number:.2f}} Å when atoms are paired by number and {{=symmetric:.2f}} Å when they are paired by symmetry. And LDDT-PLI falls as the ligand is moved from the pocket, from {{=lddt_table.iloc[2, 2]:.2f}} at 0.5 Å (the contacts all stay within the first tolerance) to {{=lddt_table.iloc[3, 2]:.2f}} at 1 Å, {{=lddt_table.iloc[4, 2]:.2f}} at 2 Å and {{=lddt_table.iloc[5, 2]:.2f}} at 4 Å, and does not change at all ({{=lddt_after_rotation:.2f}}) when the whole complex is turned in space.

**The project's measure and the organisers'.** The organisers compute the scores with **OpenStructure**, a program for structural bioinformatics (Biasini and co-workers, 2013) that is available only through the package manager conda. Every gate of this part (the restraint, the selection rules, the terms for choosing among poses) is judged on a Python re-implementation of LDDT-PLI that applies the definition above with symmetry-aware atom pairing (`cypmode/validation/lddt_to_crystal.py`, `pose_pools.py`) because it must run on thousands of poses without a separate environment. Whether it agrees with the program that scores the submission is a question that the data can answer, and it was registered as a gate before it was asked: rank correlation at least 0.9, mean absolute difference at most 0.05, and agreement on the order of the poses of one ligand in at least 90 percent of the pairs, because every selection rule uses the order.
"""),
    code(OST_AGREEMENT),
    md("""
Over {{=n_ost_poses}} poses of {{=n_ost_ligands}} ligands the project's measure ranks the poses almost as OpenStructure does (rank correlation {{=ost_rank:.3f}}) and lies higher by {{=ost_signed:.3f}} on average (mean absolute difference {{=ost_difference:.3f}}); it orders the poses of one ligand the same way in {{=ost_pairs:.1%}} of the pairs. {{=n_ost_met}} of the three conditions {{='is' if n_ost_met == 1 else 'are'}} met, and the gate is {{='MET' if n_ost_met == 3 else 'NOT MET'}}. The registered consequence is that the gates which rest on the project's measure are repeated with OpenStructure's scores: the two confirmatory restraint gates by job `241_ost_regate`, and the gates of the pools by jobs `244_ost_pools_score` and `245_ost_pool_gates` (the results available when this page was built are in 11.6 and 11.8). The median BiSyRMSD of these {{=n_ost_poses}} poses is {{=bisy_median:.1f}} Å and the largest {{=bisy_max:.1f}} Å: most poses of the prediction pipeline are a few ångströms from a crystal pose, which is the finding of 11.5 seen from the other metric.

The last cell reads the validity checks of the submission and of the pools from which it was chosen.
"""),
    code(POSEBUSTERS_TABLE),
    md("""
There are {{=n_checks_pb}} checks, in three groups ({{=group_table.iloc[0, 1]}} on whether the file and the molecule can be read, {{=group_table.iloc[1, 1]}} on the ligand's own geometry, {{=group_table.iloc[2, 1]}} against the receptor). {{=n_final_pass}} of the 20 submitted poses pass all {{=n_checks_pb}}. Among the {{=n_pool}} poses of the pools only {{=n_pool_pass}} pass, and only {{=n_pool_safe}} of those pass with a distance margin of at least 0.80, the project's own safety floor (the check fails at 0.75; the smallest margin among the passing poses is {{=smallest_margin_pool:.3f}}). A pose that passes by 0.003 today can fail when a library changes an atomic radius by a hundredth, so the choice of the final poses takes the margin into account (11.10).

*On what actually ran.* The cells above are the notebook's own, with the project's `pose_selection.lddt_pli_arrays` in the third part of the first. The agreement table is read from `ost_vs_ours.csv`, written by `cypmode/validation/ost_official.py` with `ost_worker.py` in the conda environment `ost` (OpenStructure 2.12.0), as the organisers' scorer calls it (`LDDTPLIScorer` and `SCRMSDScorer` with substructure matching). `cypmode/validation/bisy_rmsd.py` is the local BiSyRMSD of the definition above, used before OpenStructure was installed; `cypmode/validation/pose_comparison.py` is the symmetry-corrected ligand RMSD between two predictions of the same compound; `cypmode/validation/pose_validity.py` runs PoseBusters on a Boltz-2 output and `cypmode/validation/official_posebusters.py` runs it exactly as the organisers' scorer does, with the numeric margin of each check; `cypmode/validation/pose_margins.py` writes the margin of every pose of the pools. The queue job that measured the agreement is `207_ost_vs_ours`.
"""),
    figure("struct_rmsd_lddt"),
    figure("struct_ost_vs_project"),
]

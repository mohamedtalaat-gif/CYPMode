# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part XI (tenth part): the correction of the official structure set, three ligands that were published without their stereochemistry (11.12)."""

from notebooks.notebook_builder import code, md

DEMO = r'''
# A pose has a handedness. The molecule below is a public drug that is not part of the challenge (ibuprofen, one stereocentre): the two enantiomers are written, a pose of each is made as the co-folding
# models make theirs (heavy atoms with coordinates, no hydrogens), and the handedness is read back from the coordinates in two ways: the sign of the signed volume of the three neighbours of the centre, and the
# function that the recipe uses (`reproduces_stereo` of `cypmode/validation/pose_stereo.py`).
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from cypmode.validation.pose_stereo import reproduces_stereo

demo_smiles = {"first enantiomer": "CC(C)Cc1ccc(cc1)[C@H](C)C(=O)O", "second enantiomer": "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"}
no_marks = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"

def pose_of(smiles, mirrored=False):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    if mirrored:
        conformer = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            x, y, z = conformer.GetAtomPosition(i)
            conformer.SetAtomPosition(i, (-x, y, z))
    return Chem.RemoveHs(mol)

def signed_volume(mol):
    centre = next(a for a in mol.GetAtoms() if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED)
    xyz = mol.GetConformer().GetPositions()
    p = xyz[centre.GetIdx()]
    a, b, c = (xyz[n.GetIdx()] - p for n in sorted(centre.GetNeighbors(), key=lambda n: n.GetIdx())[:3])
    return float(np.dot(a, np.cross(b, c)))

rows = []
for label, smiles in demo_smiles.items():
    for mirrored in (False, True):
        pose = pose_of(smiles, mirrored)
        Chem.AssignStereochemistryFrom3D(pose)
        rows.append({"molecule": label, "pose": "mirrored" if mirrored else "as written", "label from the coordinates": Chem.FindMolChiralCenters(pose, useLegacyImplementation=False)[0][1],
                     "sign of the volume": "+" if signed_volume(pose) > 0 else "-",
                     "carries the written stereochemistry": reproduces_stereo(pose, smiles), "carries the other one": reproduces_stereo(pose, list(demo_smiles.values())[1 - list(demo_smiles).index(label)]),
                     "carries the SMILES without marks": reproduces_stereo(pose, no_marks)})
demo_table = pd.DataFrame(rows)
show_table(demo_table)
demo_flips = bool((demo_table.groupby("molecule")["sign of the volume"].nunique() == 2).all())
demo_mirror_is_other = bool(demo_table.loc[demo_table.pose == "mirrored", "carries the other one"].all() and not demo_table.loc[demo_table.pose == "mirrored", "carries the written stereochemistry"].any())
demo_marks_accept_both = bool(demo_table["carries the SMILES without marks"].all())
assert demo_flips and demo_mirror_is_other and demo_marks_accept_both  # a failing cell fails the build
'''

FACTS = r'''
# The numbers of 11.12, read from the files that the job 275 left: the ledger of the runs that were set aside, the poses of those runs read again for their stereochemistry, the table of the new pools and the
# check of the final zip. No official ligand is named in the output.
import json, re
from pathlib import Path
import posebusters
from cypmode.validation import pose_stereo as ps_module

official_path = ROOT / "data/structure_track/inputs/structure_track_test_blinded.csv"
official_table = pd.read_csv(official_path)
smiles_of = dict(zip(official_table.Molecule_Name, official_table.SMILES))
n_official = len(official_table)
n_specified = sum(ps_module.has_stereo(s) for s in official_table.SMILES)
n_open_now = sum(bool(ps_module.unspecified_stereo(s)) for s in official_table.SMILES)

config_dir = Path(posebusters.__file__).parent / "config"
def stereo_checks(name):
    return re.findall(r'^\s+stereo\w*:\s*"([^"]+)"', (config_dir / name).read_text(), re.MULTILINE)
dock_checks, redock_checks = stereo_checks("dock.yml"), stereo_checks("redock.yml")

set_aside = ROOT / "data/structure_track/outputs/out/superseded_racemic"
ledger = json.loads((set_aside / "ledger.json").read_text())[0]
corrected = ledger["ligands"]
n_corrected = len(corrected)
n_runs_set_aside = sum(len(v) for v in corrected.values())
old_flags = []
for folder in sorted((set_aside / "boltz").glob("boltz_results_cyp3a4_*")):
    ligand = folder.name.removeprefix("boltz_results_cyp3a4_").split("-")[0] + "-" + folder.name.removeprefix("boltz_results_cyp3a4_").split("-")[1]
    for cif in sorted(folder.glob("predictions/*/*.cif")):
        old_flags.append(ps_module.cif_reproduces_stereo(cif, smiles_of[ligand]) is True)
n_old, n_old_right = len(old_flags), int(sum(old_flags))
n_old_other = n_old - n_old_right

pools = pd.read_csv(ROOT / "data/structure_track/outputs/cyp3a4/final/official_posebusters_pools.csv")
new_pools = pools[pools.ligand.isin(list(corrected))]
n_new, n_new_right = len(new_pools), int(new_pools.stereo_ok.astype(bool).sum())
n_pool_valid_eligible = int((new_pools.passed.astype(bool) & new_pools.stereo_ok.astype(bool)).sum())

check_text = (ROOT / "data/structure_track/outputs/cyp3a4/logs/stereo_check.log").read_text()
zip_line = re.search(r"STEREO CHECK structures_final\.zip: (PASS|FAIL) \((\d+) of (\d+)", check_text)
zip_pass, n_zip_ok, n_zip = zip_line.group(1) == "PASS", int(zip_line.group(2)), int(zip_line.group(3))
file_line = re.search(r"STEREO CHECK official file has no unspecified stereo element: (PASS|FAIL)", check_text)
file_pass = file_line.group(1) == "PASS"
assert n_open_now == 0 and file_pass and zip_pass and n_zip_ok == n_zip == n_official  # a failing cell fails the build
print(f"{n_specified} of {n_official} official ligands carry stereo marks, {n_open_now} leave a centre open; {n_corrected} ligands corrected, {n_runs_set_aside} runs set aside ({n_old} poses, {n_old_other} of the other enantiomer); "
      f"new pools {n_new} poses, {n_new_right} with the official stereochemistry, {n_pool_valid_eligible} eligible; final zip {n_zip_ok} of {n_zip}")
'''

PICKS = r'''
# The poses that changed in the new draw (held back: they are results for the blinded compounds).
picks = pd.read_csv(ROOT / "data/structure_track/outputs/cyp3a4/final/stereo_picks.csv")
n_picks_changed = int(picks.changed.sum())
changed_picks = picks[picks.changed]
median_pick_rmsd = float(changed_picks.ligand_rmsd.median()) if len(changed_picks) else float("nan")
largest_fe_change = float((changed_picks.fe_n_new - changed_picks.fe_n_old).abs().max()) if len(changed_picks) else float("nan")
print(f"{n_picks_changed} of {len(picks)} chosen poses are not the earlier ones")
'''

CELLS = [
    md("""
## 11.12 The official set corrected: three ligands that came without their stereochemistry

> **Field:** Chemistry · Structural biology · Cheminformatics · Mathematics (vectors)

*Where this section sits:* sections 11.1 to 11.11 treated the 20 ligands of the structure track as given. On 2026-10-24 the organisers corrected the file that lists them: three of the 20 had been published without their stereochemistry. This section tells what that is, why a predicted pose carries it, why the check that the scorer applies to a pose cannot see it, and what was done in the structure track once the file was corrected. Section 2.1 showed what a stereocentre is and that a flat SMILES cannot tell an enantiomer from its mirror image; this section uses that and goes on to the three-dimensional pose.

**What was wrong.** A carbon with four different neighbours exists in two forms that are mirror images (**enantiomers**, section 2.1), and the marks `@` and `@@` of a SMILES say which one is meant. When the marks are absent the SMILES describes both at once. A mixture of equal parts of the two enantiomers is a **racemate**, and the organisers used racemic samples of three of the ligands in the assay, which is why their SMILES carried no marks. The structures that the submissions are scored against are of one enantiomer each, so the file for the structure track was changed to the SMILES of the models that are used for scoring, and the participants were asked to use those. The correction was reported by a participant on the discussion channel of the challenge and confirmed by the organisers; the two files of the time-dependent inhibition track and the training files are unchanged (the same checksums).

**What a pose says about it.** A pose is a list of coordinates, and at a stereocentre the neighbours lie in space in one of two arrangements that are mirror images. Take the centre at position $\\mathbf{p}$ and three of its neighbours, in a fixed order, at $\\mathbf{a}$, $\\mathbf{b}$ and $\\mathbf{c}$ (the fourth neighbour, a hydrogen that is not drawn, lies opposite the other three). The signed volume of the three vectors from the centre,

$$V = (\\mathbf{a}-\\mathbf{p}) \\cdot \\big[(\\mathbf{b}-\\mathbf{p}) \\times (\\mathbf{c}-\\mathbf{p})\\big],$$

is a number; $\\cdot$ is the dot product of two vectors (a number) and $\\times$ their cross product (a vector perpendicular to both). $V$ does not change when the molecule is moved or turned and changes sign when it is reflected, since a reflection turns a right-handed set of three vectors into a left-handed one. RDKit reads the chirality of a pose from this sign (`AssignStereochemistryFrom3D`), and the function `reproduces_stereo` of `cypmode/validation/pose_stereo.py` then asks whether the SMILES of the compound, with its marks, matches the pose as a substructure query with the chirality switched on; a SMILES without marks is matched by either arrangement. The cell shows it on a public drug that is not part of the challenge.
"""),
    code(DEMO),
    md("""
The table shows that the two enantiomers have volumes of opposite sign, that a mirrored pose carries the other one and not the written one, and that the SMILES without marks accepts both (`demo_flips`, `demo_mirror_is_other` and `demo_marks_accept_both` are checked by the build). A protein is made of one hand of amino acids and its pocket is itself chiral, so the two enantiomers of a ligand do not fit it in the same way, like a left and a right hand in the same glove; a pose of the wrong enantiomer is a pose of another molecule.

**Why the scorer's check does not see it.** A co-folding model that is given a SMILES without marks has to draw one of the two arrangements and does so by the shape it starts from, so nothing in the input keeps its samples to one. The samples of the three ligands that had been made from the racemic SMILES were read again from their coordinates: of {{=n_old}} poses, {{=n_old_right}} carry the stereochemistry of the corrected SMILES and {{=n_old_other}} the other. The poses of the other enantiomer are not marked by the scorer's PoseBusters call either. PoseBusters has a configuration for each use; the one that the organisers call, `dock`, contains {{=len(dock_checks)}} checks of stereochemistry, and the configuration for re-docking, `redock`, contains {{=len(redock_checks)}} ({{=', '.join(redock_checks).lower()}}), since it compares the pose with a reference that has a geometry. The scorer has no reference geometry to compare with, so a pose of the wrong enantiomer passes every check with the margin of the right one, and the only penalty for it is in the quantities that measure the geometry (the RMSD of the ligand and the LDDT-PLI), where a mirrored centre moves atoms by a few ångström. The gates of the structure track had been measured on crystal ligands whose SMILES carry their marks, so the official ligands were the only run with a racemic input.

**What was registered before the rerun.** (a) Every run of the three ligands is made again from the corrected SMILES, and the earlier runs are set aside unchanged, with the entries that made the batch runner skip them, so that no sample of these ligands derives from a racemic input (`cypmode/validation/stereo_rerun.py`: a run is stale when the SMILES of its input file is not the official one, as a canonical isomeric SMILES). (b) A pose is eligible for the submission only when it carries the official stereochemistry; when a ligand has no eligible valid pose the recipe stops with an error and does not fill the place with a pose of the other enantiomer (`final_recipe`, through the column `stereo_ok` that `pose_margins` writes for every pose of every pool). (c) The rest of the recipe is unchanged: the gates are not run again, and the choice among eligible poses is the registered one. (d) The audit and the pre-submission gate check that all 20 submitted poses carry the official stereochemistry (`cypmode/audit/pre_submission.py`), the relaxation of the final poses is rejected when it would break it (`relax_final`), the pharmacophore of the official ligands is read from the eligible poses only (`official_pharmacophore`), and a test asserts that the official file leaves no stereocentre open.
"""),
    code(FACTS),
    md("""
**What the rerun gave.** The file now leaves {{=n_open_now}} of {{=n_official}} ligands with a centre open ({{=n_specified}} carry marks); {{=n_corrected}} ligands were corrected and {{=n_runs_set_aside}} of their runs set aside. Of the {{=n_new}} poses of their new pools, {{=n_new_right}} carry the official stereochemistry and {{=n_pool_valid_eligible}} are also valid under the scorer's call, so that {{=('the model keeps to the stereochemistry that it is given: every sample is of the right enantiomer' if n_new_right == n_new else 'the model does not keep to the stereochemistry that it is given, and the eligibility rule is what keeps the poses of the other enantiomer out')}}. The check of the final zip finds {{=n_zip_ok}} of {{=n_zip}} poses of the official stereochemistry, and the check of the file finds no open centre.
""" ),
    code(PICKS, held_back=True),
    md("""
Of the 20 poses chosen by the unchanged rule, {{=n_picks_changed}} are not the poses that the racemic runs gave; for those the ligand moved by {{=median_pick_rmsd:.2f}} Å at the median between the two picks (pocket-aligned, symmetry-aware), and the largest change of the iron-nitrogen distance of a Type II ligand is {{=largest_fe_change:.2f}} Å. A new draw from a model gives a different pose even from the same input, so a change of pose is the expected result of the rerun and not a finding; it is reported so that nothing in the notebook, the figures or the film quotes a pose of the racemic runs.
""", held_back=True),
    md("""
*On what actually ran.* `cypmode/validation/pose_stereo.py` (`unspecified_stereo`, `has_stereo`, `reproduces_stereo`, `pdb_reproduces_stereo`, `cif_reproduces_stereo`, `zip_reproduces_stereo`) reads the stereochemistry of a pose and checks a zip or the pools; `cypmode/validation/stereo_rerun.py` finds the stale runs, sets them aside with a ledger and compares the earlier picks with the new ones. The job `275_stereo_official_rerun` does all of it in order (set aside, the base runs of the three ligands, the final recipe, the report of the picks, the checks and the scorer's call on the new zip) and the job `276_requeue_after_stereo` then runs the final chain again, from the final build of the time-dependent inhibition track on, so that every number, figure and page that depends on the poses is made again. The tests are `tests/test_pose_stereo.py` (the official file has no open centre, the racemic SMILES of the first publication leave one, a pose and its mirror image on the three ligands and on the cis and trans cyclohexane, the residue `LIG` of a submission file, a zip) and `tests/test_stereo_rerun.py` (a stale run, the set-aside, doing it twice, the dry run, the report of the picks).
"""),
]

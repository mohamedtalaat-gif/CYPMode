"""Part XI (eighth part): the final recipe, the pools, the margin, the relaxation and the archive (11.10)."""

from notebooks.notebook_builder import code, md

RECIPE = '''
# The decision of the final structure recipe, read from its record (final/recipe.json, written by cypmode/validation/final_recipe.py in job 180_structure_final_recipe_run): the verdict of every gate that the recipe reads and the
# choices that follow from them. The mapping from verdicts to choices is one fixed function, decide(); the cell calls it again on the recorded verdicts and checks that it gives the recorded recipe.
import json
from dataclasses import asdict

from cypmode.validation.final_recipe import decide

final_dir = ROOT / "data/structure_track/outputs/cyp3a4/final"
recipe_record = json.loads((final_dir / "recipe.json").read_text())
gate_verdicts = recipe_record["gates"]
again = asdict(decide(gate_verdicts, extra_per_ligand=recipe_record["recipe"]["extra_per_ligand"]))
recorded = dict(recipe_record["recipe"])
same_recipe = all(again[k] == recorded[k] for k in recorded if k != "reasons")
assert same_recipe  # a failing cell fails the build
gate_table = pd.DataFrame({"gate": list(gate_verdicts), "verdict": ["MET" if v else "NOT MET" for v in gate_verdicts.values()]})
show_table(gate_table, floatfmt=".0f")
n_gates, n_met = len(gate_table), int(sum(gate_verdicts.values()))
met_names = [g for g, v in gate_verdicts.items() if v]
met_text = ", ".join(met_names)
for reason in recorded["reasons"]:
    print("-", reason)
estimated_hours = float(recipe_record["estimated_hours"])
'''

POOLS = '''
# The pools and the pose chosen from each, read from the manifest of the final directory (final/manifest.json). A pool is every pose of every run of a ligand: the unguided samples, the restrained samples and the extra samples. The suffix of a
# run name says which kind it is: -rs restrained, -u unguided, -ex1 and so on extra samples, no suffix the first batch.
from cypmode.validation.official_posebusters import MARGIN_FLOOR

manifest = json.loads((final_dir / "manifest.json").read_text())
chosen = pd.DataFrame(manifest["ligands"])
chosen["kind"] = chosen["run"].str.extract(r"-(rs|u|ex\\d+)$")[0].map(lambda k: {"rs": "restrained", "u": "unguided"}.get(k, "extra sample" if isinstance(k, str) else "first batch"))
kind_table = chosen.groupby("kind").size().rename("ligands").reset_index()
show_table(kind_table, floatfmt=".0f")
pool_sizes = chosen["pool"]
n_poses, n_valid = int(pool_sizes.sum()), int(chosen["valid_in_pool"].sum())
all_chosen_valid, chosen_margin_min, chosen_margin_median = bool(chosen["valid"].all()), float(chosen["distance_margin"].min()), float(chosen["distance_margin"].median())
n_restrained = int((chosen["kind"] == "restrained").sum())
assert all_chosen_valid
print(f"{len(chosen)} ligands; pools of {pool_sizes.min()} to {pool_sizes.max()} poses (median {pool_sizes.median():.0f}); {n_poses} poses in all, {n_valid} valid")
'''

MARGIN_TOY = '''
# The margin rule on three small pools, with the project's own function (final_recipe.margin_candidates). A pose is a candidate when it passes the checks and its smallest relative distance to the receptor (11.4) is at least 0.80; if no valid
# pose clears 0.80, the valid pose with the largest margin is the only candidate; if no pose is valid, the candidates stay as they were. The rule never looks at how accurate a pose is: the selection rule chooses among the survivors.
from cypmode.validation.final_recipe import margin_candidates

margin_cases = {
    "some valid poses clear the floor": (np.array([True, True, False, True]), np.array([0.78, 0.86, 0.90, 0.81])),
    "valid poses, none clears it": (np.array([True, True, True]), np.array([0.76, 0.79, 0.77])),
    "no pose is valid": (np.array([False, False]), np.array([np.nan, np.nan])),
}
margin_rows = []
for case, (ok_toy, margins_toy) in margin_cases.items():
    survivors = margin_candidates(list(range(len(ok_toy))), ok_toy, margins_toy)
    margin_rows.append({"pool": case, "valid": " ".join("yes" if v else "no" for v in ok_toy), "margins": " ".join("-" if np.isnan(m) else f"{m:.2f}" for m in margins_toy), "poses that may be chosen": " ".join(str(i) for i in survivors)})
margin_toy_table = pd.DataFrame(margin_rows)
show_table(margin_toy_table, floatfmt=".2f")
'''

FORCE_TOY = '''
# The relaxation, in one dimension. A ligand atom sits too close to a frozen receptor atom. The energy has the two terms of a force field that matter for a clash: the Lennard-Jones term between the pair, 4 eps ((sigma/r)^12 - (sigma/r)^6) with
# the numbers of a carbon-carbon contact (eps 0.1 kcal/mol, sigma 3.4 A), and the tether that holds the atom to its predicted place, k (r - r0)^2 / 2. The minimiser is L-BFGS, the one the real relaxation uses. The table varies the tether:
# the real one is the line with k = TETHER (read from cypmode/validation/relax.py), and the relative distance is the distance over the sum of the van der Waals radii, 3.4 A, as the check of 11.4 defines it.
from scipy.optimize import minimize

relax_source = (ROOT / "cypmode/validation/relax.py").read_text()
tether_real = float(re.search(r"^TETHER_K = ([\\d.]+) \\* KCAL_PER_A2", relax_source, re.M).group(1))
eps_toy, sigma_toy, start_toy = 0.1, 3.4, 2.4

def lennard_jones(r):
    return 4 * eps_toy * ((sigma_toy / r) ** 12 - (sigma_toy / r) ** 6)

force_rows = []
for k_toy in (0.0, 0.5, tether_real, 20.0, 200.0):
    energy = lambda r, k=k_toy: lennard_jones(r[0]) + 0.5 * k * (r[0] - start_toy) ** 2
    solution = minimize(energy, x0=[start_toy], method="L-BFGS-B", bounds=[(1.5, 6.0)])
    final_r = float(solution.x[0])
    force_rows.append({"tether k (kcal/mol/A^2)": k_toy, "distance after (A)": final_r, "moved (A)": final_r - start_toy, "relative distance": final_r / sigma_toy, "energy before (kcal/mol)": lennard_jones(start_toy), "energy after (kcal/mol)": float(lennard_jones(final_r))})
force_table = pd.DataFrame(force_rows)
show_table(force_table, floatfmt=".2f")
relative_before = start_toy / sigma_toy
real_row = force_table.loc[force_table["tether k (kcal/mol/A^2)"] == tether_real].iloc[0]
'''

GATE_RESULTS = '''
# The validation of the relaxation on poses whose crystal pose is public (job 218_relaxation_gate; the module cypmode/validation/relax_gate.py): 90 predicted poses of 30 ligands, relaxed with the method of cypmode/validation/relax.py
# and scored before and after. The numbers are those of the log; the rule of the gate was written before it was run.
relax_log = (ROOT / "data/structure_track/outputs/cyp3a4/logs/relaxation_gate.log").read_text()
relax_poses = re.search(r"(\\d+) poses of (\\d+) ligands \\((\\d+) could not be relaxed\\); the ligand moved ([\\d.]+) A", relax_log)
relax_validity = re.search(r"poses passing the 22 checks: ([\\d.]+)% -> ([\\d.]+)%; official-style score ([\\d.]+) -> ([\\d.]+) \\(p = ([\\d.]+)\\)", relax_log)
relax_margin = re.search(r"margin \\(poses passing before and after, (\\d+) ligands\\): ([\\d.]+) -> ([\\d.]+) \\(p = ([\\d.]+)\\)", relax_log)
relax_fidelity = re.search(r"LDDT-PLI change on those poses: ([+-][\\d.]+) \\(non-inferiority margin ([+-][\\d.]+), p = ([\\d.]+)\\)", relax_log)
relax_verdict = re.findall(r"^STRUCTURE GATE force_field_relaxation[^:\\n]*: (MET|NOT MET)", relax_log, re.M)[-1]
manifest_rule = manifest["relaxation"]["rule"]
relax_final_source = (ROOT / "cypmode/validation/relax_final.py").read_text()
largest_validation_move = float(re.search(r"median [\\d.]+ A, largest ([\\d.]+) A", relax_final_source).group(1))
registered_on = re.search(r"registered (\\d{4}-\\d\\d-\\d\\d)", relax_final_source).group(1)
n_validation_poses, n_validation_ligands, n_unrelaxable, median_move = (int(relax_poses.group(1)), int(relax_poses.group(2)), int(relax_poses.group(3)), float(relax_poses.group(4)))
valid_before, valid_after, score_before, score_after, score_p = (float(v) for v in relax_validity.groups())
margin_before_val, margin_after_val = float(relax_margin.group(2)), float(relax_margin.group(3))
lddt_change, lddt_noninferiority = float(relax_fidelity.group(1)), float(relax_fidelity.group(2))
gate_table_relax = pd.DataFrame({"quantity": ["poses passing the 22 checks (%)", "official-style score", "smallest relative distance (ligands, poses passing before and after)", "LDDT-PLI change"],
                                 "before": [valid_before, score_before, margin_before_val, 0.0], "after": [valid_after, score_after, margin_after_val, lddt_change]})
show_table(gate_table_relax, floatfmt=".3f")
'''

DECISIONS = '''
# What was applied to the 20 submitted poses (relaxation/decisions.csv, written by cypmode/validation/relax_final.py in job 243_structure_relaxation_final) under the rule registered before any of them was relaxed. The rule is re-evaluated here
# from the table: both poses pass the checks, the margin rises and is at least the floor, the ligand moved at most 1 A, and an iron-nitrogen distance changed by at most 0.2 A.
decisions = pd.read_csv(final_dir / "relaxation/decisions.csv")
rule = manifest_rule
fe_change = (decisions["fe_n_after"] - decisions["fe_n_before"]).abs()
rule_holds = (decisions["passed_before"] & decisions["passed_after"] & (decisions["margin_after"] > decisions["margin_before"]) & (decisions["margin_after"] >= rule["margin_floor"])
              & (decisions["movement_A"] <= rule["max_movement_A"]) & (fe_change.fillna(0.0) <= rule["max_fe_n_change_A"]))
n_adopted, n_rule_holds = int(decisions["adopted"].sum()), int(rule_holds.sum())
n_with_iron = int(decisions["fe_n_before"].notna().sum())
assert n_adopted == manifest["relaxation"]["adopted"] and bool((rule_holds == decisions["adopted"]).all())  # the table and the rule agree ligand by ligand
decision_table = decisions[["compound", "margin_before", "margin_after", "movement_A", "fe_n_before", "fe_n_after"]].rename(columns={"compound": "ligand", "margin_before": "relative distance before", "margin_after": "relative distance after", "movement_A": "moved (A)", "fe_n_before": "Fe-N before (A)", "fe_n_after": "Fe-N after (A)"})
decision_table = decision_table.rename(columns={"relative distance before": "relative distance before", "margin_after": "relative distance after"})
show_table(decision_table.astype(object).where(decision_table.notna(), "-"), floatfmt=".2f")
move_median, move_max = float(decisions["movement_A"].median()), float(decisions["movement_A"].max())
margin_min_before, margin_min_after = float(manifest["relaxation"]["smallest_margin_before"]), float(manifest["relaxation"]["smallest_margin_after"])
fe_change_max = float(fe_change.max())
'''

ZIP_TOY = '''
# Why the archive is written the way it is. Two builds of an archive from the same files give different bytes when each entry carries the time of the file (the default of the usual tools), so the SHA-256 differs although nothing
# of substance does. The project's writer (cypmode.challenge.structure_submission.write_deterministic_zip) fixes the time (1 January 1980, the earliest a zip can hold) and the file mode of every entry. The cell builds the same three small
# files twice, a minute apart, in both ways.
import hashlib
import os
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from cypmode.challenge.structure_submission import write_deterministic_zip

def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

zip_rows = []
with tempfile.TemporaryDirectory() as tmp:
    files = []
    for name in ("one.pdb", "two.pdb", "three.pdb"):
        path = Path(tmp) / name
        path.write_text(f"HETATM    1  C1  LIG B   1       0.000   0.000   0.000  1.00  0.00           C\\n" * 40)
        files.append(path)
    for label, deterministic in (("ordinary, built at one time", False), ("ordinary, built a minute later", False), ("fixed time, built at one time", True), ("fixed time, built a minute later", True)):
        for k, path in enumerate(files):
            os.utime(path, (1_700_000_000 + 60 * ("later" in label) + k, 1_700_000_000 + 60 * ("later" in label) + k))
        archive = Path(tmp) / "archive.zip"
        if deterministic:
            write_deterministic_zip(files, archive)
        else:
            with ZipFile(archive, "w", ZIP_DEFLATED) as handle:
                for path in files:
                    handle.write(path, path.name)
        zip_rows.append({"build": label, "SHA-256 (first 16 digits)": sha256_of(archive)[:16]})
zip_table = pd.DataFrame(zip_rows)
show_table(zip_table, floatfmt=".0f")
ordinary_differ = zip_table.loc[0, "SHA-256 (first 16 digits)"] != zip_table.loc[1, "SHA-256 (first 16 digits)"]
fixed_equal = zip_table.loc[2, "SHA-256 (first 16 digits)"] == zip_table.loc[3, "SHA-256 (first 16 digits)"]
assert ordinary_differ and fixed_equal
'''

ARCHIVE_CHECK = '''
# The archive that is the candidate submission, checked here the way the portal and the scorer read it: its SHA-256 against the one the manifest recorded, the names of its members against the official ligands, the ligand residue and the
# chain of every file, and the last line of the scorer's own PoseBusters call on the archive (job 192_structure_margin_aware_zip wrote it, job 243_structure_relaxation_final checked the relaxed archive before it replaced anything).
from zipfile import ZipFile

archive_path = ROOT / manifest["zip"]
archive_sha = sha256_of(archive_path)
unrelaxed_path = final_dir / "structures_unrelaxed.zip"
official_ligands = pd.read_csv(ROOT / "data/structure_track/inputs/structure_track_test_blinded.csv")["Molecule_Name"].tolist()
with ZipFile(archive_path) as handle:
    members = sorted(handle.namelist())
    residues = {name: {line[17:20].strip() for line in handle.read(name).decode().splitlines() if line.startswith("HETATM")} for name in members}
    entry_times = {tuple(info.date_time) for info in handle.infolist()}
names_ok = members == sorted(f"{n}.pdb" for n in official_ligands)
ligand_residue_ok = all("LIG" in residues[name] for name in members)
official_summary = [line for line in (ROOT / "data/structure_track/outputs/cyp3a4/logs/official_posebusters_final.log").read_text().splitlines() if "pass all" in line][-1]
posebusters_found = re.search(r"(\\d+) structures; (\\d+) pass all (\\d+) checks; (\\d+) below the margin floor ([\\d.]+)", official_summary)
n_structures, n_pass, n_checks_final, n_below_floor = (int(posebusters_found.group(i)) for i in (1, 2, 3, 4))
archive_report = pd.DataFrame({"quantity": ["members", "names equal the official ligands", "ligand residue LIG in every file", "entry times in the archive", "SHA-256 equals the manifest", "SHA-256 of the archive (first 16 digits)", "SHA-256 of the unrelaxed archive (first 16 digits)"],
                               "value": [len(members), names_ok, ligand_residue_ok, ", ".join(f"{t[0]}-{t[1]:02d}-{t[2]:02d}" for t in sorted(entry_times)), archive_sha == manifest["sha256"], archive_sha[:16], sha256_of(unrelaxed_path)[:16]]})
show_table(archive_report, floatfmt=".0f")
assert names_ok and ligand_residue_ok and archive_sha == manifest["sha256"]
print(official_summary)
'''

CELLS = [
    md("""
## 11.10 The final recipe: from the verdicts to the file that is submitted

> **Field:** Structural biology · Software engineering

*Where this section sits:* 11.3 to 11.9 each ended in a verdict. This section is what those verdicts add up to: the recipe that was run, the pools of poses it produced, the rule that keeps a pose away from the edge of the scorer's checks, the force-field relaxation that was added after the poses were chosen, and the archive that is the candidate submission. Nothing in it was chosen by looking at the accuracy of the twenty poses, which nobody outside the organisers can see: the recipe is a function of the verdicts, the relaxation was applied under a rule written before any of the twenty poses was relaxed, and the archive is built so that the same poses give the same bytes.

**One recipe, decided by the gates.** The module `cypmode/validation/final_recipe.py` reads the line `STRUCTURE GATE <name>: MET|NOT MET` that every gate of the structure track wrote to its log, and applies one fixed mapping: a restraint gate that is MET switches the restraint on for the whole batch (the bond restraint is withdrawn if the registered check on new ligands found it worse); a pilot gate that is MET adds three runs of that lever to every pool; a MET engine gate adds three OpenFold3 samples; a MET selection gate replaces the model's own ranking among valid poses by minimum Bayes risk (11.8) or by that rank plus the terms that passed; the extra samples go where the pools disagree most if that gate is MET and are spread evenly if not. The base of every pool is five unguided Boltz-2 samples per ligand (samples that already existed count). The cell reads the record of the decision, calls the function again on the recorded verdicts, and requires that it gives the recorded recipe.
"""),
    code(RECIPE),
    md("""
Of the {{=n_gates}} gates the recipe reads, {{=n_met}} are MET, the two restraints of 11.6 ({{=met_text}}). Every other gate is NOT MET: the three levers that have a verdict (seeds, crystal conformers and cryo-EM templates, 11.7; the fourth lever, a lower step scale, produced only unusable poses and has no verdict, so it adds nothing), the second engine (11.9), minimum Bayes risk and the terms that describe a pose (11.8), expected-improvement sampling and the series consensus. Two gates are about the bond restraint on ligands that were not used to choose it (11.6): the confirmation of its gain is NOT MET, and so is the finding of harm, so the restraint stays on the evidence of the gate that chose it and is not withdrawn. The recipe is therefore short: the restraint, three restrained samples per ligand, two extra samples per ligand spread evenly, Boltz-2 alone, and the first valid pose in the model's own order, taken from the restrained run first for a ligand that has a donor of the Type II kind (that is the pick the restraint gate measured, so it is the pick that carries the evidence). The record estimates {{=estimated_hours:.1f}} hours of machine time for the runs it added to what already existed. The decision is a table of verdicts and not an opinion: a gate that is later re-run and changes its verdict changes the recipe, and the job `179_requeue_final_recipe` exists so that the decision can be run again on the final verdicts.

**The pools.** For each of the twenty ligands the pool is every pose of every run of that ligand. The next cell reads the manifest of the final directory: the size of each pool, how many of its poses are valid under the scorer's own PoseBusters call, which kind of run the chosen pose came from, and the smallest relative distance to the receptor of the chosen pose.
"""),
    code(POOLS),
    md("""
There are {{=n_poses}} poses in the twenty pools, {{=n_valid}} of them valid, in pools of {{=pool_sizes.min()}} to {{=pool_sizes.max()}}. The pose chosen for {{=n_restrained}} ligands comes from the restrained run, which is where the first-valid rule looks first for a ligand with a Type II donor. Every chosen pose is valid (the cell requires it), and the smallest relative distance to the receptor among the twenty is {{=chosen_margin_min:.3f}} (median {{=chosen_margin_median:.3f}}).

**The margin.** 11.4 introduced the check that a ligand atom is not closer to a receptor atom than the sum of their van der Waals radii allows, scaled: the check fails below a relative distance of 0.75. A pose at 0.76 passes, and passes by a hair: the scorer reads the file it is given, but a pose that sits on the threshold is one that a rounding or a small difference between two versions of the checker could turn into a zero (a failing pose scores zero on all three metrics). The recipe therefore keeps a margin: among the valid poses of a pool only those with a smallest relative distance of at least {{=MARGIN_FLOOR}} may be chosen. If no valid pose clears that, the one with the largest margin is chosen alone. If no pose is valid, the pool is left as it was. The rule never looks at how accurate a pose is, which nothing can know for a blinded ligand; it removes only the poses that are closest to being invalid, and the selection rule chooses among the rest. The cell shows the three cases with the project's function.
"""),
    code(MARGIN_TOY),
    md("""
In the first pool the pose with the margin 0.78 is dropped although it is valid, and the poses with 0.86 and 0.81 remain. In the second no valid pose reaches the floor, and the pose with the largest margin, 0.79, is the only one left. In the third there is nothing to choose between and the pool is unchanged.

**The relaxation.** The poses that co-folding models write are not always physically sensible: atoms can sit a little too close, bonds can be a little too long. The PoseBusters paper reports that a short energy minimisation with a force field repairs most of the poses that a deep-learning docking model leaves invalid. A force field (met in 3.6 and 10.6 as the way a shape is relaxed) writes the energy of a molecular system as a sum of simple terms: the stretching of bonds and the bending of angles as springs, the twisting of torsions, and between atoms that are not bonded the **Lennard-Jones** term and the Coulomb term. The Lennard-Jones term is $E = 4\\varepsilon\\left[(\\sigma/r)^{12} - (\\sigma/r)^{6}\\right]$ for two atoms at distance $r$, where $\\sigma$ is the distance at which the repulsion and the attraction cancel in energy ($\\varepsilon$ is the depth of the well): steeply repulsive when atoms are closer than the sum of their radii, weakly attractive beyond it. The Coulomb term, $q_i q_j / (4\\pi\\varepsilon_0\\,\\epsilon_r\\,r)$, is the force between the partial charges of the atoms, divided by the relative **dielectric constant** $\\epsilon_r$ of the medium, which weakens it. To **minimise** the energy is to move the atoms downhill until nothing moves; the method of the project is L-BFGS, which follows the slope of the energy and keeps a memory of the last steps to estimate the curvature.

Relaxing a pose must repair it and not re-dock it. The project's version (`cypmode/validation/relax.py`, in OpenMM, run in a conda environment of its own) keeps the protein and the heme fixed, so that they only supply the nonbonded terms; uses Amber ff14SB for the protein, the CHARMM template of heme b for the heme's nonbonded terms, and OpenFF 2.2.1 for the ligand, with AM1-BCC-like partial charges from a graph network (NAGL); sets the dielectric constant to 4; ties every heavy atom of the ligand to its predicted position with a weak spring (a **tether**); and, when a nitrogen of the ligand is within 3 Å of the iron in the prediction, holds that distance with a stiff spring. The tether is what keeps the repair local. The cell is the idea in one dimension: one ligand atom 2.4 Å from a frozen receptor atom, the radii of the two adding up to 3.4 Å, so that its relative distance is 2.4 / 3.4 = 0.71 and it fails the check, minimised for several strengths of the tether.
"""),
    code(FORCE_TOY),
    md("""
Without a tether the atom moves {{=force_table.loc[0, 'moved (A)']:.2f}} Å to the bottom of the well, a relative distance of {{=force_table.loc[0, 'relative distance']:.2f}}; with a stiff tether it barely leaves its place, {{=force_table.iloc[-1]['moved (A)']:.2f}} Å, and the contact stays tight, at a relative distance of {{=force_table.iloc[-1]['relative distance']:.2f}}, over the threshold of the check and under the floor. With the tether of the project ({{=tether_real}} kcal/mol/Å²) it moves {{=real_row['moved (A)']:.2f}} Å, and its relative distance rises from {{=relative_before:.2f}} to {{=real_row['relative distance']:.2f}}, over the threshold and over the floor, and the energy of the contact falls from {{=real_row['energy before (kcal/mol)']:.1f}} to {{=real_row['energy after (kcal/mol)']:.2f}} kcal/mol. Weak enough to repair, strong enough to be local: the same trade-off that the rule below enforces per pose. (The real system has thousands of frozen atoms and several hundred pairs in contact, charges, and many ligand atoms that move together; the toy shows the mechanism, not the size of an effect.)

**Was it validated?** The effect of the relaxation on accuracy cannot be measured for a blinded ligand, which has no crystal pose. It was therefore tested first, as a gate with its rule written beforehand (job `218_relaxation_gate`), on the predicted poses of ligands whose crystal pose is public. The gate required three things together, each tested one-sided and paired, with the ligand as the unit (the mean over its poses): that the share of poses passing the 22 checks does not fall and the official-style score (the LDDT-PLI of a pose, or zero when it fails the checks) rises; that the smallest relative distance of the poses that pass before and after rises; and that their LDDT-PLI to the crystal pose does not fall by more than 0.01 on average (a test of non-inferiority). The next cell reads the log of that gate.
"""),
    code(GATE_RESULTS),
    md("""
The relaxation was applied to {{=n_validation_poses}} poses of {{=n_validation_ligands}} ligands, of which {{=n_unrelaxable}} could not be relaxed; the ligand moved {{=median_move:.2f}} Å (median heavy-atom RMSD). The share of poses passing the checks rises from {{=valid_before}} to {{=valid_after}} percent; the official-style score, which counts a failing pose as zero, rises from {{=score_before:.3f}} to {{=score_after:.3f}} (p = {{=score_p:.4f}}); the smallest relative distance rises from {{=margin_before_val:.3f}} to {{=margin_after_val:.3f}}; and LDDT-PLI changes by {{=lddt_change:+.4f}}, so that accuracy is not harmed on these ligands within the margin that was set. The verdict of the gate is {{=relax_verdict}}.

**The rule applied to the twenty.** A gate that holds on average is not a guarantee for each pose, so the relaxation of the submitted poses was run under a rule of its own, registered on {{=registered_on}} before any of the twenty was relaxed. A relaxed pose replaces the original only when all four hold: both pass the 22 checks; the smallest relative distance rises and is at least the floor; the ligand moved by at most {{=manifest_rule['max_movement_A']}} Å (in the validation poses the median was {{=median_move:.2f}} Å and the largest {{=largest_validation_move:.2f}} Å, which is where the bound comes from); and, when a ligand nitrogen coordinates the iron in the original, it still does and the distance changed by at most {{=manifest_rule['max_fe_n_change_A']}} Å. Accuracy cannot be checked per pose, and the movement bound stands in for it. Otherwise the original pose stays. The cell evaluates the rule on the table of decisions.
"""),
    code(DECISIONS),
    md("""
The rule holds for {{=n_rule_holds}} of the 20 poses and {{=n_adopted}} were adopted (the table of decisions and the rule agree ligand by ligand). The ligands moved {{=move_median:.2f}} Å at the median and {{=move_max:.2f}} Å at most; the {{=n_with_iron}} ligands with a nitrogen at the iron changed that distance by at most {{=fe_change_max:.2f}} Å; and the smallest relative distance among the twenty rose from {{=margin_min_before:.3f}} to {{=margin_min_after:.3f}}. What the relaxation buys is safety against the checks, not accuracy; what it risks is a small displacement, which the rule bounds.

**The archive.** The portal takes a zip of exactly twenty PDB files, one for each official ligand, named for it, each a full complex with the ligand residue named LIG. The archive is built so that the same poses give the same bytes: an ordinary zip stores, for each entry, the modification time of the file, so that building it twice from the same files at different moments gives two archives with different SHA-256 although nothing of substance differs, and a hash that changes without a reason cannot serve as a record of what was submitted. The project's writer fixes the time of every entry (1 January 1980, the earliest a zip can hold) and the file mode. The cell shows it on small files, a minute apart.
"""),
    code(ZIP_TOY),
    md("""
The two ordinary builds differ and the two fixed ones are equal (the cell requires both). The next cell checks the candidate archive itself: its SHA-256 against the manifest, the names of its members against the list of official ligands, the ligand residue in every file, and the verdict of the scorer's own PoseBusters call on the archive. The unrelaxed archive is kept beside it, so that the relaxation can be undone by copying one file.
"""),
    code(ARCHIVE_CHECK),
    md("""
All {{=n_structures}} structures pass all {{=n_checks_final}} checks, {{=n_below_floor}} are below the margin floor, the names equal the official ligands, and the hash of the file equals the one recorded in the manifest. The hash is what the method report quotes, so that anyone holding the file can check that it is the one described.

**What is and is not claimed.** The validity of the poses is checked, per pose, by the scorer's own code. Their accuracy is not known: it is measured only by the organisers, against crystal structures that nobody else has. What the project has instead is a set of verdicts on the levers on ten crystal-anchored ligands (11.5 to 11.9), a restraint whose gain was shown on those ligands and neither confirmed nor refuted on ligands that were not used to choose it (11.6), and a relaxation that was validated on 30 ligands with public crystals. The recipe is a conservative one: it adds to the model's own output only what a pre-declared gate passed, and where no gate passed it keeps the model's own first valid pose. The earlier archive of unrestrained Boltz-2 poses is kept as the fallback.

*On what actually ran.* `cypmode/validation/final_recipe.py` (`read_gates`, `decide`, `margin_candidates`, `select_pose`, and the constants of cost), `cypmode/validation/pose_margins.py` (the audit of every pool pose under the scorer's call), `cypmode/validation/official_posebusters.py` (the scorer's 22 checks and `MARGIN_FLOOR`), `cypmode/validation/relax.py` (the relaxation), `cypmode/validation/relax_gate.py` (the gate of job 218) and `cypmode/validation/relax_final.py` (the rule and the rebuild of the archive), `cypmode/challenge/structure_submission.py` (`write_deterministic_zip` and the conversion of a prediction to the portal's format), the tests `tests/test_relax_final.py`, and the queue jobs `179_requeue_final_recipe`, `180_structure_final_recipe_run`, `185_structure_recipe_update`, `192_structure_margin_aware_zip`, `218_relaxation_gate` and `243_structure_relaxation_final`. The environment of the relaxation is described in `environment/openmm-environment.yml`. The cells of this section read the records those jobs left; the one-dimensional relaxation and the small archives are written for the notebook. The force-field parameters are those of the Amber ff14SB, CHARMM and OpenFF 2.2.1 distributions, which the repository does not redistribute.
"""),
]

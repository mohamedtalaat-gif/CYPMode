# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part XI (eighth part): the final recipe, the pools, the margin, the relaxation and the archive (11.10)."""

from notebooks.notebook_builder import code, figure, md

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

FORCE_TERMS = '''
# The two nonbonded terms of a force field, worked on one pair of atoms. The numbers are those of a carbon-carbon contact (well depth 0.1 kcal/mol, sigma 3.4 A) and of two partial charges,
# +0.3 and -0.4 of the charge of an electron. The Coulomb constant is not typed in: it is computed from the physical constants as e^2 / (4 pi eps0), per mole, in kcal A / (mol e^2), the unit
# in which a force field writes its charges. The minimisation is done twice from the same start, 2.4 A: by plain steps down the slope (gradient descent) and by L-BFGS, the method of the project.
from scipy import constants as phys
from scipy.optimize import minimize, minimize_scalar

well_eps, well_sigma = 0.1, 3.4
coulomb_constant = phys.e**2 / (4 * np.pi * phys.epsilon_0) * phys.N_A * 1e10 / (phys.calorie * 1000)
charge_i, charge_j = 0.3, -0.4
coulomb_in_relax = float(re.search(r"^COULOMB = ([\\d.]+)", (ROOT / "cypmode/validation/relax.py").read_text(), re.M).group(1))  # the same constant in the units of OpenMM, kJ nm / (mol e^2)
assert abs(coulomb_constant * phys.calorie / 10 - coulomb_in_relax) < 1e-6 * coulomb_in_relax

def lj_repulsion(r):
    return 4 * well_eps * (well_sigma / r) ** 12

def lj_attraction(r):
    return -4 * well_eps * (well_sigma / r) ** 6

def lj_total(r):
    return lj_repulsion(r) + lj_attraction(r)

def lj_push(r):
    """Minus the slope of the energy: positive when the atoms are pushed apart."""
    return 4 * well_eps / r * (12 * (well_sigma / r) ** 12 - 6 * (well_sigma / r) ** 6)

def coulomb_energy(r, dielectric):
    return coulomb_constant * charge_i * charge_j / (dielectric * r)

r_min_formula = 2 ** (1 / 6) * well_sigma
r_min_found = float(minimize_scalar(lj_total, bounds=(3.0, 6.0), method="bounded", options={"xatol": 1e-9}).x)
assert abs(r_min_found - r_min_formula) < 1e-4 and abs(lj_total(r_min_formula) + well_eps) < 1e-12 and abs(lj_total(well_sigma)) < 1e-12

term_rows = []
for r in (2.4, 3.0, well_sigma, r_min_formula, 4.5, 6.0, 10.0):
    term_rows.append({"distance r (A)": r, "repulsive part": lj_repulsion(r), "attractive part": lj_attraction(r), "Lennard-Jones sum": lj_total(r),
                      "Coulomb, dielectric 1": coulomb_energy(r, 1.0), "Coulomb, dielectric 4": coulomb_energy(r, 4.0)})
term_table = pd.DataFrame(term_rows)
show_table(term_table, floatfmt=".4f")

def descend(step, start=2.4, tolerance=1e-3, limit=200000):
    """Gradient descent on the Lennard-Jones energy: move by step times the push until the push is below the tolerance."""
    r, n, path = start, 0, [(0, start, float(lj_total(start)), float(lj_push(start)))]
    while abs(lj_push(r)) > tolerance and n < limit:
        r += step * lj_push(r)
        n += 1
        if n <= 4:
            path.append((n, float(r), float(lj_total(r)), float(lj_push(r))))
    return n, float(r), path

descent_step, descent_start = 0.002, 2.4
n_descent, r_descent, descent_path = descend(descent_step, descent_start)
descent_path.append((n_descent, r_descent, float(lj_total(r_descent)), float(lj_push(r_descent))))
descent_table = pd.DataFrame(descent_path, columns=["step", "distance r (A)", "energy (kcal/mol)", "push, minus the slope (kcal/mol/A)"])
show_table(descent_table, floatfmt=".4f")
step_rows = []
for step in (0.002, 0.008, 0.03, 0.06):
    n_steps, r_end, path = descend(step, descent_start)
    step_rows.append({"step size (A per kcal/mol/A)": step, "distance after the first step (A)": path[1][1], "steps until the push is below 0.001": n_steps, "where it stops (A)": r_end, "distance from the bottom (A)": r_end - r_min_formula})
step_table = pd.DataFrame(step_rows)
show_table(step_table, floatfmt=".4f")
bfgs = minimize(lambda x: lj_total(x[0]), x0=[descent_start], jac=lambda x: [-lj_push(x[0])], method="L-BFGS-B", bounds=[(1.5, 6.0)])
n_bfgs, r_bfgs = int(bfgs.nit), float(bfgs.x[0])
curvature_at_bottom = 4 * well_eps * (156 * well_sigma**12 / r_min_formula**14 - 42 * well_sigma**6 / r_min_formula**8)

fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.9))
r_axis = np.linspace(3.0, 6.5, 400)
left.plot(r_axis, lj_repulsion(r_axis), color=theme["red"], lw=1.6, label="repulsive part")
left.plot(r_axis, lj_attraction(r_axis), color=theme["blue"], lw=1.6, label="attractive part")
left.plot(r_axis, lj_total(r_axis), color=theme["text"], lw=2.2, label="Lennard-Jones sum")
left.axhline(0, color=theme["muted"], lw=0.8)
left.axvline(well_sigma, color=theme["muted"], ls=":", lw=1)
left.axvline(r_min_formula, color=theme["orange"], ls="--", lw=1)
left.annotate("sigma: the sum crosses zero", (well_sigma, 0.30), (3.0, 0.36), color=theme["muted"], fontsize=8)
left.annotate("r_min = 2^(1/6) sigma: the bottom, depth -epsilon", (r_min_formula, -0.1), (4.15, -0.2), color=theme["orange"], fontsize=8)
left.set(xlabel="distance r between the two atoms (A)", ylabel="energy (kcal/mol)", ylim=(-0.26, 0.42), title="Lennard-Jones")
left.legend(frameon=False, loc="upper right")
tidy(left, theme)
r_long = np.linspace(r_min_formula, 12, 400)
right.plot(r_long, np.abs(lj_total(r_long)), color=theme["text"], lw=2.2, label="Lennard-Jones sum")
right.plot(r_long, np.abs(coulomb_energy(r_long, 1.0)), color=theme["purple"], lw=1.6, label="Coulomb, dielectric 1")
right.plot(r_long, np.abs(coulomb_energy(r_long, 4.0)), color=theme["green"], lw=1.6, label="Coulomb, dielectric 4")
right.set(xlabel="distance r (A)", ylabel="size of the energy (kcal/mol)", yscale="log", title="How fast each term fades")
right.legend(frameon=False, loc="lower left")
tidy(right, theme)
fig.tight_layout()
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
move_median, move_max = float(decisions["movement_A"].median()), float(decisions["movement_A"].max())
margin_min_before, margin_min_after = float(manifest["relaxation"]["smallest_margin_before"]), float(manifest["relaxation"]["smallest_margin_after"])
fe_change_max = float(fe_change.max())
'''

DECISIONS_TABLE = '''
# The decisions ligand by ligand (the table that the rule above is checked against): a result for the blinded ligands, held back from the public edition.
show_table(decision_table.astype(object).where(decision_table.notna(), "-"), floatfmt=".2f")
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

**The relaxation.** The poses that co-folding models write are not always physically sensible: atoms can sit a little too close, bonds can be a little too long. The PoseBusters paper reports that a short energy minimisation with a force field repairs most of the poses that a deep-learning docking model leaves invalid. A **force field** (met in 3.5 and 10.5 as the way a shape is relaxed) writes the energy of a molecular system as a sum of simple terms: the stretching of bonds and the bending of angles as springs, the twisting of torsions, and between atoms that are not bonded the **Lennard-Jones** term and the Coulomb term. The Lennard-Jones term is $E = 4\\varepsilon\\left[(\\sigma/r)^{12} - (\\sigma/r)^{6}\\right]$ for two atoms at distance $r$, where $\\sigma$ is the distance at which the repulsion and the attraction cancel in energy ($\\varepsilon$ is the depth of the well): steeply repulsive when atoms are closer than the sum of their radii, weakly attractive beyond it. The Coulomb term, $q_i q_j / (4\\pi\\varepsilon_0\\,\\epsilon_r\\,r)$, is the force between the partial charges of the atoms, divided by the relative **dielectric constant** $\\epsilon_r$ of the medium, which weakens it. To **minimise** the energy is to move the atoms downhill until nothing moves; the method of the project is L-BFGS, which follows the slope of the energy and keeps a memory of the last steps to estimate the curvature. The next cell puts numbers into all three ideas for one pair of atoms, before the relaxation itself is used.
"""),
    code(FORCE_TERMS),
    md("""
**The two terms, worked on one pair of atoms.** The first table puts numbers into the terms for a carbon-carbon contact, with a well depth $\\varepsilon$ of 0.1 kcal/mol and a $\\sigma$ of 3.4 Å, and for two partial charges, $q_i = +0.3$ and $q_j = -0.4$, counted in units of $e$, the charge of one electron (a kcal/mol is the unit in which force fields write energy: the energy of one mole of such pairs, in kilocalories). Read it from the top. At $r = 2.4$ Å the repulsive half of the Lennard-Jones term is {{=term_table.loc[0, 'repulsive part']:.1f}} kcal/mol and the attractive half only {{=-term_table.loc[0, 'attractive part']:.1f}}, so the sum is {{=term_table.loc[0, 'Lennard-Jones sum']:.1f}} kcal/mol, a hard push apart. At $r = \\sigma$ = 3.4 Å the two halves are equal and opposite and the sum is zero, which is what $\\sigma$ means. The sum is lowest, $-\\varepsilon$ = -0.1 kcal/mol, at the distance $r_{\\min}$ = {{=r_min_formula:.2f}} Å, and beyond it the repulsive half has died out and only a weak pull is left, {{=-term_table.loc[5, 'Lennard-Jones sum']:.3f}} kcal/mol in size at 6 Å and {{=-term_table.loc[6, 'Lennard-Jones sum']:.4f}} at 10 Å. The place of the bottom is not a fitted number. It is where the slope of the energy is zero,

$$\\frac{dE}{dr} = 4\\varepsilon\\left[-12\\,\\frac{\\sigma^{12}}{r^{13}} + 6\\,\\frac{\\sigma^{6}}{r^{7}}\\right] = 0 \\;\\Longrightarrow\\; r^{6} = 2\\sigma^{6} \\;\\Longrightarrow\\; r_{\\min} = 2^{1/6}\\,\\sigma ,$$

and putting $r_{\\min}$ back into the term gives $4\\varepsilon\\,(1/4 - 1/2) = -\\varepsilon$, because $(\\sigma/r_{\\min})^{6} = 1/2$. The cell also searches for the lowest value between 3 and 6 Å without the formula and requires the two to agree: {{=r_min_found:.4f}} Å by search, {{=r_min_formula:.4f}} Å by formula.

**The Coulomb term in numbers.** The energy of two charges is $q_i q_j / (4\\pi\\varepsilon_0\\,\\epsilon_r\\,r)$, where $\\varepsilon_0$ is the electric constant of the vacuum. With the charges in units of $e$, the distance in Å and the energy in kcal/mol, everything except the charges, the dielectric constant and the distance folds into one number, $e^2/(4\\pi\\varepsilon_0)$ per mole, which is {{=coulomb_constant:.2f}} kcal Å/(mol $e^2$). The cell does not type it in: it computes it from the elementary charge, the electric constant, Avogadro's number and the calorie, and requires that it equals the constant that the relaxation code writes in the units of OpenMM, the library for molecular simulation that it runs in, {{=coulomb_in_relax}} kJ nm/(mol $e^2$). At $r_{\\min}$ the pair has {{=term_table.loc[3, 'Coulomb, dielectric 1']:.1f}} kcal/mol with $\\epsilon_r = 1$ (a vacuum) and {{=term_table.loc[3, 'Coulomb, dielectric 4']:.1f}} kcal/mol with $\\epsilon_r = 4$, the value that the relaxation uses (`DIELECTRIC` in `cypmode/validation/relax.py`; bulk water at room temperature is about 78). The minus sign is the attraction of opposite charges: two charges of the same sign give the same size with a plus sign, a repulsion. Two things set the term apart from the Lennard-Jones term. For charged atoms it is much larger: at $r_{\\min}$ it is {{=abs(term_table.loc[3, 'Coulomb, dielectric 4']) / well_eps:.0f}} times the depth of the Lennard-Jones well even with $\\epsilon_r = 4$. And it reaches far: it falls as $1/r$ while the Lennard-Jones attraction falls as $1/r^{6}$, so at 10 Å the Coulomb energy is still {{=-term_table.loc[6, 'Coulomb, dielectric 4']:.2f}} kcal/mol in size when the Lennard-Jones sum is {{=-term_table.loc[6, 'Lennard-Jones sum']:.4f}}. The right panel of the figure shows the two rates on a logarithmic axis.

**Minimising, step by step.** The second table is minimisation done in the simplest possible way. The push on the atom is minus the slope of the energy, $-dE/dr$, and it points downhill. The cell starts the atom at 2.4 Å, where the push is {{=descent_table.iloc[0, 3]:.1f}} kcal/mol/Å, and repeats one rule: move by a small **step size** times the push, here 0.002 Å for every kcal/mol/Å of push. The first move is {{=descent_table.iloc[1, 1] - descent_table.iloc[0, 1]:.3f}} Å, to {{=descent_table.iloc[1, 1]:.2f}} Å, and the energy falls from {{=descent_table.iloc[0, 2]:.1f}} to {{=descent_table.iloc[1, 2]:.1f}} kcal/mol. The next pushes are smaller ({{=descent_table.iloc[1, 3]:.1f}}, {{=descent_table.iloc[2, 3]:.1f}}, {{=descent_table.iloc[3, 3]:.1f}}) because the slope flattens as the atom leaves the repulsive wall. This is gradient descent, the rule of 4.1, which there lowers the loss of a model by moving its weights and here lowers an energy by moving the position of an atom: the slope of the energy with respect to the coordinates is called its gradient, and in one dimension it is the ordinary slope. The rule stops when the push falls below 0.001 kcal/mol/Å, after {{=n_descent}} steps, at {{=r_descent:.4f}} Å, which is {{=r_min_formula - r_descent:.4f}} Å short of the bottom. Thousands of steps are needed because the well is nearly flat at the bottom: its curvature there (how fast the slope changes) is {{=curvature_at_bottom:.2f}} kcal/mol/Å², so a step of {{=descent_step}} removes only about {{=descent_step * curvature_at_bottom:.2%}} of the distance that is left.

The third table shows why the step size cannot simply be made larger. A step of {{=step_table.loc[1, 'step size (A per kcal/mol/A)']}} reaches the bottom in {{=step_table.loc[1, 'steps until the push is below 0.001']}} steps. A step of {{=step_table.loc[2, 'step size (A per kcal/mol/A)']}} throws the atom to {{=step_table.loc[2, 'distance after the first step (A)']:.1f}} Å with its first move, far past the bottom, where the pull is weak, and it needs {{=step_table.loc[2, 'steps until the push is below 0.001']}} steps to crawl back. A step of {{=step_table.loc[3, 'step size (A per kcal/mol/A)']}} throws it to {{=step_table.loc[3, 'distance after the first step (A)']:.1f}} Å, where the push is already below the stopping value, so the rule stops after {{=step_table.loc[3, 'steps until the push is below 0.001']}} step, {{=step_table.loc[3, 'distance from the bottom (A)']:.1f}} Å from the bottom, as if it had arrived. A fixed step is right for one steepness only, and a well is steep on one side and flat on the other. L-BFGS removes the choice: it estimates the curvature from its last few steps and sets the length of each step itself, and from the same start it stops at {{=r_bfgs:.4f}} Å after {{=n_bfgs}} iterations, where gradient descent needed {{=n_descent}} steps. (The cell hands the exact slope to the routine; the real relaxation obtains it from the force field.)

Relaxing a pose must repair it and not re-dock it. The project's version (`cypmode/validation/relax.py`, in OpenMM, run in a conda environment of its own) keeps the protein and the heme fixed, so that they only supply the nonbonded terms; uses Amber ff14SB for the protein, the CHARMM template of heme b for the heme's nonbonded terms, and OpenFF 2.2.1 for the ligand, with AM1-BCC-like partial charges from a graph network (NAGL); sets the dielectric constant to 4; ties every heavy atom of the ligand to its predicted position with a weak spring (a **tether**); and, when a nitrogen of the ligand is within 3 Å of the iron in the prediction, holds that distance with a stiff spring. The names in that list are those of ready-made sets of numbers for the terms of a force field: Amber ff14SB is made for the atoms of proteins, CHARMM supplies the numbers for the heme (the project takes only its nonbonded part), and OpenFF, the Open Force Field, is made for drug-like molecules (its version 2.2.1 is called Sage). AM1-BCC is a recipe that gives the atoms of a molecule their partial charges from a fast approximate quantum calculation (AM1) plus a correction for each kind of bond (BCC, bond charge corrections), so that they approach the charges of a slower calculation; NAGL is a graph neural network (4.8) that has learned to give such charges without running the calculation, which is why the charges are called AM1-BCC-like. The tether is what keeps the repair local. The cell is the idea in one dimension: one ligand atom 2.4 Å from a frozen receptor atom, the radii of the two adding up to 3.4 Å, so that its relative distance is 2.4 / 3.4 = 0.71 and it fails the check, minimised for several strengths of the tether.
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
    code(DECISIONS_TABLE, held_back=True),
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
    figure("struct_relaxation_margin"),
    figure("struct_official20_pools", held_back=True),
    figure("struct_official20_recipe", held_back=True),
]

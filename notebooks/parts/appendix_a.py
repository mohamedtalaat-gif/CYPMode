"""Appendix A: the practical walkthrough. The environments, the map of the repository, every stage as a runnable path, the list of the queue jobs (generated from the jobs themselves), how a candidate is reproduced, and how a model or a structure lever is added through a gate."""

from notebooks.notebook_builder import code, md

ENVIRONMENTS = '''
# The environments, read from the files that pin them (environment/). A pinned entry is a package with the exact version it was installed at (name==version in a pip file, name=version in a conda file). The file SHA256SUMS holds the SHA-256 of
# every pin file, so that a file that changed since the checksums were written is found, and pyproject.toml lists the optional groups of packages (extras) that pip install -e ".[...]" takes.
import hashlib
import tomllib

env_dir = ROOT / "environment"
recorded_sums = {line.split()[1]: line.split()[0] for line in (env_dir / "SHA256SUMS").read_text().splitlines() if line.strip()}
env_rows = []
for path in sorted(p for p in env_dir.iterdir() if p.name != "SHA256SUMS"):
    lines = [line for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if path.suffix == ".txt":
        pinned = sum(1 for line in lines if "==" in line)
    else:
        pinned = sum(1 for line in lines if line.lstrip().startswith("- ") and "=" in line)
    env_rows.append({"file": path.name, "pinned entries": pinned, "SHA-256 equals the recorded one": hashlib.sha256(path.read_bytes()).hexdigest() == recorded_sums.get(path.name)})
env_table = pd.DataFrame(env_rows)
show_table(env_table, floatfmt=".0f")
extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"].get("optional-dependencies", {})
extras_text = ", ".join(f"{name} ({len(packages)} packages)" for name, packages in extras.items())
n_env_files, all_sums_match = len(env_table), bool(env_table["SHA-256 equals the recorded one"].all())
assert all_sums_match  # a changed pin file fails the build
'''

REPO_MAP = '''
# The map of the repository, with the number of source files in each place that holds code. The purpose of each place is written here; the count is made when the cell runs.
repo_places = [
    ("cypmode/challenge", "the TDI track: labels, features, the nested evaluation, every candidate model, the gates, the final build", "*.py"),
    ("cypmode/validation", "the structure track: Boltz-2 and OpenFold3 runs, scoring of poses, the gates, the final recipe", "*.py"),
    ("cypmode/metrics", "motif and amine detection used by the first version of the project", "*.py"),
    ("cypmode/data", "loaders of the public data sets of the first version", "*.py"),
    ("cypmode/audit", "the audit, the method report, the negative-results ledger, the checklist check, the readiness scan", "*.py"),
    ("scripts", "the queue runner and its guards, the audit and reproduction scripts, the release script", "*"),
    ("queue/jobs", "one shell script per experiment or build step, run in the order of its number", "*.sh"),
    ("tests", "the test suite", "*.py"),
    ("notebooks/parts", "the text and the code cells of this notebook", "*.py"),
    ("docs", "figures, the card and the film, with the programs that draw them", "*.py"),
]
repo_rows = []
for place, purpose, pattern in repo_places:
    folder = ROOT / place
    files = [p for p in folder.rglob(pattern) if p.is_file() and "__pycache__" not in p.parts] if folder.exists() else []
    repo_rows.append({"place": place, "what it holds": purpose, "source files": len(files)})
repo_table = pd.DataFrame(repo_rows)
show_table(repo_table, floatfmt=".0f")
n_python_files = int(repo_table.loc[repo_table["place"].isin(["cypmode/challenge", "cypmode/validation", "cypmode/metrics", "cypmode/data", "cypmode/audit"]), "source files"].sum())
n_test_files = int(repo_table.loc[repo_table["place"] == "tests", "source files"].iloc[0])
'''

JOB_LIST = '''
# The list of queue jobs, generated from the jobs themselves: the number, the name and the first comment line of every script in queue/jobs, which by the rules of the repository says what the job does and why. Nothing here is typed in; a job
# that is added to the directory is in the table the next time the notebook is built.
import re

job_rows = []
for path in sorted((ROOT / "queue/jobs").glob("*.sh")):
    comment = next((line.lstrip("# ").strip() for line in path.read_text().splitlines()[1:] if line.startswith("#")), "")
    number, _, name = path.stem.partition("_")
    job_rows.append({"job": number, "name": name, "what it does": comment[:230] + (" ..." if len(comment) > 230 else "")})
job_table = pd.DataFrame(job_rows)
show_table(job_table, floatfmt=".0f")
job_numbers = job_table["job"].astype(int)
ranges = [("000 to 099", 0, 99), ("100 to 199", 100, 199), ("200 to 299", 200, 299), ("900 to 999", 900, 999)]
range_counts = {label: int(((job_numbers >= lo) & (job_numbers <= hi)).sum()) for label, lo, hi in ranges}
n_jobs = len(job_table)
'''

PARSE_TOY = '''
# How a gate reaches the final build, on logs made for the purpose in a temporary folder. The final build (final_tdi_build.parse_gates) reads the line "CYP3A4 GATE (stacked, ...): MET" or "NOT MET" from the log of each group, under the name
# the group's pattern gives. A group of CONFIRMED_GROUPS (the neural pair) counts as passed only if the log of its 10-outer-fold replicate says MET as well; a missing replicate is a failure. A log under any other name is read by nobody.
import tempfile
from pathlib import Path

from cypmode.challenge.final_tdi_build import CONFIRMED_GROUPS, GROUPS as BUILD_GROUPS, parse_gates

MET_LINE = "CYP3A4 GATE (stacked, corrected p < 0.05 at both z0): MET\\n"
NOT_MET_LINE = "CYP3A4 GATE (stacked, corrected p < 0.05 at both z0): NOT MET\\n"
REPLICATE_MET = "CYP3A4 GATE (replicate neural, stacked): MET\\n"
gate_rows = []
with tempfile.TemporaryDirectory() as tmp:
    folder = Path(tmp)
    cases = [
        ("a group with a MET line", {"physics_experiment_CYP3A4.log": MET_LINE}),
        ("a group with a NOT MET line", {"mmp_transfer_CYP3A4.log": NOT_MET_LINE}),
        ("the neural pair, MET, no replicate log", {"aux_neural_CYP3A4.log": MET_LINE}),
        ("the neural pair, MET, replicate MET", {"aux_neural_CYP3A4.log": MET_LINE, "replicate_verdict_CYP3A4.log": REPLICATE_MET}),
        ("a MET line in a log with a name that no group reads", {"exploratory_physics_f10_CYP3A4.log": MET_LINE}),
    ]
    for label, logs in cases:
        for old in folder.iterdir():
            old.unlink()
        for name, text in logs.items():
            (folder / name).write_text(text)
        verdicts = parse_gates(folder)
        gate_rows.append({"case": label, "groups the build sees": len(verdicts), "groups that pass": ", ".join(f"{group} ({isoform})" for (group, isoform), v in verdicts.items() if v.get("stacked")) or "none"})
parse_table = pd.DataFrame(gate_rows)
show_table(parse_table, floatfmt=".0f")
n_build_groups = len(BUILD_GROUPS)
confirmed_text = ", ".join(CONFIRMED_GROUPS)
assert parse_table.loc[0, "groups that pass"].startswith("physics") and parse_table.loc[1, "groups that pass"] == "none"
assert parse_table.loc[2, "groups that pass"] == "none" and parse_table.loc[3, "groups that pass"].startswith("neural")
assert parse_table.loc[4, "groups the build sees"] == 0
'''

LEVER_NAMES = '''
# The gates the structure recipe reads, with their recorded verdicts (final/recipe.json). A new lever for the structure track becomes one more line "STRUCTURE GATE <name>: MET|NOT MET" in the log of its own job, and one more entry in the
# mapping of cypmode/validation/final_recipe.py, which is the only place that turns verdicts into choices.
import json

structure_gate_names = list(json.loads((ROOT / "data/structure_track/outputs/cyp3a4/final/recipe.json").read_text())["gates"])
n_structure_gates = len(structure_gate_names)
structure_gate_text = ", ".join(structure_gate_names)
'''

CELLS = [
    md("""
# Appendix A. Practical walkthrough

This appendix is the path through the repository for a reader who wants to run it, check it or extend it. It is not a second document next to the notebook: every table in it is produced from the repository when the notebook is built, so that it cannot describe a repository that is not the one beside it. A.1 says how the software is installed and pinned. A.2 is the map of the directories. A.3 gives every stage of the project as the command that runs it. A.4 lists every job of the queue. A.5 shows how the final build reads the verdict of a gate, on logs made for the purpose. A.6 and A.7 say how a model for the TDI track and a lever for the structure track are added, and what must be written before either is run.

## A.1 Environments and pins

> **Field:** Software engineering · Reproducible research

Software changes: a package that is updated a year later can give a different number from the same code. The remedy is to **pin** the environment, which means to record the exact version of every package that was installed, in a **lock file** that anyone can install from. The repository holds four pin files in `environment/`: `requirements-lock.txt` for the main environment (the TDI track and the tests), `requirements-lock-ml.txt` for the neural models (Chemprop, TabICL, TabPFN, CheMeleon), `openmm-environment.yml` for the force-field relaxation of 11.10 and `ost-environment.yml` for the scoring with OpenStructure. The last two are conda environments, which can install programs that are not Python packages. Boltz-2 and OpenFold3 run in **virtual environments** of their own (a folder with its own copy of Python and its own packages, so that installing one tool cannot break another), `.venv-boltz` and `.venv-openfold3`, which are outside the pin files; that is a gap in the record, and the versions that ran are in the manifests of the structure track.

The next cell reads the files in `environment/`, counts the pinned entries of each and checks each file against the checksum recorded in `SHA256SUMS`. The optional groups of packages that `pip install -e ".[...]"` accepts are listed from `pyproject.toml`.
"""),
    code(ENVIRONMENTS),
    md("""
There are {{=n_env_files}} pin files, and every one equals its recorded checksum (the cell requires it). The optional groups of the package are {{=extras_text}}. The quick start of the README installs the main environment with `pip install -e ".[challenge,validate]"`; the other environments are created from their own files. A job whose environment is missing does not run: its first lines (`need_file`) make it wait, so that a missing tool is a visible delay and not a wrong result.

The reason for several environments is practical. A package such as LightGBM, which runs on every thread of the processor, and a package such as PyTorch, which also uses a library for parallel loops, can disturb each other when both are loaded in one process; the repository learned this when a test of the Atom-JEPA readout, run beside the LightGBM tests, hung for hours. 12.3 tells the story; the consequence here is that the tests that load PyTorch or the quantum-chemistry package run in processes of their own, and `CONTRIBUTING.md` says which.

## A.2 The map of the repository

> **Field:** Software engineering · Reproducible research

The repository has the code of two tracks, the tools that check them, the queue that runs them, and the text. The next cell lists the places that hold code, says in a line what each holds, and counts the files, so that the map and the repository cannot drift apart.
"""),
    code(REPO_MAP),
    md("""
The code of the five packages is {{=n_python_files}} Python files, and the test suite is {{=n_test_files}} files. The data are not in this table because they are not code: `data/tdi_track/inputs` and `data/structure_track/inputs` hold the files of the challenge and the public data that were added to it, the `outputs` folders hold what the jobs wrote (the caches of the nested evaluation, the logs the gates read, the final candidate files), and `submissions/` holds the files in the form the portal takes. Appendix D, printed by the builder, maps every one of these files to the section of this notebook that explains it.

## A.3 Every stage as a runnable path

> **Field:** Software engineering · Reproducible research

The project is run in the order that follows; each command is resumable, which means that it keeps what it has finished and a second run continues where the first stopped.

```bash
python3 -m venv .venv && source .venv/bin/activate        # 1. the main environment
pip install -e ".[challenge,validate]"
python -m pytest tests -q                                  # 2. the tests (the isolated files as CONTRIBUTING.md says)
bash scripts/reproduce_tdi.sh                              # 3. raw challenge files -> validated TDI candidate files
bash scripts/queue_start.sh                                # 4. the queue: every other experiment, in the order of its number
bash scripts/queue_status.sh                               #    what is done, running, waiting
python -m cypmode.audit.checklist --status                 # 5. what the checklist says is done
bash scripts/run_audit.sh                                  # 6. the audit, generated twice and compared byte for byte
python -m cypmode.audit.method_report                      # 7. the method report and the negative-results ledger
python -m cypmode.audit.public_readiness                   # 8. the scan for what must not be published
python -m notebooks.build_notebook                         # 9. this notebook
```

Step 3 is the TDI path alone, from the raw files to the candidate files: it builds the nested grid for both isoforms and then runs the jobs numbered below 100. The structure track is run by the jobs of the hundreds that the table in A.4 names (the Boltz-2 and OpenFold3 runs and their scoring), and it needs the environments of A.1 and, for the runs themselves, a machine with the memory that 11.9 describes. The queue runs everything in the order of the job number, one job at a time, and the final jobs (900 to 960) wait until the notebook is written, so that the audit, the method report and the film are made from the state that the text describes.

## A.4 The jobs of the queue

> **Field:** Software engineering · Reproducible research

The table is generated from the jobs. A job is a shell script; it starts with `source queue/lib.sh`, which gives it the helpers described in 12.4, and its first comment line says what it does.
"""),
    code(JOB_LIST),
    md("""
The queue holds {{=n_jobs}} jobs: {{=range_counts['000 to 099']}} numbered below 100 (the TDI experiments and the final TDI build), {{=range_counts['100 to 199']}} from 100 to 199 and {{=range_counts['200 to 299']}} from 200 to 299 (the structure track, the additions of 9.9 and 9.10, the housekeeping jobs), and {{=range_counts['900 to 999']}} from 900 (the final chain). A job that is finished leaves an empty file with its name in `queue/state/done`; a job that is not ready ends with the status 75 and is tried again in the next round; any other failure is tried once more and then recorded in `queue/state/failed`. 12.4 explains why the queue is built so.

## A.5 How a gate reaches the final build

> **Field:** Software engineering · Reproducible research

A new idea becomes part of the final model only through the line that its log writes: "CYP3A4 GATE (stacked, corrected p < 0.05 at both z0): MET" or "NOT MET". The next cell shows how the final build (`cypmode/challenge/final_tdi_build.py`) reads those lines, on logs made for the purpose in a temporary folder, including the two rules that matter.
"""),
    code(PARSE_TOY),
    md("""
The final build knows {{=n_build_groups}} groups, each with the name of the log it reads; it forms the candidate sets of Part VIII, evaluates them by the same nested procedure as everything else and keeps the best eligible one. The groups in `CONFIRMED_GROUPS` ({{=confirmed_text}}) pass only with a replicate that is also MET, and a log under a name that no group reads is not read at all.

The first two rows are the plain rule. The third and fourth are the rule for a group that was found after the 15-fold results had been looked at (9.9): the gate alone is not enough, the replicate on the 10-outer-fold plan has to say MET as well. The last row is the one that makes a second look at an idea safe: a log named `exploratory_...` is written, listed in the ledger and read by no parser, so that however it comes out it cannot change the model. The queue has one such pair of jobs, the replicate of the Atom-JEPA readout (9.10), and it is written exactly so.

## A.6 Adding a model to the TDI track through a gate

> **Field:** Software engineering · Reproducible research

A model for the TDI track is added in four steps, and the order is the point.

1. **Declare.** Before any number of the new model exists, the item of `SUBMISSION_CHECKLIST.md` says what it is, where it comes from, the comparison that decides it (alone, and added to the stack, on both isoforms), and the rule it must pass. The commit that holds the declaration is dated before the commit that holds its result, and the history is never squashed, so that the order can be checked.
2. **Register.** The model is written as a candidate (`Candidate` in `cypmode/challenge/nested_runners.py`; the torch-side ones in `nested_runners_torch.py`, the cached ones in `cached_candidates.py`), added to a group, and the group gets its log name in `final_tdi_build.GROUPS`. The test suite checks that every registered block computes on a small set of molecules and the tests of the final build check the registered groups and the names of their logs.
3. **Run.** Two queue jobs: one that fills the cache of predictions for every task of the nested plan (in time-boxed chunks, resumable), and one that runs `python -m cypmode.challenge.aux_experiments --experiment <name> --isoform <iso>`, which compares the model alone and added to the default stack and writes the gate line.
4. **Let the build decide.** Nothing is done by hand: the next run of the final build reads the line, and the model joins the stack or it does not. A model that fails stays in the ledger of negative results with its numbers.

## A.7 Adding a lever to the structure track

> **Field:** Software engineering · Reproducible research

A lever for the structure track (a restraint, a way of sampling more poses, a rule for choosing among poses) follows the same logic with different files. The lever is a change to the input or to the selection (`build_input.py`, `pose_selection.py` and their neighbours in `cypmode/validation`); a job runs it on the crystal-anchored benchmark and scores it; the scoring job writes one line "STRUCTURE GATE <name>: MET|NOT MET" with the rule that was written before the run; and `final_recipe.decide` maps the verdicts to the choices of the final recipe. The next cell lists the gates that the recipe of this project reads.
"""),
    code(LEVER_NAMES),
    md("""
The recipe reads {{=n_structure_gates}} gates: {{=structure_gate_text}}. A lever that is MET switches something on in the recipe; a lever that is NOT MET changes nothing, which is what the recipe of 11.10 shows. Every pose of every pool is audited under the scorer's own PoseBusters call (11.10), so that a lever cannot buy accuracy with a pose that the scorer would reject.

*On what actually ran.* The cells above read `environment/`, `pyproject.toml`, `queue/jobs/` and `final/recipe.json`, and call `final_tdi_build.parse_gates` on logs made in a temporary folder. The commands of A.3 are shown, not run, by the notebook; the audit (12.2) and the test suite (12.3) are what run them.
"""),
]

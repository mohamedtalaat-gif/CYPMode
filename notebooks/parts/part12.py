# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part XII (first part): engineering for trust. Reproducibility (12.1), the audit (12.2) and the test suite with its continuous-integration workflow (12.3)."""

from notebooks.notebook_builder import code, md

FLOAT_TOY = r'''
# Why the same numbers added in a different order give a different total. A computer stores a real number in 64 bits, which is about sixteen significant digits, and rounds after every addition; rounding is not associative, so (a + b) + c and a + (b + c)
# can differ in the last digit. The first table is the classic three-number case. The second adds the same million numbers one by one in five different orders, and compares each total with math.fsum, which tracks the rounding error and returns
# the exactly rounded sum. The third splits the million numbers into four parts, as four threads would, and combines the four partial sums in two different orders.
import math

orders_rng = np.random.default_rng(12)
million = orders_rng.normal(size=1_000_000)
small_cases = pd.DataFrame({"expression": ["(0.1 + 0.2) + 0.3", "0.1 + (0.2 + 0.3)"], "value": [repr((0.1 + 0.2) + 0.3), repr(0.1 + (0.2 + 0.3))]})
show_table(small_cases, floatfmt=".0f")
exact_total = math.fsum(million)
order_rows = []
for label, values_in_order in {"as generated": million, "reversed": million[::-1], "ascending": np.sort(million), "smallest magnitude first": million[np.argsort(np.abs(million))], "shuffled": orders_rng.permutation(million)}.items():
    running = 0.0
    for value in values_in_order.tolist():
        running += value
    order_rows.append({"order of the additions": label, "total": running, "error against fsum": running - exact_total})
order_table = pd.DataFrame(order_rows)
show_table(order_table, floatfmt=".3e")
quarters = [math.fsum(part) for part in np.array_split(million, 4)]
thread_orders = {"threads finish in order 1, 2, 3, 4": ((quarters[0] + quarters[1]) + quarters[2]) + quarters[3], "threads finish in order 4, 3, 2, 1": ((quarters[3] + quarters[2]) + quarters[1]) + quarters[0], "pairs, then the pairs": (quarters[0] + quarters[1]) + (quarters[2] + quarters[3])}
thread_table = pd.DataFrame({"how the partial sums are combined": list(thread_orders), "total": list(thread_orders.values()), "error against fsum": [v - exact_total for v in thread_orders.values()]})
show_table(thread_table, floatfmt=".3e")
largest_order_error, number_of_distinct_totals = float(order_table["error against fsum"].abs().max()), int(order_table["total"].nunique())
assert number_of_distinct_totals > 1  # the order matters, or the lesson is wrong
'''

DETERMINISM_RECORD = r'''
# What the repository itself recorded about reproducibility. Nothing is typed in: the line of the checklist that records the check of every comparison after the stack was made deterministic (8.3), the determinism section of the audit report
# (the tests that the audit runs again every time), and the log of the GPU speed test of the CheMeleon fine-tuning, where determinism was required and the machine could not give it.
import re

checklist_text = (ROOT / "SUBMISSION_CHECKLIST.md").read_text()
n_recomputed = int(re.search(r"\((\d+) stacked comparisons\)", checklist_text).group(1))

audit_text = (ROOT / "audit/AUDIT_REPORT.md").read_text()
determinism_block = re.search(r"## 3\. Determinism and cache consistency\n(.*?)\n## ", audit_text, re.S).group(1)
table_row = re.compile(r"^\| (.*?) \| (PASS|FAIL|INFO|SKIP) \| (.*) \|$")
determinism_rows = [list(m.groups()) for m in (table_row.match(line) for line in determinism_block.splitlines()) if m]
determinism_table = pd.DataFrame(determinism_rows, columns=["check", "result", "detail"])
show_table(determinism_table, floatfmt=".0f")
n_determinism_checks, n_determinism_pass = len(determinism_table), int((determinism_table["result"] == "PASS").sum())
seed_line = re.search(r"Seed registry: (\d+) distinct", audit_text)
n_seed_entries = int(seed_line.group(1))

speed_log = (ROOT / "data/structure_track/outputs/cyp3a4/logs/chemeleon_speed.log").read_text()
gpu_operation = re.search(r"(\w+) does not have a deterministic implementation", speed_log).group(1)
'''

MD_12_1 = """
## 12.1 Reproducibility: seeds, threads, solvers and archives

> **Field:** Software engineering · Mathematics

*Where this section sits:* Part XI ended with a file that is submitted. Part XII asks what a reader should need in order to believe the numbers in the other parts, and the first answer is that the numbers can be produced again. A result that appears once and cannot be produced again is an anecdote; a result that can be produced again, from the code and the data in the repository, can be checked by anyone, including by the author a month later.

**What reproducible means here.** The strong form is that the same code, the same data and the same settings give the same bytes. A weaker form, the same numbers to a stated precision, is enough for a table but not for a hash, and a hash is what the project uses wherever it can: the SHA-256 of a submitted file is quoted in the method report, so that the file can be recognised. Four things stand between the two forms, and each has a place in the repository.

**Seeds.** A **pseudo-random number generator** is a fixed recipe that turns one number, the **seed**, into a long sequence that looks random; the same seed gives the same sequence. Every random choice in the project (the split of the molecules into folds, the starting point of a model, the order in which a solver visits its variables) is drawn from a generator with a seed written in the code. The nested evaluation of Part VI uses seed 0, three repeats, five outer and four inner folds, and the plan of the folds is saved with an identifier that is computed from the raw files, so that a plan rebuilt from the same files is recognised as the same plan.

**Floating-point arithmetic.** The second source is not randomness at all. A computer keeps a real number to about sixteen significant digits and rounds after every operation, and rounding does not respect the rule that the order of additions does not matter: $(a+b)+c$ is not always $a+(b+c)$. The next cell shows it on the smallest case, on a million numbers added in five different orders, and on the same numbers split into four parts, as four threads would split them.
"""

MD_12_1_B = """
Every order gives a different total ({{=number_of_distinct_totals}} different totals from five orders; the largest error against the exactly rounded sum is {{=largest_order_error:.1e}}), and none is wrong. This matters because a library that uses several **threads** (streams of work that the processor runs at the same time) splits a sum into parts, one per thread, and combines the parts in the order in which the threads finish, which can change from one run to the next. The third table does that with four parts and three orders of combination, and they differ too. The consequence for the project is a rule, not a remedy: the thread count is fixed where it can be, results are compared to a tolerance and not for equality when threads are used, and the audit reports whether a model fitted with one thread and with four gives the same predictions (it is listed as information and not required, because the library does not promise it).

**A solver that visits its variables in a random order.** The third source cost the project something, and 8.3 tells it in full. The meta-learner of the stack (8.2) is a logistic regression of the outcome on the scores of the candidates, solved by a **coordinate method**: it improves one coefficient at a time, in an order drawn at random, until the improvement is smaller than a **tolerance**. With the library's default tolerance the point at which it stops depends on the order, so two processes ended at slightly different blends of the same candidates, and the outer AUC and MCC of the stack differed from one process to the next by more than many of the differences that the gates of Part IX test. The problem is convex, so every order leads towards the same lowest point, and a tight tolerance with a fixed seed makes every order end there. It was the byte-for-byte comparison of the audit (12.2) that showed the difference.

**Archives and environments.** Two more sources are outside the arithmetic. An archive records the time at which each file was written, so that the same files packed twice give two different hashes; the writer of the submitted archive fixes the time of every entry (11.10), and the cell there shows both cases. And a package that is updated changes numbers without any change to the code, which is why the environments are pinned in files with their own checksums (Appendix A, A.1).

The next cell reads the records of all this from the repository: the determinism section of the audit, whose checks are run again each time, the seed registry, and the log of one run where determinism could not be had.
"""

MD_12_1_C = """
The repair of the solver was followed by the work it required. Job 084 evaluated again, from the prediction caches (nothing was refitted), every comparison that goes through the stack: {{=n_recomputed}} stacked comparisons. No verdict changed, and the stacked numbers moved by less than a thousandth. The lesson drawn is the one the audit encodes: a comparison between two models is only as good as the repeatability of each of them, so repeatability is tested before the comparison is believed.

The determinism section of the audit has {{=n_determinism_checks}} lines and {{=n_determinism_pass}} of them are PASS; the others are information that the audit reports without requiring it. They cover the feature blocks computed twice, the folds rebuilt from the raw files, a model fitted twice with the same seed and with one and with four threads, and a cached prediction compared with a fresh fit of the current code (they agree to about the eighth decimal, the precision of the float32 numbers in the cache). The seed registry of the audit lists {{=n_seed_entries}} distinct (file, kind, value) entries, and a seed that is not written where the audit can see it is not allowed.

*Where determinism could not be had.* One of the experiments of Part IX asked for the strongest form of it, and the machine could not give it. The speed test of the CheMeleon fine-tuning (9.10) trains with deterministic algorithms required, which makes the library refuse any operation that has no deterministic implementation; on the Apple GPU the operation `{{=gpu_operation}}` has none, and the run stopped at once. No GPU time was measured. The decision that followed was about cost on the processor, recorded in the checklist, and not a claim about the method; the log is the evidence that the stop was about the operation and not about speed.

**The record of the push.** A number is reproducible only if the repository that reaches a reader is the one the author had. `scripts/mirror_check.sh` checks that after a push: it fetches the remote branch and fails when the working tree differs from HEAD in any way (a modified, an untracked or a missing file), or when `origin/<branch>` is not at HEAD, and it prints the files that stay local on purpose, those listed in `.gitignore`. It is the other end of the chain that the pinned environments and the fixed seeds begin: the numbers can be produced again from the files, and the files are the ones that were pushed. The script is not run by the notebook, which is built before the push; its check is described here from the script itself.

*On what actually ran.* The first cell is a toy made for the lesson. The second reads the checklist, the audit report and the log of the speed test.
"""

AUDIT_SECTIONS = r'''
# The audit report, read as it stands: the counts of its summary, the checks that did not pass, and its sections with the number of lines of results in each. The report is generated by cypmode/audit/report.py in the final chain (job 900); this cell
# reads the file that the chain produced, so that the text below describes the audit of the state that is submitted. The size of a section is its number of non-empty lines.
import re

report_text = (ROOT / "audit/AUDIT_REPORT.md").read_text()
counts = dict(re.findall(r"(PASS|FAIL|INFO|SKIP) = (\d+)", re.search(r"Counts: .*", report_text).group(0)))
n_pass, n_fail, n_info = int(counts["PASS"]), int(counts["FAIL"]), int(counts["INFO"])
n_checks = n_pass + n_fail + n_info + int(counts.get("SKIP", 0))
summary_block = re.search(r"## Check summary\n(.*?)\n## ", report_text, re.S).group(1)
table_row = re.compile(r"^\| (.*?) \| (PASS|FAIL|INFO|SKIP) \| (.*) \|$")
summary_rows = [list(m.groups()) for m in (table_row.match(line) for line in summary_block.splitlines()) if m]
failed_checks = pd.DataFrame([r for r in summary_rows if r[1] == "FAIL"], columns=["check", "status", "detail"])
section_names = re.findall(r"^## (\d+b?\. .+)$", report_text, re.M)
section_rows = []
for name in section_names:
    block = re.search(rf"^## {re.escape(name)}\n(.*?)(?=^## |\Z)", report_text, re.M | re.S).group(1)
    section_rows.append({"section": name[:78], "lines": sum(1 for line in block.splitlines() if line.strip())})
section_table = pd.DataFrame(section_rows)
show_table(section_table, floatfmt=".0f")
n_sections = len(section_table)
if len(failed_checks):
    print("checks that do not pass in the report:")
    show_table(failed_checks, floatfmt=".0f")
fail_sentence = "Every check passes." if n_fail == 0 else f"{n_fail} checks do not pass; they are printed above with their detail, and the report lists them first."
'''

MD_12_2 = """
## 12.2 The audit: what it checks and how it is itself checked

> **Field:** Software engineering

*Where this section sits:* 12.1 explained why a result must be producible again. The audit is the program that produces it again, on purpose, every time the project is finished, and compares what it gets with what is written down.

**What it is.** `python -m cypmode.audit.report` regenerates a full report from the current code, data and caches. Every number in the report is computed when the program runs; the only text written by hand is the wording around the numbers. The report contains no time, no duration and no temporary path, because anything that changes from one run to the next would make two runs differ for a reason that has nothing to do with the project. `bash scripts/run_audit.sh` runs the program twice and requires the two reports to be identical byte for byte. An audit that is not itself reproducible cannot vouch for anything else.

**What it checks.** The next cell reads the report and lists its sections. In order they establish what was run (versions, hashes of the inputs), that the test suite passes now and not at some earlier time, determinism (12.1), the integrity of the data (the rule that defines the label reproduces every label, the folds of a nested plan do not overlap), the baselines against the models under each evaluation population, the committee, the calibration, the chemistry evidence, the ledger of every experiment parsed from the run logs, the structure benchmark, the readiness of the submission files, the credit hygiene (no leaderboard alias used as a credit, one author of the commits), the consistency of the queue jobs with the repository, the pinned environment and the seed registry, the leakage checks of every feature block, the standing rules that a script can check, and the hashes of the final candidate files with the portal's own checks; the limitations of the method are logged in a section of their own, and the gate ledger lists every pre-registered verdict found in the run logs.
"""

MD_12_2_B = """
The report has {{=n_sections}} sections and {{=n_checks}} checks in its summary: {{=n_pass}} pass, {{=n_fail}} fail and {{=n_info}} are information. {{=fail_sentence}}

**Why a failing check is kept in view.** A check that fails is a finding, not an embarrassment, and the audit is built so that it cannot be silenced: the summary lists every check with its status, a check is added when a mistake is found (so the same mistake cannot return unseen), and no check is removed because it fails. The check that the working tree is clean, for instance, fails whenever there are files that are not committed, which is true during the work; it is in the report because a result that was produced from files that nobody recorded cannot be reproduced.

**A needed condition.** The audit needs a quiescent state. It counts files, and the second run must find the same files as the first. A file created by another program between the two runs makes them differ, and the audit then reports a failure of reproducibility that is not a failure of the code; this happened once, and the queue therefore runs the audit alone, with nobody editing the repository while it runs.

**What it found.** Its byte-for-byte comparison is what showed that the stack was not deterministic (8.3, 12.1): the report recomputes the nested results of the stack, and two reports differed. It also holds the checks that keep each correction of the project from returning: the label rule, the leakage tests of the feature blocks, the check that the files submitted are the files built. 12.5 describes the other generated documents that read the same records.

*On what actually ran.* The cell reads `audit/AUDIT_REPORT.md` as the final chain left it; the audit itself is not run by the notebook, because it runs the test suite and takes longer than a notebook should.
"""

TEST_COUNTS = r'''
# The test suite, counted from its files. A test is a function whose name starts with test_, in a file under tests/. The count is made by reading the files, not by running them; the next cell runs one small file as a demonstration.
import ast
import re

test_files = sorted((ROOT / "tests").glob("test_*.py"))
tests_per_file = {}
for path in test_files:
    tree = ast.parse(path.read_text())
    tests_per_file[path.name] = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"))
n_test_files, n_test_functions = len(test_files), sum(tests_per_file.values())
largest_files = pd.Series(tests_per_file).sort_values(ascending=False).head(6).rename("test functions").reset_index().rename(columns={"index": "file"})
show_table(largest_files, floatfmt=".0f")

pyproject_text = (ROOT / "pyproject.toml").read_text()
ignored_files = re.findall(r"--ignore=(tests/\S+?\.py)", pyproject_text)
n_ignored = len(ignored_files)
segfault_status = re.search(r"segfault, exit (\d+)", pyproject_text).group(1)
workflow_text = (ROOT / ".github/workflows/tests.yml").read_text()
workflow_steps = re.findall(r"- name: (.+)", workflow_text)
operating_systems = re.search(r"os: \[(.+?)\]", workflow_text).group(1)
n_workflow_steps = len(workflow_steps)
step_table = pd.DataFrame({"step of the workflow": workflow_steps})
show_table(step_table, floatfmt=".0f")
'''

LIVE_TEST = r'''
# One small test file, run here in a process of its own (a child process, as the isolated files of the project are), so that the summary line that pytest prints is shown. The file checks the constants of the structure track.
import subprocess
import sys

run = subprocess.run([sys.executable, "-m", "pytest", "tests/test_constants.py", "-q", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True, timeout=300)
summary_line = [line for line in run.stdout.splitlines() if " passed" in line or " failed" in line or " error" in line][-1]
print(summary_line)
assert run.returncode == 0
'''

MD_12_3 = """
## 12.3 The test suite and the continuous-integration workflow

> **Field:** Software engineering

*Where this section sits:* the audit (12.2) runs the test suite among its checks. This section says what the suite is, what it was made to catch, and the lesson about isolation that cost the project hours of machine time.

**A test** is a small program that calls a part of the code with an input whose answer is known and fails if the answer is different. The suite is the collection of them, run with one command (`python -m pytest tests`). It does three jobs. It guards a result: the test that the label rule reproduces every label, or that the folds do not overlap, fails the moment a change breaks what a number depends on. It guards a promise: the test that a block computes on a small set of molecules, the test that no file naming a blinded compound would be pushed. And it guards a mistake already made: a test is written for each defect found, so that it cannot come back unseen. The next cell counts the test functions of the suite by reading its files, lists the largest files, and reads the settings and the workflow that run them.
"""

MD_12_3_B = """
The suite has {{=n_test_functions}} test functions in {{=n_test_files}} files, the largest of them above, and the steps of the workflow are listed in the second table.

**Continuous integration** is the practice of running the tests automatically on every change, on a machine other than the author's, so that a result that depends on something only the author's machine has is found. The workflow of the repository (`.github/workflows/tests.yml`) runs on {{=operating_systems}} and has {{=n_workflow_steps}} steps, listed above: it installs the package, runs the main group of tests, runs the consistency check of the checklist (12.5), and then runs the files that cannot share a process with the others, one step each. The comments in the file record real failures: a missing package at collection time stopped the whole run, so that none of the other tests ran, and the workflow checked almost nothing without saying so. A test run that silently runs nothing is worse than none, which is why that comment is in the workflow.

**The lesson about isolation.** Two native libraries that each carry their own copy of the runtime for parallel loops (OpenMP) can disturb one another when both are loaded into one process. The suite found this the hard way, and `pyproject.toml` records it in its markers. Loading PyTorch and LightGBM together ends in a crash of the process (the status {{=segfault_status}}, a segmentation fault), every time; loading the quantum-chemistry package after a number of LightGBM-heavy tests ends in a **deadlock**, a state in which every part of the program waits for another and nothing moves, at no use of the processor. Because pytest imports every file it collects, even one it will later deselect, a marker is not enough: the files that load the second library are named in `--ignore` options (there are {{=n_ignored}}), and each is run on its own, in a process of its own, as the last steps of the workflow and the isolated runs of the queue do.

The cost of getting this wrong was seen once more, late in the project. A test of the Atom-JEPA readout, which loads PyTorch, was not on the list, and a run of the suite on the queue stopped with no output and no use of the processor for hours, holding every later job behind it. The cause was the same, the file was added to the ignore list, and the audit's test run now has a time limit, because a hang that nobody sees is the worst failure of a long unattended run.

The next cell runs one small file of the suite in a process of its own and prints pytest's summary.
"""

MD_12_3_C = """
The summary above is the line a developer reads. The suite is the cheapest of the checks of this part, and the one that runs most often: the queue runs it in the audit (job 900) and in the lint cleanup (job 295).

*On what actually ran.* The first cell counts the test functions by reading the files with Python's parser, reads the ignore list from `pyproject.toml` and the steps from the workflow file. The second cell runs `tests/test_constants.py` as a child process. The workflow itself runs on the hosting service, not in the notebook.
"""

TESTS_TABLE = """
The table below names every file of the suite and gives a line on what each checks, taken from the file's own docstring or from the names of the tests it defines. It is the complete list and not a sample: a reader can find any file of the suite here without running the builder, and the builder's coverage check keeps the list in step with the files, failing the build when a file of the tests is named nowhere (Appendix D).

| test file | what it checks |
| --- | --- |
| `test_alert_context.py` | the alert–lipophilicity interaction block: a molecule without an alert keeps its plain descriptors and gains no interaction feature, a blocking fluorine counts only when it sits next to the alert, and the block stacks one row per molecule |
| `test_amines.py` | the trialkylamine screen: the known TDI drugs match, ketoconazole's N-aryl piperazine and non-amine and delocalised nitrogens do not, and an unparseable or empty SMILES raises |
| `test_app.py` | the pure logic of the Streamlit app — validating SMILES, screening compounds, loading the real blinded test set, merging challenge results — and that every tab renders without an exception |
| `test_applicability_domain.py` | the arithmetic of the applicability domain (the bin edges, the interval of a bin, the tier that follows from it) and of the diagnostics of 9.12, on inputs made for the purpose |
| `test_artifacts.py` | the provenance record of the card, poster, film and figures: an untouched artifact built from the current numbers passes, one built before a number changed is stale, one edited after its build fails, and a missing record is reported as a failure |
| `test_atomjepa_embeddings.py` | the Atom-JEPA block reads its cache without importing torch: label-free principal components of all cached compounds, rows in the order asked for, and a missing embedding replaced by the mean |
| `test_atomjepa_head.py` | the attention readout on the cached per-atom features: it trains, finds a planted signal wherever it sits, and reports a missing cache instead of a silent zero (needs torch, so it is skipped in the lightweight environment) |
| `test_aux_blocks.py` | the predicted blocks: one finite score per evaluation row, one column per predicted reading, and never a reading or a target of an evaluation or held-out row reaching its own prediction |
| `test_basic_centre.py` | the basic-centre block: one value per column for every kind of molecule, a tertiary amine near a ring counted as a centre, the cation classes told apart, no centre for nitro or neutral molecules, and the cache keyed by canonical SMILES |
| `test_batch.py` | the batch runner: a checkpoint after each compound, a failure recorded without stopping the batch, a restart skipping what is done, and both a successful and a failed result flattened to a table |
| `test_bindingdb_auxiliary.py` | the BindingDB auxiliary data: censoring prefixes, unparseable values and extended SMILES dropped, repeated measurements collapsed to the median, and an exact overlap with the challenge removed |
| `test_bioemu2.py` | the BioEmu2 runner: the configuration of the checkpoint switched to the pure-PyTorch kernel with chunked layers and refused when it is not the file expected, the alignment rows taken with the query first, and the samples listed in sample order |
| `test_bisy_rmsd.py` | BiSyRMSD on a structure pair: zero for identical structures, invariant to a rigid transform, sensitive to a ligand-only difference, and the binding-site residues found from the real pocket |
| `test_build_input.py` | the Boltz-2 input builder: the coordinating cysteine found where the sequence puts it (absence and ambiguity raise), and the YAML matching the hand-written reference while rejecting a wrong residue or an out-of-range value |
| `test_build_summary.py` | the structure summary: the ligand SMILES read from the real input file, and compound discovery run without a local Boltz run |
| `test_card.py` | the README card: the iron–nitrogen distance from the ligand's own coordinates, the donor nitrogen marked and the one nearest the iron in the submitted pose circled, and the README pointing to the card and the film |
| `test_chemberta_features.py` | ChemBERTa embeddings: a real finite vector that tells two molecules apart, the batch cache written to disk and resumed, and one fixed-size array per row added as a block |
| `test_chemberta_significance.py` | loading of the ChemBERTa cache: a missing cache or a missing compound raises rather than passing silently, and the join to the feature table is correct |
| `test_chemeleon_embedding_tabular.py` | TabICL and TabPFN on the frozen CheMeleon embedding are registered like the other cached candidates, wait for their inputs instead of failing, and are waited for by the long Chemprop family |
| `test_chemeleon_stage.py` | the stage rule of the fine-tuned CheMeleon: stage 2 runs only above the gain floor in enough folds, and a missing or unreadable stage-1 result stops the design |
| `test_chemprop_variants.py` | the Chemprop command runs the same two variants as before when no variant is named, and the foundation-weight variants are reached only by name (jobs 224 and 225) |
| `test_combined_features.py` | the combined-feature cross-validation: every compound covered once and the expected auxiliary columns required |
| `test_committee_uncertainty.py` | the committee's uncertainty: vote entropy zero for unanimity and maximal for a split, the spread as the per-row standard deviation, and the verdict that separates new information from redundancy and from noise |
| `test_conformers.py` | conformer pools: the pocket distance ignoring rigid motion, one record per entry, poorly resolved structures and close analogs of the compound excluded, and the conformers starting at the medoid and spreading out |
| `test_constants.py` | the sequence constants: the amino-acid strings of the published length, the known residues at the published numbering, and a coordination cutoff just above the dative-bond range |
| `test_cryoem_templates.py` | the cryo-EM templates: pockets becoming arrays, one row per run, and an entry whose ligand is the compound never serving as its own template |
| `test_diversity_pilot.py` | the diversity pilot: one row per run with the options each lever needs, names mapping back to ligand and lever, and a verdict that needs a wider pool and a closer best pose for enough ligands |
| `test_docs_state.py` | the generated blocks of the README and the specification: a block rewritten between its markers and nothing else changed, a missing marker pair reported as an error, and every marker the generator writes present |
| `test_ei_sampling.py` | the expected-improvement sampling: every policy spending the same budget and picking one per ligand, extra samples given to the pool that disagrees, and the result deterministic for a seed |
| `test_external_continuous.py` | the continuous external block: how the panel's rows become targets, that challenge compounds never reach the training table, and the Tox21 table flooring inactives, averaging replicates and removing luciferase inhibitors |
| `test_figures.py` | the figure registry: an identifier registered once, a caption computed from the data, the files attributed to their figure, and the label rule and the memory formula drawn in a figure recomputed from their closed forms |
| `test_final_assembly.py` | the CYP3A4 assembly: one row per test compound, real booleans from a valid threshold, the call matching the reported threshold, and no training label reaching a prediction |
| `test_final_assembly_cyp2d6.py` | the CYP2D6 assembly: one row per test compound, booleans matching the threshold, and missing CYP3A4 labels tolerated through masking |
| `test_final_recipe.py` | the structure recipe: the gate lines of every log read, the recipe following the verdicts, the sample estimate, the run groups, and the choice of a valid consensus pose |
| `test_final_tdi_build.py` | the final TDI build: gate lines parsed per group and isoform, every group naming real candidates and a log pattern, the written files validated, reloaded and hashed, the leakage audit, and the portal's rules |
| `test_frozen_embeddings.py` | frozen-embedding blocks: the principal components of the requested width and order, a missing cache or compound reported as an error, and the reduction label-free and deterministic |
| `test_grouping.py` | the grouping: Butina puts near duplicates together and distant compounds apart, a stricter cutoff gives more and smaller clusters, and scaffolds share an identifier across substituents |
| `test_hedged_blend.py` | the hedged blend: a subset stack using only its columns, the blend as a rank average of its two parts, and the minimax rule adopting it only when its worst case is at least as good |
| `test_heme_frame_prior.py` | the heme-frame prior: the axis as the heme normal, the descriptor reading height and radius and ignoring rigid motion and the side of the porphyrin, and the prior preferring typical poses and excluding the scored compound |
| `test_knn_features.py` | the read-across features: a clean majority vote recovered, an empty reference pool handled, the fold-internal construction excluding a training row's own label and giving a validation row the full training pool, and every compound covered once |
| `test_label_confidence.py` | label confidence: the z of a shift against the threshold, the assay floor clipping the direct value, a label on the threshold carrying no confidence, and the training weights that zero the uncertain rows |
| `test_lddt_to_crystal.py` | the distance to a crystal pose: one for an identical pose, low for a pose moved away from the pocket, symmetric atoms mapped to their best permutation, the best crystal copy counted, and nothing compared without a reference |
| `test_licensing.py` | the licence layout: the texts of the licences present, the citation file, `pyproject.toml` and the notice agreeing on the licence of the code, the additional term named in the notice, the headers and the provenance records, every kind of file mapped by `REUSE.toml` to a licence that has a text, and every tracked Python file carrying its header |
| `test_library_blocks.py` | the library blocks: the compound number from its trailing digits, the panel block as the range screened on the other isoforms whatever the row order, and no panel when nothing was measured outside CYP3A4 |
| `test_mmp_transfer.py` | matched molecular pairs: the shared constant part and the direction, the features using training labels only and never the query's own row, a compound without partners, and the shrunk and signed transformation delta |
| `test_model_comparison.py` | the corrected resampled t: the Nadeau–Bengio variance, the correction making the test conservative, the degenerate inputs, and the demonstration that a naive t over repeated cross-validation manufactures significance between identical models |
| `test_motifs.py` | the motif screen: the known azole TDI drugs matched, and the mechanism classified |
| `test_multiplicity.py` | the Holm adjustment against the textbook step-down, and the parsing of the logs of the added tests |
| `test_multitask.py` | the shared-trunk network: one logit per isoform, the masked loss ignoring unlabelled examples, training producing finite predictions, cross-validation covering both isoforms, and the threshold that maximises the MCC |
| `test_multitask_no_standardization_leak.py` | a regression guard for the standardisation leak found in an audit: the detector that flags statistics computed over the full data, and each cross-validation scaling a fold with its own training statistics |
| `test_nested.py` | the nested plan: inner folds entirely inside the outer training rows and partitioning them exactly once, the outer folds partitioning the pool once per repeat, stratified and reproducible splits, a plan identity that changes with any ingredient, and deployment tasks that never leak external rows |
| `test_nested_runners.py` | the nested runners: one finite score per evaluation row, no candidate reading the labels of the rows it scores, deterministic predictions per task, the cache filled and resumed, and an end-to-end evaluation on real runner output |
| `test_nested_runners_torch.py` | the leakage guards of the torch-side nested runners: a task trains only on its own rows plus compounds outside the pool, and the predictions do not depend on evaluation or outer-validation rows |
| `test_nonneg_stack.py` | the non-negative blend: every coefficient at least zero, the fit matching the unconstrained one when that is already non-negative, the penalty switching candidates off, and the blend ranking like the logistic one |
| `test_notebook_builder.py` | the builder itself: cells in a shared namespace with their outputs recorded, a failing cell reported, the glossary rule, numbers in markdown filled from the namespace, and the contents following the headings |
| `test_octant_benchmark.py` | the Octant benchmark: the motif columns added, the potency compared by motif, a missing pIC50 excluded, and the inhibition data read from the checked-in cache without downloading again |
| `test_official_extra.py` | the organisers' extra files: the batch identifier resolved to its compound, the guard removing only blinded rows, the arrays aligned to the requested names, no blinded compound holding a reading in any official file, the audit passing on the downloaded files, and the PXR loader dropping blinded compounds |
| `test_official_posebusters.py` | the official PoseBusters call: bond orders from the template, a mismatched template refused, a pose safe only when it passes and clears the margin, the safe poses preferred, and the scorer's exact call passing a submitted pose with all its checks |
| `test_oof_bridge.py` | the out-of-fold bridge: the real inhibition file loaded, every compound predicted by a model that never saw it, the final model fitted and applied round-trip, inverse-variance weighting, and no information from a compound's own label |
| `test_openfold3.py` | the OpenFold3 runner: the Boltz input read with its ligand and not the heme, the same chain layout, the model paths in sample order, the ranking scores read, and a prediction that refuses to overlap another |
| `test_openfold3_lowmem.py` | the low-memory chunking: it matches the original chunking and the unchunked layer, and the broadcast bias is free for one sample and a full copy for more |
| `test_ost_official.py` | the OpenStructure scoring: the poses already scored read by column name even when the table is empty, and the restraint effect read with each measure |
| `test_ost_pool_gates.py` | the OpenStructure pool gates: the verdicts read from the gate lines with the last line winning, the tables getting the OpenStructure RMSD, the choice of poses to score, and the capture of what a gate prints |
| `test_pair_transfer.py` | the pair model: the neighbour table without the compound itself, the pair-matrix layout, series SAR learned and better than plain neighbour voting, and no evaluation label reaching its own prediction |
| `test_panel.py` | the cytochrome panel: its shape, the mechanism columns, an unknown isoform raising, and the summary |
| `test_pharmacophore_fingerprint.py` | the pharmacophore fingerprint: a real three-dimensional conformer, a sparse binary vector, an all-zero vector for an unembeddable SMILES, two molecules told apart, and the batch cache written and resumed |
| `test_physics_block.py` | the GFN2-xTB part of the physics block: a complete feature vector, the chemistry of the hydrogen abstraction, ionisation potentials and proton affinities in the expected order, the Fukui function on an amine, and a resumable extraction |
| `test_physics_sites.py` | the bookkeeping of the physics block that needs no quantum calculation: the carbon–hydrogen roles, the binder roles, the protonated site at the right distance, and the pharmacophore geometry |
| `test_physics_tierb.py` | the GFN2-xTB part of physics tier B: the relaxation lowering the abstraction energy while keeping the site order, complete and deterministic features, a failure giving a NaN row, and a resumable extraction |
| `test_physics_tierb_pure.py` | the pure bookkeeping of physics tier B: Boltzmann weights summing to one, the mobile region, the top sites by vertical energy, a loader refusing a missing cache, and unique feature names |
| `test_pipeline.py` | the Boltz-2 pipeline with the subprocess mocked: the correct YAML written, the output streamed, the failure paths carrying Boltz's own output, and the affinity flag passed through |
| `test_pocket_geometry.py` | pocket geometry: the ligand burial matching a known deeply buried compound, the pocket volume positive, and neither function mutating the input structure |
| `test_pose_awareness.py` | the pose-awareness terms: the substrate-window penalty, packing and clash contacts, the pharmacophore terms for Type II ligands, the within-pool ranking, and the gate needing three ligands |
| `test_pose_comparison.py` | pose comparison: an RMSD of zero for an identical pose, invariance to a rigid translation, sensitivity to a real conformational difference, and agreement with the direct call |
| `test_pose_consensus.py` | the engine consensus: OpenFold3 playing only where Boltz-2 has nothing or fails PoseBusters, the PoseBusters gate outranking agreement, and a tie going to Boltz-2 |
| `test_pose_pools_cross_engine.py` | the cross-engine pose pools: the symmetry permutations of a tert-butyl group with the identity first, a swap of equivalent atoms costing nothing, the atoms reordered to the template, and a pool mixing two atom namings compared through the SMILES |
| `test_pose_rescoring.py` | pose rescoring: the overlap one for identical point sets and symmetric, falling as a pose is translated away, penalising a size mismatch, invariant to atom order, and an unparseable SMILES giving nothing |
| `test_pose_rescoring_validation.py` | the rescoring validation: the RMSD as the root mean square of per-atom displacements, decoys as rigid motions within the shift bound covering near-native and clearly wrong poses, one copy per compound, and the crystal pose ranked above the decoys |
| `test_pose_selection.py` | pose selection: identical complexes scoring one, atoms missing from either complex ignored, the expected LDDT favouring the pose most of the pool agrees with, and a single-pose pool returned as it is |
| `test_pose_validity.py` | pose validity: a known good structure passing, the input not mutated, a bond-length check reported, and the first valid model chosen or none returned |
| `test_pubchem_auxiliary.py` | the PubChem auxiliary data: replicate rows collapsed instead of multiplied, one row per compound after the pivot, a real single-panel schema parsed, and an exact overlap including a salt form removed |
| `test_public_readiness.py` | the readiness scan: private paths, credentials and foreign e-mail addresses found, clean text giving no finding, and copied runs detected by their shingles |
| `test_quantum_pharmacophore.py` | the quantum-pharmacophore features: a real three-dimensional conformer, electronic-structure data from GFN2-xTB, finite and sane values for ethanol, two molecules told apart, and a NaN row for an unembeddable SMILES |
| `test_queue_lib.py` | the queue helpers: the cheap jobs do not wait for the long family and the final chain does, and the requeue does not wait for the jobs it re-queues |
| `test_recompute_paired_summaries.py` | the recomputation of the paired summaries: one paired table per raw table, written without refitting |
| `test_reference_ligands.py` | the reference ligands: an active-site ligand extracted with its coordinates, iron and pocket residues, a peripheral ligand refused, fragments skipped, and a structure without a heme yielding nothing |
| `test_relax.py` | the relaxation module's file handling and the arithmetic of its gate (the force-field part needs the conda environment of openmm and is exercised by a queue job) |
| `test_relax_final.py` | the relax rule at the end: a relaxed pose adopted only when every criterion holds, the zip taking the adopted poses, a second run giving the same zip, an unexpected zip never overwritten, and nothing used when the gate is not met |
| `test_release_after_deadline.py` | the hold-back: the HELD BACK block removed and everything else left, the commented film section switched on, and nothing that names blinded results would be pushed |
| `test_reliability.py` | the reliability score: zero ensemble disagreement for a deterministic signal, the Tanimoto out-of-distribution distance, and a one-class SVM score higher for a clear outlier |
| `test_replicate_verdict.py` | the replicate rule: MET only with significance in the ten-fold design and the same sign in the five-fold, and a missing log never a pass |
| `test_restraint_expansion.py` | the restraint expansion: its plan and decision rules registered, eligibility by donor distance, size and resolution, benchmark and multi-fragment ligands excluded, one row per ligand, and the test confirming a gain and detecting a loss |
| `test_selection_validation.py` | the selection validation: minimum Bayes risk picking the consensus pose, the paired gate requiring consistent improvement, the picked pose scored by its true RMSD, and the engine gate needing different errors |
| `test_series.py` | the series benchmark: the common core of two analogs, the rigid rotation, the combined pick, pairs of distinct compounds above a similarity threshold, and the official pairs symmetric and free |
| `test_softmax_selector.py` | the softmax blend: the temperature running from the best model to the equal committee, exploiting with one informative candidate and exploring with equally informative ones, and a fixed temperature as a selector of its own |
| `test_split_validity.py` | the similarity summary of a split: high when the apply set equals the training set, low for distant compounds, and the fraction above a threshold reported correctly |
| `test_stacking_significance.py` | the stacking significance: the repeated cross-validation comparison covering every repeat and fold with and without auxiliary features, the paired summary detecting a real difference, and the corrected variance with the Holm adjustment |
| `test_structure_benchmark.py` | the structure benchmark: the drug-likeness and quality criteria applied with the best copy kept, and a diverse pick that is deterministic and never returns the seed or a duplicate |
| `test_structure_benchmark_report.py` | the benchmark report: every rule reporting the true RMSD of the pose it would submit, a rule needing an engine the compound never ran on left out rather than zero-filled, the summary counting only what a rule could score, and the variants compared across suffixes and budgets |
| `test_structure_gates.py` | the structure gates: the restraint gate needing a lower first-pick RMSD and not worse validity, and the pilot gate needing both wider pools and a closer best pose |
| `test_structure_standardization.py` | structure standardisation: a salt form matching its free base, the plain canonical SMILES that would have missed the match, and no match for an unparseable SMILES |
| `test_structure_submission.py` | the structure submission: the CIF converted to PDB with the ligand residue renamed and the source not mutated, exactly twenty PDB files and a zip, no ligand residue left in an entry, and the same bytes when rebuilt |
| `test_structures.py` | the structure parsing: a molecule built from a direct atom correspondence when the order matches and refused when it does not, the ketoconazole coordinates summarised, and the motif distance ignoring a closer non-motif nitrogen |
| `test_submission.py` | the submission file: the real blinded test set loaded, the columns and their order assembled, a length mismatch rejected, and the validator accepting a correct file and rejecting a wrong row count or a missing column |
| `test_submissions_sync.py` | the submissions record: the TDI README naming its file and its hash and saying that nothing was uploaded, a failed check shown instead of hidden, the structure README listing every member, stale files moved aside and never deleted, and the differing calls counted |
| `test_tabpfn_registration.py` | TabPFN registered like TabICL as a cached candidate group, and its jobs deferring until the licence acceptance exists instead of failing |
| `test_tdi_ablation.py` | the M0 to M3 staircase: the feature vector growing with each tier, the no-amine set never containing the amine flag, the ablation over four models and both isoforms, and the paired folds |
| `test_tdi_baseline.py` | the challenge's own baseline reproduced: the real training file loaded, metrics per isoform, the summary, and a fixed seed that is reproducible and covers every compound once |
| `test_tdi_reference.py` | the reference TDI drugs: six named drugs loaded, the challenge threshold applied, the motif screen missing the non-azole inhibitors, and structural coordination filling only the known two |
| `test_tdi_targets.py` | the label rule: the shift against the floor-clipped direct value, the two formulations of the threshold agreeing, missing values kept missing, the populations tagged without mixing imputed and measured shifts, and the single threshold reproducing every given label |
| `test_threshold_tuning.py` | threshold tuning: the MCC-maximising threshold, the inner out-of-fold threshold as a training-only property, the cross-validation covering every labelled compound once, and the confusion arithmetic |
| `test_tier3_features.py` | the Tier 3 structural features: the real schema parsed, uncovered rows left as real missing rather than zero, and the comparison scored only on the covered validation rows |
| `test_tier3_selection.py` | the budgeted selection: the motif flag added, the uncertainty threshold as a training-fold property, the combination of motif and uncertainty, the validation labels never touching the policy, and exactly the budget returned |
| `test_validation_matching.py` | the validation matching: the Tanimoto matrix against its definition, the nearest neighbour ignoring the row itself, similarity weights reshaping the validation set to the test set, prior weights moving the positive rate, and the weighted MCC |
| `test_video.py` | the film: the facts reaching the page as valid JavaScript, the gates on screen excluding the exploratory rescorings, a refusal to say the audit reproduced when it did not, the scenes covering the film without gaps, and a full-HD frame |
| `test_xiso_transfer.py` | the cross-isoform transfer block: one finite score per evaluation row, other-isoform measurements of evaluation and held-out rows never read, a training compound never predicted by a model that saw its own measurement, and the correlation table |
"""

CELLS = [
    md("""
# Part XII. Engineering for trust

The other parts of this notebook say what was found. This part says why a reader should believe it, and the answer is a set of habits that make the project hard to fool, above all itself. A result that cannot be produced again is an anecdote, so 12.1 is about reproducibility, and about the small, unglamorous sources of difference between two runs that should agree. 12.2 is the audit, a program that regenerates a full report on the state of the repository and requires it to be the same twice. 12.3 is the test suite and the automatic workflow that runs it elsewhere. 12.4 is the queue that ran every job in this notebook, with its guards and what the hardware taught. 12.5 is the documents that are generated from the logs, so that no number in them is typed. 12.6 is the rules that never change, and how each is enforced by a program and not by a promise.

The reason for all of it is the one thing the project cannot do: it cannot see the labels of the compounds it is scored on. Every honest claim it makes is therefore a claim about a procedure, and a procedure is only worth what its checks are worth.
"""),
    md(MD_12_1),
    code(FLOAT_TOY),
    md(MD_12_1_B),
    code(DETERMINISM_RECORD),
    md(MD_12_1_C),
    md(MD_12_2),
    code(AUDIT_SECTIONS),
    md(MD_12_2_B),
    md(MD_12_3),
    code(TEST_COUNTS),
    md(MD_12_3_B),
    md(TESTS_TABLE),
    code(LIVE_TEST),
    md(MD_12_3_C),
]

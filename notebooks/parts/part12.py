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
    code(LIVE_TEST),
    md(MD_12_3_C),
]

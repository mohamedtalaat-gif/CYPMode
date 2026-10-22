# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part XII (second part): the job queue and what the hardware taught (12.4), the generated documents and the checks on them (12.5), and the rules that never change with their enforcement (12.6)."""

from notebooks.notebook_builder import code, md

QUEUE_HELPERS = r'''
# The helpers that every job sources (queue/lib.sh), run for real in child processes. defer ends the job with the status 75, "not ready, try again later". chunk_status is the helper of the jobs that run in time-boxed chunks: it is given the name of the
# job and the number of tasks still to do, and answers 75 when this chunk made progress (come back for the next), 1 when it made none (a failed attempt) and 0 when nothing is left. It keeps the last number in a file, so the four calls below are
# four separate processes that remember each other, in a temporary folder so that the real state of the queue is not touched.
import subprocess
import tempfile

meaning = {75: "not ready, or progress made: try again later", 1: "no progress: a failed attempt", 0: "done"}
defer_run = subprocess.run(["bash", "-c", 'source queue/lib.sh >/dev/null 2>&1; defer "the file is missing"; echo never printed'], cwd=ROOT, capture_output=True, text=True)
helper_rows = [{"call": 'defer "the file is missing"', "prints": defer_run.stdout.strip(), "exit status": defer_run.returncode, "meaning": meaning[defer_run.returncode]}]
with tempfile.TemporaryDirectory() as scratch:
    for tasks_left in (100, 100, 40, 0):
        run = subprocess.run(["bash", "-c", f"source {ROOT}/queue/lib.sh >/dev/null 2>&1; chunk_status demo_job {tasks_left}"], cwd=scratch, capture_output=True, text=True)
        helper_rows.append({"call": f"chunk_status demo_job {tasks_left}", "prints": "", "exit status": run.returncode, "meaning": meaning[run.returncode]})
helper_table = pd.DataFrame(helper_rows)
show_table(helper_table, floatfmt=".0f")
assert list(helper_table["exit status"]) == [75, 75, 1, 75, 0]
'''

STAGE_LOG = r'''
# What the runner recorded about itself, read from its own log (queue/logs/queue_stage.txt): one line for every start, every finish, every retry, every guard action. The word RUN is written before a job is tried, including a job that then answers
# "not ready", so most RUN lines are re-checks and the number to read is DONE.
import collections
import re

stage_lines = [line for line in (ROOT / "queue/logs/queue_stage.txt").read_text().splitlines() if line.strip()]
event_counter = collections.Counter(line.split()[2].rstrip(":") for line in stage_lines if len(line.split()) > 2)
first_entry = " ".join(stage_lines[0].split()[:2])
guard_swaps_gib = [float(m.group(1)) / 1024 for line in stage_lines if " GUARD" in line for m in [re.search(r"swap used ([\d.]+)M", line)] if m]
event_table = pd.DataFrame({"event": ["DONE", "RETRY", "FAILED", "GUARD", "BATTERY", "IDLE"], "meaning": ["a job finished", "a job failed once and was tried again", "a job failed twice and was recorded as failed", "the memory guard killed the running engine", "a notification about the battery", "nothing could run"],
                            "lines in the log": [event_counter.get(k, 0) for k in ("DONE", "RETRY", "FAILED", "GUARD", "BATTERY", "IDLE")]})
show_table(event_table, floatfmt=".0f")
n_done_events, n_guard, n_failed_events, n_retry = (int(event_table.set_index("event").loc[k, "lines in the log"]) for k in ("DONE", "GUARD", "FAILED", "RETRY"))
n_log_lines, max_swap_at_guard = len(stage_lines), max(guard_swaps_gib)
state_dir = ROOT / "queue/state"
n_done_now, n_failed_now = len(list((state_dir / "done").iterdir())), len(list((state_dir / "failed").iterdir()))
n_jobs_now = len(list((ROOT / "queue/jobs").glob("*.sh")))
guard_header = [line.lstrip("# ").rstrip() for line in (ROOT / "scripts/guards.sh").read_text().splitlines()[1:8] if line.startswith("#")]
print("\n".join(guard_header))
'''

MD_12_4 = """
## 12.4 The job queue, the runner and the guards; what the hardware taught

> **Field:** Software engineering

*Where this section sits:* every experiment in Parts VI to XI, and most of the figures, ran as a job of one queue on one laptop. This section says how that queue works, because the way it fails is a large part of what the project learned, and because a result is only as trustworthy as the run that produced it.

**Why a queue.** The experiments take hours, some of them days; the machine has 16 gigabytes of memory shared between its processor and its graphics unit (unified memory, 11.9) and room for one heavy job at a time; a run that goes on through the night must continue after an error and after a power cut, and nobody should have to remember what was done last. A **job queue** answers all three: the work is written down as a list of jobs, a program takes them one at a time, and what has been done is recorded in files that outlive the program.

**The jobs and the runner.** A job is a shell script, `queue/jobs/NNN_name.sh`; the number sets the order, and a job begins by sourcing `queue/lib.sh`, which gives it a few helpers. The runner (`scripts/queue_runner.sh`) goes through the jobs in the order of their names and runs the first one that is not finished; when it ends, the next starts at once. New job files are picked up while it runs, so the queue never has to be restarted. A job tells the runner what happened by its **exit status**, the number a program returns when it ends: 0 means done, and the runner records it by creating an empty file with the job's name in `queue/state/done`. The status 75 means "not ready": the code or the inputs that the job needs do not exist yet; the job is skipped, looked at again in the next round and never marked as failed. Any other status is a failure, which the runner tries once more before it records the job in `queue/state/failed` and goes on.

The distinction between not ready and failed is the most useful rule of the design. A long chain of experiments has many jobs that cannot start before another has finished; if each of them failed when its input was missing, the queue would be full of failures that mean nothing. The helpers make the difference easy to write: `need_file` ends the job with the status 75 until a file exists, `need_done` until named jobs have finished, `all_lower_done` until every job with a smaller number has finished, and `defer` for any other reason. The next cell runs the two that are easiest to see.
"""

MD_12_4_B = """
The first row is the whole of the "not ready" mechanism. The next four show the helper of the jobs that run in **time-boxed chunks**. A fit of a model for each task of a nested plan may take an hour and a half; a job that ran until it was finished would keep every shorter job behind it waiting. So such a job works for a fixed time, reports how many tasks remain and ends; the helper turns that number into a status: progress means come back later (75), the same number twice means nothing was done (1, which counts as a failure and eventually stops a job that is stuck), and zero means finished. The work is kept in a cache file per task, so the next chunk continues where the last stopped.

**Re-running and chains.** A job runs again when its flag is removed: the jobs named `requeue` (179, 188, 299) remove the flags of the jobs after a change so that they run once more, and the lint cleanup (job 295) removes the flag of the final TDI build for the same reason. This is deliberately simple: a dependency is a file that exists or does not, and a chain is a list of numbers.

**Guards.** The runner starts a second program beside it, `scripts/guards.sh`, whose comment says why each guard exists. The text printed by the next cell is that comment.
"""

MD_12_4_C = """
Each guard is the answer to something that happened. The memory guard is the one that matters: five Boltz samples in parallel used up the memory and the swap file, and the machine took a **kernel panic**, which is the operating system stopping itself because it cannot continue. From then on one sample at a time, a ceiling on what the graphics unit may take, and a guard that kills the running structure engine when the free memory stays under a few percent. A hard ceiling of 0.9 times the memory, tried first, made every forward pass of the engines fail, so the allowance was loosened and the guard does the protecting. The battery guard asks nothing of the run: a flat battery once stopped a night's run, and the guard therefore sends a notification at two levels of charge and never waits or stops anything (the function that used to wait is kept as an empty one, so that older jobs still call it).

The runner is started under `caffeinate`, which keeps the machine awake while it runs; a run that looks hung may turn out to be a machine that went to sleep, and that mistake was made once in this project, so the first question about a quiet job is whether the machine slept.

The runner's own log shows what this meant in practice. It holds {{=n_log_lines}} lines since {{=first_entry}}. {{=n_done_events}} DONE lines are jobs that finished, {{=n_retry}} are retries and {{=n_failed_events}} are failures that were recorded; the jobs that failed were repaired and queued again, and {{=n_failed_now}} of the {{=n_jobs_now}} jobs are in the failed state as this notebook is built, while {{=n_done_now}} are done. The memory guard acted {{=n_guard}} times, with the swap file as large as {{=max_swap_at_guard:.1f}} gibibytes at the worst moment; each of these would otherwise have been a crash of the machine or of the run.

**What the hardware taught, in six rules.** A job that is not ready is not a failed job (the status 75). A long job works in chunks and remembers its place in files, so a restart costs a chunk and not a day. The state lives in the repository and not in a temporary folder, which the machine empties when it restarts. The memory of a laptop is shared between its parts, so a guard must watch the free memory and not only the load. Nothing is allowed to wait for a person (the battery), and everything that can hang is given a time limit (the tests, 12.3). And a quiet job is first suspected of sleeping, not of being broken.

*On what actually ran.* The first cell runs `defer` and `chunk_status` for real, in child processes and a temporary folder. The second reads the runner's log, the counters of `queue/state` and the header of `guards.sh`. The runner itself ran the jobs; the notebook only reads what it left.
"""

REPORT_FILES = r'''
# The two generated reports and the blocks of the other documents that are generated. A value whose source does not exist yet is written as (pending: job) in METHOD_REPORT.md, so that the report cannot contain a number nobody computed; the claim-to-evidence
# check regenerates both reports in memory and fails when the files on disk differ, so a number edited by hand is a difference. The check is run here as the audit runs it.
import re
import subprocess
import sys

method_report = (ROOT / "METHOD_REPORT.md").read_text()
negative_report = (ROOT / "NEGATIVE_RESULTS.md").read_text()
report_sections = re.findall(r"^## (\d+\. .+)$", method_report, re.M)
n_pending = len(re.findall(r"\(pending:", method_report))
n_limitations = len(re.findall(r"^- \*\*L\d+\*\*", method_report, re.M))
check_run = subprocess.run([sys.executable, "-m", "cypmode.audit.method_report", "--check"], cwd=ROOT, capture_output=True, text=True, timeout=600)
report_is_fresh = check_run.returncode == 0
check_line = (check_run.stdout.strip().splitlines() or ["(no output)"])[-1]
generated_blocks = {name: re.findall(r"<!-- generated:(\S+) begin", (ROOT / name).read_text()) for name in ("README.md", "CYPMode_Architecture_Specification.md")}
block_table = pd.DataFrame({"document": list(generated_blocks), "generated blocks": [", ".join(v) for v in generated_blocks.values()], "number": [len(v) for v in generated_blocks.values()]})
show_table(block_table, floatfmt=".0f")
section_table = pd.DataFrame({"section of METHOD_REPORT.md": report_sections})
show_table(section_table, floatfmt=".0f")
negative_gates = len(re.findall(r"\| NOT MET \|", negative_report))
print(check_line)
freshness_sentence = "At the time of this build the check passes: the two files are what the code generates from the logs." if report_is_fresh else "At the time of this build the check reports the files as stale, which is what it is for: the final chain regenerates them (job 910) before the notebook is built."
'''

CHECKLIST_AND_SCAN = r'''
# The two checks that look at the repository from outside. The first is the consistency of the checklist: every job number, every code file in backticks and every module a job calls must exist, and every open item must have a job or a tag that
# says who does it. The second is the readiness scan: three of its detectors are run on a string made for the purpose (the private path, the key and the address are put together at run time, so that this notebook does not contain what the scan looks for).
from cypmode.audit import checklist as checklist_module
from cypmode.audit import public_readiness as readiness

problems_found, untagged_items, item_counts = checklist_module.check()
n_checklist_problems, n_untagged = len(problems_found), len(untagged_items)
n_items_done, n_items_open = item_counts["done"], item_counts["open"]
sample = "see " + "/Use" + "rs/someone/project/file.csv, token " + "hf_" + "a" * 34 + ", write to " + "some.one" + "@" + "example.org"
found_by_scan = readiness.scan_text(sample)
scan_table = pd.DataFrame({"detector": list(found_by_scan), "matches in the sample": [len(v) for v in found_by_scan.values()]})
show_table(scan_table, floatfmt=".0f")
n_detectors_fired = len(found_by_scan)
held_back_lines = [line for line in (ROOT / ".gitignore").read_text().split("# >>> HELD BACK", 1)[-1].split("# <<< HELD BACK", 1)[0].splitlines() if line and not line.startswith("#")] if "# >>> HELD BACK" in (ROOT / ".gitignore").read_text() else []
n_held_back = len(held_back_lines)
'''

MD_12_5 = """
## 12.5 The documents that are generated, and the checks on them

> **Field:** Software engineering · Data science

*Where this section sits:* 12.1 to 12.4 made the runs repeatable. A repeatable run is wasted if the documents that describe it are written by hand, because a number typed into a document begins to drift away from the log the moment it is typed. The answer of the project is to generate every number, and to check the generated files against the logs they came from.

**The method report and the ledger of negative results.** `METHOD_REPORT.md` and `NEGATIVE_RESULTS.md` are produced by `cypmode/audit/method_report.py` from the repository's own logs, tables and audit report. Every number in them is copied by code from one of those sources; the prose around the numbers is fixed text. A value whose source does not exist yet is written as `(pending: job)`, so the report cannot contain a number that nobody computed. The limitations are kept as data (`cypmode/audit/limitations.yaml`): the wording is written by hand, and each value in it is filled in from the audit through a named key, so that a limitation that quotes a figure quotes the figure the audit computed. The same idea reaches the other documents: the README and the architecture specification have blocks between markers (the final stack, the verdicts of the gates, the structure recipe) that the program `cypmode/audit/docs_state.py` rewrites, and the text around them is written by hand.

**Two sections of the report that show the whole.** The report holds, before its results, a section on the architecture of the two tracks, step by step, and a section on the exploration, and both are generated: the first reads the default candidates, the feature store, the blend, the call rates and the structure recipe from the code and the final manifests; the second counts the verdicts of the ledger by the kind of idea each tested (`cypmode/audit/ledger_kinds.py`, shared with the figure of the map of what was explored), gives the accounting of the number of attempts, the controls and the audit. `tests/test_method_report_sections.py` checks that the sections exist, come in order and that no family of the ledger is left without a kind.

**Provenance of what the code generates.** A generated file is only as useful as the knowledge of where it came from. `CITATION.cff` is the single source of the author, the repository, the identifiers and the licence; `cypmode/provenance.py` reads it, prints the citation when a script that generates data starts, and writes `PROVENANCE.json` beside the predictions, the feature table and the final outputs (once; an existing file is never rewritten, and the submitted files are never touched, because their SHA-256 is recorded). Audit section 20 compares the name, the repository and the DOI across the citation file, the README, `NOTICE` and the archive metadata, and `tests/test_provenance.py` checks that every script that generates data announces its origin. The programs never refuse to run: a licence, not a program, is what asks that the notice stays (`USE_AND_CITATION.md`).

**The check on them.** `python -m cypmode.audit.method_report --check` regenerates both reports in memory and fails when the files on disk differ. This is the claim-to-evidence check of the audit. A number edited by hand is a difference, and so is a report that was not regenerated after a log changed; the check does not know which, and does not need to. The cell below runs it.
"""

MD_12_5_B = """
{{=freshness_sentence}} The method report has {{=len(report_sections)}} numbered sections, {{=n_limitations}} logged limitations and {{=n_pending}} value(s) still pending, and the ledger of negative results has {{=negative_gates}} rows marked NOT MET. The README and the specification hold the generated blocks listed in the first table.

**The artifacts that present the results.** The same discipline covers the files that show the results to a reader: the README card, the poster, the film and the figures. `cypmode/audit/artifacts.py` keeps the record `audit/artifacts.json`, written by the builders of those four artifacts: for each one the SHA-256 of every file it wrote, a fingerprint of the numbers it was built from (everything that the film's facts module reads from the final outputs, the ledger, the audit and the repository), and the builder's own digests. Its `verify`, called by the pre-submission gate, asks three questions of every artifact: whether each recorded output is still the file the builder wrote, whether the numbers it shows are the final ones (the recorded fingerprint equals the current one), and whether the artifact's own source still produces the same content. The card's HTML and the DOM of every frame of the film and of the poster are recomputed and compared exactly; the PNG and the H.264 files are not compared pixel by pixel, because rendering the same frame twice differs in a few dozen pixels by less than one level of 255 and the encoder adds its own differences. The gate fails an artifact that was edited after its build, one built from numbers that have changed since, and one whose source no longer reproduces it.

**The checklist is checked too.** `SUBMISSION_CHECKLIST.md` is the contract of the project: each open item names the job that does it or says who does. It is a document, so it can promise something that does not exist. `cypmode/audit/checklist.py` makes the three ways in which it drifts from the repository checkable by a script: a job number in the text that has no job, a code file in backticks that does not exist, and a job that calls a module that does not exist (which would make it answer "not ready" for ever). It also lists the open items that have neither a job nor a tag. The workflow of 12.3 runs it on every change. In this build it finds {{=n_checklist_problems}} problems and {{=n_untagged}} untagged items in {{=n_items_done}} items done and {{=n_items_open}} open.

**What must not be published.** The public-readiness scan (`cypmode/audit/public_readiness.py`) looks at the files that git tracks or would add. Its detectors find private paths on the author's machine, strings shaped like credentials, e-mail addresses other than the public contact lines, files near the size limit of the hosting service, long verbatim runs from the books used while writing, and, until the challenge closes, results for the blinded compounds: a file that names five or more of them, or a table with one row per blinded compound (750 for the TDI test set, 20 for the structure set), is a prediction and must not be pushed. The files that are held back are the {{=n_held_back}} patterns of the HELD BACK block of `.gitignore`, and the scan can also read every blob of the git history, because a push carries the history. The table shows three detectors firing on a sample that was made for the purpose.
"""

MD_12_5_C = """
{{=n_detectors_fired}} of the detectors fired on the sample, as they must. What the scan cannot do is find what it was not taught to look for: it is a net with a mesh, and the mesh is the list of patterns. The final chain runs it (job 930) and its findings that code can fix are fixed; the others are for a person to decide, and the push is not made before they are.

*On what actually ran.* The first cell runs the claim-to-evidence check as a child process and reads the two reports and the generated blocks. The second calls the checklist check and the scan's `scan_text` in the notebook's process.
"""

RULES = r'''
# The rules that never change, read from the checklist (section 0), each with the mechanism that enforces it. The mapping from rule to mechanism is written here by hand and every mechanism it names is checked to exist; a rule that nothing
# enforces by program is labelled as a rule of procedure, because that is what it is.
import re

checklist_text = (ROOT / "SUBMISSION_CHECKLIST.md").read_text()
rules_block = re.search(r"## 0\. Rules that never change\n(.*?)\n## ", checklist_text, re.S).group(1)
rules = [re.sub(r"\s*\[.*$|\s*\(standing rule\)\s*$", "", item).strip() for item in re.findall(r"^- \[ \] (.+)$", rules_block, re.M)]
enforcement = [
    ("No submission and no push", "procedure, with a program behind part of it", ["scripts/release_after_deadline.py", "cypmode/audit/pre_submission.py"]),
    ("No leaderboard probing", "procedure; no program of the repository submits anything", ["cypmode/challenge/submission.py"]),
    ("Numbers are quoted from the corrected procedure", "the generated report and its claim-to-evidence check", ["cypmode/audit/method_report.py", "cypmode/challenge/model_comparison.py"]),
    ("Credit another team's idea", "the credit-hygiene section of the audit and the credits section of the report", ["cypmode/audit/sections.py", "cypmode/audit/method_report.py"]),
    ("Code comments in English", "the audit (section 18 for the code, section 21 for the text files) and the readiness scan", ["cypmode/audit/sections_extra.py", "cypmode/audit/public_readiness.py"]),
    ("Machine safety", "the guards of the runner", ["scripts/guards.sh", "scripts/queue_runner.sh"]),
    ("Stop rule", "the ledger of negative results", ["cypmode/audit/method_report.py"]),
]
rule_rows = []
for rule in rules:
    match = next((e for e in enforcement if rule.startswith(e[0])), None)
    rule_rows.append({"rule (first words)": rule[:70], "enforced by": match[1] if match else "(none recorded)", "files exist": all((ROOT / f).exists() for f in match[2]) if match else False})
rule_table = pd.DataFrame(rule_rows)
show_table(rule_table, floatfmt=".0f")
n_rules = len(rule_table)
n_by_program = int(rule_table["enforced by"].str.contains("hook|audit|guard|check|ledger|report").sum())
assert rule_table["files exist"].all()
hooks_present = {"pre-commit": (ROOT / ".git/hooks/pre-commit").exists()}
'''

RELEASE_TEST = r'''
# One of the programs that enforce a rule, run live: the test file of the hold-back, which asserts that nothing that names a blinded result would be pushed, and the release script in its dry-run form, which writes nothing and says what it finds.
# The script refuses to release anything before the end of the challenge; the dry run only looks.
import subprocess
import sys

test_run = subprocess.run([sys.executable, "-m", "pytest", "tests/test_release_after_deadline.py", "-q", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True, timeout=600)
test_summary = [line for line in test_run.stdout.splitlines() if " passed" in line or " failed" in line][-1]
dry_run = subprocess.run([sys.executable, "scripts/release_after_deadline.py", "--dry-run"], cwd=ROOT, capture_output=True, text=True, timeout=120)
dry_run_line = dry_run.stdout.strip().splitlines()[0]
print(test_summary)
print(dry_run_line)
assert test_run.returncode == 0 and dry_run.returncode == 0
'''

MD_12_6 = """
## 12.6 The rules that never change, and how each is enforced

> **Field:** Software engineering

*Where this section sits:* the last four sections described mechanisms. This one lists the rules those mechanisms exist for, from the first section of the checklist, which no result or deadline changes, and says for each how it is enforced. The honest answer is that some are enforced by a program and some only by a person, and the section says which.

A rule that is only a promise is weakest at the moment it matters most: late, tired and near a deadline, the temptation is to make an exception once. The project's habit is therefore to turn every rule that can be turned into a check into a check, and to say plainly which ones cannot be. The next cell reads the rules from the checklist and pairs each with its mechanism; the mapping is written by hand, and the cell fails if a file it names does not exist.
"""

MD_12_6_B = """
There are {{=n_rules}} rules, and {{=n_by_program}} of them have a program behind them. The two that are rules of procedure are the ones that concern an act of the author, not a property of a file: nothing is submitted or pushed without the author's explicit go-ahead, and the portal is never used to read the labels. For the first there is a program that helps (the release script refuses to publish anything before the deadline, and the pre-submission gate says whether everything is finished, consistent and fresh), but the decision itself is a person's, and the checklist says so.

**A hook.** A **hook** is a small program that git runs at a fixed moment. The repository uses one, kept in the local copy and not among the files that git stores: it runs before each commit, applies the linter to the files that are staged and looks for merge-conflict markers left in them. The hook is deliberately fast, so that nobody is tempted to skip it, and skipping it is not allowed; when the new modules were first committed, it was the hook that showed how many findings they carried, and the cleanup was then made a job of the queue (job 295) with its own checks and a rollback. In this build the hook is present: pre-commit {{=hooks_present['pre-commit']}}.

**A rule that caught its author.** The rule that no result for a blinded compound is published until the challenge closes is enforced by three things together: the HELD BACK block of `.gitignore`, the readiness scan of 12.5, and a test (`tests/test_release_after_deadline.py`) that fails if any file that would be pushed names blinded compounds. The test earned its place late in the project. A log of the force-field relaxation, written by a job of the queue, named the twenty blinded ligands; nobody had listed it as held back, nobody had noticed, and the test failed on it during the lint cleanup. The log was added to the HELD BACK block the same day. The rule had not been broken, because nothing had been pushed, but the next push would have broken it, and only the check knew. The next cell runs that test file and the dry run of the release script.
"""

MD_12_6_C = """
The test file has passed and the dry run reports what the release script would change, without writing anything. After the deadline the script, run without the dry-run option, removes the HELD BACK block and switches on the parts of the README that were held back; before the deadline it refuses.

**What the checks cannot do.** They catch what they were taught to look for: the patterns of the scan, the files of the HELD BACK block, the numbers that the report copies. They do not decide whether a method is right, whether a gate was chosen fairly, or whether a sentence in a document is true. Those are the work of Parts VI and IX, and of the person who reads the text word by word. Engineering for trust is the part that makes the other parts checkable; it is not a substitute for them.

*On what actually ran.* The first cell reads the checklist and the existence of the files it names; the hooks are looked for in the local copy of the repository. The second runs the test file of the hold-back and the release script with `--dry-run` as child processes.
"""

CELLS = [
    md(MD_12_4),
    code(QUEUE_HELPERS),
    md(MD_12_4_B),
    code(STAGE_LOG),
    md(MD_12_4_C),
    md(MD_12_5),
    code(REPORT_FILES),
    code(CHECKLIST_AND_SCAN),
    md(MD_12_5_B),
    md(MD_12_5_C),
    md(MD_12_6),
    code(RULES),
    md(MD_12_6_B),
    code(RELEASE_TEST),
    md(MD_12_6_C),
]

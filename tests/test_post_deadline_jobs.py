# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JOBS = ("907_post_deadline_released", "908_post_deadline_tdi", "909_post_deadline_structure")


def _text(job: str) -> str:
    return (ROOT / f"queue/jobs/{job}.sh").read_text()


@pytest.mark.skipif(sys.platform == "win32", reason="the queue and the launchers are bash scripts of Linux and macOS; Windows has run_cypmode.bat instead")
def test_the_jobs_of_the_analysis_after_the_deadline_are_valid_shell_and_run_before_the_method_report_that_reads_their_tables():
    for job in JOBS:
        assert subprocess.run(["bash", "-n", str(ROOT / f"queue/jobs/{job}.sh")], capture_output=True, check=False).returncode == 0, job
    numbers = [int(job.split("_")[0]) for job in JOBS]
    assert numbers == sorted(numbers) and 906 < numbers[0] and numbers[-1] < 910  # after the bundle of the explorer, before the method report (job 910)
    assert "source queue/lib.sh" in _text(JOBS[0]) and all("all_lower_done" in _text(job) for job in JOBS)


def test_the_questions_wait_for_the_files_and_each_job_runs_its_own_module_and_writes_its_log_with_the_tables():
    assert "post_deadline.released --verify" in _text(JOBS[0]) and "need_done 907_post_deadline_released" in _text(JOBS[1]) and "need_done 907_post_deadline_released" in _text(JOBS[2])
    assert "post_deadline.tdi" in _text(JOBS[1]) and "outputs/tdi.log" in _text(JOBS[1])
    assert "post_deadline.structure" in _text(JOBS[2]) and "outputs/structure.log" in _text(JOBS[2])
    assert "post_deadline.structure --external" in _text(JOBS[2]) and "outputs/structure_external.log" in _text(JOBS[2])  # question 6, exploratory, runs after the questions of the plan
    assert all("git push" not in _text(job) and "git commit" not in _text(job) for job in JOBS)  # the analysis measures; it publishes nothing


def test_the_final_chain_requeues_the_jobs_of_the_analysis_with_the_rest_of_it():
    release = (ROOT / "queue/jobs/296_release_unblock.sh").read_text()
    assert "queue/jobs/9[0-7]*.sh" in release  # the pattern covers 907 to 909: the scores of the poses are kept, so a second run only scores the submitted archive again

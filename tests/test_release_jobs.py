# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JOBS = ("296_release_unblock", "970_data_record_build", "975_clean_clone_check")


@pytest.mark.skipif(sys.platform == "win32", reason="the queue and the launchers are bash scripts of Linux and macOS; Windows has run_cypmode.bat instead")
def test_the_scripts_and_jobs_of_the_release_are_valid_shell():
    for path in [ROOT / "scripts/clean_clone_check.sh", *(ROOT / f"queue/jobs/{job}.sh" for job in JOBS)]:
        done = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, check=False)
        assert done.returncode == 0, f"{path.name}: {done.stderr.strip()}"


def test_the_release_job_waits_for_the_end_of_the_challenge_the_version_and_a_committed_tree_and_requeues_only_the_final_chain():
    text = (ROOT / "queue/jobs/296_release_unblock.sh").read_text()
    assert "release_after_deadline.py" in text and "bump_version.py" in text and "tree_committed" in text and "queue/jobs/9[0-7]*.sh" in text
    assert text.index("tree_committed") < text.index("release_after_deadline.py ") < text.index("build_card.py --full") < text.index("git commit")  # nothing is released before the checks, the full card is drawn before anything is committed
    assert "build_interactive_card.py --full" in text and "HELD BACK UNTIL AFTER THE DEADLINE" in text  # a second attempt after a failed drawing does not release again
    assert "git push" not in text


def test_the_jobs_of_the_release_come_in_the_order_of_their_numbers_and_the_data_record_follows_the_release():
    numbers = [int(job.split("_")[0]) for job in JOBS]
    assert numbers == sorted(numbers) and numbers[0] < 900 < numbers[1]
    assert "HELD BACK UNTIL AFTER THE DEADLINE" in (ROOT / "queue/jobs/970_data_record_build.sh").read_text()
    assert "need_done 970_data_record_build" in (ROOT / "queue/jobs/975_clean_clone_check.sh").read_text()

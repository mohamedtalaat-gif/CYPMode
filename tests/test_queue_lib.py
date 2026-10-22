# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import subprocess
from pathlib import Path

LIB = Path(__file__).resolve().parents[1] / "queue" / "lib.sh"
JOBS = ["010_a", "096_applicability_domain", "097_why_no_gain", "203_chemprop", "204_chemprop", "219_ext", "227_surrogates", "299_requeue_final_chain", "900_audit"]


def lower_done(tmp_path: Path, me: str, pending: set[str]) -> bool:
    (tmp_path / "queue/jobs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "queue/state/done").mkdir(parents=True, exist_ok=True)
    (tmp_path / "queue/state/failed").mkdir(parents=True, exist_ok=True)
    for job in JOBS:
        (tmp_path / f"queue/jobs/{job}.sh").write_text("")
        done = tmp_path / f"queue/state/done/{job}"
        done.unlink(missing_ok=True)
        if job not in pending:
            done.write_text("")
    run = subprocess.run(["bash", "-c", f"source {LIB} 2>/dev/null; all_lower_done {me}"], cwd=tmp_path, capture_output=True, check=False)
    return run.returncode == 0


def test_the_cheap_jobs_do_not_wait_for_the_long_family_and_the_final_chain_does(tmp_path):
    long_family = {"203_chemprop", "204_chemprop", "096_applicability_domain", "097_why_no_gain"}
    assert lower_done(tmp_path, "219_ext", long_family)  # the cheap candidates run before CheMeleon, which waits for them
    assert lower_done(tmp_path, "227_surrogates", long_family)
    assert not lower_done(tmp_path, "299_requeue_final_chain", {"203_chemprop"})  # the final chain waits for CheMeleon stage 1
    assert not lower_done(tmp_path, "299_requeue_final_chain", {"219_ext"})


def test_the_requeue_does_not_wait_for_the_jobs_it_re_queues(tmp_path):
    assert lower_done(tmp_path, "299_requeue_final_chain", {"096_applicability_domain", "097_why_no_gain"})  # they need 299 first
    assert not lower_done(tmp_path, "900_audit", {"096_applicability_domain"})  # the audit does wait for them once the chain is re-queued

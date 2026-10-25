# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import subprocess
import sys
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


def committed_state(tmp_path: Path, files: dict[str, str]) -> int:
    """The exit code of ``tree_committed`` in a git repository that holds ``files`` (committed first), with the package of this repository linked in and ignored."""
    root = LIB.parents[1]
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.org"]
    subprocess.run([*git, "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "cypmode").symlink_to(root / "cypmode")
    (tmp_path / ".gitignore").write_text("cypmode\n.venv\n")
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    subprocess.run([*git, "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "state"], cwd=tmp_path, check=True)
    return tmp_path


def tree_committed_code(tmp_path: Path) -> int:
    python_dir = Path(sys.executable).parent
    run = subprocess.run(["bash", "-c", f"export PATH={python_dir}:$PATH; source {LIB} 2>/dev/null; tree_committed"], cwd=tmp_path, capture_output=True, check=False)
    return run.returncode


def test_the_audits_wait_for_a_committed_tree_and_ignore_only_their_own_files(tmp_path):
    committed_state(tmp_path, {"a.txt": "one\n", "audit/AUDIT_REPORT.md": "report\n"})
    assert tree_committed_code(tmp_path) == 0
    (tmp_path / "audit/AUDIT_REPORT.md").write_text("another report\n")  # a file that the audit writes itself
    assert tree_committed_code(tmp_path) == 0
    (tmp_path / "a.txt").write_text("two\n")
    assert tree_committed_code(tmp_path) == 75  # a modified file defers the job
    subprocess.run(["git", "checkout", "-q", "a.txt"], cwd=tmp_path, check=True)
    (tmp_path / "new.txt").write_text("new\n")
    assert tree_committed_code(tmp_path) == 75  # so does an untracked one


def test_both_audits_of_the_final_sequence_wait_for_the_tree_and_the_second_for_the_submission_sync():
    root = LIB.parents[1]
    first, second = (root / "queue/jobs/900_tests_and_audit.sh").read_text(), (root / "queue/jobs/955_final_audit.sh").read_text()
    assert "tree_committed" in first and "tree_committed" in second
    assert "need_done 950_submissions_final_sync" in second and "scripts/run_audit.sh" in second
    requeue = (root / "queue/jobs/283_requeue_final_sequence.sh").read_text()
    for job in ("900_tests_and_audit", "945_video_render", "950_submissions_final_sync", "955_final_audit", "960_pre_submission_gate"):
        assert job in requeue

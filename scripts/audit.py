#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Independent, repeatable code audit -- the on-demand, comprehensive
counterpart to .git/hooks/pre-commit's fast per-commit checks.

Runs the same tool set the original manual codebase audit used (ruff,
bandit, vulture, radon, mypy, pip-audit), from the isolated .venv-audit/
(never installed into the project's real .venv, same reasoning as
.venv-boltz's own isolation), plus the full test suite -- run as
separate process groups matching the project's own real, confirmed
process-isolation requirements (lightgbm+torch and lightgbm+tblite
segfault/deadlock in the same process; see pyproject.toml's `torch`
marker and cypmode.challenge.multitask's own docstring) -- plus two
project-specific checks that came directly out of real bugs found this
session:

- Words of a local list anywhere in the git commit history (the list is
  not part of the repository; this re-checks the whole log in case history
  was rewritten).
- Hardcoded "cyp3a4"/"cyp2d6" string literals outside an obvious
  target/isoform context, repo-wide (the pre-commit hook only checks
  the staged diff; this checks everything, catching a hardcoded literal
  that was already committed before the hook existed).

Usage: python3 scripts/audit.py [--skip-tests] [--skip-mypy]
Exit code is non-zero if any BLOCKING check fails (ruff, bandit-high,
tests). mypy/vulture/radon/pip-audit are reported but do not fail the
run by default -- this codebase was never fully typed, and vulture/radon
findings need human judgment, not an automatic gate.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT_BIN = ROOT / ".venv-audit" / "bin"
MAIN_VENV_PY = ROOT / ".venv" / "bin" / "python3"

TORCH_ISOLATED_TESTS = ["tests/test_multitask.py", "tests/test_final_assembly_cyp2d6.py"]
TBLITE_ISOLATED_TESTS = ["tests/test_quantum_pharmacophore.py"]
CHEMBERTA_ISOLATED_TESTS = ["tests/test_chemberta_features.py"]

RESULTS: list[tuple[str, bool, str]] = []  # (name, blocking_ok, detail)


def _run(cmd: list[str], cwd: Path = ROOT) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    return proc.returncode, (proc.stdout + proc.stderr)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def check_ruff() -> None:
    section("ruff (lint + real bugs, e.g. undefined names, unused imports)")
    code, out = _run([str(AUDIT_BIN / "ruff"), "check", "cypmode/", "tests/", "scripts/"])
    print(out[-4000:] or "(clean)")
    RESULTS.append(("ruff", code == 0, f"exit {code}"))


def check_bandit() -> None:
    section("bandit (security -- high/medium confidence only, to keep signal real)")
    code, out = _run([str(AUDIT_BIN / "bandit"), "-r", "cypmode/", "-ll", "-q"])
    print(out[-4000:] or "(clean)")
    # bandit exits 1 on any finding at all severity; only fail the audit on
    # a finding bandit itself rates HIGH severity, not every LOW note.
    high = "Severity: High" in out
    RESULTS.append(("bandit (high-severity)", not high, f"exit {code}, high-severity found: {high}"))


def check_vulture() -> None:
    section("vulture (dead code -- informational, needs human judgment)")
    code, out = _run([str(AUDIT_BIN / "vulture"), "cypmode/", "--min-confidence", "80"])
    print(out[-3000:] or "(clean)")
    RESULTS.append(("vulture (informational)", True, f"exit {code}"))


def check_radon() -> None:
    section("radon (cyclomatic complexity -- informational)")
    # radon's FileConfig eagerly parses pyproject.toml with configparser
    # interpolation enabled and crashes on a literal '%' in the pytest
    # markers' prose (a real radon limitation, not a project bug) -- run
    # from a cwd with no pyproject.toml in its ancestry, absolute target
    # path, so radon's upward config search never finds it.
    scratch_cwd = Path.home() / ".cache"
    scratch_cwd.mkdir(exist_ok=True)
    code, out = _run(
        [str(AUDIT_BIN / "radon"), "cc", str(ROOT / "cypmode"), "-nc", "--total-average"], cwd=scratch_cwd
    )
    print(out[-3000:] or "(clean)")
    RESULTS.append(("radon (informational)", True, f"exit {code}"))


def check_mypy(skip: bool) -> None:
    section("mypy (type checking -- informational, codebase not fully typed)")
    if skip:
        print("(skipped via --skip-mypy)")
        RESULTS.append(("mypy (informational, skipped)", True, "skipped"))
        return
    code, out = _run([str(AUDIT_BIN / "mypy"), "cypmode/", "--ignore-missing-imports", "--no-error-summary"])
    print(out[-3000:] or "(clean)")
    RESULTS.append(("mypy (informational)", True, f"exit {code}"))


def check_pip_audit() -> None:
    section("pip-audit (known CVEs in the real, installed dependency set)")
    # --local audits packages already installed in whatever environment
    # pip-audit runs AS (via importlib.metadata, no reinstall) -- the right
    # mode here, since the alternative (-r a frozen requirements.txt) makes
    # pip-audit do a strict dry-run reinstall of the whole set together,
    # which fails on this real environment's legitimately-incompatible
    # cross-pins (built up incrementally across many real `pip install`
    # calls, never resolved from one requirements file) --
    # confirmed directly, a real ResolutionImpossible on numpy, not a
    # config mistake. --local only audits the CURRENT process's own
    # environment though, and pip-audit itself only lives in .venv-audit
    # (deliberately isolated from the real project deps) -- so run it via
    # the main venv's own interpreter, borrowing pip-audit's package
    # through PYTHONPATH rather than installing it into .venv for real.
    import os

    env = dict(os.environ, PYTHONPATH=str(AUDIT_BIN.parent / "lib" / "python3.13" / "site-packages"))
    proc = subprocess.run(
        [str(MAIN_VENV_PY), "-m", "pip_audit", "--local"],
        cwd=ROOT, capture_output=True, text=True, env=env, check=False,
    )
    out = proc.stdout + proc.stderr
    print(out[-4000:] or "(clean)")
    # cypmode itself is correctly, harmlessly skipped ("not found on PyPI")
    # -- it is the local package of the project, not something with a CVE registry
    # entry to check. Only fail the audit on a real reported vulnerability.
    has_real_vuln = "Found " in out and "known vulnerabilit" in out
    RESULTS.append(("pip-audit", not has_real_vuln, f"exit {proc.returncode}, vulnerabilities found: {has_real_vuln}"))


def check_private_list_in_history() -> None:
    section("Words of the local list in the git commit history")
    terms_file = Path(".git/info/local_patterns")
    terms = [t.strip() for t in terms_file.read_text().splitlines() if t.strip()] if terms_file.exists() else []
    _code, out = _run(["git", "log", "--all", "-i", "-E", "--oneline", *[f"--grep={t}" for t in terms]])
    hits = [line for line in out.splitlines() if line.strip()]
    if hits:
        print("FOUND (real violation):")
        for h in hits:
            print(" ", h)
    else:
        print("(clean -- no matching commits)" if terms else "(the local list is not present in this checkout)")
    RESULTS.append(("local-list-in-history", len(hits) == 0, f"{len(hits)} matching commit(s)"))


def check_hardcoded_isoform_literals() -> None:
    section("hardcoded 'cyp3a4'/'cyp2d6' literals outside target/isoform context (repo-wide)")
    _code, out = _run(
        [
            "grep", "-rnE", "--include=*.py", r'"cyp3a4"|\x27cyp3a4\x27|"cyp2d6"|\x27cyp2d6\x27',
            "cypmode/",
        ]
    )
    suspect_lines = [
        line for line in out.splitlines()
        if not any(k in line for k in ("target", "TARGET", "isoform", "ISOFORM", "_SEQUENCE", "#"))
    ]
    if suspect_lines:
        print(f"{len(suspect_lines)} line(s) worth a human look (heuristic, not certain):")
        for line in suspect_lines[:30]:
            print(" ", line)
    else:
        print("(clean -- every match is in an obvious target/isoform/comment context)")
    # Informational only -- this heuristic has real false-positive risk
    # (e.g. a real default value assignment is legitimate), a human call,
    # not an automatic gate.
    RESULTS.append(("hardcoded-isoform-literals (informational)", True, f"{len(suspect_lines)} flagged"))


def check_tests(skip: bool) -> None:
    section("pytest -- main suite + isolated files as separate processes")
    if skip:
        print("(skipped via --skip-tests)")
        RESULTS.append(("pytest (skipped)", True, "skipped"))
        return

    venv_py = str(MAIN_VENV_PY)
    all_ok = True
    detail_parts = []

    code, out = _run([venv_py, "-m", "pytest", "tests/", "-q"])
    print("-- main suite --")
    print(out[-2000:])
    all_ok &= code == 0
    detail_parts.append(f"main suite exit {code}")

    for group_name, files in [
        ("torch-isolated", TORCH_ISOLATED_TESTS),
        ("tblite-isolated", TBLITE_ISOLATED_TESTS),
        ("chemberta-isolated", CHEMBERTA_ISOLATED_TESTS),
    ]:
        code, out = _run([venv_py, "-m", "pytest", *files, "-q", "--override-ini=addopts="])
        print(f"-- {group_name} --")
        print(out[-1500:])
        all_ok &= code == 0
        detail_parts.append(f"{group_name} exit {code}")

    RESULTS.append(("pytest (all groups)", all_ok, "; ".join(detail_parts)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-tests", action="store_true", help="skip the (slower) full pytest run")
    parser.add_argument("--skip-mypy", action="store_true", help="skip mypy (can be slow on first run, no cache)")
    args = parser.parse_args()

    if not AUDIT_BIN.exists():
        sys.exit(
            f"{AUDIT_BIN} does not exist -- run:\n"
            f"  python3 -m venv .venv-audit && .venv-audit/bin/pip install ruff bandit vulture radon mypy pip-audit"
        )

    check_private_list_in_history()
    check_hardcoded_isoform_literals()
    check_ruff()
    check_bandit()
    check_vulture()
    check_radon()
    check_mypy(args.skip_mypy)
    check_pip_audit()
    check_tests(args.skip_tests)

    section("SUMMARY")
    blocking_failed = []
    for name, ok, detail in RESULTS:
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {name} -- {detail}")
        if not ok:
            blocking_failed.append(name)

    if blocking_failed:
        print(f"\n{len(blocking_failed)} blocking check(s) failed: {', '.join(blocking_failed)}")
        sys.exit(1)
    print("\nAll blocking checks passed.")


if __name__ == "__main__":
    main()

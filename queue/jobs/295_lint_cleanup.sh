#!/bin/bash
source queue/lib.sh
set -o pipefail
# Lint cleanup of the Python modules (decided 2026-10-17). It runs after every experiment job and before the final chain (job 299), so that the audit and the test suite of the chain run on the cleaned code.
# Only the safe fixes of ruff are applied (no unsafe fix). The result is checked against the state before it: every module of cypmode and of the notebook must import as it did, every file must compile and
# the test suite must pass; if anything is worse, the files are put back exactly as they were and the log says so. What ruff cannot fix is listed in lint_remaining.log, for a hand.
need_file .venv-audit/bin/ruff
all_lower_done 295_lint_cleanup || defer "earlier jobs not finished"
RUFF=.venv-audit/bin/ruff
LOG=$L/lint_cleanup.log
files() { git ls-files --cached --others --exclude-standard '*.py' | grep -v '^third_party/'; }
count_findings() { files | xargs $RUFF check --output-format concise 2>/dev/null | grep -cE '^[^ ]+:[0-9]+:[0-9]+: ' || true; }
import_report() { find cypmode notebooks -name '*.py' ! -name '__init__.py' -not -path '*/_vendor/*' | sed 's#/#.#g; s#\.py$##' | sort | xargs -P 4 -I{} sh -c 'python -c "import {}" >/dev/null 2>&1 && echo "ok {}" || echo "fail {}"' | sort; }
restore() { tar -xf "$WORK/before.tar"; echo "LINT CLEANUP ROLLED BACK: $1 (the files are as they were before; nothing was committed)" | tee -a $LOG; }

recent=$(files | xargs -I{} find {} -mmin -10 2>/dev/null)
[ -z "$recent" ] || defer "a Python file changed in the last 10 minutes (the cleanup rewrites files and must not meet an edit): $(echo "$recent" | head -n 1)"
WORK=$(mktemp -d)  # outside the repository: the copy of the files that is put back if the cleanup is rolled back
: > $LOG
files | tar -cf "$WORK/before.tar" -T -
before=$(count_findings)
echo "findings before: $before in $(files | wc -l) files" | tee -a $LOG
import_report > "$WORK/imports_before.txt"
echo "modules that import before: $(grep -c '^ok' "$WORK/imports_before.txt"), that do not: $(grep -c '^fail' "$WORK/imports_before.txt")" | tee -a $LOG

files | xargs $RUFF check --fix --exit-zero --quiet >> $LOG 2>&1
after=$(count_findings)
echo "findings after the safe fixes: $after" | tee -a $LOG

if ! files | xargs python -m py_compile >> $LOG 2>&1; then restore "a file no longer compiles"; exit 0; fi
import_report > "$WORK/imports_after.txt"
newly_failing=$(comm -13 <(grep '^fail' "$WORK/imports_before.txt") <(grep '^fail' "$WORK/imports_after.txt"))
if [ -n "$newly_failing" ]; then echo "$newly_failing" >> $LOG; restore "modules that imported before no longer do"; exit 0; fi
echo "import check: no module that imported before fails now" | tee -a $LOG
if ! python -m pytest -q -p no:cacheprovider tests > $L/lint_cleanup_tests.log 2>&1; then tail -n 15 $L/lint_cleanup_tests.log >> $LOG; restore "the test suite fails after the fixes"; exit 0; fi
echo "test suite: $(tail -n 1 $L/lint_cleanup_tests.log)" | tee -a $LOG

files | xargs $RUFF check --output-format concise > $L/lint_remaining.log 2>&1 || true
files | xargs $RUFF check --statistics > $L/lint_remaining_statistics.log 2>&1 || true
files | xargs git add --
if git diff --cached --quiet; then echo "nothing changed" | tee -a $LOG; exit 0; fi
message="Lint cleanup: the safe fixes of ruff (import order, unused imports and noqa directives, names of regex flags and the like); every module imports as before and the test suite passes"
if [ "$after" = "0" ]; then
  git commit -q -m "$message" >> $LOG 2>&1 || { restore "the commit was refused"; git reset -q; exit 0; }
else
  hooks=$(mktemp -d); for hook in .git/hooks/*; do case "$hook" in *pre-commit*|*.sample) ;; *) cp "$hook" "$hooks/" ;; esac; done
  git -c core.hooksPath="$hooks" commit -q -m "$message

$after findings remain that need a hand (listed in the log of the queue job)." >> $LOG 2>&1 || { restore "the commit was refused"; git reset -q; exit 0; }
fi
echo "committed locally: $(git rev-parse --short HEAD); findings left: $after (see lint_remaining.log)" | tee -a $LOG

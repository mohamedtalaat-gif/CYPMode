#!/bin/bash
# Generate the audit report twice from the current state and require byte-identical output.
# Usage: bash scripts/run_audit.sh [--skip-tests]
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
mkdir -p audit
python -m cypmode.audit.report --out audit/AUDIT_REPORT.md "$@" 2>/dev/null | tail -n 1
python -m cypmode.audit.report --out audit/AUDIT_REPORT.run2.md "$@" 2>/dev/null | tail -n 1
if cmp -s audit/AUDIT_REPORT.md audit/AUDIT_REPORT.run2.md; then
  echo "REPRODUCIBLE: two runs are byte-identical ($(shasum -a 256 audit/AUDIT_REPORT.md | cut -c1-16))"
  rm -f audit/AUDIT_REPORT.run2.md
else
  echo "NOT REPRODUCIBLE: the two runs differ"; diff audit/AUDIT_REPORT.md audit/AUDIT_REPORT.run2.md | head -40
  exit 1
fi

#!/bin/bash
# One run from a clean clone: the repository is cloned at its last commit, the data record is restored there from its archives, the audit is run in the clone and its checks are compared with the checks of the committed report.
# Usage: bash scripts/clean_clone_check.sh ARCHIVE_DIR [WORK_DIR]
# The inputs that are fetched by name and checksum from their own sources (data/*/inputs, which the data record does not hold) are copied from this machine, and the environment is this one; the check is of the code and of the data record.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
ARCHIVES=${1:?usage: bash scripts/clean_clone_check.sh ARCHIVE_DIR [WORK_DIR]}
if [ -n "${2:-}" ]; then
  WORK=$2
  KEEP=1
else
  WORK="$(mktemp -d)/clone"
  KEEP=0
fi
git clone --quiet "$ROOT" "$WORK"
cd "$WORK"
# The environments of this machine are linked into the clone (the audit checks them against the pin files) and hidden from git there, because the ignore file names them as folders and a link is not a folder.
for env in "$ROOT"/.venv*; do
  [ -e "$env" ] && ln -s "$env" "$(basename "$env")"
done
echo ".venv*" >> .git/info/exclude
# The list of private words is not part of any checkout; the clone reads the one of this machine, so that the scan of the credit section runs there as it does here.
[ -f "$ROOT/.git/info/local_patterns" ] && cp "$ROOT/.git/info/local_patterns" .git/info/local_patterns
.venv/bin/python scripts/fetch_data.py fetch --source "$ARCHIVES"
for track in tdi_track structure_track; do
  mkdir -p "data/$track/inputs"
  rsync -a --ignore-existing "$ROOT/data/$track/inputs/" "data/$track/inputs/"
done
bash scripts/run_audit.sh
python3 - "$ROOT/audit/AUDIT_REPORT.md" "$WORK/audit/AUDIT_REPORT.md" <<'PY'
import sys

def checks(path):
    rows, section = {}, ""
    for line in open(path, encoding="utf-8").read().splitlines():
        if line.startswith("## "):
            section = line[3:]
        elif line.startswith("| ") and not line.startswith(("| check", "|---", "| ---")):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 2 and cells[1] in {"PASS", "FAIL", "INFO", "WARN"}:
                rows[(section, cells[0])] = cells[1]
    return rows

committed, clone = checks(sys.argv[1]), checks(sys.argv[2])
only_committed, only_clone = sorted(set(committed) - set(clone)), sorted(set(clone) - set(committed))
changed = sorted(k for k in set(committed) & set(clone) if committed[k] != clone[k])
print(f"checks: {len(committed)} in the committed report, {len(clone)} in the report of the clean clone")
for label, keys in (("only in the committed report", only_committed), ("only in the clean clone", only_clone)):
    for section, name in keys:
        print(f"  {label}: [{section}] {name}")
for key in changed:
    print(f"  status differs: [{key[0]}] {key[1]}: {committed[key]} -> {clone[key]}")
sys.exit(1 if (only_committed or only_clone or changed) else 0)
PY
echo "the clean clone gives the same checks as the committed report"
# A clone that was made in a temporary folder holds about 10 GB; it is removed when the checks agree and kept, for the diagnosis, when they differ.
if [ "$KEEP" -eq 0 ]; then
  cd "$ROOT"
  rm -rf "$(dirname "$WORK")"
fi

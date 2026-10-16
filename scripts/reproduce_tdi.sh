#!/bin/bash
# One command from the raw challenge files to the validated TDI candidate files (hours, resumable: every step is cached).
#   bash scripts/reproduce_tdi.sh
# Order: default-grid nested caches for both isoforms, then every TDI queue job (000-090: population experiments, physics
# block, SAR / matched-pair / committee / embedding / weighting experiments, hedged blend, final build). Seeds are fixed in the
# nested plans (seed 0, 3 repeats, 5 outer, 4 inner folds); the build writes data/tdi_track/outputs/final/manifest.json with
# the SHA-256 of every candidate file, the package versions and the gate verdicts. Nothing is uploaded or submitted.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
export PYTHONPATH=. REPRODUCE=1
for iso in CYP3A4 CYP2D6; do
  python -m cypmode.challenge.nested_runners --isoform "$iso" --repeats 3 || { echo "nested grid failed for $iso"; exit 1; }
done
for job in $(ls queue/jobs/0*.sh | sort); do
  echo "== $job"
  bash "$job"
  rc=$?
  if [ $rc -eq 75 ]; then echo "JOB NOT READY: $job (its code or inputs are missing)"; exit 75; fi
  [ $rc -eq 0 ] || { echo "JOB FAILED ($rc): $job"; exit $rc; }
done
echo "done: see data/tdi_track/outputs/final/manifest.json"

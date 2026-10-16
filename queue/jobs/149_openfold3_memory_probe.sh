#!/bin/bash
source queue/lib.sh
# OpenFold3 memory probe. The largest benchmark ligand (bench-YNV, 46 heavy atoms; its first run stopped with "Invalid buffer size: 9.27 GiB")
# must complete 3 samples with the runner YAML (smaller chunk size); if it does not, with the low-memory chunk_layer path as well.
# The first mode that completes it is written to openfold3_memory_mode.txt and used by every later OpenFold3 run (jobs 150, 185).
need_file scripts/run_openfold_lowmem.py
need_file data/structure_track/inputs/openfold3_runner.yml
MODE_FILE=data/structure_track/inputs/openfold3_memory_mode.txt
.venv-openfold3/bin/python3 scripts/run_openfold_lowmem.py --self-check > $L/openfold3_lowmem_selfcheck.log 2>&1 || exit 1
rm -f $MODE_FILE
for mode in config lowmem; do
  if python3 -m cypmode.validation.openfold3 --names bench-YNV --samples 3 --memory $mode --strict > $L/openfold3_probe_$mode.log 2>&1; then
    echo $mode > $MODE_FILE
    exit 0
  fi
done
exit 1

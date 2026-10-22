#!/bin/bash
source queue/lib.sh
# BioEmu2 timing probe in an exclusive window: one benchmark system (CYP3A4, ligand and heme, about 570 tokens), two samples, on the Apple GPU.
# It starts only when the long GPU jobs have settled, so that nothing else is running and the whole memory is free for it; the runner's memory guard
# is active. The result is the log, `timings.json` under data/structure_track/outputs/bioemu2/probe (trunk seconds, seconds per sample, peak memory) or the
# statement that the run did not complete; both are readings of what this machine can do, not failures to retry, so the job ends with status 0 either way.
need_file cypmode/validation/bioemu2.py
need_settled 239_chemeleon_addition 257_openadmet_weights_stage2_CYP3A4 258_openadmet_weights_addition
start=$(date +%s)
if .venv-bioemu2/bin/python -m cypmode.validation.bioemu2 --probe > $L/exploratory_bioemu2_probe.log 2>&1; then
  echo "BIOEMU2 PROBE: completed in $(( $(date +%s) - start )) s" >> $L/exploratory_bioemu2_probe.log
else
  echo "BIOEMU2 PROBE: did not complete (exit $?) after $(( $(date +%s) - start )) s; swap at the end: $(sysctl -n vm.swapusage)" >> $L/exploratory_bioemu2_probe.log
fi
exit 0

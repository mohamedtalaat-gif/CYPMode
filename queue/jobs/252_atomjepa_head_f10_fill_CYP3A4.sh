#!/bin/bash
source queue/lib.sh
set -o pipefail
# Exploratory replicate of the Atom-JEPA readout on CYP3A4 (checklist, Atom-JEPA item; declared 2026-10-17, after the registered 5-fold gate was NOT MET): the cached readout completed for every task of the 10-outer-fold plan
# (time-boxed chunks, resumable). The encoder does not run again.
need_file cypmode/challenge/nested_runners_torch.py
need_file data/tdi_track/outputs/caches/atomjepa_atoms.npz
[ -f data/tdi_track/outputs/nested_f10/CYP3A4/plan.npz ] || defer "10-fold grid not built yet"
out=$(PYTHONPATH=. .venv-ml/bin/python -m cypmode.challenge.nested_runners_torch --isoform CYP3A4 --root data/tdi_track/outputs/nested_f10 --groups atomjepa_head --max-minutes 90 2>&1 | tee -a $L/atomjepa_head_f10_CYP3A4.log | tail -n 1) || exit 1
chunk_status 252_atomjepa_head_f10_fill_CYP3A4 "${out##* }"

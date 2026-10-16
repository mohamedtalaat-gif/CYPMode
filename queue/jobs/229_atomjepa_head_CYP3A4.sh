#!/bin/bash
source queue/lib.sh
set -o pipefail
# A small attention readout trained on the cached per-atom features of the frozen Atom-JEPA encoder, completed for every task of the nested plan (time-boxed chunks, resumable). The encoder does not run again.
need_file cypmode/challenge/nested_runners_torch.py
need_file data/tdi_track/outputs/caches/atomjepa_atoms.npz
out=$(PYTHONPATH=. .venv-ml/bin/python -m cypmode.challenge.nested_runners_torch --isoform CYP3A4 --groups atomjepa_head --max-minutes 90 2>&1 | tee -a $L/atomjepa_head_CYP3A4.log | tail -n 1) || exit 1
chunk_status 229_atomjepa_head_CYP3A4 "${out##* }"

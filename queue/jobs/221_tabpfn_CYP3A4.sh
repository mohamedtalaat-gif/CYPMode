#!/bin/bash
source queue/lib.sh
set -o pipefail
# TabPFN (Prior Labs) completed for every task of the nested plan (time-boxed chunks, resumable; the memory guard stays on). The weights are under Prior Labs' non-commercial licence and are fetched by the
# package after the licence has been accepted at ux.priorlabs.ai; the job waits until the file data/tdi_track/inputs/tabpfn_licence_accepted.txt exists (it is created once the licence has been accepted).
need_file cypmode/challenge/nested_runners_torch.py
need_file data/tdi_track/inputs/tabpfn_licence_accepted.txt
.venv-ml/bin/python -c "import tabpfn" 2>/dev/null || defer "the package tabpfn is not installed in .venv-ml"
out=$(PYTHONPATH=. .venv-ml/bin/python -m cypmode.challenge.nested_runners_torch --isoform CYP3A4 --groups tabpfn --max-minutes 90 2>&1 | tee -a $L/tabpfn_CYP3A4.log | tail -n 1) || exit 1
chunk_status 221_tabpfn_CYP3A4 "${out##* }"

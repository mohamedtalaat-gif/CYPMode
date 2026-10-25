#!/bin/bash
source queue/lib.sh
# The affinity step of the three base runs made again from the corrected SMILES (job 275) hit the memory cap of the guards (PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.3: "ran out of memory, skipping batch"), so no
# affinity file was written and the batch runner recorded the whole row of each of the three ligands as an error (checklist item "Affinity step of the corrected base runs", registered 2026-10-25 before this job
# ran). The structure of each run exists and is not predicted again: the entries leave the checkpoint after the SHA-256 of their structure files is recorded, Boltz-2 is run again with --no-override (it skips the
# finished prediction and runs only the missing affinity step) on the CPU, which has no memory cap, and the structure files are compared with the recorded digests afterwards. HELD BACK: the logs are named
# affinity_retry_*.log, the digests are under final/.
need_file cypmode/validation/affinity_retry.py
need_file $OUT/structure_track_results.json
all_lower_done 277_affinity_base_runs || defer "earlier jobs not finished"
python3 -m cypmode.validation.affinity_retry --prepare > $L/affinity_retry_prepare.log 2>&1 || exit 1
python3 -m cypmode.validation.batch $IN/structure_track_test_blinded.csv $OUT/structure_track_results.json --protein-msa $MSA --max-parallel-samples 1 --no-override --accelerator cpu > $L/affinity_retry_batch.log 2>&1 || exit 1
python3 -m cypmode.validation.affinity_retry --verify > $L/affinity_retry_verify.log 2>&1 || exit 1

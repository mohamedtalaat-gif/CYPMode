#!/bin/bash
source queue/lib.sh
set -o pipefail
# OpenStructure's LDDT-PLI and BiSyRMSD for the 167 poses that the gates of the pools were judged on (about 4 s per pose, measured on three poses), in time-boxed chunks, resumable. The registered consequence of the verdict of job 207.
need_file cypmode/validation/ost_pools.py
[ -x ~/miniforge3/bin/mamba ] || defer "mamba is missing"
need_done 207_ost_vs_ours
out=$(python3 -m cypmode.validation.ost_pools --max-minutes 60 2>&1 | tee -a $L/ost_pools.log | tail -n 1) || exit 1
chunk_status 244_ost_pools_score "${out##* }"

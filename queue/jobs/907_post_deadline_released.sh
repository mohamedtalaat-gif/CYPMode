#!/bin/bash
source queue/lib.sh
# The files that the organisers released after the deadline (the unblinded TDI test file, the 20 structures, the evaluation code and the Boltz-2 baseline of the tutorial repository at a fixed commit), fetched by name and checked against the SHA-256 values of data/post_deadline/released_manifest.json; the analysis of docs/post_deadline/PLAN.md reads them. It runs in the final chain, before the method report (job 910), which reads its tables.
need_file cypmode/post_deadline/released.py
all_lower_done 907_post_deadline_released || defer "earlier jobs not finished"
python3 -m cypmode.post_deadline.released --verify > data/post_deadline/outputs/released.log 2>&1 || python3 -m cypmode.post_deadline.released > data/post_deadline/outputs/released.log 2>&1 || exit 1

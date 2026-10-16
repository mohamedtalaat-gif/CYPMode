#!/bin/bash
source queue/lib.sh
# The applicability domain of the 750 blinded compounds, generated from the final build: the reliability of the final stack by nearest-training similarity (held-out outer folds of the nested validation)
# and the tier of every blinded compound. The method report and the notebook read the files it writes.
need_file cypmode/challenge/applicability_domain.py
need_file data/tdi_track/outputs/final/CYP3A4_hits_only_scores.csv
need_done 299_requeue_final_chain 090_final_tdi_build
all_lower_done 096_applicability_domain || defer "earlier jobs not finished"
python3 -m cypmode.challenge.applicability_domain > $L/applicability_domain.log 2>&1

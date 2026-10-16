#!/bin/bash
source queue/lib.sh
# Older significance tables re-analysed from their saved per-fold scores with the paired corrected t (no refits).
need_file cypmode/challenge/recompute_paired_summaries.py
python -m cypmode.challenge.recompute_paired_summaries > $L/recompute_paired_summaries.log 2>&1

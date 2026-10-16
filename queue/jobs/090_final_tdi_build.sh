#!/bin/bash
source queue/lib.sh
# Gate-driven TDI candidate files, once per design (5 outer folds, then the 10-fold replicate). Never submits.
need_file cypmode/challenge/final_tdi_build.py
all_lower_done 090_final_tdi_build || defer "earlier TDI jobs not finished"
python -m cypmode.challenge.final_tdi_build > $L/final_tdi_build.log 2>&1 || exit 1
python -m cypmode.challenge.final_tdi_build --design 10 --out-dir data/tdi_track/outputs/final_f10 > $L/final_tdi_build_f10.log 2>&1

#!/bin/bash
source queue/lib.sh
# OpenStructure's own LDDT-PLI (the organisers' scorer, in the conda environment `ost`) against ours, pose by pose, on the confirmation ligands.
need_file cypmode/validation/ost_official.py
[ -x ~/miniforge3/bin/mamba ] || defer "mamba is missing"
python3 -m cypmode.validation.ost_official > $L/ost_vs_ours.log 2>&1

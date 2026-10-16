#!/bin/bash
source queue/lib.sh
# Force-field relaxation of predicted ligand poses in the frozen pocket (OpenMM, in the conda environment openmm), tested on the poses of the crystal-anchored ligands before any use on the official ones:
# the scorer's PoseBusters verdict and OpenStructure's LDDT-PLI, before and after, with the registered gate of cypmode/validation/relax_gate.py.
need_file cypmode/validation/relax_gate.py
need_file cypmode/validation/relax.py
need_file "$HOME/miniforge3/envs/openmm/bin/python"
need_file "$HOME/miniforge3/bin/mamba"
python3 -m cypmode.validation.relax_gate > $L/relaxation_gate.log 2>&1

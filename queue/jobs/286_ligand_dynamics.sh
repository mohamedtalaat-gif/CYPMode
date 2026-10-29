#!/bin/bash
source queue/lib.sh
# Ligand dynamics in the frozen pocket (the rule of docs/structure_ligand_dynamics.md, committed before any number): the pools of the static pharmacophore gate, 20 ps of the ligand in its frozen pocket for every pose, the dynamic scores and the gate.
need_file cypmode/validation/ligand_dynamics.py
OPENMM_PY="$HOME/miniforge3/envs/openmm/bin/python"
[ -x "$OPENMM_PY" ] || defer "the conda environment openmm is missing"
python -m cypmode.validation.ligand_dynamics --prepare > $L/ligand_dynamics_prepare.log 2>&1 || exit 1
for i in 0 1 2 3; do
  "$OPENMM_PY" -m cypmode.validation.ligand_dynamics --simulate --shard $i/4 --threads 2 > $L/ligand_dynamics_simulate_$i.log 2>&1 &
done
wait
planned=$(python -c "import json; print(len(json.load(open('data/structure_track/outputs/cyp3a4/ligand_dynamics/manifest.json'))))")
done_count=$(ls data/structure_track/outputs/cyp3a4/ligand_dynamics/trajectories/*.npz 2>/dev/null | wc -l)
[ "$planned" = "$done_count" ] || { echo "only $done_count of $planned trajectories exist"; exit 1; }
python -m cypmode.validation.ligand_dynamics --gate > $L/ligand_dynamics.log 2>&1

#!/bin/bash
source queue/lib.sh
# S8 part a (declared 2026-10-23 before any code): the pharmacophore model of a predicted pose, judged on the same pooled crystal-anchored benchmark as the pose terms of S4. Each pose is put in the frame of its
# own heme and scored against the fields and the sites of the crystal ligands built without the references of the scored ligand (the compound itself, and in the regime of the gate its analogues too).
need_file cypmode/validation/pose_pharmacophore.py
need_file $OUT/diversity_pilot_poses.csv
need_file data/tdi_track/outputs/caches/pocket_fields_cyp3a4.npz
python3 -m cypmode.validation.pose_pharmacophore --gate > $L/pose_pharmacophore_model.log 2>&1 || exit 1

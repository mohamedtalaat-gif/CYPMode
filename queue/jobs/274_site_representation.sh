#!/bin/bash
source queue/lib.sh
# The check of the binding-site representation (declared 2026-10-23 before any code): the variability of the pocket among the crystal entries, the pockets of the predicted complexes of the pools of job 272 against it, and the support of
# the fields. It writes the two flags that the official ligands carry in job 925. No blinded compound is involved.
need_file cypmode/validation/site_representation.py
need_done 272_pose_pharmacophore_gate
python3 -m cypmode.validation.site_representation > $L/site_representation.log 2>&1 || exit 1
need_file $OUT/site_representation_flags.json

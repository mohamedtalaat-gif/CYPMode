#!/bin/bash
source queue/lib.sh
# Physics block inside the selection-aware CYP2D6 model (P(hit) and P(label | hit)), panel-block pool.
need_file cypmode/challenge/population_physics.py
need_file data/tdi_track/outputs/caches/physics_cache.csv
python -m cypmode.challenge.population_physics --isoform CYP2D6 --repeats 3 --run --n-jobs 4 > $L/population_physics_CYP2D6.log 2>&1

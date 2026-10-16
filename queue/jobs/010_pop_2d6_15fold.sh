#!/bin/bash
source queue/lib.sh
# Selection-aware CYP2D6 (panel-block pool): 3 repeats x 5 folds, the gate for the pre-registered decision.
python -m cypmode.challenge.population_experiment --isoform CYP2D6 --repeats 3 --run --n-jobs 4 > $L/population_panel_CYP2D6_r3.log 2>&1

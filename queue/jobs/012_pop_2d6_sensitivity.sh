#!/bin/bash
source queue/lib.sh
# Same caches: sensitivity to the share of non-hits in the scored set.
need_file $L/population_panel_CYP2D6_r3.log
python -m cypmode.challenge.population_experiment --isoform CYP2D6 --repeats 3 --sensitivity > $L/population_sensitivity_panel_CYP2D6_r3.log 2>&1

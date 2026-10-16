#!/bin/bash
source queue/lib.sh
# CYP3A4 control: the test is analogs of 3A4 hits, so no selection-aware gain is expected here.
python -m cypmode.challenge.population_experiment --isoform CYP3A4 --repeats 1 --run --n-jobs 4 > $L/population_panel_CYP3A4.log 2>&1

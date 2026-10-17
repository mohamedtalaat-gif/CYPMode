#!/bin/bash
source queue/lib.sh
# Runs the calibration experiment (calibration_experiment.py); the log is calibration.log.
need_file cypmode/challenge/calibration_experiment.py
python -m cypmode.challenge.calibration_experiment > $L/calibration.log 2>&1

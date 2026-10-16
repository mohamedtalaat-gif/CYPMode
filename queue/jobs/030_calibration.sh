#!/bin/bash
source queue/lib.sh
need_file cypmode/challenge/calibration_experiment.py
python -m cypmode.challenge.calibration_experiment > $L/calibration.log 2>&1

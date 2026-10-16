#!/bin/bash
source queue/lib.sh
# The real screen inside the selection-aware CYP2D6 model (gate learned from the screen; predicted screen in the end-to-end classifier), on the panel-block pool.
need_file cypmode/challenge/screen_selection.py
python3 -m cypmode.challenge.screen_selection --isoform CYP2D6 --repeats 3 --run > $L/population_screen_CYP2D6.log 2>&1

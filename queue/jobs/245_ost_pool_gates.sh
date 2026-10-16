#!/bin/bash
source queue/lib.sh
# The gates of the pools repeated on OpenStructure's yardstick (after a check that the same code reproduces every original verdict on unchanged tables); nothing the gates wrote before is touched.
need_file cypmode/validation/ost_pool_gates.py
need_done 244_ost_pools_score
python3 -m cypmode.validation.ost_pool_gates > $L/ost_pool_gates.log 2>&1

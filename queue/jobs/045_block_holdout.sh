#!/bin/bash
source queue/lib.sh
# Most test-like validation available: hold out the analog-expansion block of the training ids and score the stack on it.
need_file cypmode/challenge/block_holdout.py
python -m cypmode.challenge.block_holdout > $L/block_holdout.log 2>&1

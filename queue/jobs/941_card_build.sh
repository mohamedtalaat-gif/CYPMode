#!/bin/bash
source queue/lib.sh
# The README card, built after the figures from the final numbers and recorded in audit/artifacts.json (the pre-submission gate checks the record against the numbers).
need_file docs/card/build_card.py
all_lower_done 941_card_build || defer "earlier jobs not finished"
python3 docs/card/build_card.py > $L/card_build.log 2>&1

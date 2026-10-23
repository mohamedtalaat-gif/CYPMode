#!/bin/bash
source queue/lib.sh
# The interactive card (docs/card/interactive/index.html, one self-contained page with a rotatable structure, the six complexes, the iron-nitrogen distances measured on the loaded coordinates and the
# numbers of the card), built from the final numbers after the film and recorded in audit/artifacts.json (the pre-submission gate checks the record against the numbers and rebuilds the page to compare it).
need_file docs/card/build_interactive_card.py
need_file docs/card/vendor/3Dmol-min.js
all_lower_done 946_interactive_card || defer "earlier jobs not finished"
python3 docs/card/build_interactive_card.py > $L/interactive_card_build.log 2>&1

#!/bin/bash
source queue/lib.sh
# The illustrations (docs/figures/PLAN.md): every generator, the catalog check, light and dark SVG and PNG, the animations. Runs after every
# result is final, so the figures show the final numbers.
need_file docs/figures/build_all.py
all_lower_done 940_figures_build || defer "earlier jobs not finished"
python3 docs/figures/build_all.py > $L/figures_build.log 2>&1

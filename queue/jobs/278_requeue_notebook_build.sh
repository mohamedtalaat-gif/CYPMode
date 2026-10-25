#!/bin/bash
source queue/lib.sh
# The first build of the notebook with the sections 9.13 to 9.15, 11.11 and 11.12 (job 942, 2026-10-25, two attempts) stopped at a Markdown cell that writes a number into the text before the cell that computes it
# (NameError: name 'pocket_sets' is not defined); the cells that compute the numbers of those sections now come before the text that reads them, and tests/test_notebook_template_order.py checks the order of
# all the cells. The notebook jobs run again: the complete edition (942) and the public edition (947), which reads the same cells.
all_lower_done 278_requeue_notebook_build || defer "earlier jobs not finished"
for id in 942_build_notebook 947_notebook_public; do
  rm -f queue/state/done/$id queue/state/failed/$id queue/state/attempts/$id
done
echo "re-queued: 942 and 947"

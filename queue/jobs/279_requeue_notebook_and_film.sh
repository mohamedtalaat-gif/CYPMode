#!/bin/bash
source queue/lib.sh
# The second build of the notebook (job 942, 2026-10-25) stopped at the paragraph of section 12.1 that read a log which the speed test of 2026-10-22 had overwritten; the cell now reads the two records of
# the speed test, and a collector that ran all the cells found nothing else. The render of the film (job 945) lost its headless Chrome on the first frame in both attempts of the same morning, while
# the same Chrome class rendered frames by hand in the scratch folder; it runs again here, alone on the machine, between the build of the notebook and its public edition.
all_lower_done 279_requeue_notebook_and_film || defer "earlier jobs not finished"
for id in 942_build_notebook 945_video_render 947_notebook_public; do
  rm -f queue/state/done/$id queue/state/failed/$id queue/state/attempts/$id
done
echo "re-queued: 942, 945 and 947"

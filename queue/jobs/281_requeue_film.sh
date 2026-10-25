#!/bin/bash
source queue/lib.sh
# The render of the film (job 945) lost its browser 31 to 33 seconds after every start of the browser and failed in all four attempts of 2026-10-25. The driver of the browser now starts a new one and opens the page again
# when the browser has gone (docs/video/anime/chrome.py, tests/test_film_chrome.py), and the film is drawn again here, after the build of the notebook, on the audit of this chain.
all_lower_done 281_requeue_film || defer "earlier jobs not finished"
rm -f queue/state/done/945_video_render queue/state/failed/945_video_render queue/state/attempts/945_video_render
echo "re-queued: 945"

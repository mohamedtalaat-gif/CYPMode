#!/bin/bash
source queue/lib.sh
# The second render of the film (job 945, 2026-10-25) stopped after about 500 frames because another browser, started by a test, used the same debugging port while the render ran and the page of the new
# browser had no drawing function. The driver now takes a free port for every start (docs/video/anime/chrome.py) and starts the browser again when it is ended while it starts; the film is drawn
# again here, on a machine on which no other browser runs.
all_lower_done 282_requeue_film_again || defer "earlier jobs not finished"
rm -f queue/state/done/945_video_render queue/state/failed/945_video_render queue/state/attempts/945_video_render
echo "re-queued: 945"

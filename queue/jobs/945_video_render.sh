#!/bin/bash
source queue/lib.sh
# The animation (docs/video/STORYBOARD.md): every frame drawn by code, encoded to mp4, plus a gif preview and a poster frame.
need_file docs/video/render_video.py
all_lower_done 945_video_render || defer "earlier jobs not finished"
# the spoken lines are read from the film itself and spoken first (the Kokoro environment .venv-voice; without it the film stays silent)
if [ -x .venv-voice/bin/python ]; then
  python3 docs/video/voices.py harvest > $L/voices_harvest.log 2>&1 || exit 1
  .venv-voice/bin/python docs/video/voices.py speak > $L/voices_speak.log 2>&1 || exit 1
fi
python3 docs/video/render_video.py > $L/video_render.log 2>&1

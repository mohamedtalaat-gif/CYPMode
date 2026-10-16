#!/bin/bash
source queue/lib.sh
# The cinematic molecule clips of the film (docs/video/STORYBOARD.md): the CYP3A4 enzyme with its heme and a Type II ligand drawn by Blender with the Molecular Nodes add-on from the real
# coordinates; the film falls back to the PyMOL pictures of job 944 when a clip is missing.
need_file docs/video/blender_clips.py
need_file /Applications/Blender.app/Contents/MacOS/Blender
all_lower_done 943_blender_clips || defer "earlier jobs not finished"
python3 docs/video/blender_clips.py > $L/blender_clips.log 2>&1

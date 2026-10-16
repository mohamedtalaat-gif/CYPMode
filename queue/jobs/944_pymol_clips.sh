#!/bin/bash
source queue/lib.sh
# The real-structure stills and turntables of the film (docs/video/STORYBOARD.md): the heme pocket of CYP3A4 and the pose pools of the official ligands, rendered with PyMOL from the coordinates.
need_file docs/video/pymol_clips.py
all_lower_done 944_pymol_clips || defer "earlier jobs not finished"
python3 docs/video/pymol_clips.py > $L/pymol_clips.log 2>&1

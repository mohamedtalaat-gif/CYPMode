#!/bin/bash
source queue/lib.sh
# The public edition of the notebook (notebooks/public/cypmode_notebook_public.ipynb, .html, .pdf): the same cells run the same way, the outputs and figures that show results for the blinded compounds replaced
# by a note, built after the complete edition (job 942) from the final numbers; the build fails when a cell still names an official structure ligand or five or more blinded compounds. The complete edition
# replaces it after the challenge closes.
need_file notebooks/build_notebook.py
need_done 942_build_notebook
all_lower_done 947_notebook_public || defer "earlier jobs not finished"
python3 -m notebooks.build_notebook --public > $L/notebook_public_build.log 2>&1

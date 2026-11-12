---
title: CYPMode explorer
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: cc-by-nc-sa-4.0
short_description: The CYPMode notebook made interactive, with a simulator for each of its 129 sections.
---

# CYPMode explorer

The complete explorer of [CYPMode](https://github.com/mohamedtalaat-gif/CYPMode): the notebook of the project made interactive, with one simulator for each of its 129 sections. A simulator runs the project's own functions on small bundled data (a molecule, a pair of potencies, a noise level, a pocket in three dimensions) or replays what an engine produced. It never trains one of the models of the project and never starts a job.

The Space builds from the folders `cypmode` and `explorer/full` of the main branch of the repository, with the packages pinned in `explorer/full/requirements.txt`. The files that build it are `explorer/space/Dockerfile` and `explorer/space/README.md` in the repository. The same explorer runs on a computer with the launcher of the repository, and the explorer that runs in the browser without a server, on the pages of the project site, shows the ledger of gates, the selection rules, the poses of the benchmark, and the out-of-fold predictions.

The notebook, the explorer, and the method report are licensed under CC BY-NC-SA 4.0 and the code under AGPL-3.0-or-later with an additional term (see the `NOTICE` file of the repository).

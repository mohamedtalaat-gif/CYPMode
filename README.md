<!-- public:card begin -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/card/cypmode_card-dark.gif">
    <img src="docs/card/cypmode_card-light.gif" alt="CYPMode: the CYP3A4 enzyme turning once, with its heme and fluconazole, six public crystal complexes with their compounds and the headline numbers" width="100%">
  </picture>
</p>

<p align="center"><a href="https://mohamedtalaat-gif.github.io/CYPMode/card/">Open the interactive card</a>: rotate the enzyme, load each of the six public crystal complexes and read the iron&ndash;nitrogen distance measured on the deposited coordinates. GitHub does not run scripts in a README, so the interactive version is a page of its own. The predicted structures of the project are published after the challenge closes.</p>
<!-- public:card end -->

<!-- held-back:card
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/card/cypmode_card_full-dark.gif">
    <img src="docs/card/cypmode_card_full-light.gif" alt="CYPMode: the CYP3A4 enzyme turning once, with its heme and a ligand, six complexes with their compounds and the headline numbers" width="100%">
  </picture>
</p>
<p align="center"><a href="https://mohamedtalaat-gif.github.io/CYPMode/card/">Open the interactive card</a>: rotate the enzyme, load each of the six complexes and read the iron&ndash;nitrogen distance measured on the coordinates. GitHub does not run scripts in a README, so the interactive version is a page of its own.</p>
held-back:card-end -->

## What CYPMode is

**CYPMode is a set of reproducible predictors, with the evidence behind them, for one question: will a drug-like molecule switch off a cytochrome P450 enzyme, and does it keep it off?**
Cytochrome P450 enzymes such as CYP3A4 and CYP2D6 clear most medicines from the body. A compound can block an enzyme while it is present (direct inhibition) or after the enzyme has
processed it (time-dependent inhibition, TDI: a reactive product stays bound), and TDI is the form that makes drug-drug interactions dangerous. Many inhibitors bind the iron of the
enzyme's heme through a nitrogen atom (a Type II ligand), which is also why the geometry of a bound compound carries information.

The repository answers the question for the two tasks of the OpenADMET CYP Inhibition Blind Challenge:

- **TDI classification (CYP3A4 and CYP2D6).** For each of the 750 blinded compounds: is it a time-dependent inhibitor? The label is a function of two measured potencies: it is positive
  exactly when the pIC50 after pre-incubation with the enzyme exceeds the direct-inhibition pIC50 (floored at 4) by more than log10 2, a factor of two. The rule was reproduced exactly from the
  training file.
- **CYP3A4 structure prediction.** For each of 20 ligands: the three-dimensional complex with the enzyme (Boltz-2 co-folding, with an iron-nitrogen restraint for ligands that have a Type II
  donor), every pose checked under the organisers' own PoseBusters call.

**How it works.** For TDI, label-free features (fingerprints, descriptors, motif flags, GFN2-xTB physics) feed a stack of diverse models; the blend is cut by transferring the expected rate of
positive calls of the scored population. CYP2D6 has one more candidate, a selection-aware model, because compounds were not tested on every enzyme at random. For structures, a recipe of restraint,
pose-validity and margin rules in which every lever was decided by a gate on crystal-structure benchmarks.

**What is different here.**

1. *Evaluation without selection bias.* Models are judged by nested cross-validation that reproduces the scored population (similarity to the blinded set, test prevalence, only confidently labelled
   compounds), with every selection step inside the loop, and compared with the corrected resampled t test of Nadeau and Bengio.
2. *Pre-registered gates.* The rule an idea must pass is written down before the idea is tested. An idea that fails stays in a ledger of negative results; it is not tuned until it passes. The
   ledger and its counts are generated from the run logs: [NEGATIVE_RESULTS.md](NEGATIVE_RESULTS.md) and [METHOD_REPORT.md](METHOD_REPORT.md).
3. *Generated and audited.* The reports are produced by code from the logs; an automated audit runs twice and must agree byte for byte; a leakage audit checks that no test label and no
   board-derived label is used.
4. *Uncertainty by construction.* Reliability tiers by similarity to the training data (`applicability_domain.py`), probabilities calibrated on inner out-of-fold scores, and the disagreement of the
   committee, reported with every prediction.

**What each of these rests on.** The table is written by code from the logs, the manifests and the audit; a number that changes in a log changes here.

<!-- generated:innovation begin (python -m cypmode.audit.docs_state; the text between the markers is rewritten) -->
| claim | evidence | source |
| --- | --- | --- |
| The label is a function of two measurements; the models learn the measurements | 0 and 0 mismatches of the label rule over 2334 and 1493 rows with both arms | audit section 4 |
| A slice of the CYP3A4 labels follows another convention and is left out | 1249 rows without a direct measurement | audit section 4 |
| Selection-aware CYP2D6 model; the file is chosen by a rule fixed before any upload | AUC 0.5618 to 0.7278 (+0.1660, corrected p = 0.026) on a scored set that looks like the training panel; the rule gives file A | `population_panel_CYP2D6_r3.log`, `selection_decision_CYP2D6.log` |
| Every idea passes a gate written before its result | 187 verdicts in 59 families, 9 MET | `NEGATIVE_RESULTS.md` |
| Reliability tiers by similarity to the training data, intervals over chemical series | blinded compounds supported: 750 of 750 (CYP3A4), 0 of 750 (CYP2D6) | `applicability_domain.log` |
| Positive rates of the scored set recovered from the public boards | 0.310 (CYP3A4), 0.059 (CYP2D6) | limitation L13 |
| The same rates estimated without a board, by label shift, under a rule fixed before the first run | 0.135 (0.043 to 0.286) against 0.059 for CYP2D6: NOT INFORMATIVE; 0.352 (0.274 to 0.451) against 0.310 for CYP3A4: NOT INFORMATIVE | `label_shift.log` |
| Restraint of the iron-nitrogen bond for Type II ligands | gate MET on the crystal benchmark, MET under the OpenStructure scorer | `structure_gates.log`, `ost_pool_gates.log` |
| The pharmacophore of a pose read over 20 ps of ligand dynamics in the frozen pocket, under a rule fixed before any number | mean rho with minus the true RMSD over 10 ligands, dynamic pharmacophore -0.214, stability +0.042, dynamic combined -0.071 against -0.240 for the static model: gate NOT MET for all 3 | `ligand_dynamics.log` |
| Validity margin of the scorer's own call and a relaxation under a registered rule | relaxation adopted for 20 of 20; smallest margin 0.801 to 0.849 | `relax_final.log` |
<!-- generated:innovation end -->

**What it does not claim.** Cross-validation estimates are not the challenge score: the challenge scores hidden labels, and its test set is closer to the training set than a cross-validation
fold is. A predicted structure is a model, not a measurement. The numbers reported here come from nested cross-validation on the training data and from checks of the predicted coordinates. One piece of aggregate information about the hidden labels is used: the positive rate of the scored set, which the public leaderboards make recoverable from the accuracy, precision and recall they publish (it weights the yardstick and sets the call rate); no label of any compound and no probing submission is used, and the method report lists it as a limitation.

## Reader's guide

| Question | Where to read or open |
| --- | --- |
| What is this, in two minutes? | the section "What CYPMode is" above and [the interactive card](https://mohamedtalaat-gif.github.io/CYPMode/card/) |
| What is the method, with every gate and number? | [METHOD_REPORT.md](METHOD_REPORT.md): "Summary and disclosures" and "Innovation summary" first, then section 3 (the two architectures step by step), section 4 (what was explored and how the number of attempts is accounted for) and section 5 (how validation is made to resemble the scored set) |
| What did not work? | [NEGATIVE_RESULTS.md](NEGATIVE_RESULTS.md), and sections 4.7 and 12 of the method report |
| How does it all work, from first principles, with no background assumed? | the public edition of the notebook: `notebooks/public/cypmode_notebook_public.html` (one file, opens in a browser), also as `.pdf` and `.ipynb`. Part 0 (Orientation) comes first; then the part that is needed: the data (II), validation without selection bias (VI), the final model (VIII), the additions that were tested (IX), quantum chemistry and physics (X), the structure track (XI). Every section says where it sits in the story and teaches the science it uses; Appendix A is the practical walkthrough and Appendix D says which section explains which file. The full edition is released after the challenge closes |
| How to explore the numbers? | the interactive explorer, a marimo notebook that reads a small public data bundle and never trains a model or runs a job: `explorer/run_explorer.command` (macOS), `explorer/run_explorer.bat` (Windows) or `bash scripts/explorer.sh run`; the first start builds the environment `.venv-marimo` by itself (a few minutes, once), so nothing has to be installed or activated first |
| Where is everything, and what depends on what? | the section "Map of the project" below and the four pages of [docs/map](docs/map/README.md), written from the files of the repository by `python -m cypmode.audit.project_map` |
| How to check the claims? | [audit/AUDIT_REPORT.md](audit/AUDIT_REPORT.md) (generated twice and compared byte for byte), the tests in `tests/` and `bash scripts/run_audit.sh` |
| How to reproduce it? | Quick start below, `bash scripts/reproduce_tdi.sh` and the jobs of `queue/jobs/` |
| Where is the design record? | [CYPMode_Architecture_Specification.md](CYPMode_Architecture_Specification.md) and the architecture sections of this file |

<!-- generated:project-map begin (python -m cypmode.audit.docs_state; the text between the markers is rewritten) -->
## Map of the project

The pages of [docs/map](docs/map/README.md) show where everything is and what depends on what: the code, the queue, the flow of the data and the evidence. The interactive map draws the same four views as one page that can be explored: a graph of the modules, a grid of the jobs, a search box, and for each module and job what it imports, what imports it, which job runs it, which test imports it and which section of the notebook names it. The loop below is the page itself.

<p align="center"><a href="https://mohamedtalaat-gif.github.io/CYPMode/map/"><img src="docs/map/preview/explorer.gif" alt="The interactive map of the project: a search for a word, a module with what it imports and what imports it, the job that runs it, the flow of the data and the kinds of idea" width="90%"></a></p>

**Open the interactive map:** [on the Pages site](https://mohamedtalaat-gif.github.io/CYPMode/map/). **Download it:** [explorer.html](https://github.com/mohamedtalaat-gif/CYPMode/raw/main/docs/map/explorer.html) (save the file with a right click and open it in a browser; it is one file and needs no network).

The pages list every module with what it does and what it imports, and every job of the queue with what it waits for.
<!-- generated:project-map end -->

<!-- held-back:film
## The film

A 6.5-minute animated film explains the project from the question to the results, scene by scene, with nine characters who each stand for one part of the work (the assay, the data scout, the
features, the stack, the statistician, the physics, the pretrained models, the structure track and the audit). Every number on screen is read from this repository's own files, every molecule and pocket is
drawn from the coordinates, and the voices are synthesised (Kokoro-82M). Click the picture to play the film (H.264, `docs/video/cypmode_overview_web.mp4`); the storyboard and the build are in
[`docs/video/STORYBOARD.md`](docs/video/STORYBOARD.md).

<p align="center">
  <a href="docs/video/cypmode_overview_web.mp4">
    <img src="docs/video/cypmode_overview_poster.png" alt="The poster of the CYPMode film: the title, the question, four numbers and the whole team" width="90%">
  </a>
</p>

held-back:film-end -->

## Origin of the project and the two parts of this file

CYPMode began as a heme-coordination motif screen for CYP450 inhibitors with a structure-based check (described in the second part of
this file). It now also contains an entry for the OpenADMET CYP blind challenge:

- **TDI classification** (time-dependent inhibition of CYP3A4 and CYP2D6): a label-process-aware model evaluated with a nested,
  selection-corrected cross-validation on the same population the challenge scores.
- **CYP3A4 structure prediction**: Boltz-2 (and optionally OpenFold3) co-folding with a gated pose-selection recipe.

Every addition to a model passes through a pre-registered gate (corrected resampled t, both confidence cut-offs, alone and inside the
assembled stack). Results are generated by code from the run logs: [METHOD_REPORT.md](METHOD_REPORT.md),
[NEGATIVE_RESULTS.md](NEGATIVE_RESULTS.md) and [audit/AUDIT_REPORT.md](audit/AUDIT_REPORT.md). The queue (`queue/jobs/`) holds the job that runs each experiment, `bash scripts/queue_status.sh` prints what is done, running and waiting, and
`python -m cypmode.audit.checklist` checks that every job names modules and scripts that exist.

## Architecture: TDI classification

```
                         SMILES  (training compounds + the blinded test structures; features are label-free)
                            |
        +-------------------+----------------------------+
        v                                                v
 2D FEATURE BLOCKS                              PHYSICS BLOCKS (GFN2-xTB via tblite, own process)
 Morgan counts, RDKit descriptors,              tier A   molecule: conformer ensemble, vertical IP / EA, hardness,
 MACCS, Avalon, ErG, torsion,                              solvation, shape
 feature-Morgan, motifs, alerts                          atom: charges, Fukui f- / f+, C-H abstraction energy,
                                                           proton affinity of N atoms, SASA
 opt-in blocks:                                          site: per-role statistics (benzylic, alpha-N, azole N ...)
 CheMeleon embeddings (frozen, 64 PCs)                   pharmacophore: heme-binder <-> oxidation-site distances
                                                tier B   Boltzmann-weighted conformers, locally relaxed hydrogen abstraction
                                                compact  the 20 physics columns with the largest gain, chosen from the
                                                         training rows of every task (evaluation rows never pick columns)
        |                                                |
        +-------------------+----------------------------+
                            v
          MODEL MEMBERS  (nested CV; A to D are candidates of the default stack, and an added group enters a stack only when its gate is met)
 +----------------------+---------------------+--------------------------+--------------------------+
 | A  Classifiers       | B  SAR-aware pairs  | C  Label-process model   | D  Selection-aware       |
 | LightGBM (A, A+fps,  | each compound vs    | committee of two-arm     | CYP2D6 only              |
 | A+physics, A+compact | its nearest labelled| regressors (TDI arm and  | P(label) =               |
 | physics, A+arms of   | neighbours: deltas  | direct arm, floor-clipped| P(hit) x P(label | hit)  |
 | the other isoforms), | of descriptors and  | at 4, noise-weighted)    | gate fitted on the       |
 | ExtraTrees, RF,      | of physics; matched | -> exact P(shift >       | all-isoform panel block  |
 | kNN / SVM (Tanimoto),| molecular pairs     | log10 2) and expected    | (4,352 compounds); the   |
 | two-arm regressors   |                     | improvement              | physics block is an      |
 |                      |                     |                          | optional input of its    |
 |                      |                     |                          | gate and label models    |
 +----------+-----------+----------+----------+------------+-------------+------------+-------------+
 +--------------------------------------------------------------------------------------------------+
 | E  Learned representations: joined the CYP3A4 stack (for CYP2D6 no group passed its gate)        |
 |    neural: multitask network over the assay arms + plain Chemprop (graph network, arm shift)     |
 |    chemprop_foundation: Chemprop started from two pretrained OpenADMET models, fine-tuned        |
 +--------------------------------------------------------------------------------------------------+
            +----------------------+-----------------------+--------------------------+
                                   |   optional training weights: similarity to the test structures
                                   |   (density ratio, tempered); measurement-variance weights in C
                                   v
            STACK    logistic blend of the members' inner out-of-fold scores (L1 / L2, chosen by inner CV)
                                   v
            CYP2D6 only   HEDGED BLEND  rank blend of the hits-only stack and the selection-aware stack, judged across
                          regimes of the share of never-promoted compounds in the scored set (label-shift estimate of that
                          share from the test structures: test_nonhit_share); written as a candidate file only if its
                          minimax gate is met
                                   v
            CALIBRATION   raw, Platt or isotonic, whichever meets ECE <= 0.05 and slope in [0.8, 1.2]
                                   v
            DECISION      positive calls = a pre-declared multiple of the scored prevalence (rate transfer),
                          the top-scoring compounds of the test file are called
                                   v
            candidate files (schema checked with the challenge portal's own rules) + SHA-256 manifest
```

How the sketch is evaluated and assembled:

- **Nested evaluation.** 5 outer folds x 3 repeats; every selection (members, stack weights, calibration, threshold rule) happens inside
  the training part of each outer fold. Comparisons use the corrected resampled t of Nadeau and Bengio with Holm adjustment.
- **Gate.** An addition counts only when it beats its reference with corrected p < 0.05 at both confidence cut-offs, alone and inside the
  assembled stack (`final_tdi_build.py` reads the verdict lines and applies one fixed rule; nothing is chosen by hand).
- **Replicate.** The physics additions are re-judged with 10 outer folds (`--design 10`, variance factor 0.144 instead of 0.317); an addition
  counts as replicated only if the 10-fold gate is met and the 5-fold sign is positive (`replicate_verdict.py`).
- **Reproduction.** `scripts/reproduce_tdi.sh` rebuilds the candidate files from the raw files and checks them against the portal rules.

Label rule (recovered from the data and reproduced exactly): `s = t - max(d, 4) > log10 2`, with `t` the TDI-condition pIC50 and `d`
the direct-inhibition pIC50. Evaluation weights follow similarity to the test structures, the estimated test prevalence and the
confidence of each label (`|s - log10 2| / sd(s)` at two cut-offs).

### Final stack and verdicts

<!-- generated:final-stack begin (python -m cypmode.audit.docs_state; the text between the markers is rewritten) -->
Final stack per isoform, from the nested cross-validation on the population the challenge scores (z0 is the confidence cut-off of the label weights; MCC is at the rate transferred from the scored prevalence):

| isoform | stack | members | groups added by a met gate | AUC (z0 1.0) | AUC (z0 1.96) | MCC (z0 1.0 / 1.96) |
| --- | --- | --- | --- | --- | --- | --- |
| CYP3A4 | default+all_survivors | 28 | ['chemprop_foundation', 'neural'] | 0.834 | 0.862 | 0.476 / 0.539 |
| CYP2D6 | default | 23 | none | 0.731 | 0.754 | 0.167 / 0.204 |

Members of the default stack: k-nearest neighbours (Tanimoto) (3), support vector machine (Tanimoto kernel) (3), LightGBM classifier (11), extremely randomised trees (1), random forest (1), LightGBM regressor of the arm shift (2), LightGBM regressors of the two arms (2); 23 members in all.
<!-- generated:final-stack end -->

<!-- generated:gate-verdicts begin (python -m cypmode.audit.docs_state; the text between the markers is rewritten) -->
Every addition was judged by a rule fixed before its result. 187 pre-registered verdicts from 59 families of ideas: 9 MET and 178 NOT MET. The logs print 29 more gate lines, which are not in that count: the negative control (a MET verdict there means that the pipeline gives chance on permuted labels, as it must) and the exploratory second looks. The gates that were met: Chemprop from the two OpenADMET pretraining weights (staged), bond restraint, contact restraint, force-field relaxation of poses, pool gates repeated with OpenStructure's yardstick, replicates with 10 outer folds, the neural pair (multitask network and plain Chemprop) in the stack. An addition to the TDI stack counts only when its gate is met alone and inside the stack; groups added to the final stack: CYP3A4: ['chemprop_foundation', 'neural']; CYP2D6: none.

<details><summary>All verdicts</summary>

| experiment | TDI CYP3A4 | TDI CYP2D6 | structure |
| --- | --- | --- | --- |
| Atom-JEPA readout in the stack | NOT MET | NOT MET | - |
| Chemprop from the two OpenADMET pretraining weights (staged) | MET | - | - |
| Emax of both arms as a predicted block | NOT MET | NOT MET | - |
| Monroe (GRIT) embeddings | NOT MET | NOT MET | - |
| PXR-induction readings as a predicted block | NOT MET | NOT MET | - |
| SAR pair transfer | NOT MET | NOT MET | - |
| SAR pair transfer with physics | NOT MET | NOT MET | - |
| TabICL in the stack | NOT MET | NOT MET | - |
| TabICL on Monroe embeddings | NOT MET | NOT MET | - |
| TabICL on frozen CheMeleon embeddings | NOT MET | NOT MET | - |
| TabPFN in the stack | NOT MET | NOT MET | - |
| TabPFN on frozen CheMeleon embeddings | NOT MET | NOT MET | - |
| Tox21 CYP-screen readings as a predicted block | NOT MET | NOT MET | - |
| aux neural f10 | MET | - | - |
| basic-centre block (protonation states, centre classes) | NOT MET | NOT MET | - |
| bond restraint confirmed on new ligands | - | - | NOT MET |
| committee label-process model | NOT MET | NOT MET | - |
| compact physics block | NOT MET | NOT MET | - |
| compact physics block, replicate with 10 outer folds | NOT MET | NOT MET | - |
| compound relative to the CYP2D6 pocket | - | NOT MET | - |
| compound relative to the CYP3A4 pocket | NOT MET | - | - |
| compounds without a direct curve | NOT MET | - | - |
| consensus of poses across a chemical series | - | - | NOT MET |
| continuous external-panel responses as a predicted block | NOT MET | NOT MET | - |
| cross-isoform transfer of the assay arms | NOT MET | NOT MET | - |
| expected-improvement sampling of poses | - | - | NOT MET |
| exploratory atomjepa head f10 | NOT MET | - | - |
| first run of the pocket block, void: control pocket | NOT MET | NOT MET | - |
| first run of the pocket block, void: own pocket | NOT MET | NOT MET | - |
| force-field relaxation of poses | - | - | MET |
| frozen Atom-JEPA embeddings as a block | NOT MET | NOT MET | - |
| frozen CheMeleon embeddings | NOT MET | NOT MET | - |
| gate-shift features | NOT MET | NOT MET | - |
| hedged blend of the hits-only and the selection-aware stacks | - | NOT MET | - |
| heme-frame prior over poses | - | - | NOT MET |
| hypothesis model with permuted labels (control) | NOT MET | NOT MET | - |
| ligand dynamics in the frozen pocket | - | - | NOT MET |
| ligand-based pharmacophore hypothesis model | NOT MET | NOT MET | - |
| logD and protein binding from the OpenADMET surrogate models | NOT MET | NOT MET | - |
| matched-molecular-pair transfer | NOT MET | NOT MET | - |
| minimum Bayes risk and restraint judged by LDDT-PLI (exploratory) | - | - | NOT MET |
| minimum Bayes risk selection and engine consensus | - | - | NOT MET |
| multitask network in the stack | NOT MET | NOT MET | - |
| multitask network with the screen and Emax as auxiliary heads | NOT MET | NOT MET | - |
| negative control (permuted labels) | MET | MET | - |
| non-negative stacking | NOT MET | NOT MET | - |
| pharmacophore model of the poses | - | - | NOT MET |
| physics block tier A, replicate with 10 outer folds | NOT MET | NOT MET | - |
| physics block tier B, replicate with 10 outer folds | NOT MET | NOT MET | - |
| physics block, tier A | NOT MET | NOT MET | - |
| physics block, tier B | NOT MET | NOT MET | - |
| physics, pharmacophore and combined pose terms | - | - | NOT MET |
| plain Chemprop in the stack | NOT MET | NOT MET | - |
| pocket block of the other enzyme (control) | NOT MET | NOT MET | - |
| pool gates repeated with OpenStructure's yardstick | - | - | mbr_selection_ost NOT MET; engine_consensus_ost NOT MET; ei_sampling_ost NOT MET; heme_frame_prior_ost NOT MET; pose_physics_ost NOT MET; pose_pharmacophore_ost NOT MET; pose_combined_ost NOT MET; series_consensus_ost NOT MET; restraint_bond_ost MET; restraint_contact_ost MET; pilot_conformers_ost NOT MET; pilot_cryoem_ost NOT MET; pilot_seeds_ost NOT MET; mbr_lddt_exploratory_ost NOT MET; restraint_bond_lddt_exploratory_ost NOT MET; mbr_on_restrained_lddt_exploratory_ost NOT MET |
| pose choice with BioEmu2 ensembles | - | - | NOT MET |
| replicates with 10 outer folds | replicate physics NOT MET; replicate physics_compact NOT MET; replicate physics_b NOT MET; replicate neural MET | NOT MET | - |
| restraint confirmation repeated with OpenStructure's LDDT-PLI | - | - | NOT MET |
| restraints and pose-diversity pilots | - | - | restraint_bond MET; restraint_contact MET; pilot_conformers NOT MET; pilot_cryoem NOT MET; pilot_seeds NOT MET |
| selection-aware CYP2D6 model with physics | - | NOT MET | - |
| selection-aware CYP2D6 model with the screen | - | NOT MET | - |
| selection-aware model, second version | - | NOT MET | - |
| single-concentration screen as a predicted block | NOT MET | NOT MET | - |
| site model of the other enzyme (control) | NOT MET | NOT MET | - |
| softmax-temperature selector | NOT MET | NOT MET | - |
| structure-based site model of the CYP2D6 pocket | - | NOT MET | - |
| structure-based site model of the CYP3A4 pocket | NOT MET | - | - |
| test-likeness weighting | NOT MET | NOT MET | - |
| the LDDT-PLI of the project against that of OpenStructure | - | - | NOT MET |
| the neural pair (multitask network and plain Chemprop) in the stack | MET | NOT MET | - |
| two-arm regressors, floor-clipped | NOT MET | NOT MET | - |

</details>
<!-- generated:gate-verdicts end -->

## Architecture: structure track

```
 ligand SMILES + CYP3A4 sequence (UniProt P08684, precomputed MSA)
        |
        v
 Boltz-2 co-folding  <---- optional levers, each used only when its gate is met ------+
        |                  Fe-donor bond / contact restraint (Type II donors)         |
        |                  crystal and cryo-EM conformer templates, seeds, step scale |
        |                  OpenFold3 as a second engine (low-memory chunking)        |
        v                                                                            |
 pose pool per ligand  <---------------------------------------------------------+
        |
        v
 PoseBusters validity filter -> selection among valid poses, mean rank of:
        expected LDDT-PLI (minimum Bayes risk)                                        (selection_validation)
        physics / molecule terms (match to the xTB conformer ensemble and its         each term enters
          energy, burial, contacts, clashes)                                           only when its
        pharmacophore terms (Fe-donor distance error, oxidation site in the           gate is met
          substrate window)                                                           (pose_awareness)
        SAR term (core agreement with the poses of the ligand's analogs, maximum
          common substructure after superposing the C-alpha atoms)                    (series, series_consensus)
        heme-frame pose prior from 148 crystal ligands                                (heme_frame_prior)
        |
        v
 extra samples allocated by pool disagreement (optional, ei_sampling) -> one pose per ligand -> zip
        |
        v
 final_recipe.py   reads the gate verdicts, applies one fixed mapping to the recipe, runs the 20 official ligands, validates with the
                   portal's rules and lists, per ligand, the donor status, the closest analog and the measured CYP3A4 values of
                   the same compound in the TDI training file
```

Benchmark for every structure gate: crystal-anchored CYP3A4 complexes (true ligand RMSD of the picked pose, PoseBusters validity),
plus a series benchmark of analog pairs that have crystal structures. The ten ligands of the crystal-anchored benchmark were picked from the CYP3A4 co-crystal ligands of the PDB by a rule fixed in advance (20 to 60 heavy atoms, resolution at most 3.0 A, organic, one fragment, then the most mutually dissimilar by a maximin pick on Morgan fingerprints); they were chosen for diversity and not for likeness to the 20 official ligands, and the models may have seen their entries in training, so the benchmark compares settings and does not forecast a score. The method report (section 11) and the notebook (section 11.5) list the ten with their PDB entries. The earlier Boltz-only zip stays untouched as the fallback.

### Final recipe

<!-- generated:structure-recipe begin (python -m cypmode.audit.docs_state; the text between the markers is rewritten) -->
How the 20 poses of the submitted zip were chosen, from the verdicts of the gates and the recipe that reads them (`cypmode/validation/final_recipe.py`):

- restraint 'bond': gate MET
- diversity levers: none (no pilot gate MET)
- Boltz only: engine_consensus not MET
- selection 'first_valid': mbr_selection not MET, so the model's own ranking among valid poses is kept; for a ligand with a Type II donor the restrained run is read first (the pick the restraint gate measured)
- extra samples allocated uniformly: ei_sampling not MET
- margin: among valid poses only those with a relative distance to the receptor of at least 0.8 under the scorer's own PoseBusters call are eligible (else the largest margin)
- force-field relaxation of the chosen poses in the frozen pocket (gate met on 30 crystal-anchored ligands): 20 of 20 poses replaced under a rule fixed before the use (both poses pass the 22 checks, the smallest relative distance to the receptor rises and stays at or above 0.8, the ligand moves by at most 1.0 A, a coordinated iron-nitrogen distance changes by at most 0.2 A); smallest relative distance 0.801 to 0.849
<!-- generated:structure-recipe end -->

## Repository map

| Path | Content |
| --- | --- |
| `cypmode/challenge/` | TDI engine: feature store, nested cross-validation harness (`nested.py`, `nested_runners.py`), label confidence, library blocks, physics blocks (`physics_block.py`, `physics_tierb.py`, `population_physics.py`), SAR models (`pair_transfer.py`, `mmp_transfer.py`), label-process committee, cross-isoform arm transfer (`xiso_transfer.py`), selection-aware models (`selection_aware_v2.py`, `hedged_blend.py`, `test_nonhit_share.py`), frozen embeddings, calibration, `replicate_verdict.py`, `final_tdi_build.py` |
| `cypmode/validation/` | structure tools: input builder, Boltz-2 / OpenFold3 runners, PoseBusters checks, pose pools, templates (`conformers.py` crystal, `cryoem_templates.py` cryo-EM), selection (`selection_validation.py`), gates (`structure_gates.py`), pose terms (`pose_awareness.py`), analog series (`series.py`, `series_consensus.py`), heme-frame prior, expected-improvement sampling, `final_recipe.py` |
| `cypmode/audit/` | automated audit report, gate ledger, checklist consistency check (`checklist.py`), method-report generator (`method_report.py`), public-readiness scan (`public_readiness.py`) |
| `cypmode/metrics/`, `cypmode/data/` | the original motif screen and panel loaders |
| `queue/`, `scripts/` | the permanent job queue (`queue/jobs/NNN_*.sh`, helpers in `queue/lib.sh`), runner, memory / battery guards (`scripts/guards.sh`), reproduction and audit scripts, the low-memory OpenFold3 launcher (`scripts/run_openfold_lowmem.py`) |
| `notebooks/` | the notebook `cypmode_notebook`: the teaching text and the laboratory notebook of the project, built from `notebooks/parts/` by `python -m notebooks.build_notebook` (ipynb, html, pdf); the public edition is in `notebooks/public/`, the full edition is released after the challenge closes |
| `environment/` | pinned package lists for the Python environments (main, torch and explorer) with checksums |
| `explorer/`, `cypmode/explorer/` | the interactive explorer of the finished results, a reactive marimo notebook that reads only a small public data bundle (`explorer/public`, built by `cypmode/explorer/data.py`): the gate ledger, what a different threshold would have let through, the selection rules fold by fold, the benchmark poses and the stored out-of-fold predictions |
| `tests/` | test suite (the torch and tblite files run in their own processes, see below) |
| `data/` | inputs, caches and outputs of both tracks |
| `app.py` | Streamlit app: the original screen and a TDI results tab that reads the generated final report and gate ledger |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[challenge,validate]"
python -m pytest tests -q                       # main suite
python -m pytest tests/test_quantum_pharmacophore.py tests/test_physics_block.py tests/test_physics_tierb.py -q   # tblite, own process
bash scripts/run_audit.sh                       # audit report, generated twice and compared byte for byte
bash scripts/reproduce_tdi.sh                   # raw files -> validated TDI candidate files (hours, resumable)
bash scripts/queue_start.sh                     # start the permanent queue; bash scripts/queue_status.sh to inspect it
python -m cypmode.audit.checklist --status      # what is done, running, queued
python -m cypmode.audit.method_report           # regenerate METHOD_REPORT.md and NEGATIVE_RESULTS.md from the logs
python -m cypmode.audit.public_readiness        # scan for private paths, credentials, large files, third-party text
bash scripts/mirror_check.sh                      # after a push: remote branch, HEAD and working tree are identical
python -m notebooks.build_notebook              # rebuild the notebook (ipynb, html, pdf)
bash scripts/explorer.sh run                    # the interactive explorer; the first start builds .venv-marimo by itself; also edit, html, ipynb, wasm and plan
```

Torch-based tests (`tests/test_multitask.py`, `tests/test_chemberta_features.py`) and the tblite tests must not share a process with
LightGBM; the section "Contributing" at the end of this file explains why and how each is run.

## Credits and sources

Published sources are listed in [CITATION.cff](CITATION.cff) and in section 14 of [METHOD_REPORT.md](METHOD_REPORT.md).
How to cite this work, what counts as using it and what the licence can and cannot do are in [USE_AND_CITATION.md](USE_AND_CITATION.md).

---

# Original scope: mode-aware reinterpretation of TDC's CYP450 inhibition benchmarks

## Problem

[Therapeutics Data Commons](https://tdc.hub.biolib.com) provides binary inhibitor and non-inhibitor labels for the five major cytochrome P450 isoforms (`CYP1A2_Veith`, `CYP2C9_Veith`, `CYP2C19_Veith`, `CYP2D6_Veith`, `CYP3A4_Veith`, from Veith et al., *Nat. Biotechnol.* 2009). Every ADMET model trained on them inherits the same limitation: the label states *whether* a compound inhibits a CYP and not *how*. Pharmacologically, the "how" is the difference between Type I inhibition (reversible, with the substrate in the active site and the axial water of the heme in place) and Type II inhibition (a nitrogen lone pair coordinates the heme iron directly and displaces that water), a much tighter and often clinically more significant interaction (de Groot, *Drug Discov. Today*, 2006, 11:601-606). A binary "1" collapses both into the same bit.

The framing was checked against the data. Across the five isoforms only 26 to 31% of the labeled inhibitors of TDC carry any heme-coordinating motif (`cypmode/data/panel.py`, `summarize_panel()`: CYP1A2 28.1%, CYP2C9 30.0%, CYP2C19 28.2%, CYP2D6 26.5%, CYP3A4 30.6%; n = 2,514 to 5,829 labeled inhibitors per isoform). Most of the compounds that carry the "inhibitor" label of the panel are therefore not expected by a motif screen to coordinate the heme directly, which is the information that a plain binary label discards.

The project is scoped as a direct extension of a newly solved CYP3A4 cryo-EM structure paper (ketoconazole, ritonavir, azamulin and vardenafil bound) and not as a re-annotation of the whole panel by assertion. `cypmode/validation/structures.py` checks the calls of the motif screen against physics-based structure prediction (Boltz-2) for exactly the four compounds for which that paper solved structures. Physics-based validation of the full panel (about 62,000 compounds) would take on the order of 2,000 GPU-hours on this hardware, and it would be a different, unvalidated claim and not a rerun of the method of that paper.

## Contents of the original scope

- `cypmode/metrics/motifs.py`: a fast RDKit/SMARTS screen for the classic heme-coordinating substructures (imidazole, triazole, pyridine, thiazole), in the structural-alert tradition of Leach, *Molecular Modelling: Principles and Applications* (2001, ch. 12, section 12.2). It takes seconds for the whole panel and is not a mechanism predictor on its own.
- `cypmode/data/panel.py`: loads and mechanism-annotates the five-isoform CYP inhibition panel of TDC.
- `cypmode/data/octant_benchmark.py`: checks the motif screen along a different, independent axis, the CYP3A4 dose-response potency from a diversity-library screen of about 1,200 compounds (OpenADMET/Octant), in place of the binary labels of TDC or the four structurally validated reference compounds (see "Independent large-N potency check" below).
- `cypmode/validation/build_input.py`: builds a Boltz-2 YAML input (protein and heme, covalently bonded) for any ligand SMILES and locates the coordinating cysteine on any given protein sequence instead of assuming a fixed residue number. None of the four original reference inputs had been generated by code, and this module closes that gap.
- `cypmode/validation/pipeline.py`: `run_structural_validation(name, smiles)` is one call from a SMILES string to a coordination result for a compound that the four reference inputs do not cover. It wraps `build_input`, a `boltz predict` subprocess run (tens of minutes, not skipped or simulated) and the summary of `structures.py` in one function instead of three manual steps.
- `cypmode/validation/structures.py`: parses the structures predicted by Boltz-2 (the heme is modeled explicitly and bonded covalently to the axial cysteine) and measures the distance of the heme iron to the ligand nitrogens that match a Type II motif, and not only to the geometrically closest one, which is not always the same atom (see the methodology note below).
- `cypmode/validation/batch.py`: runs `run_structural_validation` over a compound set (name to SMILES) in one call, checkpoints the results after every compound so that a run of several hours survives an interruption, and isolates the failure of one compound from the rest of the batch.
- `cypmode/validation/build_summary.py`: regenerates `data/structure_track/outputs/cyp3a4/validation_summary.json` by discovering finished runs under `data/structure_track/outputs/out/` and reading the SMILES of each from its own input YAML (see "Reproducing this" below).
- `app.py`: a Streamlit tool in which a pasted compound receives the motif call and, for the four reference compounds, the structural check.
- `docs/figures/generate_validation_figure.py`: regenerates the Fe-N bar chart below (light and dark variants) from `data/structure_track/outputs/cyp3a4/validation_summary.json`; it needs `pip install matplotlib`, which is not otherwise a dependency of the project.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows (cmd/PowerShell): .venv\Scripts\activate
pip install -e .
```

The five-isoform panel is cached under `data/tdi_track/inputs/tdc_cache/`, so the default install (RDKit, pandas and Streamlit) is sufficient to run `cypmode.data.panel` and the motif screen of the app without a network call. To refresh the cache from the source, the `tdc` extra is installed together with `PyTDC` itself as a separate `--no-deps` step, because the dependency tree of PyTDC includes packages that fail to build on Windows and that the project does not need:

```bash
pip install -e ".[tdc]"
pip install --no-deps "PyTDC==1.1.14"
```

`pip install -e ".[validate]"` adds Biopython, which is needed only to parse the Boltz-2 structures in `cypmode/validation/structures.py`.

## Run the app

Double-click `run_cypmode.command` (macOS) or `run_cypmode.bat` (Windows) and choose the screening tool or the interactive explorer. The explorer also has its own launcher in its folder, `explorer/run_explorer.command` (macOS) and `explorer/run_explorer.bat` (Windows). The same two can be started from a terminal:

```bash
streamlit run app.py                     # the screening tool and the results tab of the TDI track
bash scripts/explorer.sh run             # the interactive explorer of the finished results (builds its own environment on the first start)
```

## Hosting

`requirements.txt` is prepared for Hugging Face Spaces (Streamlit SDK) and for Streamlit Community Cloud. The Structural validation tab of `app.py` reads the checked-in `data/structure_track/outputs/cyp3a4/validation_summary.json` and does not need a local Boltz-2 run, so it behaves identically hosted and local.

## Validation against crystal and cryo-EM structures

Boltz-2 ([jwohlwend/boltz](https://github.com/jwohlwend/boltz), MIT) was used to predict CYP3A4-ligand complexes for the four compounds for which a recent cryo-EM paper (co-authored by Pat Walters) solved structures. The first attempt left the heme cofactor out of the model; this was found before any conclusion was drawn from it, because the pose of a ligand docked into the apo protein can neither confirm nor refute heme coordination when the structure contains no heme. The corrected input includes the heme (PDB ligand code `HEM`) as a third chain, bonded covalently (`constraints: bond`) to Cys442, the axial thiolate ligand, which was located by finding the conserved P450 signature motif `FxxGxxxCxG` in the UniProt P08684 sequence and not assumed from memory. No constraint is placed on the position of the *inhibitor* relative to the heme, so a short predicted iron-to-ligand-nitrogen distance is a structural prediction that the input did not force.

### Reproducing this, and validating a new compound

The `data/structure_track/inputs/*.yaml` files are the exact inputs. The predicted structures and the confidence and affinity JSON that they produce are not committed (they can be regenerated and are large); only the small numeric summary derived from them is. `boltz` needs its own virtual environment (Python 3.10 to 3.12; it pins PyTorch Lightning independently of the other dependencies of the project), which is set up once:

```bash
python3 -m venv .venv-boltz
source .venv-boltz/bin/activate
pip install boltz
```

For a compound that the project does not already cover, a blind-challenge entry for instance, `cypmode.validation.pipeline` is the one call from a SMILES string to a result. It runs in the `.venv` of the project and not in `.venv-boltz`, and it calls `.venv-boltz/bin/boltz` as a subprocess:

```bash
pip install -e ".[validate]"
python -m cypmode.validation.pipeline azamulin_v2 "CC[C@]1(C)C[C@@H](OC(=O)..."
```

The command builds the YAML (against CYP3A4 by default; `--protein-sequence` selects a different target, and a different construct can shift the residue numbering, so Cys442 is not assumed to carry over unchanged), runs `boltz predict` (tens of minutes, not skipped; the output is streamed while it runs and not held until the process exits) and prints the structural-validation result as JSON. The same function is available in Python as `run_structural_validation()`.

The runs use `--override` by default. It was confirmed on a fresh compound without any earlier cache that Boltz-2 can exit with status 0 after predicting the structure and having silently skipped the affinity step; `--override` makes the step run, and it still reuses the expensive MSA and preprocessing cache when the input has not changed, so it costs nothing beyond forcing the prediction step to complete. `--no-override` opts out.

To reproduce one of the four checked-in reference compounds by hand, or to run `boltz predict` with different flags, `cypmode.validation.build_input` builds the same YAML that `cypmode.validation.pipeline` generates internally:

```python
from cypmode.validation.build_input import build_boltz_yaml, find_coordinating_cysteine

cys = find_coordinating_cysteine(protein_sequence)  # to be verified per target; 442 does not necessarily carry over
yaml_text = build_boltz_yaml(protein_sequence, ligand_smiles, cys)
```

`--cache` is where the model weights and the CCD chemical-component cache are downloaded on the first run (about 7.6 GB), a one-time cost of using Boltz-2 that is not specific to this project. `--use_msa_server` generates the MSA through the public MMseqs2 server in place of a precomputed one. `--accelerator gpu` resolves to Metal/MPS on Apple Silicon or to CUDA on NVIDIA. Each compound took about 40 minutes on an Apple M-series chip.

Once result directories exist under `data/structure_track/outputs/out/`, the checked-in summary and the figure are regenerated. `build_summary` discovers the compounds from that directory and reads the SMILES of each from its own input YAML, in place of a separate hard-coded list:

```bash
pip install -e ".[validate]"   # in the .venv of the project, not in .venv-boltz
python -m cypmode.validation.build_summary
python docs/figures/generate_validation_figure.py   # needs matplotlib
```

**Methodology note (2026-09-24).** The distance first reported here was that of "the closest ligand nitrogen, whichever one that is". This is wrong in general, as a fifth case showed: for vardenafil, the geometrically closest ligand nitrogen (4.84 Å) is not part of any matched Type II motif; it is a different nitrogen that is chemically irrelevant to heme coordination. `cypmode/validation/structures.py` now reconstructs the bonds of the ligand through the `AssignBondOrdersFromTemplate` function of RDKit (the technique that `cyp-heatmap` uses), so that the reported distance is that of the closest **motif-matched** nitrogen, and the distance of every candidate nitrogen remains available in `all_ligand_nitrogen_distances` for review. The `coordinated` call for all four compounds is unchanged; the distance reported for vardenafil moved from 4.84 Å to 10.54 Å, so the evidence for "not coordinated" became stronger once the correct atom was measured.

Results, from `summarize_compound()` of `cypmode/validation/structures.py` run against the local output in `data/structure_track/outputs/out/`:

| Compound | Motif call | Closest motif-N distance | Coordinated | `affinity_probability_binary` | `confidence_score` |
|---|---|---|---|---|---|
| Ketoconazole | Type II (imidazole) | 2.05 Å | yes | 0.86 | 0.92 |
| Ritonavir | Type II (thiazole) | 2.19 Å | yes | 0.40 | 0.92 |
| Azamulin | Type II (triazole) | 13.41 Å | no | 0.73 | 0.93 |
| Vardenafil | Type II (imidazole_Nsub) | 10.54 Å | no | 0.42 | 0.93 |

![Bar chart of heme iron to closest motif-matched ligand nitrogen distance for ketoconazole, ritonavir, azamulin, and vardenafil, with a dashed line at the 2.6 Å dative-coordination cutoff. Ketoconazole (2.05 Å) and ritonavir (2.19 Å) fall below the cutoff and are colored as coordinated; azamulin (13.41 Å) and vardenafil (10.54 Å) fall above it and are colored as not coordinated.](docs/figures/fe_n_validation-light.svg#gh-light-mode-only)
![Bar chart of heme iron to closest motif-matched ligand nitrogen distance for ketoconazole, ritonavir, azamulin, and vardenafil, with a dashed line at the 2.6 Å dative-coordination cutoff. Ketoconazole (2.05 Å) and ritonavir (2.19 Å) fall below the cutoff and are colored as coordinated; azamulin (13.41 Å) and vardenafil (10.54 Å) fall above it and are colored as not coordinated.](docs/figures/fe_n_validation-dark.svg#gh-dark-mode-only)

Two of the four calls of the motif screen hold up structurally, and the other two are informative in different ways and are not simple failures.

- **Ketoconazole and ritonavir** fall inside the range of a dative bond (2.0 to 2.3 Å is the range shown by solved Type II CYP structures; Poulos and Johnson, ch. 3 in Ortiz de Montellano (ed.), *Cytochrome P450*, 3rd ed., 2005), with no constraint forcing that outcome. Ritonavir is the compound for which the thiazole motif was added (see the git history), and its coordination confirms that change structurally and not only by substructure match. The `affinity_probability_binary` of Boltz-2 for ritonavir (0.40) is lower than that for ketoconazole (0.86), although both coordinate the heme at a similar distance and confidence: the geometric signal and the learned affinity-probability signal do not fully agree here, and the project has no explanation for the difference.
- **Azamulin** does not coordinate (13.41 Å, far from the pocket in this pose) although it matches the triazole motif. The reason was checked and the result was not taken as a false negative: azamulin is an established mechanism-based (suicide-substrate) inactivator of CYP3A4 and not a reversible Type II coordinator; CYP3A4 metabolizes it into a reactive intermediate that modifies the enzyme covalently, a turnover-dependent process that a static co-folding prediction cannot represent (Lim et al., *Drug Metab Dispos*, 2005, [10.1124/dmd.104.003475](https://doi.org/10.1124/dmd.104.003475); Sevrioukova, *Int J Mol Sci*, 2019, [10.3390/ijms20174245](https://doi.org/10.3390/ijms20174245)). The "Type II" call of the motif screen is misleading here in a specific and now documented way: the inhibition is real and the mechanism category is wrong. One caveat cannot be excluded: the announcement of the cryo-EM paper states that current co-folding models struggle with CYP3A4 because of conformational plasticity and F/G-loop remodeling that the crystal structures in their training data do not capture (Walters et al., LinkedIn, 2026-09; preprint "Structural Plasticity and Ligand Promiscuity of CYP3A4 Revealed by Cryo-EM", bioRxiv). The large Fe-N distance of azamulin is consistent with its documented suicide-substrate mechanism, but a co-folding failure of the kind that the announcement describes is an alternative or compounding explanation that a single static prediction cannot exclude, and the question is left open.
- **Vardenafil** does not coordinate either (10.54 Å). This resolves the question that was open for it in favor of a false positive: its `imidazole_Nsub` match lies inside a fused pyrazolo-pyrimidinone core, and the structural check shows that it does not behave as a discrete coordinating azole.

### Generalization check, not a reference compound

The four results above are the validated claim: motivated by TDC, scoped to compounds with independently solved cryo-EM structures, and cross-checked below. Separately, `cypmode.validation.pipeline` was run end to end on **itraconazole** (the ChEMBL SMILES, another triazole antifungal of the same class as ketoconazole, never used to build or tune the motif patterns, the cysteine search or the atom matching of the project) purely as a test of the wiring: whether the pipeline works on a compound for which there is no answer key.

The run completed (Fe-N = 2.02 Å, coordinated) and exposed a defect: Boltz-2 exited with status 0 after predicting the structure and silently skipped the affinity step on a fresh compound, with no stale cache to explain it (fixed by defaulting to `--override`; see the git history). The result of 2.02 Å is chemically unsurprising for a triazole of the class of ketoconazole, which is the purpose of the check: it confirms that the machinery generalizes and found a reliability gap, and it does not independently validate the coordination of itraconazole in the way that the four compounds above are validated. Itraconazole is therefore not in `data/structure_track/inputs/` as a tracked reference input; the YAML generated by `build_input.py` for it can be reproduced from the command above and is not presented as a result.

### Independent cross-check

[Walters, `cyp-heatmap`](https://github.com/PatWalters/cyp-heatmap) (pushed 2026-09-20) fingerprints 122 CYP3A4 structures with [ProLIF](https://prolif.readthedocs.io), a pipeline entirely different from that of the project: experimental and modeled structures in place of Boltz-2 co-folding, and interaction-type fingerprinting in place of a raw Fe-N distance. Its manifest includes the same four compounds under the same mechanistic question. The table gives its `HEM601.A_MetalAcceptor` column (the metal-coordination contact with the heme iron) from `CYP3A4_heatmap_rich_data.csv`:

| Compound | `MetalAcceptor` (cyp-heatmap) | Closest motif-N distance (this project) | Agree? |
|---|---|---|---|
| Ketoconazole (`ketaconazole-a`/`-b`) | 1 | 2.05 Å, coordinated | yes |
| Ritonavir | 1 | 2.19 Å, coordinated | yes |
| Azamulin | 0 | 13.41 Å, not coordinated | yes |
| Vardenafil (`vardenafil21`) | 0 (`HBAcceptor`=1 instead) | 10.54 Å, not coordinated | yes |

The two independent methods agree on the binary coordination call for all four compounds. The correction exposed one gap: the `HBAcceptor` hit of cyp-heatmap on vardenafil is an interaction with some heme-adjacent atom, but the Type-II-motif-matched nitrogen of this project (10.54 Å) is too far to be that atom; the more likely candidate is one of the other, closer nitrogens of vardenafil in `all_ligand_nitrogen_distances` (for example 4.84 Å), which is not part of a Type II motif and was never checked by the screen. Neither pipeline validates the numerical details of the other (a contact fingerprint is not a distance, and the structures of this project are predicted and not deposited), but the binary coordination call agrees across two unrelated approaches that were built independently, days apart, from the same cryo-EM release.

### Independent large-N potency check

The cross-check above confirms the binary coordination call of the motif screen for 4 compounds. The question of whether motif presence tracks inhibition *potency* across many more compounds than the four validated structurally was examined separately.

Source: [OpenADMET/Octant, CYP inhibition and reactivity blog release](https://huggingface.co/datasets/openadmet/Octant_CYP_inhibition_reactivity_blog_release) (CC BY 4.0), a diversity-library CYP3A4 dose-response screen of about 1,200 compounds with a 30-minute pre-incubation with the active enzyme (so the measured potency reflects combined reversible and time-dependent inhibition). It is a different chemical library from both the Veith panel of TDC and the four structural reference compounds of the project; a comparison of the SMILES confirms that only vardenafil overlaps with it.

`python -m cypmode.data.octant_benchmark` (`cypmode/data/octant_benchmark.py`) compares the CYP3A4 pIC50 of motif-positive and motif-negative compounds, restricted to the dose-response curves that pass QC, with a one-sided Mann-Whitney U test. The one compound of the release that is also in the blinded test set of the challenge is excluded and never read. Result (`data/tdi_track/inputs/octant_cache/motif_potency_validation.json`):

| | n | mean pIC50 | median pIC50 |
|---|---|---|---|
| Motif-positive | 450 | 5.32 | 5.43 |
| Motif-negative | 633 | 5.17 | 5.30 |

**p = 0.0085** (one-sided, motif-positive > motif-negative): a statistically significant difference on a dataset against which the motif screen was never tuned.

**Reading.** The difference is small (under half a log unit) between two heavily overlapping distributions. It is independent evidence that the structural feature sought by the screen tracks a pharmacological property. It is not proof that the screen predicts potency: as "Scope and limitations" below states, it was built to flag the *capability* for heme coordination and not to rank how tightly a compound binds. The result is one more independent data point in support of the underlying chemistry and does not extend what the screen claims to do.

## OpenADMET CYP Blind Challenge, Structure Prediction Track

During the preparation for the Direct Inhibition track of the [OpenADMET CYP Inhibition Blind Challenge](https://huggingface.co/spaces/openadmet/cyp-challenge), [the changelog of the challenge dataset](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test) was checked directly (and not only the announcement timeline of the challenge). It showed a third dataset configuration, `structure`, added on 2026-09-23: a blinded test set of 20 compounds for a CYP3A4 structure-prediction (co-folding) task. On 2026-09-24 this track was not yet visible in the interface or the FAQ of the challenge Space, and the data were public (Apache 2.0, as the rest of the challenge data) a day ahead of any announcement.

The task overlaps with the structural-validation pipeline of the project (Boltz-2 with an explicit heme and a covalent axial-cysteine bond, `cypmode/validation/pipeline.py`), which had been built for an earlier purpose (checking Type II motif calls against physics-based structure prediction for the 4 solved reference compounds above) and applies to any CYP3A4-ligand SMILES pair without modification.

`data/structure_track/inputs/structure_track_test_blinded.csv` is the public blinded test set of this track, saved as published and as corrected by the organisers on 2026-10-24: three ligands had first been published without their stereochemistry, every run of them was made again from the corrected SMILES, and a pose is eligible for the submission only when it carries the official stereochemistry (`cypmode/validation/pose_stereo.py`). The Type II motif screen applied to it finds that **12 of the 20 compounds (60%) carry a coordinating motif** (reproducible with `cypmode.metrics.motifs.detect_motifs`). `cypmode/validation/batch.py` runs `run_structural_validation` on all 20 compounds, not only on the motif-positive ones, because the coordination status cannot be known without running the structure, which is the purpose of the pipeline; the results are written to `data/structure_track/outputs/cyp3a4/structure_track_results.json`.

On 2026-09-24 the scoring rules, the submission format and the deadline of the track had not been published and only the blinded SMILES were public; the rules were published on 2026-09-25 (see `CYPMode_Architecture_Specification.md`). The work of that date was preparation with a pipeline that the project already had, applied to the compounds of a part of the challenge that had not yet been formally announced.

### TDI reference panel: quantitative validation and a gap

The independent checks above validate the binary coordination call of the motif screen and its correlation with potency at scale. The question of whether the screen says anything about *time-dependent* inhibition (TDI), the mechanism-based inactivation as distinct from reversible coordination, was examined separately.

Source: [OpenADMET/Octant, CYP Data Engine blog post repository](https://github.com/OpenADMET/Octant_CYP_blog_post) (CC BY 4.0), a small panel of 6 named drugs with measured preincubation-versus-direct pIC50 shifts (`cypmode/validation/tdi_reference.py`, `data/tdi_track/inputs/octant_cache/tdi_reference/`). Two of the six, ketoconazole and azamulin, are already in the structural reference set of the project, so both their motif call and their static heme-coordination geometry are known independently of their TDI status:

| Drug | TDI shift (Δ pIC50) | TDI-positive (>2-fold)? | Motif match | Coordinated (this project)? |
|---|---|---|---|---|
| Ketoconazole | 0.03 | No | Yes (imidazole) | Yes, 2.05 Å |
| Clotrimazole | 0.17 | No | Yes (imidazole) | not run |
| Diltiazem | 0.35 | **Yes** | **No** | not run |
| Verapamil | 0.84 | **Yes** | **No** | not run |
| Azamulin | 0.99 | **Yes** | Yes (triazole) | **No, 13.41 Å** |
| Troleandomycin | 1.25 | **Yes** | **No** | not run |

Two entries agree with the structural claims of the project: ketoconazole (motif-positive, coordinated, TDI shift of about 0, a reversible Type II inhibitor) and azamulin (motif-positive, *not* coordinated, a strong TDI shift, consistent with mechanism-based inactivation and not with static coordination).

**The gap.** Diltiazem, verapamil and troleandomycin are documented, clinically relevant time-dependent CYP3A4 inhibitors, and all three carry **no Type II motif at all**: their mechanisms go through a tertiary amine (verapamil, diltiazem) or a macrolide nitroso-heme complex (troleandomycin), chemistry that the screen was never built to detect (it looks only for azole, pyridine and thiazole nitrogens). Half of the TDI-positive compounds of this small panel would therefore be invisible to the motif screen. This is a different and larger failure mode than the "motif-positive but non-coordinating" case of azamulin, and no tuning of the Type II motif can remove it, because the chemistry lies outside what the screen looks for.

## Scope and limitations

- The motif screen is a substructure alert and not a mechanism predictor: a match means "capable of heme coordination" and not "confirmed to coordinate in the active site of this isoform". The `imidazole_Nsub` match of vardenafil, inside a fused ring system and not in a discrete azole, is a false positive that the structural check confirmed.
- The screen covers only direct heme-iron coordination. It says nothing about mechanism-based (quasi-irreversible) inactivation or other non-coordinating modes of inhibition. Azamulin is a confirmed example: a CYP3A4 inhibitor with a "Type II" motif match and no structural coordination, because its mechanism is suicide inactivation, a category that the screen was never built to detect.
- Structural validation is restricted by design to the four compounds with independently solved reference structures (see "Problem" above for why panel-wide physics-based validation is not attempted).
- A short Fe-ligand-N distance is evidence of a geometrically plausible coordination pose and not proof of the biological mechanism. Boltz-2 has no notion of catalytic turnover or covalent chemistry, so it can confirm a static coordination geometry, but, as azamulin shows, the absence of coordination does not exclude a different mechanism.

## References

The tools and data on which the project depends are cited here in full.

- Passaro, S., Corso, G., Wohlwend, J., Reveiz, M., Thaler, S., Somnath,
  V.R., Getz, N., Portnoi, T., Roy, J., Stark, H., Kwabi-Addo, D., Beaini,
  D., Jaakkola, T., Barzilay, R. "Boltz-2: Towards Accurate and Efficient
  Binding Affinity Prediction." *bioRxiv* 2025.06.14.659707 (2025).
  https://www.biorxiv.org/content/10.1101/2025.06.14.659707
- Huang, K., Fu, T., Gao, W., Zhao, Y., Roohani, Y., Leskovec, J., Coley,
  C.W., Xiao, C., Sun, J., Zitnik, M. "Therapeutics Data Commons: Machine
  Learning Datasets and Tasks for Drug Discovery and Development."
  *NeurIPS 2021 Datasets and Benchmarks*. arXiv:2102.09548.
- OpenADMET. "CYP Inhibition Blind Challenge."
  https://huggingface.co/spaces/openadmet/cyp-challenge. The training
  data, the blinded test set and the TDI task definition on which the
  `cypmode/challenge/` package works come from this source.
- OpenADMET/Octant. `Octant_CYP_blog_post`
  (github.com/OpenADMET/Octant_CYP_blog_post), CC BY 4.0. The 6-drug
  quantitative TDI preincubation-shift panel against which
  `cypmode/validation/tdi_reference.py` cross-checks.
- Landrum, G. et al. RDKit: Open-source cheminformatics.
  https://www.rdkit.org. The descriptors, fingerprints and SMARTS matching
  of the project are built on it.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu,
  T.-Y. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree."
  *NeurIPS* (2017). The classifier used throughout `cypmode/challenge/`.
- Castellanos, M.A., MacDermott-Opeskin, H., Ainsley, J., et al., Walters,
  W.P. "Assessing the state of the art in ADMET prediction: Lessons from
  the OpenADMET-ExpansionRx community blind challenge." *ChemRxiv*
  (2026). https://doi.org/10.26434/chemrxiv.15009342.v1. A separate,
  completed OpenADMET blind challenge (not the CYP challenge itself),
  whose findings on statistical methodology are compared with
  `cypmode/challenge/stacking_significance.py`; see
  `CYPMode_Architecture_Specification.md`.
- Huddleston, K. "Cofolding? I hardly know her! - Cofolding Methods on
  ADMET Targets." *OpenADMET Blog* (2026).
  https://doi.org/10.5281/zenodo.21785463. Quantified evidence that
  Boltz-2 and OpenFold3 (both used in the Structure track of the project)
  place CYP3A4 ligands correctly in only 57.8% and 31.5% of the cases,
  even under an optimistic oracle-ranking evaluation; see
  `CYPMode_Architecture_Specification.md`.

## Citation

If you use CYPMode, please cite it as software, and say which part you relied on. The machine-readable form is [CITATION.cff](CITATION.cff) (GitHub shows it as "Cite this repository"); how to cite, what counts as using the work and
what the licence asks are in [USE_AND_CITATION.md](USE_AND_CITATION.md). A ready entry:

```bibtex
@software{talaat_cypmode_2026,
  author  = {Talaat, Muhammad},
  title   = {CYPMode: time-dependent CYP inhibition and CYP3A4 structure prediction for the OpenADMET CYP blind challenge},
  year    = {2026},
  version = {0.2.0},
  doi     = {10.5281/zenodo.23816451},
  url     = {https://github.com/mohamedtalaat-gif/CYPMode},
  license = {AGPL-3.0-or-later}
}
```

The DOI of the archived release 0.2.0 is [10.5281/zenodo.23816451](https://doi.org/10.5281/zenodo.23816451); it is the identifier to cite.

## Contributing

Issues and pull requests are welcome; scope changes are easiest to agree on before writing code, so open an issue first for anything beyond a small fix.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,challenge]"
```

Extras (the Install section says what each is for):

- `dev` - the motif screen, data loaders and `app.py`.
- `challenge` - LightGBM, scikit-learn, statsmodels, tblite and the rest of the TDI engine.
- `validate` - Biopython and PoseBusters for the structure tools.
- `multitask` - PyTorch, for the shared-trunk model and ChemBERTa features.
- `tdc` - refreshes the CYP panel cache from source instead of the checked-in `data/tdi_track/inputs/tdc_cache/`.

Boltz-2, OpenFold3 and chemprop each live in their own virtual environment (`.venv-boltz`, `.venv-openfold3`, `.venv-ml`); the package
versions of `.venv` and `.venv-ml` are pinned in `environment/`.

### Running tests

```bash
python -m pytest tests -q
python -m pytest tests/test_multitask.py -q
python -m pytest tests/test_chemberta_features.py -q
python -m pytest tests/test_quantum_pharmacophore.py tests/test_physics_block.py tests/test_physics_tierb.py -q
```

The last three commands each run in their own process. PyTorch and tblite bundle OpenMP runtimes that crash or deadlock when they are
loaded into the same process as LightGBM, so `pyproject.toml` ignores those files in the main run. Tests that need the TDC panel
cache, Boltz-2 outputs or the large caches skip cleanly when those are absent.

### Checks before a pull request

```bash
python -m cypmode.audit.checklist       # every queue job names modules and scripts that exist
bash scripts/run_audit.sh               # full audit report, generated twice and compared byte for byte
python -m cypmode.audit.method_report --check   # METHOD_REPORT.md and NEGATIVE_RESULTS.md equal a fresh generation
python -m cypmode.audit.public_readiness        # private paths, credential-shaped strings, large files, third-party text
bash scripts/queue_status.sh            # done, running and waiting queue jobs
```

Documents that describe the architecture (the sketches and the repository map of this README) are
updated in the same change as the code they describe.

### Experiments

A date in a comment such as "declared on 2026-10-23 before any result" records when an experiment was registered in the author's working notes, a private file that is kept out of the repository; the rule of each gate is in the module that runs it and in the log that it writes.

A new model component is a candidate group in `cypmode/challenge/nested_runners.py` with a driver built on
`cypmode/challenge/addition_experiment.py`: it is evaluated alone and inside the assembled stack, with the corrected resampled t at both
confidence cut-offs, and it enters the final stack only through the gate (`cypmode/challenge/final_tdi_build.py`). Long runs are queue
jobs (`queue/jobs/NNN_name.sh`, started with `bash scripts/queue_start.sh`; helpers in `queue/lib.sh`, memory and battery guards in
`scripts/guards.sh`). Physics additions are also judged with 10 outer folds (`python -m cypmode.challenge.replicate_verdict`, candidate files
from `final_tdi_build.py --design 10`); `bash scripts/reproduce_tdi.sh` rebuilds the candidate files from the raw files.

Structure-track additions follow the same pattern: a term or lever gets a gate on the crystal-anchored benchmark (`cypmode/validation/pose_awareness.py`,
`series_consensus.py`, `selection_validation.py`, `structure_gates.py`), prints a `STRUCTURE GATE name: MET|NOT MET` line, and enters the
recipe only through `cypmode/validation/final_recipe.py`.

The notebook is rebuilt with `python -m notebooks.build_notebook` (executes every cell, checks that every defined term is defined where it is first used and that every file of the repository is explained, then exports HTML and PDF). A new module, script, queue job or test file has to be named in the text of the notebook before it builds.

### Conventions

- No premature abstraction: concrete functions rather than shared base classes or generic frameworks.
- Every heuristic or motif choice traces back to a citable source (a paper, a named method) in a comment or docstring, as in
  `cypmode/metrics/motifs.py`.
- Comments explain why, not what, and are written in English.
- Prose (docstrings, comments, commit messages, this README) is plain and direct: the change and its rationale.
- Numbers quoted in documents come from the corrected procedure (nested cross-validation, corrected resampled t, permutation nulls) and are
  produced by code, not typed in.

### Reporting bugs / requesting features

Use the issue templates. For anything touching `cypmode/metrics/`, include a concrete SMILES string and the motif call you got against
what you expected.

## License

The code is licensed under the [GNU Affero General Public License, version 3 or later](LICENSE) with an additional term, under section 7(b), that the attribution and the citation notice stay with every copy and every
generated output (see [NOTICE](NOTICE)). The documents (the notebook, the method report, the figures, the film, the card and the texts) are licensed under
[CC BY-NC-SA 4.0](LICENSES/CC-BY-NC-SA-4.0.txt), and the data that the code generates under [CC BY 4.0](LICENSES/CC-BY-4.0.txt); third-party material keeps its own licence. [REUSE.toml](REUSE.toml) says which licence applies to
which files. A use of the code in a product that cannot be released under these terms needs a separate written licence from the author. The versions published up to the first 27 commits were published under MIT OR Apache-2.0 and remain
available under those terms. Contributions are accepted under the licence of the part of the repository that they change.

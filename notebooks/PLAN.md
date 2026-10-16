# CYPMode notebook: plan and table of contents

One notebook, `cypmode_notebook` (`.ipynb`, `.html`, `.pdf`), written in English. It is both the teaching text and the laboratory notebook of
the project: everything that was built, everything that was tried and not used, and the reason for each outcome. The practical walkthrough is
not a second document. Executed code cells sit next to the idea they illustrate, and Appendix A collects the complete, runnable path
through the repository.

Voice: impersonal, in the register of a paper and a textbook. The text speaks of "this project" and addresses the reader as "you".

Reference for form and style: `Gabr Lab - BBB Engine/docs/notebook/gabr_bbb_engine_notebook` (parts with dividers, numbered sections, table
of contents, bold terms, display equations rendered for the PDF, "On what actually ran" paragraphs, numbered references).

## Standards every section meets

1. **Zero background.** A reader with no chemistry, biology, statistics or machine learning reads it from the first page. Every term is
   defined where it first appears; the notebook builder checks that no defined term is used earlier than its definition.
2. **Every equation is built, not quoted.** Each symbol is named and its unit or range given, the equation is derived or assembled step
   by step from simpler ones, and a numerical example is computed in a code cell.
3. **Origin of every idea.** Each method, formula, constant and rule is traced to where it comes from (who proposed it and in which work).
   Where an idea in this project is its own, that is said and the evidence for it is shown.
4. **Self-contained.** The text explains enough that the reader needs no outside source. A numbered reference list (Appendix C) credits
   the origins; it is for attribution, not for understanding.
5. **Everything is shown, including what did not work.** Every experiment appears with its pre-declared rule, its numbers and its verdict.
   An experiment that failed its gate is explained as carefully as one that passed, including why it did not help.
6. **No typed-in numbers.** Every number, table and figure is produced by a code cell that reads the repository's own files at build time;
   a failing cell fails the build. The notebook is rebuilt from the repository whenever results change.
7. **Honest scope.** Where the code does less than its name suggests, an "On what actually ran" paragraph says so.
8. **Whole-project coverage, enforced.** The builder fails when any module of `cypmode/`, any file of `scripts/`, any job of `queue/jobs/`
   or any test file is not named in the notebook, so nothing in the repository can go unexplained, and every later addition has to be
   explained before the notebook builds. Appendix D prints the resulting map from file to section.

## Front matter

- Title page, abstract, how to read the notebook, table of contents, notation conventions, list of symbols.

## Part 0. Orientation

- 0.1 The problem: medicines in the body, the liver, enzymes that can be switched off, and why predicting their interactions early matters.
- 0.2 How this project grew: from a heme-binding motif screen to the OpenADMET CYP blind challenge (two tracks).
- 0.3 The challenge: the data, the two tracks, the scoring, the deadline, the rules about open code and method reports.
- 0.4 The whole system on two pages: the TDI architecture and the structure architecture, box by box, with the section that explains each box.
- 0.5 The ledger in one table: every idea tried, its pre-declared gate, its verdict (generated from the logs).
- 0.6 How the notebook is built and how to run it.

## Part I. Biology and chemistry

- 1.1 Drugs in the body: absorption, distribution, metabolism, excretion.
- 1.2 Enzymes: what a catalyst does; the Michaelis-Menten equation derived from the reaction scheme.
- 1.3 Cytochrome P450: the heme, the iron, the catalytic cycle; CYP3A4 and CYP2D6 and their active sites.
- 1.4 Drug-drug interactions and why inhibition matters clinically.
- 1.5 Reversible inhibition: competitive, non-competitive, mixed; the equations and the IC50.
- 1.6 Time-dependent inhibition: mechanism-based inactivation, the kinetic scheme, k_inact and K_I, the observed inactivation rate derived.
- 1.7 How potency is measured: dose-response, the Hill equation, IC50, pIC50 = -log10(IC50), the assay floor.
- 1.8 The two assay arms of the challenge (direct inhibition and the TDI condition) and the shift between them.
- 1.9 Heme chemistry: iron coordination, Type I and Type II ligands, why an azole or pyridine nitrogen binds the iron.
- 1.10 Reactive metabolites and structural alerts: why some substructures (tertiary amines, thiophenes, furans, methylenedioxy) are linked with inactivation.

## Part II. The data

- 2.1 A molecule as text and as a graph: atoms, bonds, valence, aromaticity, SMILES (grammar and canonical form), stereochemistry.
- 2.2 The training and blinded test files, column by column; what is missing and why.
- 2.3 The label rule: how s = t - max(d, 4) > log10 2 was recovered from the data and checked exactly; why 4, why log10 2.
- 2.4 The contamination found in the first version of the labels and how it was fixed.
- 2.5 Measurement noise: the posterior standard deviations, error propagation for a difference, the confidence z = |s - log10 2| / sd(s).
- 2.6 Which compounds were measured on which isoform: selection, "hits", the all-isoform panel block.
- 2.7 How the test set differs from the training set: similarity, prevalence, the density ratio.
- 2.8 Other data sources: the TDC CYP panel, ChEMBL, BindingDB, PubChem AID 1851, the Octant set, the Hugging Face boards.
- 2.9 Standardising structures to match the same compound across sources.
- 2.10 The public leaderboard as data: recovering the test set's positive rate from accuracy, precision and recall.

## Part III. Molecules as numbers

- 3.1 Morgan (circular) fingerprints: the algorithm, hashing, counts versus bits.
- 3.2 Tanimoto similarity and why it suits fingerprints.
- 3.3 RDKit descriptors: molecular weight, Crippen logP (the atom-contribution scheme), TPSA, rotatable bonds, hydrogen-bond counts.
- 3.4 MACCS keys, Avalon, ErG, topological torsion, feature-Morgan, the motif and structural-alert blocks.
- 3.5 Three-point pharmacophore fingerprints.
- 3.6 Chemical series: Butina clustering, scaffolds, and why whole series must stay together in validation.
- 3.7 The feature store: how blocks are built once and reused, and why features never see a label.

## Part IV. Machine learning from zero

- 4.1 Supervised learning, loss functions, log-loss derived from likelihood, the bias-variance trade-off.
- 4.2 Decision trees: impurity (Gini, entropy), the split search.
- 4.3 Gradient boosting: functional gradient descent, the second-order objective used by LightGBM, histograms, leaf-wise growth, regularisation.
- 4.4 Random forests and extremely randomised trees: bagging, variance reduction.
- 4.5 Nearest neighbours and support vector machines with a Tanimoto kernel.
- 4.6 Regression for the two assay arms; floor-clipping.
- 4.7 Neural networks: the perceptron, backpropagation, the shared-trunk multitask network.
- 4.8 Message-passing networks on molecular graphs (Chemprop) and pretrained embeddings (transformers and self-attention for ChemBERTa; CheMeleon).
- 4.9 Tabular in-context learning (TabICL) and principal component analysis.
- 4.10 Metrics: ROC and AUC (and its Mann-Whitney meaning), average precision, MCC derived from the confusion matrix, Brier score, calibration error.

## Part V. The Type II motif screen

- 5.1 The idea: a heme-coordinating nitrogen as a predictor of inhibition, and the SMARTS patterns that encode it.
- 5.2 The trialkylamine alert and the false positives it had to exclude.
- 5.3 The panel data and how it is annotated.
- 5.4 Testing the screen against real potency: the large Octant benchmark and the statistics used.
- 5.5 Checking the screen against predicted structures: the iron-nitrogen distance in Boltz-2 complexes.
- 5.6 Pocket geometry from a predicted structure: solvent-accessible area and pocket volume.
- 5.7 Cross-checking the screen against time-dependent shift measurements.
- 5.8 The Streamlit application and its figures.

## Part VI. Measuring honestly

- 6.1 Why a score on the training data lies.
- 6.2 Cross-validation and its optimism when it also chooses.
- 6.3 Nested cross-validation: the construction, every loop, what is fitted where, the prediction cache.
- 6.4 Making validation resemble the test: importance weights, similarity to the test structures, prevalence, confident rows only.
- 6.5 Comparing two models: why folds are correlated; the Nadeau-Bengio corrected resampled t derived (the factor 1/J + n_test/n_train); Holm adjustment; permutation nulls; bootstrap intervals.
- 6.6 Why Tukey's HSD was dropped and how the older results were recomputed.
- 6.7 Pre-declared gates: the rule, why it is fixed before results, the 10-fold replicate and the reason for it.
- 6.8 Checking that a split is informative; the test-like hold-out block; leakage controls.

## Part VII. The first TDI models and what they taught

- 7.1 Reproducing the challenge's own baseline from the raw file.
- 7.2 The M0 to M3 ablation (motif features, amine features) and its verdict.
- 7.3 Nearest-neighbour read-across features.
- 7.4 The direct-inhibition bridge (out-of-fold) and why it was retracted.
- 7.5 Decision thresholds tuned inside folds; reliability from ensemble disagreement and out-of-distribution distance.
- 7.6 Query-by-committee uncertainty judged against real errors.
- 7.7 External data as auxiliary signal: ChEMBL, BindingDB, PubChem, external activity models.
- 7.8 Quantum-surrogate reactivity features and global GFN2 descriptors (the early nulls).
- 7.9 ChemBERTa embeddings and the multitask shared-trunk network.
- 7.10 Tier 3: structural evidence from Boltz-2 and the selection of compounds worth predicting.
- 7.11 The first assembled models, the independent-head architecture test and the CYP2D6 tuning levers.

## Part VIII. The final TDI model

- 8.1 The candidate grid: 23 members and their feature sets.
- 8.2 Stacking: the logistic meta-learner, L1 and L2 penalties, the objective, how the inner out-of-fold scores are built.
- 8.3 A reproducibility bug in the stack: liblinear's random coordinate order, the evidence, the fix (tolerance and seed), the test.
- 8.4 Turning scores into calls: rate transfer, the bootstrap-smoothed threshold, the call-rate multiple.
- 8.5 Calibration: Platt scaling, isotonic regression (the pool-adjacent-violators algorithm), ECE and slope.
- 8.6 The candidate files, the portal's own checks, the manifest and hashes, the leakage audit.
- 8.7 Results with their uncertainty.

## Part IX. Additions that were tested

Each section: the idea and its origin, the exact construction, the pre-declared gate, the alone and stacked results in both designs, and why
it did or did not help.

- 9.1 The selection-aware CYP2D6 model: P(label) = P(hit) x P(label | hit), the Heckman correction (probit, inverse Mills ratio) derived; the hedged blend; the non-hit share and label shift.
- 9.2 SAR-aware pair model and matched molecular pairs.
- 9.3 The label-process committee: P(shift > log10 2) from a Gaussian, expected improvement derived, noise-weighted fitting.
- 9.4 Frozen foundation-model embeddings (CheMeleon).
- 9.5 Test-likeness weighting.
- 9.6 Floor-clipped arms.
- 9.7 Cross-isoform transfer of the assay arms.
- 9.8 Confidence-weighted training.
- 9.9 The neural pair (the multitask network with plain Chemprop): the 15-fold result that met the gate, the replicate declared before any 10-fold result, and its verdict.
- 9.10 The other additions, tested the same way: pretrained tables and representations (TabICL, TabPFN, Monroe, Atom-JEPA, CheMeleon fine-tuned in two stages), outside measurements (NCATS continuous, Tox21 with its luciferase counter-screen, predicted logD and protein binding, PXR induction), the organisers' extra readings (single-concentration screen, Emax, rows without a direct curve, the label in two parts), the basic-centre block, the non-negative and softmax blends, and the negative control.
- 9.11 The ledger of the gates with the correction for the number of attempts (Holm over the whole family), and what Part IX showed.
- 9.12 Why the ideas that did not join added nothing: the diagnostics generated by code (`why_no_gain`), read from its output when the queue has produced it.

## Part X. Quantum chemistry and the physics features

- 10.1 The Schrodinger equation, the wavefunction, the Born-Oppenheimer approximation.
- 10.2 Hartree-Fock and the self-consistent field; basis sets.
- 10.3 Density functional theory: Hohenberg-Kohn, Kohn-Sham equations, exchange-correlation.
- 10.4 Semi-empirical and tight-binding methods; the GFN2-xTB energy expression term by term.
- 10.5 The quantities used: vertical ionisation potential and electron affinity, hardness and electrophilicity, Fukui functions by finite differences, hydrogen-abstraction energies, proton affinity, ALPB solvation, Boltzmann weighting of conformers.
- 10.6 Tier A, the compact selection, tier B, the physics inside the selection-aware model, the 10-fold replicate, and why none passed its gate.

## Part XI. The structure track

- 11.1 Proteins, folding, and the structure of CYP3A4; what a complex file contains.
- 11.2 Co-folding models: the idea behind AlphaFold-style prediction, diffusion generative models (forward and reverse process equations), the Pairformer and triangle attention, MSAs; Boltz-2 and OpenFold3.
- 11.3 Building a Boltz-2 input: protein, heme with its covalent bond, ligand, the MSA, restraints, templates; the batch runner.
- 11.4 How a pose is scored: RMSD with symmetry, LDDT and LDDT-PLI, BiSyRMSD, the PoseBusters checks.
- 11.5 The reference ligands from the PDB, the crystal-anchored benchmark and the series benchmark.
- 11.6 The Fe-donor restraint: what it adds to the input, why it works for Type II ligands (gate met).
- 11.7 Levers that did not pass: seeds, crystal conformer templates, step scale, cryo-EM templates.
- 11.8 Choosing among poses: minimum Bayes risk, experimental-prior rescoring, the heme-frame prior, expected-improvement sampling, the physics, pharmacophore and SAR terms (all gates NOT MET).
- 11.9 OpenFold3 in full: what it is, how it was run on this machine, the out-of-memory failure traced to its cause, the exact fix and the test that it does not change the model, the benchmark, and why it did not help (engine consensus NOT MET).
- 11.10 The final recipe, the pools, the validation of every structure, the candidate zip (deterministic archive).

## Part XII. Engineering for trust

- 12.1 Reproducibility: seeds, threads, solver randomness, deterministic archives, pinned environments.
- 12.2 The audit: what it checks and how it caught the stack's non-determinism.
- 12.3 The test suite and the continuous-integration workflow.
- 12.4 The job queue, the runner and the guards; what the hardware taught (memory, swap, battery, sleep).
- 12.5 The method report, the negative-results ledger, the checklist consistency check, the public-readiness scan.
- 12.6 The rules that never change and how they are enforced.

## Part XIII. Closing

- 13.1 What the project shows and what it does not.
- 13.2 Limitations.
- 13.3 Open questions.
- 13.4 Final results and the submission state (filled from the final files).

## Appendices

- A. Practical walkthrough: environments, repository map, every stage as a runnable path, reproducing the candidates, adding a model through a gate, adding a structure lever.
- B. Symbols, abbreviations and glossary.
- C. References: the origin of every method, formula and constant.
- D. Coverage map: every module, script, queue job and test file and the section that explains it (printed by the builder).

## Coverage map by part (the exact file-to-section map is printed in Appendix D)

- Part II: `challenge/tdi_targets`, `library_blocks`, `label_confidence`, `leaderboard_prior`, `population_share`, `structure_standardization`, `external_activity`, `chembl_auxiliary`, `bindingdb_auxiliary`, `pubchem_auxiliary`, `data/octant_benchmark`, `data/panel`.
- Part III: `challenge/feature_store`, `grouping`, `pharmacophore_fingerprint`.
- Part IV: `challenge/multitask`, `chemberta_features`, `nested_runners_torch`, `nested_runners_chemprop`.
- Part V: `metrics/motifs`, `metrics/amines`, `data/panel`, `data/octant_benchmark`, `validation/structures`, `pocket_geometry`, `tdi_reference`, `app.py`.
- Part VI: `challenge/nested`, `nested_runners`, `nested_deploy`, `nested_report`, `model_comparison`, `addition_experiment`, `validation_matching`, `split_validity`, `block_holdout`, `stacking_significance`, `recompute_paired_summaries`, `replicate_verdict`, `chemberta_significance`.
- Part VII: `challenge/tdi_baseline`, `tdi_ablation`, `knn_features`, `oof_bridge`, `threshold_tuning`, `reliability`, `committee_uncertainty`, `external_auxiliary_significance`, `qm_reactivity_features`, `qm_reactivity_significance`, `combined_features`, `tier3_features`, `tier3_selection`, `tier3_knn_significance`, `architecture_significance`, `cyp2d6_tuning_significance`, `final_assembly`, `final_assembly_cyp2d6`, `confidence_experiment`.
- Part VIII: `challenge/final_tdi_build`, `calibration_experiment`, `submission`.
- Part IX: `challenge/population_experiment`, `selection_aware_v2`, `hedged_blend`, `test_nonhit_share`, `pair_transfer`, `mmp_transfer`, `committee_label_model`, `frozen_embeddings`, `test_likeness_training`, `arms_experiment`, `xiso_transfer`, `aux_blocks`, `aux_experiments`, `cached_candidates`, `official_extra`, `external_continuous`, `openadmet_surrogates`, `monroe_embeddings`, `atomjepa_embeddings`, `chemeleon_stage`, `foundation_weights`, `basic_centre`, `screen_selection`, `nonneg_stack`, `softmax_selector`, `permutation_control`, `multiplicity`, `why_no_gain`, `confidence_experiment`, `label_confidence`.
- Part X: `challenge/physics_block`, `physics_tierb`, `physics_experiment`, `physics_compact_experiment`, `physics_tierb_experiment`, `population_physics`, `quantum_pharmacophore`, `quantum_pharmacophore_significance`.
- Part XI: `challenge/structure_submission`; `validation/batch`, `bisy_rmsd`, `build_input`, `build_summary`, `conformer_benchmark`, `conformers`, `constants`, `cryoem_templates`, `diversity_pilot`, `ei_sampling`, `final_recipe`, `heme_frame_prior`, `openfold3`, `pipeline`, `pose_awareness`, `pose_comparison`, `pose_consensus`, `pose_pools`, `pose_rescoring`, `pose_rescoring_validation`, `pose_selection`, `pose_validity`, `reference_ligands`, `selection_validation`, `series`, `series_consensus`, `structure_benchmark`, `structure_benchmark_report`, `structure_gates`; `scripts/run_openfold_lowmem.py`.
- Part XII: `audit/*`, `scripts/*` (queue runner, guards, mirror check, reproduction and audit scripts), `queue/jobs/*`, `tests/*`, `environment/*`, `.github/workflows/tests.yml`, `notebooks/*`.

## Build

- Content in `notebooks/notebook_cells.py`, assembled by `notebooks/build_notebook.py`, exported by `notebooks/build_exports.py` (HTML, and a PDF in the style of the reference, equations rendered to images and cached in `notebooks/_math_cache/`).
- `python -m notebooks.build_exports` rebuilds everything from the repository; results and figures come from the project's own files.
- The notebook is updated whenever results, models or tools change, and once more when the queue has finished, with everything that changed since the last build.

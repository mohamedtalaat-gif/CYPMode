# Illustrations: plan and catalog

Goal: a complete set of figures that explains the project and its results to a reader with no background: graphs of every result,
sketches of every method, and illustrations of every process that follows an equation (kinetics, dynamics, modelling, docking). The same
figures feed the notebook (`notebooks/PLAN.md`) and the README.

## Rules

1. **Drawn by code, from the repository.** Every graph reads the project's own result files; every sketch is drawn by a generator script.
   No figure is a pasted picture, and no number in a figure is typed in.
2. **One style.** `docs/figures/style.py` holds the palette, fonts and sizes (the palette of the existing figures: blue `#0F4D92`, red `#B64342`,
   green `#1B7F3A`, dark theme `#0D1117`). Each figure is written as light and dark SVG (GitHub's `#gh-light-mode-only` / `#gh-dark-mode-only`
   convention) and as a 2x PNG for the PDF export of the notebook.
3. **Every figure is catalogued.** `docs/figures/catalog.py` lists, for each figure, its id, title, caption, the data files it reads and
   the notebook section that uses it. The builder fails when a catalogued figure is not produced or when a produced figure is not
   catalogued; `docs/figures/CATALOG.md` is generated from it.
4. **Dynamic processes also get an animation.** Where a process unfolds in time or in steps (enzyme kinetics, boosting, diffusion
   denoising, docking, the self-consistent field, the chunked attention), the static figure is accompanied by a short looping animation
   (GIF) drawn from the same code.
5. **Figures that keep to the data.** Axes start where the data says, uncertainty is shown, and a result that failed its gate is drawn as clearly as one
   that passed.

## Catalog by notebook part

**Part 0, orientation**
- System architecture of the TDI track and of the structure track as block diagrams; the project timeline from the motif screen to the
  challenge; the ledger of every experiment as one heat map (idea by isoform by verdict); the final results on one page.

**Part I, biology and chemistry**
- Absorption, distribution, metabolism and excretion; the liver and the enzyme; the P450 catalytic cycle; the heme with its iron and axial
  cysteine; Type I and Type II binding (water displacement versus nitrogen-iron coordination), drawn on a real structure.
- Kinetics from the equations: Michaelis-Menten saturation and its Lineweaver-Burk form; competitive, non-competitive and mixed
  inhibition; the time-dependent inhibition scheme with progress curves integrated from its differential equations; the observed
  inactivation rate as a function of inhibitor concentration (k_inact and K_I); dose-response and Hill curves; the pIC50 scale; the two
  assay arms and the shift between them. (Animated: progress curves filling in as the inhibitor concentration rises.)

**Part II, the data**
- The label rule drawn on the real data (t against max(d, 4), the decision band, positives and negatives); the distribution of the shift and
  of the confidence z with the thresholds; how many compounds were measured on how many isoforms; the all-isoform panel block; how the
  test set differs from the training set (nearest-neighbour similarity, prevalence); the recovery of the test prevalence from the board.

**Part III, molecules as numbers**
- The Morgan algorithm step by step on a real molecule (the environments of radius 0, 1 and 2 and the bits they set); Tanimoto similarity
  on two molecules; the descriptor blocks and their sizes; the Butina clusters; a three-point pharmacophore.

**Part IV, machine learning**
- A decision tree fitted to real data; gradient boosting fitting residuals stage by stage (animated); bagging versus boosting; the
  Tanimoto kernel; ROC, precision-recall and calibration curves of the final stack from its out-of-fold predictions; the confusion
  matrix and MCC; bias and variance; a neural network and backpropagation; message passing on a molecular graph; self-attention; PCA.

**Part V, the motif screen**
- The SMARTS patterns with example hits; the enrichment against potency; the iron-nitrogen distance distribution (updated version of the
  existing figure); the pocket geometry of a predicted complex.

**Part VI, measuring without selection bias**
- Nested cross-validation as a picture (outer and inner folds); the optimism of in-sample, plain and nested scores on the real data; the
  evaluation weights; the corrected resampled t (the variance factor against the number of folds and the test fraction); the gate as a
  flow chart; every stacked comparison of the project as one forest plot with its corrected p, in the 5-fold and the 10-fold designs.

**Part VII, the first TDI models**
- MCC progression (updated); the M0 to M3 ablation; the retracted direct-inhibition bridge shown as the artifact it was; the reliability
  correlation (updated).

**Part VIII, the final TDI model**
- The stack as a diagram; the meta-learner coefficients; the non-determinism of the solver and its removal (AUC across seeds at the loose and
  at the tight tolerance, from re-running the stack); rate transfer and the call-rate choice; reliability diagrams (raw, Platt, isotonic);
  the final results with their uncertainty.

**Part IX, additions that were tested**
- P(label) = P(hit) x P(label | hit) as a diagram, and the AUC against the non-hit weight from the sensitivity runs; the hedged blend;
  the pair model and a matched molecular pair drawn on real molecules; the committee's Gaussian P(shift > log10 2) and expected
  improvement; the arm correlations between isoforms as a heat map; the embeddings projected with PCA.
- The pharmacophore representations (9.13): the pockets of CYP3A4 and CYP2D6 as density fields in the frame of the heme, and one forest plot of every pharmacophore representation of the project with its gain, p and verdict.

**Part X, quantum chemistry and physics**
- The self-consistent field loop (animated convergence); the orbital energy diagram with HOMO, LUMO, IP, EA and Koopmans; Fukui functions
  drawn on a real molecule; the hydrogen-abstraction energy profile; Boltzmann weights of conformers; the GFN2-xTB energy terms; the
  importance of the physics columns.

**Part XI, the structure track**
- The co-folding pipeline; diffusion denoising on a toy point cloud with the forward and reverse equations (animated); the Pairformer's
  triangle updates; the heme and a ligand in the real predicted and crystal poses with the Fe-N distance; RMSD and LDDT-PLI as pictures;
  the pose pools per ligand; the restraint's effect ligand by ligand (paired plot); the minimum-Bayes-risk matrix and its pick;
  the OpenFold3 memory arithmetic against the Metal limit and the chunking that fixes it; the engine comparison; the pool composition of the
  final recipe; the PoseBusters checks (updated).

**Part XII, engineering for trust**
- The job queue as a timeline built from the queue logs; the machine's battery, swap and sleep history from the power log; the audit's two
  runs and what differed before the fix; the test counts by module; the repository map.

## Added on 2026-10-16: figure types the first plan did not have

The project grew after this plan was written (official extra files, a leak guard, pretrained models, staged designs, a confirmed-by-replicate rule, a voiced film, a README card). These types
are added; each is catalogued in `gen/exploration.py` and drawn from the repository's own files like the others.

- **The leak guard** (`data_leak_guard`, section 2.8): the files the organisers released, how many compounds each holds, how many are in the training set and which test compounds were removed;
  only identifier columns are read.
- **The test prevalence recovered from the public board** (`data_prevalence_recovery`, section 2.10): the implied positive rate of every board entry, its median and spread; the picture that
  goes with limitation L13.
- **The design floor** (`ml_design_floor`, section 6.5): the smallest certifiable gain against the number of folds, with the three designs of the project marked and every logged comparison placed
  against the certification limit; why 5 folds with one repeat is a screen and not a verdict.
- **The neural pair** (`ml_neural_pair`, section 9.9): two near-misses, the pair that joined them (a discovery) and its pre-declared replicate on the 10-outer-fold plan; shows "not run yet" until the
  replicate exists.
- **The exploration map** (`ml_exploration_map`, section 9.9): the ledger grouped by the kind of idea tested (data, representation and physics, learners, label process, structure levers).
- **The reliability of the final stack by similarity** (`ml_applicability_domain`, section 8.7) and **the choice of the call rate** (`ml_call_rate`, section 8.4): two planned figures of Part VIII, now built.
- **Every submitted structure** (`struct_all20_gallery`, section 11.10): all twenty pockets with the donor flag and the iron-nitrogen distance, so that the six on the README card are not the only ones seen.
- **The exploration-exploitation dial** (`ml_exploration_exploitation`, section 8.2): the outer AUC of a softmax blend of the cached candidates as the temperature moves from the best model to an equal-weight committee, against the logistic blend; the pre-registered softmax selector is not met on either enzyme.
- `ml_gate_forest` now reads the logs of every later addition (a log that does not exist yet is skipped) and the replicate of the neural pair.

## Added on 2026-10-26: the figure groups that the plan still lacked

Each is catalogued like the others and drawn from the repository's own files (the generators are `gen/ml_more.py`, `gen/structure_more.py` and `gen/data_more.py`; the computations behind them are tested in `tests/test_figures.py`).

- **Machine-learning concepts on real compounds** (Part IV): `ml_bias_variance` (4.1, the depth of a tree), `ml_bagging_boosting` (4.4), `ml_tanimoto_kernel` (4.5, the kernel matrix of six chemical series and its eigenvalues), `ml_backprop` (4.7, a small network trained by backpropagation and checked against finite differences),
  `ml_message_passing` (4.8, what an atom of fluconazole knows after k rounds), `ml_attention` (4.8, scaled dot-product attention over a SMILES string), `ml_pca` (4.9, the descriptors and the CheMeleon embedding) and `ml_confusion_mcc` (4.10, the confusion matrix and the MCC of the default stack).
- **Structure-track mechanics** (Part XI): `struct_pairformer_triangle` (11.2), `struct_rmsd_lddt` (11.4, RMSD and LDDT-PLI on a real pose against its crystal pose), `struct_ost_vs_project` (11.4, the LDDT-PLI of the project against OpenStructure), `struct_pose_pools` (11.8, the pool of every benchmark ligand),
  `struct_relaxation_margin` (11.10, the margin and the agreement with the crystal pose before and after the relaxation) and, held back, `struct_official20_pools` (11.10, the pools of the 20 official ligands and the margin of the chosen pose before and after the relaxation).
- **Data and experiments** (Parts II and IX): `data_nodirect_coverage` (2.4), `data_screen_vs_pic50` and `data_external_coverage` (2.8, the PXR and Tox21 panels), `ml_hit_gate_vs_screen` (9.1, the screen as the gate of the CYP2D6 hits), `ml_pretrained_forest`, `ml_chemeleon_stage1` and `ml_foundry_table` (9.4),
  and `ml_permuted_control` (9.10, the negative control).
- **Overview figures** (`gen/overview_more.py`): `arch_ledger_heatmap` (9.11, every idea by place and verdict), `arch_final_results` (8.7, the results on one page), `eng_commit_timeline` (12.4), `mol_butina_clusters` (3.6), `ml_evaluation_weights` (6.4), `ml_gate_flow` (6.7, the rule of the gate as a flow chart),
  `ml_m0_m3_ablation` (7.2), `ml_calibration_methods` (8.5), `ml_selection_sensitivity` (9.1), `data_label_gaussian` (9.3, the label as a Gaussian around a threshold), `data_arm_correlations` (9.7), `struct_engine_comparison` (11.9), `eng_repository_map` (12.5) and, held back, `struct_official20_recipe` (11.10, the run of each final pose).
- **Every catalogued figure is in the notebook.** The builder (`notebooks/build_notebook.py`) puts a figure that no part shows at the end of the section that the catalog names (the end of its part when the section does not exist), with the held-back flag for the figures of the HELD BACK block, so that a drawn figure can never be missing from the notebook.
- **The animated card of the README** (`docs/card/build_card.py`, job 941): the card with the enzyme turning once, each frame drawn from the same page of the card by the headless Chrome of the film.

## Build

- `python docs/figures/build_all.py` runs every generator, writes the files, checks the catalog and regenerates `CATALOG.md`.
- Queue job `940_figures_build` runs it after the other jobs have finished, so that the figures show the final results; the notebook
  build (job 942) then embeds them.
- The pharmacophore model of the predicted poses (11.11): the energy window that the bound conformation needs and the coverage of it by the ensembles of 9.13 and 9.14 (`struct_conformer_window`), the pocket RMSD and the support of the fields of the predicted complexes against the crystal range (`struct_site_check`), and the expected pharmacophore of the 20 official ligands (`struct_pharm_official20`, held back).
- The pharmacophore atlas (9.15), both tracks, in 2D and 3D, with the functional groups and the SAR: `pharm_atlas` (one compound in every representation), `pharm_functional_groups` (the rate of time-dependent inhibition by functional group), `pharm_mmp_sar` (the SAR of a change: matched pairs grouped by the group they gain), `pharm_features_examples` (the surest compounds with their groups and features), `pharm_hypotheses_2d3d` (the hypotheses of the ligand-based model on molecules), `pharm_pocket_3d` (the sites of the pocket model in the crystal structures), `struct_pharm_pose_3d` (the model on the best and the worst pose of three benchmark ligands), `struct_pharm_series` (the SAR of the series benchmark in 2D and 3D) and, held back, `struct_pharm_official20_2d` and `struct_pharm_official20_3d`.

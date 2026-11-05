# After the deadline: what the released labels and structures can say about the submitted entries

This plan is written before any label of the test set and any released structure has been read by a program of the project. The only numbers it uses are those that the organisers published on the final leaderboards on 5 November 2026 and in the post that announces them. Every question below has its measure and its decision rule here, so that no rule is chosen after a result is seen.

## Sources read for this plan

- The post of the organisers that announces the final leaderboards (https://openadmet.ghost.io/take-me-heme-binding-modes-wrapping-up-the-cyp-inhibition-challenge/): the tiers, the numbers of the entries, the statement that the final boards score the whole test set while the live boards scored half of it, and the definition of the hidden compounds, which are the compounds that the live board did not score.
- The card and the files of the dataset (https://huggingface.co/datasets/openadmet/cyp-challenge-train-test): the unblinded test files of 4 November 2026 and the folder of the 20 released structures, whose headers record a resolution and a refinement with PHENIX.
- The code of the Space of the challenge (https://huggingface.co/spaces/openadmet/cyp-challenge): the scoring rules in its FAQ, and the release of the ground truth for the head-to-head tab, which flags each test compound as used for the live board or held out.
- The evaluation code of the organisers (the folder `evaluation` of https://github.com/OpenADMET/CYP-Challenge-Tutorial, commit cc02086 of 1 November 2026), whose configuration states that it is ported from the backend and matches the leaderboard, and the 20 Boltz-2 baseline structures in the folder of that repository that holds its inputs.

## What is fixed and what is not touched

The submitted files (the two TDI candidates and the structure archive, identified by the SHA-256 values in `submissions/manifest.json`), the code and the reports of the releases 0.2.0 and 0.3.0, and every number already reported stay as they are. The analysis only measures; it fits no model, chooses no threshold and selects no candidate. A result is reported in the table that this plan defines, whichever way it falls, with its verdict. A question is not changed, a metric is not replaced and a subset is not added after the first result of the question has been computed.

The organisers published the test labels and the 20 structures, so any model whose choices were influenced by them is no longer blind. Nothing in this plan trains on them.

## Numbers already public (the reference for the first gate)

| entry | measure | value on the final leaderboard |
| --- | --- | --- |
| time-dependent inhibition, both isoforms | macro MCC | 0.3723 +- 0.0378 (rank 14 of 62, tier 2) |
| time-dependent inhibition, CYP2D6 | MCC | 0.277 +- 0.068 (rank 19) |
| time-dependent inhibition, CYP3A4 | MCC | 0.467 +- 0.033 (rank 19) |
| structure | LDDT-PLI | 0.4133 +- 0.0301 (rank 15 of 38, tier 2) |
| structure | BiSyRMSD | 3.970 +- 0.276 A |
| structure | LDDT-LP | 0.826 +- 0.014 |

The +- values are the standard deviations of the bootstrap of the organisers, and each board value is the mean over 1,000 bootstrap samples. The last entry of tier 1 in the structure track has an LDDT-PLI of 0.4148, which is 0.0014 above this entry.

## Question 0: does the scoring of the project reproduce the board (a gate)

The scoring is the organisers' own: their functions as published at the commit named above, run without change. For the classification track the function that scores the activity predictions takes the released label file, drops the compounds whose label is missing, scores each isoform with its bootstrap of 1,000 samples and forms the macro value of the two isoforms inside each sample. For the structure track the function that scores structure predictions runs OpenStructure on each pose against the released structure and applies the PoseBusters rule (a pose that fails any check scores zero and receives a BiSyRMSD of 20 A). If these functions cannot be imported in an environment of the project, the scorer of the project that was compared with OpenStructure earlier is used instead, and the gate decides whether that is acceptable.

For the classification track, each of the two candidate files is scored. The submitted file is the candidate whose macro MCC lies within 0.01 of 0.3723; if both or neither do, the analysis stops there and reports that. For the structure track, the submitted archive is scored; the gate is a mean LDDT-PLI within 0.005 of 0.4133 and a mean BiSyRMSD within 0.1 A of 3.970.

The bootstrap of the organisers has a fixed seed, so the board value is expected to be reproduced to the digits when the labels and the predictions are the same; the tolerance is wider only for safety.

If a gate fails, no later question is answered; the difference is investigated and written down first.

## Question 1: which TDI candidate is better on the released labels

Macro MCC and the MCC of each isoform for candidate A (hits only) and candidate B (selection aware), with a paired bootstrap of the compounds (1,000 resamples, seed 42) for the difference B minus A. Verdict: B is better if the 95% interval of the difference lies above zero, A is better if it lies below zero, and no difference is shown otherwise. The positive rate of each file is reported next to the true positive rate of each isoform among the scored compounds.

## Question 2: the label rule and the scored set

Question 2a. The rule recovered from the training data (a positive when the TDI-condition pIC50 exceeds the larger of the direct-inhibition pIC50 and 4 by more than 0.301) is applied to the values in the released test file and compared with the organisers' labels, compound by compound. The count of disagreements is reported with each disagreeing compound listed. The expectation is zero disagreements; any disagreement is a finding about the rule and is explained before it is used.

Question 2b. The compounds that carry a label in the released file are the scored set. The confidence rule of the project, which was built without knowing that set, selects a set of its own at each of the two cut-offs of the final build, 1.0 and 1.96; the sets are compared with the organisers' set in a table of agreement (both, only one, neither). The table closes the limitation of the report that the organisers' rule for the scored set was not published. No rule of the project is changed by it.

## Question 3: how much of the distance to the leading entries is a matter of the threshold

For the submitted file, the MCC at the threshold that was submitted is set beside the highest MCC that any threshold would give on the released labels. The second number is a ceiling that uses the labels and is labelled so; it is not a result and it is not used to change a file. The scores are those that the final build stored for the submitted candidate. The question measures how much of the gap to the entries of tier 1 could come from the call rate alone, given that the post reports 44 CYP2D6 positives among 748 test compounds (5.9%) against 21.6% in the training data.

## Question 4: was the submitted file chosen with the help of the live board

The live board scored half of the test set and the final board all of it. The organisers' post states that the leading teams lost ground when only the other half, the hidden compounds, is scored, and attributes this to teams that tried variants, watched the live score and reversed output classes. The release of the organisers flags each test compound as live or held out. If that flag can be read from what the organisers publish, the macro MCC of the submitted file is computed on each half, with a bootstrap of the difference. If it cannot be read, the question is dropped; no substitute split is made.

The entry was uploaded twice: first as a check of the portal and then with the link of the repository. Which file to upload was decided before the first upload by the rule of `cypmode/challenge/selection_decision.py`, so no choice of the file depended on a live score, and no difference between the halves beyond the bootstrap noise is expected. A difference larger than that is reported as it is.

## Question 5: structure track, compound by compound

The released structures are the reference. Each file records the resolution of its map and a refinement with PHENIX; the resolution of each compound is tabulated next to its results, the rank correlation between resolution and the LDDT-PLI of the submitted pose is reported, and no compound is excluded. The table also marks which of the 20 compounds are in the training file of the challenge; this is a descriptive column and no subset is analysed.

The pool of a compound is the set of poses that the final recipe chose from, as listed in the table of pools of the final structure folder (9 to 14 poses per compound, 200 in all), and the pose that unguided Boltz-2 ranks first is the first sample of the base run of the compound. For each of the 20 compounds, LDDT-PLI and BiSyRMSD are computed for these poses: the submitted one; the Boltz-2 baseline structure that the organisers published for the compound; the pose that the Boltz-2 runs of the project without a contact restraint and with default sampling rank first, where such runs exist for the compound (secondary, reported only); the best pose of the pool of the compound (a ceiling that uses the reference); and the expected value of a pose drawn at random from the pool.

- Question 5a. Mean paired difference of LDDT-PLI (and, as a secondary measure treated the same way with its sign reversed, of BiSyRMSD) between the submitted pose and the published baseline pose, paired bootstrap of the 20 compounds (10,000 resamples, seed 42). Verdict: the recipe is better if the 95% interval lies above zero, worse if below, and no difference is shown otherwise. The number of compounds in which the recipe is higher is reported next to it.
- Question 5b. The gap between the submitted pose and the best pose of the pool, set against the gap between the random pose and the best pose. It says whether the limit lay in producing a good pose or in choosing it.
- Question 5c. In the released structure of each compound, whether a donor atom of the ligand is bound to the heme iron (a donor atom is a nitrogen atom of the ligand, the distance is the shortest one between an iron of a heme and such a nitrogen in the file, and the cut-off is the one that the project used for the earlier cryo-EM structures, 2.6 A), against the call that the submitted pose makes. The two-by-two table is reported.

With 20 compounds the error bar of the LDDT-PLI of one entry is about 0.04, so a difference below that size is reported as not shown, whatever its sign.

## Deviations from the plan

Question 0, time-dependent inhibition track (written after its result was computed). The tolerance of 0.01 around the board value did not separate the two candidates: candidate A gave a macro MCC of 0.3723 and candidate B 0.3725, both inside the tolerance, so the gate as registered was not met and the analysis stopped. The two files differ only in some CYP2D6 calls. The bootstrap of the organisers has a fixed seed, and the file that was submitted reproduces the board value to the last digits: candidate A reproduces the mean and the standard deviation of all five metrics of the macro entry with a difference below 1e-15, and candidate B differs by 1.8e-4 in the MCC and by up to 7.6e-4 in the other metrics. The identification of the submitted file was therefore changed to an exact match (1e-6) of the mean and of the standard deviation of the macro MCC. Nothing else in the plan was changed. The result of the registered rule is kept in the output of the run, and questions 1 to 3 were computed after the change.

Question 0, structure track, and the version of OpenStructure (written after the result of the gate was computed). In a reply in the challenge channel of their Discord on 5 November 2026 the organisers name OpenStructure 2.11.1 and PoseBusters 0.6.5 as the versions behind the board. The first run used OpenStructure 2.12.0 with PoseBusters 0.6.5 and gave a mean LDDT-PLI of 0.41278 against the board value 0.41335 (difference -0.00057), a mean BiSyRMSD of 3.9724 against 3.9696 (+0.0029) and no pose that fails PoseBusters, so the gate as registered (0.005 and 0.1 A) was met. A second run with OpenStructure 2.11.1, built for this purpose, gave the same values to seven digits, so the difference to the board does not come from the version of OpenStructure; it is reported and not explained. The poses of question 5 were scored on both versions, the two runs agree for every one of the 260 poses (the largest difference is zero), and the tables of the run with 2.11.1 are the ones reported. The correlation of the resolution with LDDT-PLI in question 5 leaves out the released files whose header records no resolution (three), because the first run returned no value when they were included.

## What follows from the results

Nothing in this plan changes a submitted file or a number of an earlier release. If a result points at a cause that an experiment could address, the experiment is written as a new section of this plan, with its rule, before it runs. It is judged by cross-validation on the training data, as every experiment of the project is. The test labels are used for it at most once, at the end, and the result is reported as obtained after the deadline and with knowledge of the labels.

## Reporting

The results are produced by programs of the project from the released files, written to tables and logs, and read from them by the method report, the notebook and the audit, like every other number. The release that carries them is 0.3.1.

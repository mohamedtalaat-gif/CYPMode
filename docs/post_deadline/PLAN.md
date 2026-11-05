# After the deadline: what the released labels and structures can say about the submitted entries

This plan is written before any label of the test set and any released structure has been read by a program of the project. The only numbers it uses are those that the organisers published on the final leaderboards on 5 November 2026 and in the post that announces them. Every question below has its measure and its decision rule here, so that no rule is chosen after a result is seen.

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
| structure | LDDT-LP | 0.826 |

The +- values are the standard deviations of the bootstrap of the organisers. The last compound of tier 1 in the structure track has an LDDT-PLI of 0.4148, which is 0.0014 above the entry.

## Question 0: does the scoring of the project reproduce the board (a gate)

The scoring is the organisers' own: the evaluation code of their tutorial repository for the classification track, and OpenStructure with the settings of the structure track, run in the environment that the project already uses for it. The commit of the evaluation code that is read is recorded with the result.

For the classification track, each of the two candidate files is scored on the released labels with the scored-set rules of the Space (positives and inferred positives against negatives and assigned negatives, only compounds whose label can be assigned with confidence). The submitted file is the candidate whose macro MCC lies within 0.01 of 0.3723; if both or neither do, the analysis stops there and reports that. For the structure track, the submitted archive is scored; the gate is a mean LDDT-PLI within 0.005 of 0.4133 and a mean BiSyRMSD within 0.1 A of 3.970.

If a gate fails, no later question is answered; the difference is investigated and written down first.

## Question 1: which TDI candidate is better on the released labels

Macro MCC and the MCC of each isoform for candidate A (hits only) and candidate B (selection aware), with a paired bootstrap of the compounds (1,000 resamples, seed 42) for the difference B minus A. Verdict: B is better if the 95% interval of the difference lies above zero, A is better if it lies below zero, and no difference is shown otherwise. The positive rate of each file is reported next to the true positive rate of each isoform.

## Question 2: do the scored compounds reproduce the label rule of the project

The rule recovered from the training data is applied to the test file and compared with the organisers' labels, compound by compound. The count of disagreements is reported with each disagreeing compound listed. The expectation is zero disagreements; any disagreement is a finding about the rule and is explained before it is used.

## Question 3: how much of the distance to the leading entries is a matter of the threshold

For the submitted file, the MCC at the threshold that was submitted is set beside the highest MCC that any threshold would give on the released labels. The second number is a ceiling that uses the labels and is labelled so; it is not a result and it is not used to change a file. The question measures how much of the gap to the entries of tier 1 could come from the call rate alone, given that the test set holds 5.9% CYP2D6 positives against 21.6% in training.

## Question 4: was the submitted file affected by the live board

The live leaderboard scored half of the test set. If the organisers publish which compounds that half holds, the macro MCC of the submitted file is computed on that half and on the other half, with a bootstrap of the difference. The file was submitted once and was never tuned on the board, so no difference beyond the bootstrap noise is expected. If the list is not published by the organisers, the question is dropped; no substitute split is made.

## Question 5: structure track, compound by compound

The released structures are the reference. For each of the 20 compounds, LDDT-PLI and BiSyRMSD are computed for four poses: the submitted one; the pose that unguided Boltz-2 ranks first (no contact restraint, default sampling; taken from the runs that exist, and generated once with the pinned version and the seed of the benchmark validation if they do not exist); the best pose of the pool of the compound (a ceiling that uses the reference); and the expected value of a pose drawn at random from the pool.

- Question 5a. Mean paired difference of LDDT-PLI between the submitted pose and the unguided Boltz-2 pose, paired bootstrap of the 20 compounds (10,000 resamples, seed 42). Verdict: the recipe is better if the 95% interval lies above zero, worse if below, and no difference is shown otherwise. The number of compounds in which the recipe is higher is reported next to it.
- Question 5b. The gap between the submitted pose and the best pose of the pool, set against the gap between the random pose and the best pose. It says whether the limit lay in producing a good pose or in choosing it.
- Question 5c. In the released structure of each compound, whether a donor atom of the ligand is bound to the heme iron by the distance criterion that the project used for the earlier cryo-EM structures, against the call that the submitted pose makes. The two-by-two table is reported.

With 20 compounds the error bar of the LDDT-PLI of one entry is about 0.04, so a difference below that size is reported as not shown, whatever its sign.

## What follows from the results

Nothing in this plan changes a submitted file or a number of an earlier release. If a result points at a cause that an experiment could address, the experiment is written as a new section of this plan, with its rule, before it runs. It is judged by cross-validation on the training data, as every experiment of the project is. The test labels are used for it at most once, at the end, and the result is reported as obtained after the deadline and with knowledge of the labels.

## Reporting

The results are produced by programs of the project from the released files, written to tables and logs, and read from them by the method report, the notebook and the audit, like every other number. The release that carries them is 0.3.1; it holds code and reports only, and no archive of the data record is made for it.

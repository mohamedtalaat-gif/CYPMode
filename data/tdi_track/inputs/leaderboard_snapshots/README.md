# Leaderboard snapshots

Per-isoform TDI leaderboard exports pulled on 2026-10-04 from the challenge Space's public CSV
endpoints (interim = full test set frozen at the 2026-09-24 deadline; live = half of it). They are
used only by `cypmode.challenge.leaderboard_prior`, which recovers the test set's positive rate
from each row's accuracy / precision / recall -- see that module for the derivation and the
cross-row spread check. Nothing here contains test labels.

# Submissions

This directory holds the actual, competition-ready submission files for the
[OpenADMET CYP Inhibition Blind Challenge](https://huggingface.co/spaces/openadmet/cyp-challenge)
— kept separate from `data/*/outputs/`, which holds this project's own
exploration, ablations, calculation sheets, and figures. Nothing in
`data/*/outputs/` is submission-ready as-is; the files that actually go to
the challenge portal live here, once they exist.

**Nothing has been submitted yet.** Per this project's own standing rule,
no submission is made unilaterally — these folders are scaffolding, ready
to receive the real file once a track's pipeline actually produces one.

## Per-track requirements (verified directly against the live HF Space
this session, 2026-09-26 — `config.py` and `submission.py` at
`huggingface.co/spaces/openadmet/cyp-challenge`, not just recalled from
memory)

- [`tdi_track/`](tdi_track/README.md) — Time-Dependent Inhibition (classification). Our track.
- [`structure_track/`](structure_track/README.md) — Structure Prediction. Our track.
- [`direct_inhibition_track/`](direct_inhibition_track/README.md) — Direct Inhibition (regression). Not started; schema kept for reference only.

## Submission-mechanics details that apply across all three tracks

Confirmed from the live `submission.py`, not previously documented in this
project in this much detail:

- **Rate limit**: one submission per track per 12 hours (`HOURS_BETWEEN_SUBMISSIONS`); only the latest valid submission per track counts toward the leaderboard.
- **Required metadata at submission time**: HuggingFace `username` (validated), `user_alias`, an `anonymous` flag, and the track name.
- **Optional metadata**: `model_report_link` (validated as a reachable HTTPS URL — but only required *when* `open_source_code=True`), `include_in_publication`, `used_proprietary_data`, `open_source_code`, plus contact fields (`participant_name`, `discord_username`, `email`, `affiliation`).
- Submissions are stored server-side as `s3://{bucket}/submissions/{track}/{username}/{submission_id}/` containing `metadata.json` plus the predictions file (or `structures.zip` for the structure track).

The submission portal itself (uploading, disclosing proprietary-data use,
supplying the report link) is a manual, one-at-a-time action on the actual
HF Space UI — not something this repo automates or should automate.

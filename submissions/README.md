# Submissions

The files that would be uploaded to the [OpenADMET CYP Inhibition Blind Challenge](https://huggingface.co/spaces/openadmet/cyp-challenge), kept apart from `data/*/outputs/`, which holds the project's
own experiments. Each track folder has its own README with the required format, the checks every file passed and the file hashes, and is rewritten by `cypmode.challenge.submissions_sync`
from the final results. **No file here has been uploaded to the portal.**

- [`tdi_track/`](tdi_track/README.md): time-dependent inhibition (classification), CYP3A4 and CYP2D6.
- [`structure_track/`](structure_track/README.md): structure prediction, 20 CYP3A4 complexes.

## Mechanics that apply to both tracks (read from the portal's `submission.py`, 2026-09-26)

- One submission per track every 12 hours; only the latest valid submission of a track counts for the leaderboard.
- Required at submission: the Hugging Face `username`, a `user_alias`, an `anonymous` flag and the track.
- Optional: `model_report_link` (a reachable HTTPS URL, required when `open_source_code` is ticked), `include_in_publication`, `used_proprietary_data`, `open_source_code` and contact fields.
- The upload itself is a manual action on the portal.

`_superseded/` (when present) holds files that earlier versions of these folders contained and that are not part of the final set.

# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

from cypmode.audit import public_readiness as pr


def test_private_paths_credentials_and_foreign_emails_are_found():
    text = "see /Users/" + "someone/project and /Volumes/" + "Data/x/ token hf_" + "a" * 34 + " mail me" + "@lab.org and ok@example.com"
    found = pr.scan_text(text)
    assert set(found) == {"private path", "credential", "e-mail"} and found["e-mail"] == ["me" + "@lab.org"]


def test_clean_text_has_no_findings_and_shingles_detect_copied_runs():
    assert pr.scan_text("relative paths like data/tdi_track/inputs are fine") == {}
    book = " ".join(f"word{i} alpha{i % 7} beta{i % 5}" for i in range(40))
    assert pr.shingles(book) & pr.shingles("intro text. " + book + " closing words")
    assert not (pr.shingles(book) & pr.shingles("completely different sentence about enzymes and heme iron coordination in cytochromes"))

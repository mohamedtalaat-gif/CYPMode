# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

from cypmode.audit import prose_rules


def write(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return name


def test_a_line_that_refers_to_an_owner_is_found_with_its_place(tmp_path):
    files = [
        write(tmp_path, "SUBMISSION_CHECKLIST.md", "- [ ] An item.\n- [ ] The owner decided on 2026-10-25.\n"),
        write(tmp_path, "notebooks/parts/part1.py", "text = 'a step of the Owner'\n"),
        write(tmp_path, "cypmode/module.py", "# the owner's choice\n"),
    ]
    assert prose_rules.mentions_of_the_owner(tmp_path, files) == ["SUBMISSION_CHECKLIST.md:2", "notebooks/parts/part1.py:1", "cypmode/module.py:1"]


def test_a_word_that_only_contains_the_letters_is_not_a_reference(tmp_path):
    files = [write(tmp_path, "README.md", "ownership of the data and a co-owned file are not a person\n")]
    assert prose_rules.mentions_of_the_owner(tmp_path, files) == []


def test_licence_texts_the_audit_report_and_other_file_kinds_are_left_alone(tmp_path):
    files = [
        write(tmp_path, "LICENSE", "the copyright owner\n"),
        write(tmp_path, "audit/AUDIT_REPORT.md", "no text refers to an owner\n"),
        write(tmp_path, "data/outputs/README.md", "owner\n"),
        write(tmp_path, "docs/figure.svg", "<text>owner</text>\n"),
    ]
    assert prose_rules.mentions_of_the_owner(tmp_path, files) == []


def test_no_text_of_this_repository_refers_to_an_owner():
    assert prose_rules.mentions_of_the_owner() == []  # the checklist, the notebook, the reports and the comments are written without a person named by a role

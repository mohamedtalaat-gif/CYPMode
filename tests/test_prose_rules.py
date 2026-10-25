# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

from types import SimpleNamespace

from cypmode.audit import prose_rules
from cypmode.audit.core import FAIL


def write(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return name


ROLE = prose_rules.ROLE


def test_a_line_that_names_a_person_by_a_role_is_found_with_its_place(tmp_path):
    files = [
        write(tmp_path, "SUBMISSION_CHECKLIST.md", f"- [ ] An item.\n- [ ] The {ROLE} decided on 2026-10-25.\n"),
        write(tmp_path, "notebooks/parts/part1.py", f"text = 'a step of the {ROLE.title()}'\n"),
        write(tmp_path, "cypmode/module.py", f"# the {ROLE}'s choice\n"),
    ]
    assert prose_rules.role_mentions(tmp_path, files) == ["SUBMISSION_CHECKLIST.md:2", "notebooks/parts/part1.py:1", "cypmode/module.py:1"]


def test_a_word_that_only_contains_the_letters_is_not_a_reference(tmp_path):
    files = [write(tmp_path, "README.md", f"{ROLE}ship of the data and a co-{ROLE}ed file are not a person\n")]
    assert prose_rules.role_mentions(tmp_path, files) == []


def test_licence_texts_vendored_libraries_the_audit_report_and_other_kinds_of_file_are_left_alone(tmp_path):
    files = [
        write(tmp_path, "LICENSES/AGPL-3.0-or-later.txt", f"the copyright {ROLE}\n"),
        write(tmp_path, "audit/AUDIT_REPORT.md", f"no text names the {ROLE}\n"),
        write(tmp_path, "docs/card/vendor/lib-min.js", "an honest — line\n"),
        write(tmp_path, "data/outputs/README.md", f"{ROLE}\n"),
        write(tmp_path, "docs/figure.svg", f"<text>{ROLE}</text>\n"),
    ]
    assert all(not hits for hits in prose_rules.violations(tmp_path, files).values())


def test_each_rule_finds_the_wording_it_is_named_for(tmp_path):
    files = [
        write(tmp_path, "a.md", f"the user decided on a plan and the {ROLE} agreed\n"),
        write(tmp_path, "b.md", "I added the table and I'll check it\n"),
        write(tmp_path, "c.md", "fixed this session, your call\n"),
        write(tmp_path, "d.md", "We checked it and our result held for us\n"),
        write(tmp_path, "e.md", "it doesn't work and that's all\n"),
        write(tmp_path, "f.md", "NOT WRITTEN YET\n"),
        write(tmp_path, "g.md", "a result \u2014 and a tick \u2713\n"),
        write(tmp_path, "h.md", "an honest look, genuinely different\n"),
        write(tmp_path, "i.md", "it is worth noting that, moreover, it works\n"),
    ]
    found = prose_rules.violations(tmp_path, files)
    assert list(found) == list(prose_rules.RULES)
    assert [lines for lines in found.values()] == [[f"{name}.md:1"] for name in "abcdefghi"]


def test_a_quotation_in_a_document_keeps_the_words_of_its_source(tmp_path):
    files = [write(tmp_path, "a.md", 'The text reads "a shift can\'t be measured directly" and no more.\n'), write(tmp_path, "b.py", 'message = "a shift can\'t be measured"\n')]
    found = prose_rules.violations(tmp_path, files)
    assert found["contractions"] == ["b.py:1"]  # in code the string is text that is shown, so it is matched


def test_the_two_letters_of_a_country_and_the_possessive_of_a_variable_are_not_the_plural_first_person(tmp_path):
    files = [write(tmp_path, "a.py", "locale = 'en-US'\nours = load()\n"), write(tmp_path, "b.md", "the US Food and Drug Administration\n")]
    assert prose_rules.violations(tmp_path, files)["first-person plural"] == []
    assert prose_rules.violations(tmp_path, [write(tmp_path, "c.md", "compared with ours\n")])["first-person plural"] == ["c.md:1"]


def test_an_established_term_that_contains_a_listed_word_is_not_a_manner_of_writing(tmp_path):
    files = [write(tmp_path, "a.md", "Tukey's honestly significant difference compares every pair of means.\n")]
    assert all(not hits for hits in prose_rules.violations(tmp_path, files).values())


def test_the_audit_section_reports_every_rule_with_its_count(tmp_path, monkeypatch):
    from cypmode.audit import sections_extra

    files = [write(tmp_path, "a.md", "an honest look — and nothing else\n")]
    monkeypatch.setattr(prose_rules, "ROOT", tmp_path)
    monkeypatch.setattr(prose_rules, "project_files", lambda root=tmp_path: files)
    monkeypatch.setattr(sections_extra, "run", lambda command, *a, **k: "A commit message.\n")
    ctx = SimpleNamespace(metrics={})
    section = sections_extra.wording(ctx)
    failed = [check for check in section.checks if check.status == FAIL]
    assert len(section.checks) == len(prose_rules.RULES) and len(failed) == 2  # the dash and the intensifier
    assert ctx.metrics["wording_lines"] == 2


def test_no_text_of_this_repository_breaks_a_wording_rule():
    found = prose_rules.violations()
    assert {rule: lines for rule, lines in found.items() if lines} == {}  # the checklist, the notebook, the reports and the comments are the plain writing of the researcher

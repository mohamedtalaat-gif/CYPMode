# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import re

from cypmode.audit import project_map as pm
from cypmode.audit import prose_rules


def test_every_module_is_in_the_code_page_exactly_once_and_every_job_in_the_queue_page():
    code, queue = pm.render_code(), pm.render_queue()
    for module, path in pm.module_paths().items():
        assert len(re.findall(rf"^\| `{re.escape(path.relative_to(pm.ROOT).as_posix())}` \|", code, re.MULTILINE)) == 1, module
    for path in (pm.ROOT / "queue/jobs").glob("*.sh"):
        assert len(re.findall(rf"^\| `{path.stem}` \|", queue, re.MULTILINE)) == 1, path.name  # one row in the table of its family
    data = pm.explorer_data()  # the page that can be explored lists every module and every job once as well
    assert sorted(m["id"] for m in data["modules"]) == sorted(pm.module_paths()) and sorted(j["name"] for j in data["jobs"]) == sorted(p.stem for p in (pm.ROOT / "queue/jobs").glob("*.sh"))


def test_the_arrows_between_packages_count_the_imports_that_cross_them():
    facts = pm.code_facts()
    crossing = sum(1 for m, targets in facts["graph"].items() for t in targets if pm.package_of(t) != pm.package_of(m))
    assert sum(facts["cross"].values()) == crossing
    stated = sum(int(n) for n in re.findall(r"  \w+ -->\|(\d+)\| \w+", pm.render_code()))
    assert stated == crossing
    modules = pm.explorer_data()["modules"]  # and the lists of the page that can be explored agree with the graph, in both directions
    assert sum(len(m["imports"]) for m in modules) == sum(len(t) for t in facts["graph"].values()) == sum(len(m["importedBy"]) for m in modules)


def test_an_import_is_found_wherever_it_stands_and_a_relative_one_is_resolved(tmp_path):
    source = tmp_path / "m.py"
    source.write_text("import cypmode.a\n\ndef f():\n    from cypmode.b import c\n    from . import d\n    from .e import g\n")
    known = {"cypmode.a", "cypmode.b.c", "cypmode.d", "cypmode.e", "cypmode.zz"}
    assert pm.imports_of("cypmode.m", source, known) == {"cypmode.a", "cypmode.b.c", "cypmode.d", "cypmode.e"}


def test_a_long_first_sentence_is_cut_at_a_clause_and_still_ends_in_a_full_stop():
    long = "The module does one thing in a very careful way: " + "and then another thing " * 30
    short = pm.first_sentence(long)
    assert len(short) <= 421 and short == "The module does one thing in a very careful way." and pm.first_sentence("") == "no docstring"
    assert pm.first_sentence("It reads e.g. a table. It then writes.") == "It reads e.g. a table."


def test_every_mermaid_block_is_closed_and_its_nodes_are_defined_once():
    for name, text in pm.build().items():
        assert text.count("```mermaid") == text.count("```") // 2, name
        for block in re.findall(r"```mermaid\n(.*?)```", text, re.DOTALL):
            defined = re.findall(r"^  (\w+)\[", block, re.MULTILINE)
            assert len(defined) == len(set(defined)), f"{name}: a node is defined twice"


def test_the_pages_follow_the_wording_rules_and_do_not_depend_on_the_run():
    first, second = pm.build(), pm.build()
    assert first == second
    for name, text in first.items():
        path = pm.ROOT / "docs/map" / name
        found = prose_rules.violations(files=[path.relative_to(pm.ROOT).as_posix()]) if path.exists() else {}
        assert not any(found.values()), (name, found)
        assert "(pending" not in text
    page = first["explorer.html"]  # the page that can be explored is one file: no script, style or link comes from elsewhere, and its data are the data of the code
    assert not re.findall(r'(?:src|href)="https?://', page) and "<script src" not in page and "<link " not in page
    assert page.count("/*DATA*/") == 0 and pm.explorer_data()["meta"]["modules"] == len(pm.module_paths())


def test_the_committed_pages_match_the_code():
    assert pm.main(["--check"]) == 0


def test_the_readme_shows_the_two_diagrams_of_the_map_and_they_are_the_ones_the_code_draws():
    from cypmode.audit import docs_state

    text = (pm.ROOT / "README.md").read_text()
    assert docs_state.render(text, ("project-map",)) == text
    assert pm.dataflow_diagram(pm.dataflow_facts()) in text and pm.code_diagram(pm.code_facts()) in text

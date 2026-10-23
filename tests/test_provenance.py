# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The provenance record: read from the citation file, written once next to generated data, never over an existing file."""

import json

from cypmode import provenance


def test_the_record_carries_the_author_the_repository_and_the_licence_of_the_citation_file():
    cff = provenance.load()
    rec = provenance.record(["python", "-m", "demo"])
    assert rec["authors"][0]["name"].endswith(cff["authors"][0]["family-names"])
    assert rec["repository"] == cff["repository-code"] and rec["command"] == ["python", "-m", "demo"]
    assert rec["licence"] and rec["cite_as"].startswith(provenance.authors(cff))


def test_the_notice_names_the_author_and_the_repository(capsys):
    provenance.announce()
    err = capsys.readouterr().err
    cff = provenance.load()
    assert cff["authors"][0]["family-names"] in err and cff["repository-code"] in err


def test_a_directory_gets_its_sidecar_once_and_a_file_gets_one_beside_it(tmp_path):
    (tmp_path / "cache").mkdir()
    first = provenance.stamp(tmp_path / "cache")
    first.write_text(json.dumps({"kept": True}))
    assert provenance.stamp(tmp_path / "cache") == first and json.loads(first.read_text()) == {"kept": True}
    data = tmp_path / "table.csv"
    data.write_text("a,b\n1,2\n")
    before = data.read_bytes()
    side = provenance.stamp(data)
    assert side.name == "table.csv.provenance.json" and data.read_bytes() == before
    assert provenance.stamp(tmp_path / "missing") is None


def test_the_doi_enters_the_citation_text_when_the_citation_file_has_one():
    cff = {"title": "T", "version": "1", "date-released": "2026-11-04", "repository-code": "https://example.org/r", "authors": [{"given-names": "A", "family-names": "B"}], "identifiers": [{"type": "doi", "value": "10.5281/zenodo.1"}]}
    assert provenance.citation_text(cff).endswith("https://doi.org/10.5281/zenodo.1")
    cff["identifiers"] = []
    assert provenance.citation_text(cff).endswith("https://example.org/r")


ENTRY_POINTS = (
    "challenge/nested_runners", "challenge/nested_runners_torch", "challenge/nested_runners_chemprop", "challenge/aux_experiments", "challenge/final_tdi_build", "challenge/physics_block", "challenge/frozen_embeddings",
    "challenge/monroe_embeddings", "challenge/atomjepa_embeddings", "challenge/openadmet_surrogates", "challenge/external_continuous", "challenge/all_in_one", "challenge/population_experiment", "challenge/physics_tierb",
    "validation/batch", "validation/final_recipe", "validation/relax_final", "validation/structure_benchmark",
)


def test_every_entry_point_that_generates_data_announces_its_origin():
    for name in ENTRY_POINTS:
        text = (provenance.ROOT / "cypmode" / f"{name}.py").read_text()
        assert 'if __name__ == "__main__":\n    from cypmode import provenance\n\n    provenance.announce()' in text, name


def test_the_prediction_cache_writes_its_sidecar_with_the_first_prediction(tmp_path):
    from types import SimpleNamespace

    import numpy as np

    from cypmode.challenge.nested import PredictionCache

    cache = PredictionCache(tmp_path / "cache", SimpleNamespace(plan_id="plan0"))
    task = SimpleNamespace(key="r0_k0", evaluate=[0, 1, 2])
    cache.put("model_a", task, np.array([0.1, 0.2, 0.3]))
    sidecar = tmp_path / "cache" / provenance.SIDECAR
    assert sidecar.exists() and json.loads(sidecar.read_text())["repository"]
    first = sidecar.read_text()
    cache.put("model_b", task, np.array([0.3, 0.2, 0.1]))
    assert sidecar.read_text() == first


def test_a_command_that_points_inside_the_repository_is_recorded_relative_to_it_and_other_arguments_are_kept():
    inside = str(provenance.ROOT / "cypmode" / "challenge" / "aux_experiments.py")
    rec = provenance.record([inside, "--isoform", "CYP3A4", "/somewhere/else/file.csv"])
    assert rec["command"] == ["cypmode/challenge/aux_experiments.py", "--isoform", "CYP3A4", "/somewhere/else/file.csv"]
    assert str(provenance.ROOT) not in json.dumps(rec)

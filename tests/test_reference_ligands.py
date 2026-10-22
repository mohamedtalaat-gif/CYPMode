# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("Bio")
pytest.importorskip("requests")

import requests
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

from cypmode.validation import reference_ligands
from cypmode.validation.reference_ligands import (
    MAX_FE_DISTANCE_A,
    MIN_HEAVY_ATOMS,
    POCKET_CA_RADIUS_A,
    download_mmcif,
    extract_active_site_ligands,
    fetch_entry_ids,
    fetch_ligand_inventory,
    load_reference_set,
)


def _atom(name, xyz, element, serial):
    return Atom(name, np.asarray(xyz, dtype=np.float32), 0.0, 1.0, " ", name.ljust(4), serial, element)


def _ligand_residue(comp_id: str, seq: int, center, n_atoms: int = 16, seed: int = 0, extra_element: str | None = None) -> Residue:
    rng = np.random.default_rng(seed)
    residue = Residue((f"H_{comp_id}", seq, " "), comp_id, " ")
    for i, xyz in enumerate(rng.normal(size=(n_atoms, 3)) * 1.5 + np.asarray(center, dtype=float)):
        residue.add(_atom(f"C{i}", xyz, "C", 1000 + seq * 100 + i))
    if extra_element:  # a hydrogen, which must never count as a heavy atom
        residue.add(_atom("H1", np.asarray(center) + 1.0, extra_element, 2000 + seq))
    return residue


def _structure(*, with_heme: bool = True, ligands: list[Residue] = (), n_protein: int = 210) -> Structure:
    structure = Structure("s")
    model = Model(0)
    structure.add(model)
    rng = np.random.default_rng(0)
    protein = Chain("A")
    model.add(protein)
    directions = rng.normal(size=(n_protein, 3))
    xyz = directions / np.linalg.norm(directions, axis=1, keepdims=True) * rng.uniform(4, 28, size=(n_protein, 1))
    for i, point in enumerate(xyz, start=1):
        residue = Residue((" ", i, " "), "ALA", " ")
        residue.add(_atom("CA", point, "C", i))
        protein.add(residue)
    if with_heme:
        heme_chain = Chain("H")
        model.add(heme_chain)
        heme = Residue(("H_HEM", 1, " "), "HEM", " ")
        heme.add(_atom("FE", (0.0, 0.0, 0.0), "FE", 5000))
        heme_chain.add(heme)
    ligand_chain = Chain("L")
    model.add(ligand_chain)
    for residue in ligands:
        ligand_chain.add(residue)
    return structure


def _inventory(*comp_ids: str, resolution: float | None = 2.1) -> dict:
    return {
        "resolution": resolution,
        "ligands": {c: {"smiles": "c1ccccc1", "inchikey": f"KEY-{c}", "name": f"name {c}", "mw": 250.0} for c in comp_ids},
    }


# --- extract_active_site_ligands -------------------------------------------------------------------


def test_an_active_site_ligand_is_extracted_with_its_coordinates_iron_and_pocket_ca_atoms():
    structure = _structure(ligands=[_ligand_residue("DRG", 1, (0.0, 0.0, 5.0), extra_element="H")])
    [record] = extract_active_site_ligands(structure, "1ABC", _inventory("DRG"))

    assert (record["pdb_id"], record["comp_id"], record["ligand_chain"]) == ("1ABC", "DRG", "L")
    assert record["n_heavy"] == 16 and len(record["coords"]) == 16 and set(record["elements"]) == {"C"}  # hydrogen excluded
    assert record["fe"] == [0.0, 0.0, 0.0]
    assert record["resolution"] == 2.1 and record["name"] == "name DRG" and record["inchikey"] == "KEY-DRG"
    assert 0 < record["min_fe_distance"] <= MAX_FE_DISTANCE_A

    distances = {k: np.linalg.norm(v) for k, v in record["pocket_ca"].items()}
    assert distances and max(distances.values()) <= POCKET_CA_RADIUS_A + 0.01  # only the pocket, not the whole protein
    assert all(isinstance(k, str) for k in record["pocket_ca"])  # JSON-serialisable residue numbers


def test_a_ligand_in_the_peripheral_site_far_from_the_heme_is_not_an_active_site_reference():
    far = _ligand_residue("PER", 2, (0.0, 0.0, MAX_FE_DISTANCE_A + 20.0))
    near = _ligand_residue("DRG", 1, (0.0, 0.0, 5.0))
    records = extract_active_site_ligands(_structure(ligands=[near, far]), "1ABC", _inventory("DRG", "PER"))
    assert [r["comp_id"] for r in records] == ["DRG"]


def test_components_outside_the_ligand_inventory_are_ignored_even_when_they_sit_in_the_site():
    buffer = _ligand_residue("GOL", 3, (0.0, 0.0, 5.0))
    assert extract_active_site_ligands(_structure(ligands=[buffer]), "1ABC", _inventory("DRG")) == []


def test_a_fragment_below_the_heavy_atom_floor_is_skipped():
    tiny = _ligand_residue("FRG", 4, (0.0, 0.0, 5.0), n_atoms=MIN_HEAVY_ATOMS - 1)
    assert extract_active_site_ligands(_structure(ligands=[tiny]), "1ABC", _inventory("FRG")) == []


def test_a_structure_without_a_heme_has_no_active_site_and_yields_nothing():
    assert extract_active_site_ligands(_structure(with_heme=False, ligands=[_ligand_residue("DRG", 1, (0, 0, 5))]), "1ABC", _inventory("DRG")) == []


def test_a_model_without_a_real_protein_chain_yields_nothing():
    assert extract_active_site_ligands(_structure(n_protein=50, ligands=[_ligand_residue("DRG", 1, (0, 0, 5))]), "1ABC", _inventory("DRG")) == []


def test_every_copy_of_a_ligand_in_the_asymmetric_unit_is_returned_separately():
    copies = [_ligand_residue("DRG", 1, (0.0, 0.0, 5.0), seed=1), _ligand_residue("DRG", 2, (1.0, 0.0, 5.0), seed=2)]
    records = extract_active_site_ligands(_structure(ligands=copies), "1ABC", _inventory("DRG"))
    assert len(records) == 2 and records[0]["coords"] != records[1]["coords"]


def test_records_are_json_serialisable():
    [record] = extract_active_site_ligands(_structure(ligands=[_ligand_residue("DRG", 1, (0, 0, 5))]), "1ABC", _inventory("DRG"))
    assert json.loads(json.dumps(record)) == record


# --- RCSB access (HTTP mocked) -----------------------------------------------------------------------


class _Response:
    def __init__(self, payload=None, text="", status=200):
        self._payload, self.text, self.status_code = payload, text, status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


def test_fetch_entry_ids_posts_a_uniprot_query_and_returns_sorted_ids(monkeypatch):
    seen = {}

    def fake_post(url, json, timeout):
        seen["url"], seen["body"] = url, json
        return _Response({"result_set": ["5A1R", "1TQN", "3NXU"]})

    monkeypatch.setattr(reference_ligands.requests, "post", fake_post)
    assert fetch_entry_ids("P08684") == ["1TQN", "3NXU", "5A1R"]
    assert seen["url"] == reference_ligands.RCSB_SEARCH
    assert seen["body"]["query"]["parameters"]["value"] == "P08684"


def _graphql_entry(rcsb_id, resolution, comps):
    return {
        "rcsb_id": rcsb_id,
        "rcsb_entry_info": {"resolution_combined": resolution},
        "nonpolymer_entities": [
            {"nonpolymer_comp": {"chem_comp": {"id": cid, "name": f"n {cid}", "formula_weight": mw}, "rcsb_chem_comp_descriptor": {"SMILES": "CCO", "InChIKey": f"K{cid}"}}}
            for cid, mw in comps
        ],
    }


def test_inventory_keeps_drug_sized_components_and_drops_blocklisted_and_light_ones(monkeypatch):
    payload = {
        "data": {
            "entries": [
                _graphql_entry("1TQN", [2.05], [("TQN", 420.0), ("HEM", 616.0), ("GOL", 92.0), ("SMA", 120.0), ("PEG", 400.0)]),
                _graphql_entry("1XYZ", None, [("DRG", 300.0)]),
            ]
        }
    }
    monkeypatch.setattr(reference_ligands.requests, "post", lambda url, json, timeout: _Response(payload))
    inventory = fetch_ligand_inventory(["1TQN", "1XYZ"])
    assert set(inventory["1TQN"]["ligands"]) == {"TQN"}  # HEM/GOL/PEG blocklisted, SMA below the MW floor
    assert inventory["1TQN"]["resolution"] == 2.05
    assert inventory["1XYZ"]["resolution"] is None and set(inventory["1XYZ"]["ligands"]) == {"DRG"}


def test_an_entry_with_no_nonpolymer_entities_is_an_empty_inventory(monkeypatch):
    payload = {"data": {"entries": [{"rcsb_id": "1APO", "rcsb_entry_info": {"resolution_combined": [3.0]}, "nonpolymer_entities": None}]}}
    monkeypatch.setattr(reference_ligands.requests, "post", lambda url, json, timeout: _Response(payload))
    assert fetch_ligand_inventory(["1APO"]) == {"1APO": {"resolution": 3.0, "ligands": {}}}


def test_download_is_cached_so_a_second_call_makes_no_request(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        return _Response(text="data_1ABC\n")

    monkeypatch.setattr(reference_ligands.requests, "get", fake_get)
    first = download_mmcif("1ABC", tmp_path)
    second = download_mmcif("1ABC", tmp_path)
    assert first == second == tmp_path / "1ABC.cif" and first.read_text() == "data_1ABC\n"
    assert len(calls) == 1


def test_an_empty_cached_file_is_re_downloaded(tmp_path, monkeypatch):
    (tmp_path / "1ABC.cif").write_text("")
    monkeypatch.setattr(reference_ligands.requests, "get", lambda url, timeout: _Response(text="data_1ABC\n"))
    assert download_mmcif("1ABC", tmp_path).read_text() == "data_1ABC\n"


def test_a_transient_download_failure_is_retried_then_succeeds(tmp_path, monkeypatch):
    attempts = []

    def flaky(url, timeout):
        attempts.append(1)
        if len(attempts) < 3:
            raise requests.ConnectionError("dns")
        return _Response(text="data_1ABC\n")

    monkeypatch.setattr(reference_ligands.requests, "get", flaky)
    monkeypatch.setattr(reference_ligands.time, "sleep", lambda s: None)
    assert download_mmcif("1ABC", tmp_path).read_text() == "data_1ABC\n"
    assert len(attempts) == 3


def test_a_persistent_download_failure_raises_after_the_last_attempt_and_leaves_no_file(tmp_path, monkeypatch):
    def always_down(url, timeout):
        raise requests.ConnectionError("dns")

    monkeypatch.setattr(reference_ligands.requests, "get", always_down)
    monkeypatch.setattr(reference_ligands.time, "sleep", lambda s: None)
    with pytest.raises(requests.ConnectionError):
        download_mmcif("1ABC", tmp_path, attempts=2)
    assert not (tmp_path / "1ABC.cif").exists()


def test_a_http_error_status_is_a_failure_not_cached_content(tmp_path, monkeypatch):
    monkeypatch.setattr(reference_ligands.requests, "get", lambda url, timeout: _Response(text="<html>404</html>", status=404))
    monkeypatch.setattr(reference_ligands.time, "sleep", lambda s: None)
    with pytest.raises(requests.HTTPError):
        download_mmcif("9ZZZ", tmp_path, attempts=2)
    assert not (tmp_path / "9ZZZ.cif").exists()


# --- build / load -------------------------------------------------------------------------------------


def test_build_reference_set_downloads_only_entries_with_a_qualifying_ligand_and_writes_the_json(tmp_path, monkeypatch):
    inventory = {"1AAA": _inventory("DRG"), "1BBB": {"resolution": 2.0, "ligands": {}}}
    downloaded = []
    monkeypatch.setattr(reference_ligands, "fetch_entry_ids", lambda: ["1AAA", "1BBB"])
    monkeypatch.setattr(reference_ligands, "fetch_ligand_inventory", lambda ids: inventory)
    monkeypatch.setattr(reference_ligands, "download_mmcif", lambda pid, cache_dir: downloaded.append(pid) or Path(f"{pid}.cif"))

    parser_module = importlib.import_module("Bio.PDB.MMCIFParser")  # the module, not the class Bio.PDB re-exports under the same name

    class _Parser:
        def __init__(self, QUIET=True):
            pass

        def get_structure(self, pid, path):
            return _structure(ligands=[_ligand_residue("DRG", 1, (0, 0, 5))])

    monkeypatch.setattr(parser_module, "MMCIFParser", _Parser)

    out = tmp_path / "refs.json"
    records = reference_ligands.build_reference_set(out_json=out, cache_dir=tmp_path)

    assert downloaded == ["1AAA"]  # 1BBB has nothing to extract, so it is never fetched
    assert [r["pdb_id"] for r in records] == ["1AAA"]
    document = json.loads(out.read_text())
    assert document["n_entries"] == 2 and document["ligands"] == records
    assert load_reference_set(out) == records


def test_load_reference_set_returns_just_the_ligand_list(tmp_path):
    path = tmp_path / "refs.json"
    path.write_text(json.dumps({"source": "x", "n_entries": 1, "ligands": [{"pdb_id": "1ABC", "comp_id": "DRG"}]}))
    assert load_reference_set(path) == [{"pdb_id": "1ABC", "comp_id": "DRG"}]


def test_the_committed_reference_set_is_well_formed_and_usable_by_the_rescoring_code():
    path = Path("data/structure_track/inputs/cyp3a4_reference_ligands.json")
    if not path.exists():
        pytest.skip("reference set not built (python -m cypmode.validation.reference_ligands)")
    references = load_reference_set(path)
    assert len(references) >= 50
    for ref in references:
        assert ref["n_heavy"] >= MIN_HEAVY_ATOMS and len(ref["coords"]) == ref["n_heavy"] == len(ref["elements"])
        assert ref["min_fe_distance"] <= MAX_FE_DISTANCE_A
        assert len(ref["pocket_ca"]) >= 25 and ref["smiles"] is not None
        assert np.asarray(ref["coords"]).shape == (ref["n_heavy"], 3)

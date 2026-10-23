# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Experimental CYP3A4 co-crystal ligands from the PDB -- the reference set
for cypmode.validation.pose_rescoring.

Why: the Structure Track's accuracy metrics are scored against sequestered
experimental structures, so no predicted pose can be scored locally. What
CAN be used is prior experimental evidence about where drug-like ligands
sit in this pocket: RCSB holds ~130 CYP3A4 entries (UniProt P08684), ~100
distinct drug-like ligands among them. This module turns those into a
compact, versioned reference set of active-site ligand positions plus the
pocket C-alpha atoms needed to superpose each one into any predicted
complex's frame.

Only ACTIVE-SITE ligands are kept: the nearest heavy atom must lie within
MAX_FE_DISTANCE_A of a heme iron. CYP3A4 also binds ligands at a peripheral
site (e.g. progesterone in 1W0F) -- real, but not where an inhibitor docks,
and including them would pull the pose prior away from the catalytic site.
Buffers, ions, detergents and other small molecules are excluded by a
component blocklist plus a size floor (MIN_HEAVY_ATOMS, MW >= MIN_MW).

The raw mmCIFs are cached under data/structure_track/inputs/pdb_cache/
(gitignored, regenerable); the compact REFERENCE_JSON is what's tracked.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

UNIPROT = "P08684"
RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
RCSB_FILES = "https://files.rcsb.org/download/{pdb_id}.cif"

CACHE_DIR = Path("data/structure_track/inputs/pdb_cache")
REFERENCE_JSON = Path("data/structure_track/inputs/cyp3a4_reference_ligands.json")

MAX_FE_DISTANCE_A = 12.0
POCKET_CA_RADIUS_A = 15.0
MIN_HEAVY_ATOMS = 12
MIN_MW = 180.0

# Crystallization additives, buffers, ions and cofactors -- never inhibitors.
NON_LIGAND_COMPS = frozenset(
    ["HEM", "GOL", "EDO", "SO4", "PEG", "PGE", "PG4", "P6G", "1PE", "ACT", "CL", "NA", "MG", "ZN", "CA", "K", "FMT", "DMS", "BME", "TRS", "MPD", "OCT", "BOG", "PO4", "NO3", "IMD", "MES", "EPE", "TAM", "CIT", "TLA", "DTT", "SCN", "IOD", "BR", "NI", "CD", "CO", "MN", "FE", "CU", "MLI", "BCT", "PGO", "PGR", "ACY", "FLC", "15P", "2PE", "7PE", "PE4", "PE5", "PE8", "PG0", "PG6", "12P", "HOH"]
)

_GRAPHQL = """
query($ids:[String!]!){ entries(entry_ids:$ids){ rcsb_id
  rcsb_entry_info{ resolution_combined }
  nonpolymer_entities{ nonpolymer_comp{ chem_comp{ id name formula_weight }
    rcsb_chem_comp_descriptor{ SMILES InChIKey } } } } }"""


def fetch_entry_ids(uniprot: str = UNIPROT, timeout: int = 60) -> list[str]:
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match",
                "value": uniprot,
            },
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": 1000}, "results_verbosity": "compact"},
    }
    response = requests.post(RCSB_SEARCH, json=query, timeout=timeout)
    response.raise_for_status()
    return sorted(response.json()["result_set"])


def fetch_ligand_inventory(pdb_ids: list[str], timeout: int = 120) -> dict:
    """{pdb_id: {"resolution": float | None, "ligands": {comp_id: {"smiles", "inchikey", "name", "mw"}}}} --
    only components that clear the blocklist and the MW floor.
    """
    response = requests.post(RCSB_GRAPHQL, json={"query": _GRAPHQL, "variables": {"ids": pdb_ids}}, timeout=timeout)
    response.raise_for_status()
    inventory = {}
    for entry in response.json()["data"]["entries"]:
        resolution = (entry["rcsb_entry_info"].get("resolution_combined") or [None])[0]
        ligands = {}
        for npe in entry.get("nonpolymer_entities") or []:
            comp = npe["nonpolymer_comp"]["chem_comp"]
            if comp["id"] in NON_LIGAND_COMPS or (comp["formula_weight"] or 0) < MIN_MW:
                continue
            descriptor = npe["nonpolymer_comp"].get("rcsb_chem_comp_descriptor") or {}
            ligands[comp["id"]] = {
                "smiles": descriptor.get("SMILES"),
                "inchikey": descriptor.get("InChIKey"),
                "name": comp["name"],
                "mw": comp["formula_weight"],
            }
        inventory[entry["rcsb_id"]] = {"resolution": resolution, "ligands": ligands}
    return inventory


def download_mmcif(pdb_id: str, cache_dir: Path = CACHE_DIR, attempts: int = 4) -> Path:
    path = cache_dir / f"{pdb_id}.cif"
    if path.exists() and path.stat().st_size > 0:
        return path
    cache_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(RCSB_FILES.format(pdb_id=pdb_id), timeout=60)
            response.raise_for_status()
            path.write_text(response.text)
            return path
        except requests.RequestException:
            if attempt == attempts:
                raise
            time.sleep(2 * attempt)
    raise AssertionError("unreachable")


def _heavy(atoms):
    return [a for a in atoms if (a.element or "").strip().upper() not in ("H", "D", "")]


def extract_active_site_ligands(structure, pdb_id: str, entry_inventory: dict) -> list[dict]:
    """Every drug-like ligand copy in this model whose nearest heavy atom is
    within MAX_FE_DISTANCE_A of a heme iron, with the pocket C-alpha atoms
    (by PDB residue number) of the protein chain that iron belongs to --
    what cypmode.validation.pose_rescoring needs to superpose the ligand
    into another structure's frame.
    """
    model = structure[0]
    wanted = entry_inventory["ligands"]

    irons = [
        atom
        for residue in model.get_residues()
        if residue.resname == "HEM"
        for atom in residue
        if atom.get_name().strip().upper() == "FE"
    ]
    if not irons:
        return []

    protein_chains = [c for c in model if sum(1 for r in c if r.id[0] == " " and "CA" in r) >= 200]
    records = []
    for residue in model.get_residues():
        comp_id = residue.resname.strip()
        if residue.id[0] == " " or comp_id not in wanted:
            continue
        atoms = _heavy(residue.get_atoms())
        if len(atoms) < MIN_HEAVY_ATOMS:
            continue
        coords = np.array([a.coord for a in atoms], dtype=float)
        nearest_fe = min(irons, key=lambda fe: float(np.linalg.norm(coords - fe.coord, axis=1).min()))
        min_fe_distance = float(np.linalg.norm(coords - nearest_fe.coord, axis=1).min())
        if min_fe_distance > MAX_FE_DISTANCE_A:
            continue

        def _near(chain, fe=nearest_fe):
            return sum(1 for r in chain if "CA" in r and np.linalg.norm(r["CA"].coord - fe.coord) <= 20.0)

        chain = max(protein_chains, key=_near, default=None)
        if chain is None or _near(chain) < 40:
            continue
        pocket_ca = {
            r.id[1]: [round(float(x), 2) for x in r["CA"].coord]
            for r in chain
            if r.id[0] == " " and r.id[2] == " " and "CA" in r and np.linalg.norm(r["CA"].coord - nearest_fe.coord) <= POCKET_CA_RADIUS_A
        }
        records.append(
            {
                "pdb_id": pdb_id,
                "comp_id": comp_id,
                "name": wanted[comp_id]["name"],
                "smiles": wanted[comp_id]["smiles"],
                "inchikey": wanted[comp_id]["inchikey"],
                "resolution": entry_inventory["resolution"],
                "ligand_chain": residue.get_parent().id,
                "n_heavy": len(atoms),
                "min_fe_distance": round(min_fe_distance, 2),
                "elements": [a.element.strip().upper() for a in atoms],
                "coords": [[round(float(x), 2) for x in c] for c in coords],
                "fe": [round(float(x), 2) for x in nearest_fe.coord],
                "pocket_ca": {str(k): v for k, v in pocket_ca.items()},
            }
        )
    return records


def build_reference_set(out_json: Path = REFERENCE_JSON, cache_dir: Path = CACHE_DIR, workers: int = 4, uniprot: str = UNIPROT) -> list[dict]:
    """End to end: list the CYP3A4 entries, fetch their ligand inventory,
    download the structures that have a qualifying ligand, extract the
    active-site ligands, write the compact JSON. Resumable at the download
    level (cached CIFs are reused).
    """
    from Bio.PDB.MMCIFParser import MMCIFParser

    pdb_ids = fetch_entry_ids(uniprot)
    inventory = fetch_ligand_inventory(pdb_ids)
    todo = [pid for pid in pdb_ids if inventory.get(pid, {}).get("ligands")]
    with ThreadPoolExecutor(workers) as pool:
        paths = dict(zip(todo, pool.map(lambda pid: download_mmcif(pid, cache_dir), todo), strict=True))

    parser = MMCIFParser(QUIET=True)
    records = []
    for pdb_id in todo:
        structure = parser.get_structure(pdb_id, str(paths[pdb_id]))
        records.extend(extract_active_site_ligands(structure, pdb_id, inventory[pdb_id]))

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"source": f"RCSB, UniProt {uniprot}", "n_entries": len(pdb_ids), "ligands": records}) + "\n")
    return records


def load_reference_set(path: Path = REFERENCE_JSON) -> list[dict]:
    return json.loads(Path(path).read_text())["ligands"]


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uniprot", default=UNIPROT, help="the enzyme whose crystal ligands are collected (P08684 CYP3A4, P10635 CYP2D6)")
    parser.add_argument("--out", default=str(REFERENCE_JSON))
    args = parser.parse_args()
    records = build_reference_set(Path(args.out), uniprot=args.uniprot)
    distinct = {r["comp_id"] for r in records}
    entries = {r["pdb_id"] for r in records}
    print(f"{len(records)} active-site ligand copies | {len(distinct)} distinct ligands | {len(entries)} PDB entries")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    _main()

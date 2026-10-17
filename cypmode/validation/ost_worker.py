"""One OpenStructure scoring of a predicted complex against a reference complex, run inside the ``ost`` conda environment.

``mamba run -n ost python cypmode/validation/ost_worker.py --model model.pdb --reference reference.pdb --smiles SMILES`` prints one JSON object with the LDDT-PLI,
BiSyRMSD and LDDT-LP of the best ligand pairing, built the way the challenge's scorer builds it (``evaluate_predictions.score_single_structure`` of
OpenADMET/CYP-Challenge-Tutorial): both complexes read with ``PDBPrep(fault_tolerant=True)``, every copy of the ``LIG`` residue given bond orders from the SMILES and
handed to ``SCRMSDScorer`` and ``LDDTPLIScorer`` (``substructure_match=True``, ``coverage_delta=0.1``), the pairing ranked by LDDT-PLI and then BiSyRMSD.

It needs only OpenStructure and RDKit, so it loads ``official_posebusters`` (ligand copies with bond orders) from its file instead of importing the project package.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load_ligand_copies():
    spec = importlib.util.spec_from_file_location("official_posebusters", HERE / "official_posebusters.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module.ligand_copies


def score(model_path: str, reference_path: str, smiles: str) -> dict:
    from ost import io
    from ost.mol.alg.ligand_scoring import LDDTPLIScorer, SCRMSDScorer
    from ost.mol.alg.scoring_base import PDBPrep
    from rdkit import Chem

    ligand_copies = _load_ligand_copies()
    model, reference = PDBPrep(model_path, fault_tolerant=True), PDBPrep(reference_path, fault_tolerant=True)
    model_ligands, reference_ligands = [], []
    for copies, target in ((ligand_copies(model_path, smiles), model_ligands), (ligand_copies(reference_path, smiles), reference_ligands)):
        for i, copy in enumerate(copies):
            entity = io.SDFStrToEntity(Chem.MolToMolBlock(Chem.RemoveHs(copy)))
            entity.EditXCS().RenameChain(entity.chains[0], f"LIG{i}")
            target.append(entity)
    if not model_ligands or not reference_ligands:
        return {"error": "no ligand copy with bond orders on one side"}
    kwargs = {"model": model, "target": reference, "model_ligands": model_ligands, "target_ligands": reference_ligands, "substructure_match": True, "coverage_delta": 0.1}
    scrmsd, lddt_pli = SCRMSDScorer(**kwargs), LDDTPLIScorer(**kwargs)

    def key(ligand) -> str:
        return f"{ligand.chain.name}.{ligand.number}"

    pli_ref, pli_mdl = [key(x) for x in lddt_pli.target_ligands], [key(x) for x in lddt_pli.model_ligands]
    sc_ref, sc_mdl = [key(x) for x in scrmsd.target_ligands], [key(x) for x in scrmsd.model_ligands]
    results = []
    for i, j in lddt_pli.assignment:
        if pli_ref[i] not in sc_ref or pli_mdl[j] not in sc_mdl:
            continue
        a, b = sc_ref.index(pli_ref[i]), sc_mdl.index(pli_mdl[j])
        ok = scrmsd.state_matrix[a, b] == 0
        results.append({"lddt_pli": float(lddt_pli.score_matrix[i, j]), "bisyrmsd": float(scrmsd.score_matrix[a, b]) if ok else float("nan"), "lddt_lp": float(scrmsd.aux_matrix[a, b]["lddt_lp"]) if ok else float("nan")})
    if not results:
        return {"error": "no ligand assignment"}
    best = sorted(results, key=lambda r: (-r["lddt_pli"], r["bisyrmsd"] if np.isfinite(r["bisyrmsd"]) else 1e9))[0]
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--smiles", required=True)
    args = parser.parse_args()
    try:
        out = score(args.model, args.reference, args.smiles)
    except Exception as exc:
        out = {"error": f"{type(exc).__name__}: {exc}"}
    print("OST_RESULT " + json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

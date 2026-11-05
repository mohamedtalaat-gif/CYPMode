# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The small tables that the simulators of the complete explorer read, and a copy of the public bundle of the results: ``python explorer/full/build_data.py`` (main environment).

Everything here comes from the public training file of the challenge through the loaders and the functions of the project, so that a simulator that shows a real compound shows what the notebook computed:
``label_points.csv`` has the two measurements, their standard deviations, the label, the shift and the confidence z of every compound with both arms; ``compounds.csv`` has the name, the SMILES and the labels
of every compound of the training file, for the molecule explorer.
"""

import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cypmode.challenge.label_confidence import STD_SUFFIX, confidence_z
from cypmode.challenge.library_blocks import OTHER_ISOFORM_COLUMNS
from cypmode.challenge.tdi_baseline import TDI_TRAIN_PATH
from cypmode.challenge.tdi_targets import (
    BOTH_ARMS,
    NO_DIRECT,
    TDI_ISOFORMS,
    direct_column,
    label_column,
    tdi_column,
    training_view,
)

OUT = Path(__file__).parent / "data"


def label_points(train: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for isoform in TDI_ISOFORMS:
        view = training_view(train, isoform)
        z = confidence_z(train, isoform)
        keep = (view["population"] == BOTH_ARMS).to_numpy()
        frames.append(pd.DataFrame({
            "isoform": isoform, "d": view["direct"], "t": view["tdi"], "sd_d": train[direct_column(isoform) + STD_SUFFIX], "sd_t": train[tdi_column(isoform) + STD_SUFFIX],
            "label": view["label"].astype("Int64"), "shift": view["shift"], "z": z,
        })[keep])
    return pd.concat(frames, ignore_index=True)


def compounds(train: pd.DataFrame) -> pd.DataFrame:
    out = train[["Molecule_Name", "SMILES"]].copy()
    for isoform in TDI_ISOFORMS:
        out[label_column(isoform)] = train[label_column(isoform)]
    return out


def jobs() -> pd.DataFrame:
    """The queue: the number, the name and the first comment line of every job, as the notebook's Appendix A lists them."""
    rows = []
    for path in sorted((ROOT / "queue/jobs").glob("*.sh")):
        lines = path.read_text().splitlines()
        rows.append({"number": int(re.match(r"\d+", path.name).group()), "job": path.stem, "title": next((line.lstrip("# ").strip() for line in lines[1:] if line.startswith("#")), "")})
    return pd.DataFrame(rows)


def coverage_table() -> pd.DataFrame:
    """Every module, script, queue job and test file with the sections of the notebook that name it (the coverage appendix of the notebook)."""
    from notebooks.build_notebook import ALL_CELLS
    from notebooks.notebook_builder import coverage

    mapping, _missing = coverage(ALL_CELLS)
    return pd.DataFrame([{"kind": kind, "path": path, "sections": ", ".join(sections)} for kind, entries in mapping.items() for path, sections in entries.items()])


def claims_table(sections: list[dict]) -> pd.DataFrame:
    """The claims of the innovation summary with the section of the notebook that explains each (section 0.7)."""
    from cypmode.audit.method_report import NOTEBOOK_OF_CLAIM

    titles = {s["number"]: s["title"] for part in sections for s in part["sections"]}
    return pd.DataFrame([{"claim": claim, "section": number, "title": titles[number]} for claim, numbers in NOTEBOOK_OF_CLAIM for number in numbers.split()])


TEST_PATH = "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv"


def column_table(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Every column of the training file and of the blinded test file: its type, how many compounds have a value and the first value (section 2.2)."""
    rows = []
    for name, frame in (("training file", train), ("blinded test file", test)):
        for column in frame.columns:
            filled = frame[column].dropna()
            rows.append({"file": name, "column": column, "type": str(frame[column].dtype), "compounds with a value": len(filled), "of": len(frame), "first value": str(filled.iloc[0])[:60] if len(filled) else ""})
    return pd.DataFrame(rows)


def no_direct_table(train: pd.DataFrame) -> pd.DataFrame:
    """The rows that have a label but no direct arm, with the TDI-arm potency and the label they carry (section 2.4)."""
    frames = []
    for isoform in TDI_ISOFORMS:
        view = training_view(train, isoform)
        keep = (view["population"] == NO_DIRECT).to_numpy()
        frames.append(pd.DataFrame({"isoform": isoform, "t": view["tdi"], "label": view["label"].astype("Int64")})[keep])
    return pd.concat(frames, ignore_index=True)


def numbering_table(train: pd.DataFrame) -> pd.DataFrame:
    """The names of the compounds and, for each other enzyme, 1 where the compound has a measurement on it: what the block finder of section 2.6 reads."""
    return pd.concat([train[["Molecule_Name"]], train[list(OTHER_ISOFORM_COLUMNS)].notna().astype(float).where(lambda f: f > 0)], axis=1)


def similarity_table(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """The highest Tanimoto similarity (Morgan, radius 2, 2048 bits) of each test compound to the training compounds and of each training compound to the other training compounds (section 2.7)."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = lambda smiles: generator.GetFingerprint(Chem.MolFromSmiles(smiles))
    train_fps = [fp(s) for s in train["SMILES"]]
    rows = []
    for i, f in enumerate(train_fps):
        sims = DataStructs.BulkTanimotoSimilarity(f, train_fps)
        sims[i] = 0.0
        rows.append({"set": "training compounds against the other training compounds", "max similarity": max(sims)})
    for smiles in test["SMILES"]:
        rows.append({"set": "test compounds against the training compounds", "max similarity": max(DataStructs.BulkTanimotoSimilarity(fp(smiles), train_fps))})
    return pd.DataFrame(rows)


def keys_table(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """The connectivity keys of the training and of the test compounds, to ask whether a compound of another source is one of them (section 2.8)."""
    from cypmode.challenge.external_activity import connectivity_key

    rows = [{"set": name, "key": connectivity_key(s)} for name, frame in (("training", train), ("test", test)) for s in frame["SMILES"]]
    return pd.DataFrame(rows).dropna()


def sources_table(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """The other public sources and their overlap with the challenge compounds, computed by the code of the notebook's own cell (section 2.8)."""
    from notebooks.parts import part02b

    namespace = {"pd": pd, "ROOT": ROOT, "train": train, "test": test, "show_table": lambda *args, **kwargs: None}
    exec(part02b.SOURCES, namespace)  # noqa: S102
    return pd.DataFrame(namespace["source_rows"])


def board_rows() -> pd.DataFrame:
    """The accuracy, precision and recall of every row of the latest board export of each enzyme, without the names (section 2.10)."""
    frames = []
    for isoform in ("CYP3A4", "CYP2D6"):
        latest = max((ROOT / "data/tdi_track/inputs/leaderboard_snapshots").glob(f"live_{isoform}_*.csv"))
        frames.append(pd.read_csv(latest)[["Accuracy", "Precision", "Recall"]].assign(isoform=isoform))
    return pd.concat(frames, ignore_index=True)


def _motif_names(smiles: str) -> str:
    from cypmode.metrics.motifs import detect_motifs

    return ";".join(sorted(detect_motifs(smiles)))


def panel_groups() -> pd.DataFrame:
    """The panel of yes-or-no inhibitor labels (five enzymes) grouped by label and by the motifs that the compound carries (section 5.3)."""
    from cypmode.data.panel import load_panel

    panel = load_panel()
    panel["motifs"] = panel["motifs"].apply(lambda found: ";".join(sorted(found)))
    return panel.groupby(["isoform", "inhibitor", "motifs"]).size().rename("n").reset_index()


def octant_table() -> pd.DataFrame:
    """The potency and the motif flag of the compounds of the Octant screen that passed the quality control, without the compound that is also in the blinded test set (section 5.4)."""
    from cypmode.data.octant_benchmark import annotate_motifs, load_inhibition_data

    frame = annotate_motifs(load_inhibition_data(ROOT / "data/tdi_track/inputs/octant_cache/inhibition.parquet"))
    excluded = set(json.loads((ROOT / "data/tdi_track/inputs/octant_cache/motif_potency_validation.json").read_text())["excluded_test_compounds"])
    id_column = next(c for c in frame.columns if frame[c].astype(str).str.startswith("OCNT-").any())
    frame = frame[~frame[id_column].isin(excluded)]
    frame = frame[(frame["drc_qc_status"] == "PASS") & frame["CYP3A4_pIC50"].notna()]
    return frame[["CYP3A4_pIC50", "has_motif"]].rename(columns={"CYP3A4_pIC50": "pIC50", "has_motif": "motif"})


def fe_n_table() -> pd.DataFrame:
    """The distance from the heme iron to every ligand nitrogen of the four Boltz-2 predictions of the compounds with a cryo-EM structure (section 5.5), with the motif that the nitrogen matches, if any."""
    summary = json.loads((ROOT / "data/structure_track/outputs/cyp3a4/validation_summary.json").read_text())
    rows = []
    for name, record in summary.items():
        motif_of = {m["atom"]: m["motif"] for m in record["motif_nitrogen_distances"]}
        rows += [{"compound": name, "atom": n["atom"], "distance": n["distance_angstrom"], "motif": motif_of.get(n["atom"], ""), "affinity probability": record.get("affinity_probability_binary"), "confidence": record.get("confidence_score")} for n in record["all_ligand_nitrogen_distances"]]
    return pd.DataFrame(rows)


def pocket_table() -> pd.DataFrame:
    """The burial of the ligand in each of the four predictions: its surface in the complex and alone (section 5.6)."""
    from cypmode.validation.pocket_geometry import ligand_burial_sasa
    from cypmode.validation.structures import load_structure

    rows = []
    for name in json.loads((ROOT / "data/structure_track/outputs/cyp3a4/validation_summary.json").read_text()):
        cif = ROOT / f"data/structure_track/outputs/out/cyp3a4/boltz_results_cyp3a4_{name}/predictions/cyp3a4_{name}/cyp3a4_{name}_model_0.cif"
        rows.append({"compound": name, **{k: v for k, v in ligand_burial_sasa(load_structure(cif)).items() if isinstance(v, (int, float))}})
    return pd.DataFrame(rows)


def flags_table(train: pd.DataFrame) -> pd.DataFrame:
    """For every training compound, the Type II motifs that it carries and whether it has a trialkylamine, next to its two labels (section 5.7)."""
    from cypmode.metrics.amines import has_trialkylamine

    rows = pd.DataFrame({"Molecule_Name": train["Molecule_Name"], "motifs": [_motif_names(s) for s in train["SMILES"]], "amine": [has_trialkylamine(s) for s in train["SMILES"]]})
    return rows.join(train[[label_column(i) for i in TDI_ISOFORMS]])


def _pocket_lines(lines: list[str], ligand_names: tuple[str, ...], radius: float = 10.0) -> list[str]:
    """The lines of a PDB file that keep the ligand, the heme and the protein residues that have an atom within ``radius`` angstrom of the ligand."""
    import numpy as np

    atoms = [(line, np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])) for line in lines if line.startswith(("ATOM", "HETATM"))]
    ligand = np.array([xyz for line, xyz in atoms if line.startswith("HETATM") and line[17:20].strip() in ligand_names])
    keep = {line[21:27] for line, xyz in atoms if line.startswith("ATOM") and np.min(np.linalg.norm(ligand - xyz, axis=1)) < radius}
    return [line for line, _xyz in atoms if line.startswith("HETATM") or line[21:27] in keep]


def _pdb_lines(structure) -> list[str]:
    """The atoms of the first model as fixed-width PDB lines (the chain, the residue name and the numbers are cut to what the format holds)."""
    lines, serial = [], 0
    for chain in structure[0]:
        for residue in chain:
            record = "ATOM  " if residue.id[0] == " " else "HETATM"
            for atom in residue:
                serial += 1
                name = f" {atom.get_name():<3}" if len(atom.get_name()) < 4 and len(atom.element) == 1 else f"{atom.get_name()[:4]:<4}"
                x, y, z = atom.coord
                lines.append(f"{record}{serial % 100000:>5} {name} {residue.resname[:3]:>3} {str(chain.id)[0]}{residue.id[1] % 10000:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{0.0:>6.2f}          {atom.element:>2}")
    return lines


def pocket_pdbs() -> int:
    """The pockets of the four validation predictions and of the twenty submitted structures, as small PDB files for the viewer (sections 11.1 to 11.10, 5.5)."""
    from cypmode.validation.structures import load_structure

    folder = OUT / "structures"
    folder.mkdir(exist_ok=True)
    count = 0
    for name in json.loads((ROOT / "data/structure_track/outputs/cyp3a4/validation_summary.json").read_text()):
        cif = ROOT / f"data/structure_track/outputs/out/cyp3a4/boltz_results_cyp3a4_{name}/predictions/cyp3a4_{name}/cyp3a4_{name}_model_0.cif"
        lines = _pdb_lines(load_structure(cif))
        residues = {line[17:20].strip() for line in lines if line.startswith("HETATM")}
        (folder / f"validation_{name}.pdb").write_text("\n".join(_pocket_lines(lines, tuple(residues - {"HEM", "HOH"}))) + "\nEND\n")
        count += 1
    for pdb in sorted((ROOT / "submissions/structure_track/pdb_files").glob("*.pdb")):
        (folder / f"submitted_{pdb.stem}.pdb").write_text("\n".join(_pocket_lines(pdb.read_text().splitlines(), ("LIG",))) + "\nEND\n")
        count += 1
    vendor = OUT.parent / "vendor"
    vendor.mkdir(exist_ok=True)
    for name in ("3Dmol-min.js", "3Dmol.LICENSE.txt"):
        shutil.copy(ROOT / "docs/card/vendor" / name, vendor / name)
    return count


def audit_checks() -> pd.DataFrame:
    """Every check of the audit report with the section it belongs to (section 12.2)."""
    rows, section = [], ""
    for line in (ROOT / "audit/AUDIT_REPORT.md").read_text().splitlines():
        if line.startswith("## "):
            section = line[3:]
        elif line.startswith("| ") and not line.startswith(("| check", "|---", "| ---")):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 2 and cells[1] in {"PASS", "FAIL", "INFO", "WARN"}:
                rows.append({"section": section, "check": cells[0][:140], "status": cells[1], "detail": (cells[2] if len(cells) > 2 else "")[:140]})
    return pd.DataFrame(rows)


def tests_table() -> pd.DataFrame:
    """The test files of the repository with the number of tests in each (section 12.3)."""
    return pd.DataFrame([{"file": path.name, "tests": len(re.findall(r"^def test_", path.read_text(), re.MULTILINE))} for path in sorted((ROOT / "tests").glob("test_*.py"))])


def rules_table() -> pd.DataFrame:
    """The rules that never change with the mechanism that enforces each, from the cell of section 12.6."""
    from notebooks.parts import part12b

    namespace = {"pd": pd, "ROOT": ROOT, "show_table": lambda *args, **kwargs: None}
    exec(part12b.RULES, namespace)  # noqa: S102
    return namespace["rule_table"]


def limitations_table() -> pd.DataFrame:
    """The limitations of the method report: the identifier, the kind and the text (section 13.2)."""
    rows = []
    for line in (ROOT / "METHOD_REPORT.md").read_text().splitlines():
        m = re.match(r"^- \*\*(L\d+)\*\* \(([^)]*)\): (.*)$", line)
        if m:
            rows.append({"id": m.group(1), "kind": m.group(2), "text": m.group(3)})
    return pd.DataFrame(rows)


def artifacts_table() -> pd.DataFrame:
    """The artifacts that present the results, with the time they were built, how many files each wrote and the fingerprint of the numbers behind them (section 12.5)."""
    records = json.loads((ROOT / "audit/artifacts.json").read_text())
    return pd.DataFrame([{"artifact": name, "built (UTC)": r.get("built_utc", ""), "files written": len(r.get("outputs", {})), "fingerprint": r.get("fingerprint", "")[:12]} for name, r in records.items()])


def posebusters_table() -> pd.DataFrame:
    """The PoseBusters checks of the twenty submitted structures: each compound, whether every check passed and how many failed (section 11.10)."""
    frame = pd.read_csv(ROOT / "data/structure_track/outputs/cyp3a4/posebusters_validity_official20.csv")
    checks = [c for c in frame.columns if c not in {"compound", "all_passed", "model_index"} and frame[c].dtype == bool]
    return pd.DataFrame({"compound": frame["compound"], "all passed": frame["all_passed"], "checks failed": (~frame[checks]).sum(axis=1), "checks made": len(checks)})


def sequence_text() -> str:
    """The sequence of CYP3A4 as the inputs of the structure track carry it (section 11.3)."""
    text = (ROOT / "data/structure_track/inputs/cyp3a4_vardenafil.yaml").read_text()
    return re.search(r"sequence: ([A-Z]+)", text).group(1)


def open_questions() -> pd.DataFrame:
    """The open questions of section 13.3: the question in bold and the paragraph that follows it."""
    text = (ROOT / "notebooks/parts/part13.py").read_text()
    section = text[text.index("## 13.3 Open questions"):]
    section = section[: section.index('"""')]
    return pd.DataFrame([{"question": m.group(1), "what it says": m.group(2).strip().replace("\n", " ")[:600]} for m in re.finditer(r"\*\*(.+?\?)\*\* (.+?)(?=\n\n\*\*|\Z)", section, re.DOTALL)])


def final_stacks() -> pd.DataFrame:
    """The nested cross-validated estimates of the final stacks, as the method report states them (section 13.4)."""
    rows = []
    for line in (ROOT / "METHOD_REPORT.md").read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[0] in {"CYP3A4", "CYP2D6"} and re.fullmatch(r"\d\.\d+", cells[3]):
            rows.append({"isoform": cells[0], "final stack": cells[1], "candidates": int(cells[2]), "AUC at z0 = 1.96": float(cells[3]), "AUC at z0 = 1.0": float(cells[4])})
    return pd.DataFrame(rows)


def submission_files() -> pd.DataFrame:
    """The files that were submitted, with the number of rows, the share of positive calls and the SHA-256 (section 13.4)."""
    manifest = json.loads((ROOT / "submissions/manifest.json").read_text())
    rows = [{"track": "TDI", "file": e["file"], "rows": e["rows"], "CYP3A4 positive rate": e["CYP3A4_positive_rate"], "CYP2D6 positive rate": e["CYP2D6_positive_rate"], "SHA-256": e["sha256"]} for e in manifest["tdi_track"]]
    structure = manifest["structure_track"]
    rows.append({"track": "structure", "file": structure["file"], "rows": structure["members"], "SHA-256": structure["sha256"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    train = pd.read_csv(ROOT / TDI_TRAIN_PATH)
    OUT.mkdir(exist_ok=True)
    points = label_points(train)
    table = compounds(train.dropna(subset=[label_column(i) for i in TDI_ISOFORMS], how="all"))
    points.to_csv(OUT / "label_points.csv", index=False)
    table.to_csv(OUT / "compounds.csv", index=False)
    test = pd.read_csv(ROOT / TEST_PATH)
    column_table(train, test).to_csv(OUT / "columns.csv", index=False)
    no_direct_table(train).to_csv(OUT / "no_direct.csv", index=False)
    numbering_table(train).to_csv(OUT / "numbering.csv", index=False)
    similarity_table(train, test).to_csv(OUT / "similarity.csv", index=False)
    keys_table(train, test).to_csv(OUT / "keys.csv", index=False)
    sources_table(train, test).to_csv(OUT / "sources.csv", index=False)
    board_rows().to_csv(OUT / "board_rows.csv", index=False)
    panel_groups().to_csv(OUT / "panel_groups.csv", index=False)
    octant_table().to_csv(OUT / "octant.csv", index=False)
    fe_n_table().to_csv(OUT / "fe_n.csv", index=False)
    pocket_table().to_csv(OUT / "pocket.csv", index=False)
    flags_table(train).to_csv(OUT / "flags.csv", index=False)
    test[["Molecule_Name", "SMILES"]].to_csv(OUT / "test_names.csv", index=False)
    print("pockets:", pocket_pdbs())
    posebusters_table().to_csv(OUT / "posebusters.csv", index=False)
    open_questions().to_csv(OUT / "open_questions.csv", index=False)
    final_stacks().to_csv(OUT / "final_stacks.csv", index=False)
    submission_files().to_csv(OUT / "submission_files.csv", index=False)
    (OUT / "cyp3a4_sequence.txt").write_text(sequence_text() + "\n")
    audit_checks().to_csv(OUT / "audit_checks.csv", index=False)
    tests_table().to_csv(OUT / "tests.csv", index=False)
    rules_table().to_csv(OUT / "rules.csv", index=False)
    limitations_table().to_csv(OUT / "limitations.csv", index=False)
    artifacts_table().to_csv(OUT / "artifacts.csv", index=False)
    jobs().to_csv(OUT / "jobs.csv", index=False)
    coverage_table().to_csv(OUT / "coverage.csv", index=False)
    claims_table(json.loads((OUT / "sections.json").read_text())["parts"]).to_csv(OUT / "claims.csv", index=False)
    shutil.copytree(ROOT / "explorer/public", OUT / "public", dirs_exist_ok=True)  # the tables of the finished results that job 906 writes for the explorer of the public edition
    print(f"label_points.csv: {len(points)} rows; compounds.csv: {len(table)} rows; the public bundle copied; {sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) / 1e3:.0f} kB in all")

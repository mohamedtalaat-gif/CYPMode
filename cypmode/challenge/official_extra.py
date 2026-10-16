"""The official files of the challenge dataset that the first release of this project did not read, audited before anything uses them.

``openadmet/cyp-challenge-train-test`` holds more than the TDI file: the single-concentration screen (log2 fold change at 49.5 uM, four enzymes), the Emax
file (maximal effect of both arms, with the TDI label of all four enzymes) and the direct-inhibition file. ``openadmet/Octant_CYP_inhibition_reactivity_blog_release``
holds a CYP3A4/CYP2J2 reaction-phenotyping tier. They live in ``data/tdi_track/inputs/official_extra`` (one directory per source, checksums in ``SHA256SUMS``).

``python -m cypmode.challenge.official_extra`` audits them (identifiers, overlap with the blinded test set, agreement with the TDI file) and exits non-zero on a
failed check. The loaders align every table to a list of compound names (the training file followed by the blinded test file), with NaN where a compound has no
reading, so a model can ask for the reading of a training row and never receives one for a test row: no official file contains a blinded compound, which the audit
verifies by name and by structure.

The screen reading of a compound is a measurement of the same compound in the same assay family as the labels. Inside the nested evaluation it is read only for
rows the task may train on (see ``aux_blocks``); the reading of an evaluation row never reaches a model.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("data/tdi_track/inputs")
EXTRA = ROOT / "official_extra"
CYP_DIR = EXTRA / "cyp_challenge"
OCTANT_DIR = EXTRA / "octant_reactivity"
TRAIN_CSV = ROOT / "cyp-challenge-TRAIN_TDI.csv"
TEST_CSV = ROOT / "cyp-challenge-TEST-BLINDED.csv"
ENZYMES = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
SCREEN_ACTIVE_LOG2FC = -0.5  # a compound is screen-active when its log2 fold change is below this and its FDR below SCREEN_ACTIVE_FDR
SCREEN_ACTIVE_FDR = 0.05
SCREEN_FILE = CYP_DIR / "cyp-challenge-single-concentration-TRAIN.csv"
EMAX_FILE = CYP_DIR / "cyp-challenge-TRAIN_Emax.csv"
INHIBITION_FILE = CYP_DIR / "cyp-challenge-TRAIN_inhibition.csv"
REACTIVITY_FILE = OCTANT_DIR / "reactivity.tsv"
PXR_DIR = EXTRA / "pxr_challenge"
PXR_FILES = ("pxr-challenge_TRAIN.csv", "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv", "pxr-challenge_TEST_PHASE_2_UNBLINDED.csv")
PXR_SINGLE_FILE = PXR_DIR / "pxr-challenge_single_concentration_TRAIN.csv"
PXR_SINGLE_CONCENTRATIONS = {"PXR_fc_8uM": 8.3e-6, "PXR_fc_33uM": 3.3e-5}  # the two concentrations that cover almost every library compound (10,747 and 9,523 readings)


def compound_of_batch(batch: str) -> str:
    """``OCNT-0017486-AA-002`` -> ``OCNT-0017486``: the compound number of a batch identifier."""
    return batch.rsplit("-", 2)[0]


def load_screen() -> pd.DataFrame:
    """The single-concentration screen as released: one row per compound and enzyme."""
    return pd.read_csv(SCREEN_FILE)


def screen_arrays(names: list[str]) -> dict[str, np.ndarray]:
    """``{"<enzyme>_screen": log2 fold change aligned to names}`` (NaN where the compound was not screened) and ``"<enzyme>_screen_active"`` (0/1, NaN where unscreened)."""
    table = load_screen()
    out: dict[str, np.ndarray] = {}
    for enzyme in ENZYMES:
        part = table[table.enzyme == enzyme].drop_duplicates("Molecule_Name").set_index("Molecule_Name").reindex(names)
        out[f"{enzyme}_screen"] = part.log2fc_estimate.to_numpy(dtype=float)
        active = (part.log2fc_estimate < SCREEN_ACTIVE_LOG2FC) & (part.log2fc_fdr < SCREEN_ACTIVE_FDR)
        out[f"{enzyme}_screen_active"] = np.where(part.log2fc_estimate.isna(), np.nan, active.to_numpy(dtype=float))
    return out


def emax_arrays(names: list[str]) -> dict[str, np.ndarray]:
    """``{"<enzyme>_emax_direct": ..., "<enzyme>_emax_tdi": ...}``: the maximal effect against the positive control in each arm, aligned to names."""
    table = pd.read_csv(EMAX_FILE).drop_duplicates("Molecule_Name").set_index("Molecule_Name").reindex(names)
    out: dict[str, np.ndarray] = {}
    for enzyme in ENZYMES:
        out[f"{enzyme}_emax_direct"] = table[f"{enzyme}_EmaxVsPosCtrl_direct_inhibition"].to_numpy(dtype=float)
        out[f"{enzyme}_emax_tdi"] = table[f"{enzyme}_EmaxVsPosCtrl_TDI_condition"].to_numpy(dtype=float)
    return out


def load_pxr() -> pd.DataFrame:
    """The PXR-induction challenge's measured compounds (training set and the two unblinded test phases): one row per compound with its pEC50 and Emax against the positive control.

    PXR (the pregnane X receptor) switches on the genes of CYP3A4 and other drug-metabolising enzymes; a PXR activator is typically a lipophilic, drug-sized compound that also meets CYP3A4.
    The compounds are the same OCNT library as the CYP challenge: about 1,900 of them are CYP training compounds and none is a blinded CYP test compound.
    """
    frames = []
    for name in PXR_FILES:
        path = PXR_DIR / name
        if path.exists():
            frame = pd.read_csv(path)
            frames.append(pd.DataFrame({"compound": frame["OCNT Batch"].map(compound_of_batch), "pEC50": frame["pEC50"], "emax": frame["Emax.vs.pos.ctrl_estimate (dimensionless)"]}))
    return pd.concat(frames).groupby("compound", as_index=False).mean() if frames else pd.DataFrame(columns=["compound", "pEC50", "emax"])


def load_pxr_single() -> tuple[pd.DataFrame, list[str]]:
    """The PXR single-concentration screen (log2 fold change of the reporter at two concentrations, library compounds only), one row per compound with the median over its readings, and the
    blinded CYP test compounds that the file also holds.

    The file holds readings of four blinded CYP test compounds. Like the Octant compound they are removed before anything is computed (``exclude_test_compounds``): their values are never read.
    """
    if not PXR_SINGLE_FILE.exists():
        return pd.DataFrame(columns=["compound", *PXR_SINGLE_CONCENTRATIONS]), []
    raw = pd.read_csv(PXR_SINGLE_FILE, usecols=["OCNT_ID", "compound_class", "concentration_M", "log2_fc_estimate"])
    raw = raw[raw.compound_class == "Library"]
    kept, dropped = exclude_test_compounds(raw, "OCNT_ID", batch=False)
    columns = {}
    for name, concentration in PXR_SINGLE_CONCENTRATIONS.items():
        part = kept[np.isclose(kept.concentration_M, concentration, rtol=0.02)]
        columns[name] = part.groupby("OCNT_ID").log2_fc_estimate.median()
    return pd.DataFrame(columns).rename_axis("compound").reset_index(), dropped


def pxr_arrays(names: list[str]) -> dict[str, np.ndarray]:
    """``{"PXR_pEC50", "PXR_emax", "PXR_fc_8uM", "PXR_fc_33uM"}`` aligned to names (NaN where the compound was not measured): the fitted induction and the single-concentration log2 fold changes."""
    table = load_pxr().set_index("compound").reindex(names)
    single = load_pxr_single()[0].set_index("compound").reindex(names)
    return {"PXR_pEC50": table["pEC50"].to_numpy(dtype=float), "PXR_emax": table["emax"].to_numpy(dtype=float), **{k: single[k].to_numpy(dtype=float) for k in PXR_SINGLE_CONCENTRATIONS}}


def aux_arrays(names: list[str]) -> dict[str, np.ndarray]:
    """Every auxiliary reading (screen, Emax and PXR induction) aligned to names."""
    return {**screen_arrays(names), **emax_arrays(names), **pxr_arrays(names)}


def all_names() -> list[str]:
    """Training compounds followed by the blinded test compounds: the row order of the feature store and of the nested context."""
    return list(pd.read_csv(TRAIN_CSV).Molecule_Name) + list(pd.read_csv(TEST_CSV).Molecule_Name)


def test_compounds() -> set[str]:
    return set(pd.read_csv(TEST_CSV).Molecule_Name)


def exclude_test_compounds(frame: pd.DataFrame, id_column: str, batch: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """The guard for any same-laboratory release keyed by compound or batch: rows of a blinded test compound are removed and their identifiers returned.

    A release from the laboratory that measured the challenge can contain a compound that is also in the blinded test set. Its value is a measurement of a test
    compound and is never read by a model or a validation (disclosed in the method report); every other row stays in use.
    """
    ids = frame[id_column].map(compound_of_batch) if batch else frame[id_column]
    drop = ids.isin(test_compounds())
    return frame[~drop].copy(), sorted(set(ids[drop]))


@dataclass
class Check:
    name: str
    ok: bool | None
    detail: str = ""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit() -> list[Check]:
    checks: list[Check] = []

    def add(name: str, ok: bool | None, detail: str = "") -> None:
        checks.append(Check(name, ok, detail))

    train, test = pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)
    screen, emax, inhibition = load_screen(), pd.read_csv(EMAX_FILE), pd.read_csv(INHIBITION_FILE)
    reactivity = pd.read_csv(REACTIVITY_FILE, sep="\t")

    sums = {}
    for line in (EXTRA / "SHA256SUMS").read_text().splitlines():
        digest, _, rel = line.partition("  ")
        sums[rel.strip()] = digest
    bad = [rel for rel, digest in sums.items() if sha256(EXTRA / rel) != digest]
    add("every downloaded file matches its recorded SHA-256", not bad and len(sums) >= 6, f"{len(sums)} files" + (f"; changed: {bad}" if bad else ""))

    names, smiles = set(test.Molecule_Name), set(test.SMILES)
    for label, frame in (("single-concentration", screen), ("Emax", emax), ("direct-inhibition", inhibition)):
        add(f"{label} file holds no blinded test compound", not (names & set(frame.Molecule_Name)) and not (smiles & set(frame.SMILES)), f"{len(set(frame.Molecule_Name))} compounds")
    overlap = sorted(names & {compound_of_batch(b) for b in reactivity.ocnt_batch})
    add("reaction-phenotyping file: blinded test compounds in it (excluded by the guard, never read)", None, f"{len(overlap)}: {', '.join(overlap) or 'none'}")
    kept, dropped = exclude_test_compounds(reactivity, "ocnt_batch")
    add("the guard removes exactly those compounds and nothing else", len(reactivity) - len(kept) == 2 * len(dropped), f"{len(kept)} of {len(reactivity)} rows kept")

    same = list(train.Molecule_Name) == list(emax.Molecule_Name) and list(train.SMILES) == list(emax.SMILES)
    add("Emax file lists the same compounds in the same order as the TDI file", same, f"{len(emax)} rows")
    for enzyme in ("CYP2D6", "CYP3A4"):
        column = f"{enzyme}_is_TDI"
        both = train[column].notna() & emax[column].notna()
        agree = int((train.loc[both, column] == emax.loc[both, column]).sum())
        add(f"{enzyme}: the TDI label of the Emax file equals the TDI file where both exist", agree == int(both.sum()) and train[column].notna().sum() == emax[column].notna().sum(), f"{agree} of {int(both.sum())} agree; labelled rows {int(train[column].notna().sum())} / {int(emax[column].notna().sum())}")
    for enzyme in ENZYMES:
        column = f"{enzyme}_pIC50_direct_inhibition"
        joined = train.set_index("Molecule_Name")[column].reindex(inhibition.Molecule_Name).to_numpy()
        both = ~np.isnan(joined) & inhibition[column].notna().to_numpy()
        same_values = bool(np.allclose(joined[both], inhibition.loc[both, column].to_numpy(), atol=1e-9))
        extra = int((inhibition[column].notna().to_numpy() & np.isnan(joined)).sum())
        add(f"{enzyme}: the direct-inhibition file repeats the direct pIC50 of the TDI file and adds none", same_values and extra == 0, f"{int(both.sum())} values identical, {extra} new")

    per = screen.groupby("enzyme").Molecule_Name.nunique()
    add("single-concentration file: four enzymes, one reading each for the same compounds", set(per.index) == set(ENZYMES) and per.nunique() == 1 and screen.groupby(["Molecule_Name", "enzyme"]).size().max() == 1, f"{int(per.iloc[0])} compounds per enzyme at {screen.concentration_M.iloc[0] * 1e6:.1f} uM")
    shared = screen.drop_duplicates("Molecule_Name").set_index("Molecule_Name").SMILES.reindex(train.Molecule_Name).dropna()
    add("screened compounds carry the same structure as in the TDI file", bool((shared.to_numpy() == train.set_index("Molecule_Name").SMILES.reindex(shared.index).to_numpy()).all()), f"{len(shared)} compared")
    unscreened = sorted(set(train.Molecule_Name) - set(screen.Molecule_Name))
    add("training compounds without a screen reading", None, f"{len(unscreened)} of {len(train)}")
    arrays = screen_arrays(list(train.Molecule_Name))
    for enzyme in ("CYP2D6",):
        has = train[f"{enzyme}_pIC50_direct_inhibition"].notna().to_numpy()
        active = arrays[f"{enzyme}_screen_active"]
        add(f"{enzyme}: every compound with a direct curve is screen-active", bool(np.nanmin(active[has]) == 1.0), f"{int(np.nansum(active[has]))} of {int(has.sum())}; screen-active without a curve {int(np.nansum(active[~has]))} of {int(np.sum(~has & ~np.isnan(active)))}")
    octant_file = ROOT / "octant_cache" / "inhibition.parquet"
    if octant_file.exists():
        from rdkit import Chem, RDLogger
        from rdkit.Chem import inchi

        RDLogger.DisableLog("rdApp.*")

        def connectivity(smiles: str) -> str | None:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            largest = max(Chem.GetMolFrags(mol, asMols=True), key=lambda m: m.GetNumHeavyAtoms())
            key = inchi.MolToInchiKey(largest)
            return key.split("-")[0] if key else None

        octant = pd.read_parquet(octant_file)
        test_keys = {connectivity(s) for s in test.SMILES}
        by_structure = sorted(compound_of_batch(b) for b, s in zip(octant.ocnt_batch, octant.standardized_smiles, strict=True) if connectivity(s) in test_keys)
        by_identifier = sorted(names & {compound_of_batch(b) for b in octant.ocnt_batch})
        add("Octant release: blinded test compounds found by structure (connectivity of the InChIKey) are exactly those found by identifier, so the guard misses none", by_structure == by_identifier, f"by structure {by_structure}, by identifier {by_identifier}")
    pxr = load_pxr()
    add("PXR files hold no blinded CYP test compound", not (set(pxr.compound) & names), f"{len(pxr)} PXR compounds, {len(set(pxr.compound) & set(train.Molecule_Name))} are CYP training compounds")
    single, held_out = load_pxr_single()
    if PXR_SINGLE_FILE.exists():
        add("PXR single-concentration file: the blinded CYP test compounds it holds are excluded before any value is read", not (set(single.compound) & names) and len(held_out) > 0,
            f"{len(held_out)} test compounds excluded ({', '.join(held_out)}); {len(single)} compounds kept, {len(set(single.compound) & set(train.Molecule_Name))} of them CYP training compounds")
    heads = emax_arrays(list(train.Molecule_Name))
    add("Emax arrays align to the TDI file", all(len(v) == len(train) for v in heads.values()), f"{len(heads)} columns")

    from cypmode.challenge.tdi_targets import NO_DIRECT, training_view

    for enzyme in ("CYP3A4", "CYP2D6"):
        view = training_view(train, enzyme)
        rows = (view.population == NO_DIRECT).to_numpy()
        read = ~np.isnan(arrays[f"{enzyme}_screen"]) & rows & view.tdi.notna().to_numpy()
        add(f"{enzyme}: rows labelled by convention without a direct curve that carry a screen reading (could their label be recovered from it?)", None, f"{int(read.sum())} of {int(rows.sum())}: " + ("too few to recover labels" if read.sum() < 50 else "enough rows to try"))
    from sklearn.metrics import roc_auc_score

    batch_rows = reactivity.assign(Molecule_Name=reactivity.ocnt_batch.map(compound_of_batch))
    substrate = batch_rows[batch_rows.enzyme == "CYP3A4"].groupby("Molecule_Name").pct_remaining.mean()
    labelled = train.set_index("Molecule_Name").CYP3A4_is_TDI.dropna()
    joined = pd.concat([substrate.rename("remaining"), labelled], axis=1, join="inner")
    if joined.CYP3A4_is_TDI.nunique() == 2:
        add("reaction phenotyping (CYP3A4 depletion) as a predictor of the CYP3A4 TDI label", None, f"{len(joined)} compounds, {int(joined.CYP3A4_is_TDI.sum())} TDI positives, AUC {roc_auc_score(joined.CYP3A4_is_TDI.astype(int), 100 - joined.remaining):.2f}")
    return checks


def main() -> int:
    checks = audit()
    for c in checks:
        status = "INFO" if c.ok is None else "PASS" if c.ok else "FAIL"
        print(f"{status:4}  {c.name}  [{c.detail}]")
    failed = [c for c in checks if c.ok is False]
    print(f"\n{sum(c.ok is True for c in checks)} passed, {len(failed)} failed, {sum(c.ok is None for c in checks)} reported")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

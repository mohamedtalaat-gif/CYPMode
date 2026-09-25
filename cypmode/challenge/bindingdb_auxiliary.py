"""Auxiliary direct-CYP-binding signal from BindingDB (bindingdb.org) --
targeted REST pulls for CYP3A4 (UniProt P08684) and CYP2D6 (UniProt
P10635) specifically, not a bulk database download (same narrowly-scoped
pattern as cypmode.challenge.pubchem_auxiliary's PubChem AID 1851
integration: BindingDB's own full dumps run 240MB-3GB; the
getLigandsByUniprots endpoint returns only one isoform's own real data --
confirmed directly: ~3MB/~2MB for CYP3A4/CYP2D6 respectively, no bulk
download needed).

Motivation and scope: same as pubchem_auxiliary -- a model trained on a
broader, literature-curated compound set than this project's own curated
TDI training data, applied to our own compounds as an auxiliary feature.
BindingDB gives real Ki/IC50/EC50 binding-affinity measurements (not
qHTS screening activity calls like AID 1851), literature-curated across
many independent papers -- a different, complementary source, not a
duplicate of AID 1851's signal. Like AID 1851, this has no TDI labels of
its own (BindingDB records general binding/inhibition potency, not a
preincubation-vs-direct TDI comparison) -- the feature this module
produces is "predicted general CYP-isozyme binding activity from a broad
external literature source," a plausible correlate of TDI risk, not a
TDI signal itself. No fold-internal OOF construction is needed for the
same reason given in pubchem_auxiliary's module docstring: this is a
fully external dataset, safe to fit once and apply identically to every
TDI compound.

Real data quirk, confirmed directly: BindingDB is curated across many
independent papers, so the same compound can appear with several
different reported affinity values (up to 64 for a single SMILES in this
project's own real CYP2D6 pull) -- these are independent real
measurements, not same-assay replicate rows like AID 1851's duplicates,
so they're combined via median affinity rather than deduplicated by
dropping rows.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMClassifier
from rdkit import Chem
from rdkit.Chem import AllChem

from cypmode.challenge.structure_standardization import standardized_parent_smiles
from cypmode.challenge.tdi_baseline import compute_descriptors

BINDINGDB_ISOFORM_UNIPROTS = {"CYP3A4": "P08684", "CYP2D6": "P10635"}
BINDINGDB_RAW_DIR = Path("data/bindingdb_cache")
ACTIVE_THRESHOLD_UM = 10.0


def fetch_bindingdb_isoform(
    isoform: str, uniprot: str, out_dir: str | Path = BINDINGDB_RAW_DIR, timeout: int = 90
) -> Path:
    """Real, targeted REST pull for one isoform's own binding data --
    cached to disk, not re-fetched if already present.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{isoform}_{uniprot}.json"
    if not path.exists():
        url = f"https://www.bindingdb.org/rest/getLigandsByUniprots?uniprot={uniprot}&response=application/json"
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        path.write_text(resp.text)
    return path


def _parse_affinity_nM(value: str) -> float | None:
    """BindingDB's real convention for a censored measurement is a leading
    >, <, or ~ on the numeric string -- stripped here rather than treated
    as part of the number. Returns None (not raised) for anything else
    unparseable, since a handful of malformed values shouldn't crash a
    17k-row real-world literature-curated file.
    """
    value = value.strip().lstrip("><~=")
    try:
        return float(value)
    except ValueError:
        return None


def load_bindingdb_isoform(path: str | Path) -> pd.DataFrame:
    """Parses one isoform's real getLigandsByUniprots JSON into one row per
    unique SMILES, with the median reported affinity (uM) across every
    real measurement found for that compound -- see module docstring for
    why median-combining, not row-deduplication, is the right call here.

    Real data quirk, confirmed directly: a small fraction of BindingDB's
    own "smile" strings (7 of 8,679 for a real CYP3A4 pull) carry a
    trailing extended-annotation suffix (e.g. "...|r,THB:11:9:3:5.6|" --
    BindingDB/ChemAxon torsion/stereo metadata for complex macrocycles)
    that plain RDKit's Chem.MolFromSmiles doesn't parse, returning None --
    not raising, so this would otherwise crash downstream (confirmed: a
    real run crashed inside Morgan fingerprinting on exactly this). Rows
    that fail to parse are dropped here rather than passed through.
    """
    data = json.loads(Path(path).read_text())
    affinities = data["getLindsByUniprotsResponse"]["affinities"]
    rows = []
    for a in affinities:
        nM = _parse_affinity_nM(a["affinity"])
        if nM is None:
            continue
        if Chem.MolFromSmiles(a["smile"]) is None:
            continue
        rows.append({"SMILES": a["smile"], "affinity_uM": nM / 1000.0})
    if not rows:
        return pd.DataFrame(columns=["SMILES", "affinity_uM"])
    df = pd.DataFrame(rows)
    return df.groupby("SMILES", as_index=False)["affinity_uM"].median()


def exclude_exact_overlap(bindingdb_df: pd.DataFrame, challenge_smiles: pd.Series) -> pd.DataFrame:
    """Same real safety check as pubchem_auxiliary's -- standardized-parent
    matching, not plain canonical SMILES (see structure_standardization.py's
    module docstring for why: a salt form and its free base canonicalize
    differently but are the same real compound)."""
    challenge_canonical = set(challenge_smiles.map(standardized_parent_smiles).dropna())
    bindingdb_canonical = bindingdb_df["SMILES"].map(standardized_parent_smiles)
    return bindingdb_df[~bindingdb_canonical.isin(challenge_canonical)].reset_index(drop=True)


def _morgan_fp_bits(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


def _featurize(df: pd.DataFrame) -> np.ndarray:
    described = compute_descriptors(df)
    descriptors = np.stack(described["descriptors"])
    morgan = np.stack([_morgan_fp_bits(s) for s in df["SMILES"]])
    return np.hstack([descriptors, morgan])


def fit_auxiliary_model(
    bindingdb_df: pd.DataFrame, active_threshold_uM: float = ACTIVE_THRESHOLD_UM, random_state: int = 42
) -> LGBMClassifier:
    """Fits one LightGBM classifier on ALL of bindingdb_df's real median
    potency values, thresholded to active/inactive at active_threshold_uM
    (10 uM by default -- a standard, defensible cutoff for "pharmacologically
    relevant" CYP binding, not a tuned choice). No fold-internal OOF
    treatment needed -- see module docstring.
    """
    y = (bindingdb_df["affinity_uM"] <= active_threshold_uM).astype(int).to_numpy()
    X = _featurize(bindingdb_df)
    model = LGBMClassifier(verbose=-1, random_state=random_state)
    model.fit(X, y)
    return model


def predict_auxiliary_activity(model: LGBMClassifier, target_df: pd.DataFrame) -> np.ndarray:
    X = _featurize(target_df)
    return model.predict_proba(X)[:, 1]


def _main() -> None:
    from sklearn.metrics import matthews_corrcoef
    from sklearn.model_selection import StratifiedKFold

    from cypmode.challenge.tdi_ablation import add_ablation_features
    from cypmode.challenge.tdi_baseline import TDI_ISOFORMS, load_tdi_training_data

    tdi_df = add_ablation_features(compute_descriptors(load_tdi_training_data()))

    all_results = []
    for cyp in TDI_ISOFORMS:
        uniprot = BINDINGDB_ISOFORM_UNIPROTS[cyp]
        raw_path = fetch_bindingdb_isoform(cyp, uniprot)
        bdb_df = load_bindingdb_isoform(raw_path)
        n_before = len(bdb_df)
        bdb_df = exclude_exact_overlap(bdb_df, tdi_df["SMILES"])
        print(f"{cyp} (BindingDB): {n_before} compounds, {n_before - len(bdb_df)} exact-SMILES overlap excluded")

        aux_model = fit_auxiliary_model(bdb_df)
        aux_proba = predict_auxiliary_activity(aux_model, tdi_df)
        tdi_df[f"bdb_{cyp}_activity_proba"] = aux_proba

        cyp_df = tdi_df.dropna(subset=[f"{cyp}_is_TDI"]).reset_index(drop=True)
        y = cyp_df[f"{cyp}_is_TDI"].astype(int).to_numpy()
        m3_features = np.stack(cyp_df["m3_features"])
        m5_features = np.hstack([m3_features, cyp_df[[f"bdb_{cyp}_activity_proba"]].to_numpy()])

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rows = []
        for fold, (train_idx, val_idx) in enumerate(kfold.split(m3_features, y)):
            m3_model = LGBMClassifier(verbose=-1, random_state=42)
            m3_model.fit(m3_features[train_idx], y[train_idx])
            m3_pred = m3_model.predict(m3_features[val_idx])

            m5_model = LGBMClassifier(verbose=-1, random_state=42)
            m5_model.fit(m5_features[train_idx], y[train_idx])
            m5_pred = m5_model.predict(m5_features[val_idx])

            rows.append(
                {
                    "isoform": cyp,
                    "fold": fold,
                    "n_val": len(val_idx),
                    "mcc_m3": matthews_corrcoef(y[val_idx], m3_pred),
                    "mcc_m3_plus_bindingdb": matthews_corrcoef(y[val_idx], m5_pred),
                }
            )
        results = pd.DataFrame(rows)
        all_results.append(results)
        print(f"{cyp}: mean MCC M3={results['mcc_m3'].mean():.4f}, M3+BindingDB={results['mcc_m3_plus_bindingdb'].mean():.4f}")

    combined = pd.concat(all_results, ignore_index=True)
    out_path = Path("data/cyp_challenge/tdi_track/bindingdb_auxiliary_ablation.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    _main()

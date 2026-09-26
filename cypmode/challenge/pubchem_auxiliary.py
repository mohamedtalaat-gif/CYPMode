"""Auxiliary direct-CYP-activity signal from PubChem AID 1851 ("Cytochrome
panel assay with activity outcomes", NCATS) -- a real, verified qHTS assay
of 17,143 diverse NCATS screening-library compounds against five CYP
isozymes (1A2, 2C9, 2C19, 2D6, 3A4), single-incubation inhibitor/substrate
activity calls (no preincubation comparison, so it has no TDI labels of
its own -- see module docstring's Note on scope below).

Motivation: this project's own TDI training set comes from a curated,
medicinal-chemistry-series-heavy compound pool (see
CYPMode_DATA_AUDIT.md's scaffold analysis). AID 1851's compounds are
drawn from a broader, more chemically diverse screening library, largely
disjoint from that pool. A model trained on AID 1851 to predict general
CYP2D6/CYP3A4 direct-activity from structure, then applied to this
project's own TDI compounds, gives an auxiliary feature informed by a much
larger and more diverse chemical space than our own training data alone
covers -- not a replacement for direct evidence, a broader prior.

Note on scope: AID 1851 measures direct inhibitor/substrate activity in a
single-incubation assay, NOT time-dependent (mechanism-based) inhibition
specifically -- there is no preincubation-vs-direct comparison in this
assay, so it cannot and does not provide TDI labels. The feature this
module produces is "predicted general CYP-isozyme activity from a broad
external assay," a plausible correlate of TDI risk (potent inhibitors are
more likely, not certain, to also be mechanism-based ones), not a TDI
signal itself.

No fold-internal OOF construction is needed here, unlike
cypmode.challenge.oof_bridge's direct-inhibition bridge: that module's
training data and this project's own TDI-labeled compounds substantially
overlap (same underlying challenge dataset, multiple label types per
compound), so predicting one from a model touching the other needs OOF
treatment to avoid leaking correlated information. AID 1851 is a fully
external dataset with (verified below) essentially no compound overlap
with this project's own TDI compounds -- fitting one auxiliary model on
all of it and applying it identically to every TDI compound (train and
blind test alike) is the same kind of fixed, external prior information
as an RDKit descriptor, not a leak.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from rdkit import Chem
from rdkit.Chem import AllChem

from cypmode.challenge.structure_standardization import standardized_parent_smiles
from cypmode.challenge.tdi_baseline import compute_descriptors

AID1851_RAW_PATH = Path("data/pubchem_cache/aid1851_raw.csv")
AID1851_ISOFORMS = {"CYP1A2": "p450-cyp1a2", "CYP2C9": "p450-cyp2c9", "CYP2C19": "p450-cyp2c19", "CYP2D6": "p450-cyp2d6", "CYP3A4": "p450-cyp3a4"}


def load_aid1851(path: str | Path = AID1851_RAW_PATH) -> pd.DataFrame:
    """Parses the raw long-format PubChem bioactivity CSV (one row per
    compound-isozyme pair, plus 4 leading PubChem metadata rows) into one
    row per compound, with `{isoform}_active` (True/False, NaN for
    Inconclusive or a missing panel result) and `{isoform}_potency_uM`
    columns per isoform in AID1851_ISOFORMS.

    Real data quirk, confirmed directly: a given (CID, panel) pair can have
    more than one row (up to 7 seen for a single compound-isozyme pair --
    replicate measurements), affecting 2,315 (CID, panel) pairs in the raw
    file. Merging against those duplicates without collapsing them first
    multiplies row count per merge, and since 5 isoform merges run
    sequentially, the effect compounds -- confirmed directly: an earlier,
    unfixed version of this function inflated 16,560 real unique compounds
    into 86,721 rows this way, silently over-weighting whichever compounds
    happened to have replicate rows when fitting the auxiliary model.
    `drop_duplicates(subset=["PUBCHEM_CID"])` per panel, before merging,
    is what prevents that -- keeping first-seen is an arbitrary tie-break
    among replicates, not a claim that it's the "right" one, but it fixes
    the row-multiplication bug either way.
    """
    raw = pd.read_csv(path, skiprows=[1, 2, 3, 4], low_memory=False)
    raw = raw.dropna(subset=["PUBCHEM_EXT_DATASOURCE_SMILES"]).rename(columns={"PUBCHEM_EXT_DATASOURCE_SMILES": "SMILES"})

    out = raw[["PUBCHEM_CID", "SMILES"]].drop_duplicates(subset=["PUBCHEM_CID"]).reset_index(drop=True)
    for cyp, panel_name in AID1851_ISOFORMS.items():
        panel = raw[raw["Panel Name"] == panel_name][["PUBCHEM_CID", "PUBCHEM_ACTIVITY_OUTCOME", "Potency"]]
        panel = panel.drop_duplicates(subset=["PUBCHEM_CID"], keep="first")
        active = panel["PUBCHEM_ACTIVITY_OUTCOME"].map({"Active": True, "Inactive": False})  # Inconclusive -> NaN
        panel = panel.assign(**{f"{cyp}_active": active, f"{cyp}_potency_uM": panel["Potency"]})
        out = out.merge(panel[["PUBCHEM_CID", f"{cyp}_active", f"{cyp}_potency_uM"]], on="PUBCHEM_CID", how="left")
    return out


SINGLE_ISOFORM_AID_PATHS = {
    "CYP2D6": Path("data/pubchem_cache/aid891_cyp2d6_raw.csv"),  # AID 891 -- CYP2D6's own individual qHTS assay
    "CYP3A4": Path("data/pubchem_cache/aid884_cyp3a4_raw.csv"),  # AID 884 -- CYP3A4's own individual qHTS assay
}


def load_single_isoform_aid(path: str | Path, isoform: str) -> pd.DataFrame:
    """Parses a real, single-isoform NCATS qHTS assay (AID 891 for CYP2D6,
    AID 884 for CYP3A4 -- the individual per-isoform screens the later
    AID 1851 panel assay draws from, confirmed via a real compound-list
    overlap check: only 2,193/9,385 AID891 compounds and 10/13,076 AID884
    compounds overlap with AID1851 -- 77% and 99.9% genuinely new
    compounds respectively, not a duplicate of that data). Same real
    replicate-row quirk as AID 1851 (verified directly: 935 duplicate CIDs
    in AID891, 1,079 in AID884) -- collapsed the same way, first-seen per
    CID, before this becomes a feature.

    Returns one row per compound with a `{isoform}_active` column, the
    same schema fit_auxiliary_model/predict_auxiliary_activity already
    expect -- reused unchanged, not reimplemented for this data source.
    """
    raw = pd.read_csv(path, skiprows=[1, 2, 3, 4], low_memory=False)
    raw = raw.dropna(subset=["PUBCHEM_EXT_DATASOURCE_SMILES"]).rename(columns={"PUBCHEM_EXT_DATASOURCE_SMILES": "SMILES"})
    raw = raw.drop_duplicates(subset=["PUBCHEM_CID"], keep="first")

    active = raw["PUBCHEM_ACTIVITY_OUTCOME"].map({"Active": True, "Inactive": False})  # Inconclusive -> NaN
    out = raw[["PUBCHEM_CID", "SMILES"]].copy()
    out[f"{isoform}_active"] = active
    return out.reset_index(drop=True)


def exclude_exact_overlap(aid_df: pd.DataFrame, challenge_smiles: pd.Series) -> pd.DataFrame:
    """Drops any AID 1851 compound whose standardized parent structure
    matches a compound already in this project's own challenge data -- a
    real safety check, not a formality, even though the two datasets are
    expected to be near-disjoint (verified in _main). Matches on the
    standardized parent (see structure_standardization.py), not plain
    canonical SMILES: a salt form and its free base canonicalize
    differently but are the same real compound, so plain-canonical
    matching would silently miss that overlap.
    """
    challenge_canonical = set(challenge_smiles.map(standardized_parent_smiles).dropna())
    aid_canonical = aid_df["SMILES"].map(standardized_parent_smiles)
    return aid_df[~aid_canonical.isin(challenge_canonical)].reset_index(drop=True)


def _morgan_fp_bits(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


def _featurize(df: pd.DataFrame) -> np.ndarray:
    """RDKit descriptors + Morgan FP, concatenated -- the same feature
    composition M3 uses (minus the TDI-specific motif flag, not meaningful
    as a generic direct-activity predictor feature)."""
    described = compute_descriptors(df)
    descriptors = np.stack(described["descriptors"])
    morgan = np.stack([_morgan_fp_bits(s) for s in df["SMILES"]])
    return np.hstack([descriptors, morgan])


def fit_auxiliary_model(aid_df: pd.DataFrame, isoform: str, random_state: int = 42) -> LGBMClassifier:
    """Fits one LightGBM classifier on ALL of aid_df's real, non-Inconclusive
    labels for `isoform` -- see module docstring for why no fold-internal
    OOF treatment is needed for this external dataset.
    """
    labeled = aid_df.dropna(subset=[f"{isoform}_active"]).reset_index(drop=True)
    X = _featurize(labeled)
    y = labeled[f"{isoform}_active"].astype(int).to_numpy()
    model = LGBMClassifier(verbose=-1, random_state=random_state)
    model.fit(X, y)
    return model


def predict_auxiliary_activity(model: LGBMClassifier, target_df: pd.DataFrame) -> np.ndarray:
    """Applies a fitted auxiliary model to this project's own compounds
    (target_df must have a SMILES column), returning P(active) per row."""
    X = _featurize(target_df)
    return model.predict_proba(X)[:, 1]


def _run_ablation_for_isoform(tdi_df: pd.DataFrame, aid_df: pd.DataFrame, cyp: str, aux_col: str) -> pd.DataFrame:
    """Shared M3-vs-M3+aux 5-fold CV comparison, used for both AID1851 and
    the single-isoform AIDs (891/884) -- same protocol, different source
    data.
    """
    from sklearn.metrics import matthews_corrcoef
    from sklearn.model_selection import StratifiedKFold

    aux_model = fit_auxiliary_model(aid_df, cyp)
    tdi_df[aux_col] = predict_auxiliary_activity(aux_model, tdi_df)

    cyp_df = tdi_df.dropna(subset=[f"{cyp}_is_TDI"]).reset_index(drop=True)
    y = cyp_df[f"{cyp}_is_TDI"].astype(int).to_numpy()
    m3_features = np.stack(cyp_df["m3_features"])
    m5_features = np.hstack([m3_features, cyp_df[[aux_col]].to_numpy()])

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
                "mcc_m3_plus_aux": matthews_corrcoef(y[val_idx], m5_pred),
            }
        )
    return pd.DataFrame(rows)


def _main() -> None:
    from cypmode.challenge.tdi_ablation import add_ablation_features
    from cypmode.challenge.tdi_baseline import TDI_ISOFORMS, load_tdi_training_data

    tdi_df = add_ablation_features(compute_descriptors(load_tdi_training_data()))
    aid_df = load_aid1851()

    n_before = len(aid_df)
    aid_df = exclude_exact_overlap(aid_df, tdi_df["SMILES"])
    print(f"AID1851: {n_before} compounds, {n_before - len(aid_df)} exact-SMILES overlap with our TDI data excluded")

    all_results = []
    for cyp in TDI_ISOFORMS:
        results = _run_ablation_for_isoform(tdi_df, aid_df, cyp, f"aux_{cyp}_activity_proba")
        all_results.append(results)
        print(f"{cyp}: mean MCC M3={results['mcc_m3'].mean():.4f}, M3+aux={results['mcc_m3_plus_aux'].mean():.4f}")

    combined = pd.concat(all_results, ignore_index=True)
    out_path = Path("data/cyp_challenge/tdi_track/aid1851_auxiliary_ablation.csv")
    combined.to_csv(out_path, index=False)
    print(f"wrote {out_path}")

    print("\n-- single-isoform assays (AID891/AID884), the individual screens AID1851's panel draws from --")
    single_results = []
    for cyp in TDI_ISOFORMS:
        single_aid_df = load_single_isoform_aid(SINGLE_ISOFORM_AID_PATHS[cyp], cyp)
        n_before = len(single_aid_df)
        single_aid_df = exclude_exact_overlap(single_aid_df, tdi_df["SMILES"])
        print(f"{cyp} single-isoform AID: {n_before} compounds, {n_before - len(single_aid_df)} exact-SMILES overlap excluded")

        results = _run_ablation_for_isoform(tdi_df, single_aid_df, cyp, f"aux_single_{cyp}_activity_proba")
        single_results.append(results)
        print(f"{cyp}: mean MCC M3={results['mcc_m3'].mean():.4f}, M3+single-isoform-aux={results['mcc_m3_plus_aux'].mean():.4f}")

    single_combined = pd.concat(single_results, ignore_index=True)
    single_out_path = Path("data/cyp_challenge/tdi_track/aid891_884_auxiliary_ablation.csv")
    single_combined.to_csv(single_out_path, index=False)
    print(f"wrote {single_out_path}")


if __name__ == "__main__":
    _main()

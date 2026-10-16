"""Continuous readings of the NCATS qHTS cytochrome panel (PubChem AID 1851) as a label-free feature block.

``external_activity`` already feeds the stack five inhibitor probabilities per compound, trained on the binary active/inactive calls of the same panel (the TDC version). A binary call throws
away how strongly a compound inhibited and how far from active an "inactive" one was. AID 1851 keeps it: every compound-isoform pair has the maximal response of its concentration-response
curve (signed percent; -100 is complete inhibition, 0 none), and the compounds with a fitted curve have an AC50. This module trains, per isoform, a regressor of the response and a regressor of
the potency on the external compounds only, and predicts both for every challenge compound, so the block has ten columns: ``resp_<iso>`` (the response, clipped to [-20, 120] % of inhibition)
and ``pac50_<iso>`` (-log10 AC50 in molar; inactive compounds enter the potency regressor at the floor 4.0, inconclusive compounds without a fit are left out of it).

As for ``external_activity``, no challenge label is involved and every external compound that is also a challenge training or test compound (matched on the connectivity layer of the InChIKey) is
dropped before fitting, so the block scores the challenge compounds out of sample by construction and needs no nesting. Replicate rows of one compound-isoform pair are collapsed to their median.

A second panel is handled the same way (``ext_tox``): the Tox21 CYP screen (NCATS and the EPA, P450-Glo luciferase qHTS of about 7,700 compounds, mostly environmental chemicals and drugs, public domain; PubChem AIDs
1671196 CYP2D6, 1671197 CYP2C19, 1671198 CYP2C9, 1671199 CYP1A2, 1671201 CYP3A4). The compounds are far from the challenge library (the median nearest-neighbour similarity of a challenge compound to Tox21 is 0.30
and 4 to 7% reach 0.5), so what it can add is a population-level picture of weak inhibition, the range that the challenge's own pool lacks, not read-across from neighbours. The assay reads out light from a luciferase,
so a compound that inhibits the luciferase itself looks like a CYP inhibitor: the counter-screen (AID 1224835, firefly luciferase inhibitors) removes those readings before anything is fitted.

Two processes are not needed: the extraction runs under the project environment (LightGBM, RDKit) and writes ``caches/ext_continuous.npz`` and ``caches/ext_tox21.npz`` keyed by canonical SMILES; the evaluation only reads them.

  download:  .venv/bin/python -m cypmode.challenge.external_continuous --download-tox21 [--raw-dir DIR]   (6 PubChem tables of about 27 MB; the compact table of 1 to 2 MB is kept in the repository)
  extract:   .venv/bin/python -m cypmode.challenge.external_continuous --extract
  evaluate:  .venv/bin/python -m cypmode.challenge.aux_experiments --experiment ext_cont --isoform CYP3A4 --repeats 3 --run
GATE (addition_experiment): corrected p < 0.05 at both z0, alone (added to the sets with the binary external block; each of the two sources, and both together) and inside the assembled stack, per isoform.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from cypmode.challenge.external_activity import CHALLENGE_FILES, connectivity_key

RAW = Path("data/tdi_track/inputs/pubchem_cache/aid1851_raw.csv")
CACHE = Path("data/tdi_track/outputs/caches/ext_continuous.npz")
PANELS = {"cyp1a2": "p450-cyp1a2", "cyp2c9": "p450-cyp2c9", "cyp2c19": "p450-cyp2c19", "cyp2d6": "p450-cyp2d6", "cyp3a4": "p450-cyp3a4"}
RESPONSE_RANGE = (-20.0, 120.0)  # percent of inhibition kept for the regressor; the raw maxima run from -110 to +195 and a few outliers reach +1,330
POTENCY_FLOOR = 4.0  # pAC50 given to an inactive compound: the highest tested concentration of the panel is about 57 uM
COLUMNS = tuple(f"{kind}_{iso}" for kind in ("resp", "pac50") for iso in PANELS)
TOX_DIR = Path("data/tdi_track/inputs/tox21_cache")
TOX_COMPACT = TOX_DIR / "tox21_cyp_compact.csv.gz"
TOX_CACHE = Path("data/tdi_track/outputs/caches/ext_tox21.npz")
TOX_AIDS = {"cyp2d6": 1671196, "cyp2c19": 1671197, "cyp2c9": 1671198, "cyp1a2": 1671199, "cyp3a4": 1671201}
TOX_COUNTER_AID = 1224835  # firefly luciferase inhibitors
TOX_COLUMNS = tuple(f"{kind}_{iso}" for kind in ("tresp", "tpac50") for iso in TOX_AIDS)
LGBM = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.3, "verbose": -1, "n_jobs": 4, "random_state": 0}


def external_table() -> pd.DataFrame:
    """One row per compound with the median response and potency of every isoform (NaN where the panel gave no usable reading); challenge compounds removed."""
    raw = pd.read_csv(RAW, skiprows=[1, 2, 3, 4], low_memory=False).dropna(subset=["PUBCHEM_EXT_DATASOURCE_SMILES"])
    raw = raw.assign(
        SMILES=raw["PUBCHEM_EXT_DATASOURCE_SMILES"], response=-pd.to_numeric(raw["Max_Response"], errors="coerce").clip(*[-v for v in RESPONSE_RANGE[::-1]]),
        pac50=-pd.to_numeric(raw["Fit_LogAC50"], errors="coerce"),
    )
    raw.loc[raw["PUBCHEM_ACTIVITY_OUTCOME"] == "Inactive", "pac50"] = POTENCY_FLOOR
    raw.loc[(raw["PUBCHEM_ACTIVITY_OUTCOME"] == "Inconclusive") & raw["pac50"].isna(), "pac50"] = np.nan
    out = raw[["PUBCHEM_CID", "SMILES"]].drop_duplicates(subset=["PUBCHEM_CID"]).reset_index(drop=True)
    for iso, panel in PANELS.items():
        part = raw[raw["Panel Name"] == panel].groupby("PUBCHEM_CID").agg(response=("response", "median"), pac50=("pac50", "median"))
        out = out.merge(part.rename(columns={"response": f"resp_{iso}", "pac50": f"pac50_{iso}"}), left_on="PUBCHEM_CID", right_index=True, how="left")
    exclude: set[str] = set()
    for path in CHALLENGE_FILES:
        exclude |= {k for k in (connectivity_key(s) for s in pd.read_csv(path)["SMILES"]) if k}
    keys = [connectivity_key(s) for s in out["SMILES"]]
    keep = [i for i, k in enumerate(keys) if k is not None and k not in exclude]
    return out.iloc[keep].reset_index(drop=True)


def download_tox21(raw_dir: Path) -> list[Path]:
    """The six PubChem tables of the Tox21 CYP screen and of its luciferase counter-screen, saved into ``raw_dir`` (outside the repository)."""
    import urllib.request

    raw_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for aid in (*TOX_AIDS.values(), TOX_COUNTER_AID):
        path = raw_dir / f"a{aid}.csv"
        if not path.exists():
            urllib.request.urlretrieve(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/CSV", path)
        out.append(path)
    return out


def _tox_table(path: Path) -> pd.DataFrame:
    """One Tox21 assay table reduced to one row per compound: the aggregated outcome and score, the mean maximal response over the replicates (percent, signed like AID 1851: negative is inhibition) and the mean
    fitted log10 AC50 over the replicates whose curve was fitted."""
    header = pd.read_csv(path, nrows=0).columns
    wanted = ["PUBCHEM_CID", "PUBCHEM_EXT_DATASOURCE_SMILES", "PUBCHEM_ACTIVITY_OUTCOME", "PUBCHEM_ACTIVITY_SCORE"] + [c for c in header if c.startswith(("Max_Response-Replicate", "Fit_LogAC50-Replicate"))]
    raw = pd.read_csv(path, skiprows=[1, 2, 3, 4], usecols=wanted, low_memory=False).dropna(subset=["PUBCHEM_EXT_DATASOURCE_SMILES"])
    response = raw[[c for c in raw.columns if c.startswith("Max_Response")]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    logac50 = raw[[c for c in raw.columns if c.startswith("Fit_LogAC50")]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return pd.DataFrame({"cid": raw.PUBCHEM_CID, "smiles": raw.PUBCHEM_EXT_DATASOURCE_SMILES, "outcome": raw.PUBCHEM_ACTIVITY_OUTCOME, "score": raw.PUBCHEM_ACTIVITY_SCORE, "response": response, "logac50": logac50}).drop_duplicates("cid")


def build_tox21_compact(raw_dir: Path, target: Path = TOX_COMPACT) -> pd.DataFrame:
    """The compact Tox21 table: one row per compound with, per isoform, the response and the potency (pAC50, inactive compounds at the floor, inconclusive compounds without a fit left out), and the
    flag of the luciferase counter-screen. A reading of a compound that inhibits firefly luciferase is set to missing."""
    counter = _tox_table(raw_dir / f"a{TOX_COUNTER_AID}.csv").set_index("cid")
    luciferase_inhibitor = set(counter.index[counter.outcome == "Active"])
    merged = None
    for iso, aid in TOX_AIDS.items():
        t = _tox_table(raw_dir / f"a{aid}.csv")
        pac50 = np.where(t.outcome == "Inactive", POTENCY_FLOOR, -t.logac50)
        pac50 = np.where((t.outcome == "Inconclusive") & t.logac50.isna(), np.nan, pac50)
        part = pd.DataFrame({"cid": t.cid, "smiles": t.smiles, f"tresp_{iso}": -t.response.clip(*[-v for v in RESPONSE_RANGE[::-1]]), f"tpac50_{iso}": pac50})
        merged = part if merged is None else merged.merge(part.drop(columns="smiles"), on="cid", how="outer")
    merged["luciferase_inhibitor"] = merged.cid.isin(luciferase_inhibitor)
    for column in TOX_COLUMNS:
        merged.loc[merged.luciferase_inhibitor, column] = np.nan
    target.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(target, index=False, compression="gzip")
    return merged


def tox_table() -> pd.DataFrame:
    """The compact Tox21 table with every compound that is also a challenge training or test compound removed (matched on the connectivity layer of the InChIKey)."""
    table = pd.read_csv(TOX_COMPACT).dropna(subset=["smiles"])
    exclude: set[str] = set()
    for path in CHALLENGE_FILES:
        exclude |= {k for k in (connectivity_key(s) for s in pd.read_csv(path)["SMILES"]) if k}
    keys = [connectivity_key(s) for s in table["smiles"]]
    keep = [i for i, k in enumerate(keys) if k is not None and k not in exclude]
    return table.iloc[keep].reset_index(drop=True)


def extract_tox21(smiles: list[str], cache: Path = TOX_CACHE) -> None:
    from lightgbm import LGBMRegressor

    table = tox_table()
    keys = list(dict.fromkeys(Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in smiles))
    target = features([Chem.MolFromSmiles(s) for s in keys])
    train = features([Chem.MolFromSmiles(s) for s in table["smiles"]])
    out = np.zeros((len(keys), len(TOX_COLUMNS)), dtype=np.float32)
    for j, column in enumerate(TOX_COLUMNS):
        y = table[column].to_numpy(dtype=float)
        have = np.isfinite(y)
        out[:, j] = LGBMRegressor(**LGBM).fit(train[have], y[have]).predict(target)
        print(f"{column}: trained on {int(have.sum())} Tox21 compounds", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, smiles=np.array(keys, dtype=object), values=out, columns=np.array(TOX_COLUMNS))
    print(f"{len(keys)} challenge compounds scored on {len(table)} Tox21 compounds; wrote {cache}")


def load_ext_tox21(smiles: list[str], cache: Path = TOX_CACHE) -> np.ndarray:
    return load_ext_continuous(smiles, cache)


def features(mols) -> np.ndarray:
    from cypmode.challenge.feature_store import BLOCKS

    return np.hstack([BLOCKS["morgan_c2"](mols), BLOCKS["maccs"](mols)])


def extract(smiles: list[str], cache: Path = CACHE) -> None:
    from lightgbm import LGBMRegressor

    table = external_table()
    keys = list(dict.fromkeys(Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in smiles))
    target = features([Chem.MolFromSmiles(s) for s in keys])
    train = features([Chem.MolFromSmiles(s) for s in table["SMILES"]])
    out = np.zeros((len(keys), len(COLUMNS)), dtype=np.float32)
    for j, column in enumerate(COLUMNS):
        y = table[column].to_numpy(dtype=float)
        have = np.isfinite(y)
        out[:, j] = LGBMRegressor(**LGBM).fit(train[have], y[have]).predict(target)
        print(f"{column}: trained on {int(have.sum())} external compounds", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, smiles=np.array(keys, dtype=object), values=out, columns=np.array(COLUMNS))
    print(f"{len(keys)} challenge compounds scored on {len(table)} external compounds; wrote {cache}")


def load_ext_continuous(smiles: list[str], cache: Path = CACHE) -> np.ndarray:
    """Read-only lookup of the block for ``smiles`` (rows follow ``smiles``); never trains or imports LightGBM."""
    if not Path(cache).exists():
        raise FileNotFoundError(f"{cache} is missing: run `python -m cypmode.challenge.external_continuous --extract` first")
    z = np.load(cache, allow_pickle=True)
    index = {k: i for i, k in enumerate(z["smiles"])}
    wanted = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in smiles]
    missing = [w for w in wanted if w not in index]
    if missing:
        raise ValueError(f"{len(missing)} compounds are not in the continuous external block, e.g. {missing[0]!r}")
    return z["values"][[index[w] for w in wanted]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--download-tox21", action="store_true", help="download the six PubChem tables into --raw-dir and write the compact table")
    parser.add_argument("--raw-dir", type=Path, default=Path("/tmp/tox21_raw"))
    args = parser.parse_args()
    if args.download_tox21:
        download_tox21(args.raw_dir)
        print(f"{len(build_tox21_compact(args.raw_dir))} Tox21 compounds; wrote {TOX_COMPACT}")
    if args.extract:
        train, test = pd.read_csv(CHALLENGE_FILES[0]), pd.read_csv(CHALLENGE_FILES[1])
        smiles = list(train.SMILES) + list(test.SMILES)
        extract(smiles)
        if TOX_COMPACT.exists():
            extract_tox21(smiles)


if __name__ == "__main__":
    main()

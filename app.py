"""Streamlit tool for screening compounds against the Type II CYP450 motif set.

Takes SMILES (pasted or uploaded as a text file) and reports which classic
heme-coordinating motif, if any, each compound matches. A second tab shows
the physics-based structural check for the four reference compounds this
screen is validated against, read from a small checked-in summary so this
tab works the same in a hosted deployment as it does locally -- the full
Boltz-2 output (multi-GB weights, MSA downloads, predicted structures)
isn't something a hosted demo can run or ship. See README.md's Validation
section and cypmode/validation/structures.py to reproduce those numbers
from a local Boltz-2 run.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from rdkit import Chem

from cypmode.metrics.motifs import classify_mechanism, detect_motifs

VALIDATION_SUMMARY_PATH = Path("data/boltz_test/validation_summary.json")
CHALLENGE_COMPOUNDS_PATH = Path("data/cyp_challenge/structure_track_test_blinded.csv")
CHALLENGE_RESULTS_PATH = Path("data/cyp_challenge/structure_track_results.csv")


def read_smiles(file, text_input) -> list[str]:
    if file is not None:
        try:
            content = file.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            st.warning("Could not read that file as UTF-8 text.")
            return []
        return [line.strip() for line in content.splitlines() if line.strip()]
    if text_input:
        return [line.strip() for line in text_input.splitlines() if line.strip()]
    return []


def validate_smiles(smiles_list: list[str]) -> tuple[list[str], list[str]]:
    valid, invalid = [], []
    for smiles in smiles_list:
        # RDKit parses "" as a valid empty molecule rather than rejecting it;
        # treat it as invalid input here regardless.
        if smiles.strip() and Chem.MolFromSmiles(smiles) is not None:
            valid.append(smiles)
        else:
            invalid.append(smiles)
    return valid, invalid


def screen_compounds(smiles_list: list[str]) -> pd.DataFrame:
    rows = []
    for smiles in smiles_list:
        motifs = detect_motifs(smiles)
        rows.append(
            {
                "SMILES": smiles,
                "Motifs matched": ", ".join(motifs) if motifs else "none",
                "Mechanism call": classify_mechanism(smiles),
            }
        )
    return pd.DataFrame(rows)


def motif_screen_tab():
    st.caption(
        "Fast substructure screen for the classic Type II (heme-coordinating) "
        "motifs — imidazole, triazole, pyridine, thiazole. A match means "
        "'capable of heme coordination,' not 'confirmed to coordinate in a "
        "specific isoform's active site' — see the Structural validation tab "
        "and README for what's actually been checked against physics."
    )

    col1, col2 = st.columns(2)
    with col1:
        smiles_file = st.file_uploader("Upload a text file (one SMILES per line)", type=["txt", "csv", "smi"])
    with col2:
        text_input = st.text_area("Or paste SMILES, one per line")

    if st.button("Screen compounds"):
        smiles_list = read_smiles(smiles_file, text_input)
        if not smiles_list:
            st.warning("Upload a file or paste at least one SMILES string.")
            return

        valid, invalid = validate_smiles(smiles_list)
        if invalid:
            st.warning(f"Skipped {len(invalid)} unparseable SMILES string(s).")
        if not valid:
            st.error("No valid SMILES to screen.")
            return

        df = screen_compounds(valid)
        st.subheader("Results")
        st.dataframe(df, width="stretch")
        st.download_button(
            "Download results (CSV)",
            data=df.to_csv(index=False),
            file_name="cypmode_motif_screen.csv",
            mime="text/csv",
        )


def structural_validation_tab():
    st.caption(
        "Boltz-2 structure prediction for the four compounds a recent CYP3A4 "
        "cryo-EM paper solved structures for, with the heme cofactor "
        "explicitly modeled and covalently bonded to the real axial "
        "cysteine (Cys442). No constraint is placed on the inhibitor's "
        "position relative to the heme, so a short predicted iron-to-"
        "ligand-nitrogen distance is a genuine structural prediction, not "
        "something the input forced to happen."
    )

    if not VALIDATION_SUMMARY_PATH.exists():
        st.warning(f"{VALIDATION_SUMMARY_PATH} not found — see README.md's Validation section.")
        return

    try:
        results = json.loads(VALIDATION_SUMMARY_PATH.read_text())
        rows = [
            {
                "compound": name,
                "closest_motif_nitrogen_distance_angstrom": r["closest_motif_nitrogen_distance_angstrom"],
                "coordinated": r["coordinated"],
                "affinity_pred_value": r["affinity_pred_value"],
                "affinity_probability_binary": r["affinity_probability_binary"],
                "confidence_score": r["confidence_score"],
            }
            for name, r in results.items()
        ]
    except (json.JSONDecodeError, KeyError) as e:
        st.error(f"{VALIDATION_SUMMARY_PATH} is malformed ({e}) — try regenerating it with cypmode.validation.build_summary.")
        return

    st.dataframe(pd.DataFrame(rows), width="stretch")


def load_challenge_compounds() -> pd.DataFrame:
    """The CYP Blind Challenge structure-prediction track's blinded test set,
    with the motif screen applied live (screen_compounds).
    """
    compounds_df = pd.read_csv(CHALLENGE_COMPOUNDS_PATH)
    motif_df = screen_compounds(compounds_df["SMILES"].tolist())
    return compounds_df.merge(motif_df, on="SMILES", how="left")


def merge_challenge_results(compounds_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join the batch structural-validation results (cypmode.validation.batch's
    CSV export) onto the compound table by name -- compounds the batch run
    hasn't reached yet simply get null result columns, not an error, since
    that run can take hours and this table should reflect whatever's done
    so far.
    """
    results_df = results_df.rename(columns={"compound": "Molecule_Name"}).drop(columns=["smiles"], errors="ignore")
    return compounds_df.merge(results_df, on="Molecule_Name", how="left")


def challenge_tab():
    st.caption(
        "The OpenADMET CYP Blind Challenge's structure-prediction track "
        "blinded test set (20 compounds, CYP3A4 co-folding) -- see "
        "README.md's 'OpenADMET CYP Blind Challenge' section for how this "
        "track was found and what it is. Motif calls are computed live; "
        "structural-validation columns (from a real Boltz-2 run via "
        "cypmode.validation.batch, not something this hosted app can run "
        "itself) fill in as that run finishes each compound -- this table "
        "reflects whatever's completed so far, not a final result set."
    )

    if not CHALLENGE_COMPOUNDS_PATH.exists():
        st.warning(f"{CHALLENGE_COMPOUNDS_PATH} not found.")
        return

    df = load_challenge_compounds()
    if CHALLENGE_RESULTS_PATH.exists():
        df = merge_challenge_results(df, pd.read_csv(CHALLENGE_RESULTS_PATH))
        n_done = df["coordinated"].notna().sum() if "coordinated" in df else 0
    else:
        n_done = 0
    st.metric("Structural validation complete", f"{n_done} / {len(df)}")

    st.dataframe(df, width="stretch")
    st.download_button(
        "Download table (CSV)",
        data=df.to_csv(index=False),
        file_name="cyp_challenge_structure_track.csv",
        mime="text/csv",
    )


def app():
    st.title("CYPMode: Mode-Aware CYP450 Inhibition Screening")
    tab1, tab2, tab3 = st.tabs(["Motif screen", "Structural validation", "CYP Blind Challenge"])
    with tab1:
        motif_screen_tab()
    with tab2:
        structural_validation_tab()
    with tab3:
        challenge_tab()


if __name__ == "__main__":
    st.set_page_config(page_title="CYPMode", page_icon="🧬", layout="wide")
    app()

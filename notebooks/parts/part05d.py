"""Part V (last part): the six-drug panel of time-dependent shifts (5.7) and the Streamlit application (5.8)."""

from notebooks.notebook_builder import code, md

PANEL6 = '''
# The reference panel: six named drugs whose potency was measured with and without a pre-incubation (cypmode/validation/tdi_reference.py). For each drug: the shift (pIC50 with pre-incubation minus pIC50 without), its
# standard deviation and its 95 % interval as published with the data, what the interval says about the challenge's threshold of log10 2, the motif screen of 5.1, the amine alert of 5.2 and, for the two drugs that are
# in the structural reference set, the iron-nitrogen distance of 5.5.
import numpy as np
import pandas as pd

from cypmode.metrics.amines import has_trialkylamine
from cypmode.validation.tdi_reference import TDI_SHIFT_LOG2_THRESHOLD, load_tdi_reference, with_structural_coordination

reference_panel = with_structural_coordination(load_tdi_reference())
reference_panel["amine alert"] = [has_trialkylamine(s) for s in reference_panel["smiles"]]
reference_panel["reading"] = np.where(reference_panel["delta_q2.5"] > TDI_SHIFT_LOG2_THRESHOLD, "above the threshold",
                                      np.where(reference_panel["delta_q97.5"] < TDI_SHIFT_LOG2_THRESHOLD, "below the threshold", "interval straddles it"))
panel_view = pd.DataFrame({"drug": reference_panel["drug"], "shift": reference_panel["delta_mean"], "95 % interval": [f"{lo:.2f} to {hi:.2f}" for lo, hi in zip(reference_panel["delta_q2.5"], reference_panel["delta_q97.5"])],
                           "against log10 2": reference_panel["reading"], "Type II motif": [", ".join(m) or "none" for m in reference_panel["motifs"]], "amine alert": np.where(reference_panel["amine alert"], "yes", "no"),
                           "iron to nitrogen (A)": reference_panel["closest_motif_nitrogen_distance_angstrom"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")})
show_table(panel_view, floatfmt=".2f")

# the two-by-two counts of 5.3, on six compounds: the flag against the point estimate of the shift
positive = reference_panel["delta_mean"] > TDI_SHIFT_LOG2_THRESHOLD
counts_motif = {"motif and TDI": int((reference_panel["has_motif"] & positive).sum()), "motif, not TDI": int((reference_panel["has_motif"] & ~positive).sum()),
                "no motif, TDI": int((~reference_panel["has_motif"] & positive).sum()), "no motif, not TDI": int((~reference_panel["has_motif"] & ~positive).sum())}
counts_amine = {"amine and TDI": int((reference_panel["amine alert"] & positive).sum()), "amine, not TDI": int((reference_panel["amine alert"] & ~positive).sum()),
                "no amine, TDI": int((~reference_panel["amine alert"] & positive).sum()), "no amine, not TDI": int((~reference_panel["amine alert"] & ~positive).sum())}
print("motif screen against the shift:", counts_motif)
print("amine alert against the shift:", counts_amine)

fig, axis = plt.subplots(figsize=(7.0, 3.6))
order = reference_panel.sort_values("delta_mean").reset_index(drop=True)
positions = np.arange(len(order))
axis.hlines(positions, order["delta_q2.5"], order["delta_q97.5"], color=theme["muted"], lw=2.2)
colours = [theme["blue"] if a else (theme["orange"] if m else theme["muted"]) for a, m in zip(order["amine alert"], order["has_motif"])]
axis.scatter(order["delta_mean"], positions, c=colours, s=70, zorder=3)
axis.axvline(TDI_SHIFT_LOG2_THRESHOLD, color=theme["red"], ls="--", lw=1.3)
axis.set_yticks(positions, order["drug"])
axis.set(xlabel="shift of the pIC50 after pre-incubation (blue: amine alert, orange: Type II motif only)")
tidy(axis, theme)
fig.tight_layout()
n_drugs = len(reference_panel)
n_positive = int(positive.sum())
n_straddle = int((reference_panel["reading"] == "interval straddles it").sum())
straddling_drug = reference_panel.loc[reference_panel["reading"] == "interval straddles it", "drug"].tolist()
shift_of = dict(zip(reference_panel["drug"], reference_panel["delta_mean"]))
'''

APP = '''
# The application, exercised the way Streamlit's own test runner exercises it (no browser): open app.py, list the tabs and the tables each one shows, type the four reference molecules into the first tab and press its button,
# and list the functions of the file. The tabs read the same files as the notebook.
import ast

from streamlit.testing.v1 import AppTest

app_run = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
tab_labels = [t.label for t in app_run.tabs]
table_shapes = [frame.value.shape for frame in app_run.dataframe]
reference_smiles = "\\n".join(read_boltz_inputs(name)["ligand_smiles"] for name in REFERENCE_NAMES)
app_run.text_area[0].set_value(reference_smiles)
app_run.button[0].click().run()
screen_result = app_run.dataframe[0].value
screen_result["SMILES"] = [s[:28] + "..." if len(s) > 31 else s for s in screen_result["SMILES"]]
show_table(screen_result)
functions = [node.name for node in ast.parse((ROOT / "app.py").read_text()).body if isinstance(node, ast.FunctionDef)]
print("tabs:", tab_labels)
print("tables on the page (rows, columns):", table_shapes)
print("exceptions while the page ran:", len(app_run.exception))
print("functions of app.py:", functions)
n_app_functions = len(functions)
'''

CELLS = [
    md("""
## 5.7 Cross-checking the screen against time-dependent shift measurements

> **Field:** Pharmacology · Biochemistry

*Where this section sits:* sections 5.3 and 5.4 asked whether a motif goes with inhibition and with potency. The question of the challenge, however, is about neither: it is about *time dependence*, the shift of section 1.8, and a rule that finds strong reversible binders need not find time-dependent ones. This section puts the two rules of this part, the motif screen and the amine alert, side by side with a small set of drugs for which that shift was measured, and it
closes the part by saying what each rule can and cannot tell about the label that the rest of the notebook predicts.

**The panel.** The post "Building the OpenADMET Data Engine" has a companion repository with the measured potency of a few named drugs with and without a pre-incubation of the enzyme (CC BY 4.0). The project uses six of them: azamulin, ketoconazole, clotrimazole, troleandomycin, verapamil and diltiazem. For each there is the **shift**, $\\Delta$, the pIC50 measured after pre-incubation minus the pIC50 measured without it (section 1.8): a positive $\\Delta$ means that the compound
becomes more potent the longer it stays with the enzyme. The data give its mean, its standard deviation and the 2.5 and 97.5 percent quantiles, which bound a 95 percent interval (section 2.5). The challenge calls a compound time-dependent when $\\Delta$ exceeds $\\log_{10} 2 = 0.301$, a more than two-fold gain in potency. The structures of the six drugs were written into the project's code one by one and each was checked
against a chemical database, not guessed (`REFERENCE_SMILES` in `cypmode/validation/tdi_reference.py`). The cell below puts each drug's shift, what its interval says about the threshold (wholly above, wholly below, or straddling it), the calls of the two rules and, for the two drugs that are in the structural reference set, the distance of 5.5 in one table, and draws the shifts with their intervals.
"""),
    code(PANEL6),
    md("""
Read the table in two steps, first for the screen of Type II motifs and then for the amine alert. *The motif screen* flags three of the six drugs, azamulin, ketoconazole and clotrimazole, and only one of them is time-dependent: its flagged drugs are {{=counts_motif['motif and TDI']}} time-dependent and {{=counts_motif['motif, not TDI']}} not, and none of the three drugs without a motif is below the threshold ({{=counts_motif['no motif, TDI']}} time-dependent, {{=counts_motif['no motif, not TDI']}} not). The reason is the one section 1.8 gave for the two arms.
A drug that binds the iron through a nitrogen binds strongly from the start, and ketoconazole and clotrimazole are well-known strong reversible inhibitors of CYP3A4, so they are already potent in the arm without pre-incubation and pre-incubation adds little: ketoconazole's shift is {{=shift_of['ketoconazole']:.2f}}, clotrimazole's {{=shift_of['clotrimazole']:.2f}}, both far below the threshold. A motif tells the enzyme-binding story and the label of the challenge is about change over time, so a molecule can score high on the first and low on the second.
Azamulin is the exception, with a shift of {{=shift_of['azamulin']:.2f}}, and for the opposite reason to the one the motif suggests: it matches a triazole pattern but, as 5.5 showed, its prediction is not near the iron; it inactivates the enzyme by turning into a reactive species (section 1.10), and the screen is right about the molecule being a CYP3A4 inhibitor and wrong about the mechanism.

*The amine alert* flags troleandomycin, verapamil and diltiazem, the three drugs with a trialkylamine, and their shifts are {{=shift_of['troleandomycin']:.2f}}, {{=shift_of['verapamil']:.2f}} and {{=shift_of['diltiazem']:.2f}}, all above or at the threshold. Diltiazem is the borderline case: its point estimate is above the threshold, and its interval ({{=', '.join(straddling_drug)}} is the one drug in the table whose interval straddles it) includes values below it, so whether it counts as time-dependent depends on which side of the line the true shift falls, and the data cannot say.
Taking the point estimates, the alert flags {{=counts_amine['amine and TDI']}} of the {{=n_positive}} time-dependent drugs and none of the {{=n_drugs - n_positive}} that are not; the one time-dependent drug it misses is azamulin, whose route is not the amine route. This is what the chemistry of section 1.10 predicts, and it is the reason the alert was written. It is also six drugs, chosen because their behaviour is documented, so it is a check that the rule does what it was designed to do and not an estimate of how well it works on new compounds; for that the
collection of 5.2, with its modest lift, is the evidence.

**What the part adds up to.** The Type II motif goes with inhibition in general (5.3), goes weakly with potency (5.4), places the nitrogen at the iron in a predicted structure when the compound really does bind that way (5.5), and does not go with time dependence, the property the challenge asks about (this section). The amine alert goes with time dependence for the route it was built for, and is modest in the collection at large (5.2). Neither rule is a model, and the project did not treat them as one: they became two columns among hundreds in the feature store, and Part VII
tests, by the procedures of Part VI, whether they add anything to a learned model. They also left two quantities that are used far beyond this part: the iron-nitrogen distance, which Part XI measures for every pose and later uses as a restraint, and the habit of testing each claim on data that did not shape it.

*On what actually ran.* `load_tdi_reference` reads the six-drug table (`data/tdi_track/inputs/octant_cache/tdi_reference/tdi_pic50_shift.tsv`), attaches the structures, the motif calls and the label `tdi_positive` (the point estimate above the threshold); `with_structural_coordination` joins the coordination results of the structural summary onto the two drugs that have them (ketoconazole and azamulin); the amine alert, the reading against the interval and the counts are computed in the cell above. The table is the whole of the data: six rows.
"""),
    md("""
## 5.8 The Streamlit application and its figures

> **Field:** Software engineering · Cheminformatics

*Where this section sits:* the rules and checks of this part started as scripts, and the project wanted one place where a person could try the screen on a molecule of their own and see what had been checked against structures and what the final models do. That place is a small web application. This section says what it is, what it shows and how it is tested, because it is the face of the project that does not need a reader to run any code, and because it is the one part of the repository that reads
files from almost every other part.

**What Streamlit is.** **Streamlit** is a Python library that turns a script into a web page. The script is ordinary Python; a call such as `st.title(...)` puts a heading on the page, `st.text_area(...)` puts a box in which a person can type and returns what was typed, `st.button(...)` returns true on the run in which it was pressed, and `st.dataframe(...)` shows a table. The library has one unusual rule that explains how an application written this way behaves: *every time a person
interacts with the page, the whole script is run again from the top*, with the widgets returning their new values. There is no program that waits and responds; there is a script that is re-run, which is why a function that is expensive to compute is cached, and why the page can be tested without a browser, by running the script with chosen widget values, which is what the cell below does.

**What the application shows.** Its page has four tabs (`app.py`, run with `streamlit run app.py`). *Motif screen* is the original tool: paste or upload SMILES, and for each molecule the table says which of the six patterns of 5.1 match and gives the mechanism sentence (an unreadable string is skipped with a warning); the result can be downloaded as a CSV file. *Structural validation* reads the summary of the four reference compounds (the distances of 5.5) from a small file in the repository, so that it
works the same on a machine without Boltz-2, whose weights and predictions are far too large to ship. *CYP Blind Challenge* lists the twenty molecules of the structure track with their motif calls computed live and, when the files of the structure track exist, the structural results next to them. *TDI Track Results* shows the final report of the TDI build and the ledger of every gate verdict, built from the same logs as the method report (Part XII). A tab whose file is missing
shows a note instead of failing: in a public copy of the repository, from which the results of the two tracks are held back until the challenge closes, the last two tabs are the ones that say so.

**How it is tested.** The pure functions of the file (reading SMILES, checking them, screening a list, merging tables) are tested like any other code (`tests/test_app.py`), without Streamlit running. Two further tests added when this section was written, run the page with Streamlit's own test runner: one checks that the summary the structural tab reads exists, and one that the page renders all four tabs without an exception. They exist because the first run of that check found a real fault: the
summary and the structure-track table had been moved into a folder for CYP3A4 when the outputs were reorganised, and the application still looked for them at their old paths, so two tabs showed a warning and an empty count instead of their tables. The cell runs the same check, then types the four reference molecules into the first tab and presses its button.
"""),
    code(APP),
    md("""
The page has the tabs {{=', '.join(tab_labels)}}, and shows {{=len(table_shapes)}} tables after the first run (rows by columns: {{=', '.join(f'{r} by {c}' for r, c in table_shapes)}}); {{=len(app_run.exception)}} exceptions were raised while it ran. After the four reference molecules are typed into the first tab, the table above is what the application itself returns: ketoconazole, ritonavir, azamulin and vardenafil each carry a Type II pattern, the call that 5.5
then put to the test with structures, and which section 5.7 qualified. The file defines {{=n_app_functions}} functions, one per tab and the small helpers around them.

**The figures.** The bar chart of 5.5 and the bar charts of the structure track were first drawn by scripts that live next to the application, `docs/figures/generate_validation_figure.py` (the iron-nitrogen distances, in a light and a dark variant), `docs/figures/generate_structure_track_figures.py` (the PoseBusters checks) and `docs/figures/generate_tdi_figures.py` (the stages of the TDI model); each reads a file of results and writes a picture, and none computes a result of its own. Later in the project they were joined by one system that registers every figure with its caption,
the data files it was drawn from and the section of this notebook that uses it, and rebuilds them all in one command (`docs/figures/build_all.py`, Part XII); the figures that open the sections of this notebook come from there.

*On what actually ran.* The application is run locally; the repository's `requirements.txt` lists what a hosted copy would install, and no hosted copy is claimed in this notebook. The test runner executes the same script that `streamlit run` would, so what the cell reports is what a person would see; what it cannot report is how the page looks in a browser, which is a matter of taste and not of correctness.
"""),
]

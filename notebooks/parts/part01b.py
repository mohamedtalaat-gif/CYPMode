# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part I (second half): time-dependent inhibition, potency, the two assay arms, heme chemistry, reactive metabolites and structural alerts."""

from notebooks.notebook_builder import code, figure, md

TDI_SIMULATION = '''
# The scheme E + I <-> EI -> E-I* integrated as differential equations (the rate constants of the figure below), and compared with the closed form E_active(t) = E0 exp(-k_obs t),
# k_obs = k_inact [I] / (K_I + [I]), where E_active = E + EI is the enzyme that is not yet dead.
from scipy.integrate import solve_ivp

K_ON, K_OFF, K_INACT = 50.0, 500.0, 0.8            # per uM per min, per min, per min
K_I = K_OFF / K_ON
def scheme(_, y, conc):
    E, EI, dead = y
    bind, unbind, kill = K_ON * conc * E, K_OFF * EI, K_INACT * EI
    return [-bind + unbind, bind - unbind - kill, kill]
times = np.linspace(0, 12, 121)
rows = []
for conc in (2.0, 10.0, 50.0, 200.0):
    run = solve_ivp(scheme, (0, 12), [1.0, 0.0, 0.0], t_eval=times, args=(conc,), method="LSODA", rtol=1e-10, atol=1e-12)
    active = run.y[0] + run.y[1]
    slope = -np.polyfit(times[20:], np.log(active[20:]), 1)[0]          # the slope of ln(activity) against time once the fast binding has settled
    rows.append({"[I] (uM)": conc, "k_obs from the simulation (1/min)": slope, "k_inact [I] / (K_I + [I]) (1/min)": K_INACT * conc / (K_I + conc),
                 "half-life of the activity (min)": np.log(2) / slope})
print(f"K_I = k_off / k_on = {K_I:.0f} uM, k_inact = {K_INACT} per min")
show_table(pd.DataFrame(rows), floatfmt=".4f")
'''

DOSE_RESPONSE = '''
# A simulated dose-response experiment: ten concentrations in three-fold steps, true pIC50 = 6 (IC50 = 1 uM), Hill slope 1, 6 % measurement noise. The logistic curve is fitted in pIC50 units.
from scipy.optimize import curve_fit

def activity_left(log_conc, pic50, hill=1.0):
    return 1 / (1 + 10 ** (hill * (log_conc + pic50)))          # log_conc = log10 of the concentration in mol/L, so [I]/IC50 = 10^(log_conc + pIC50)
rng = np.random.default_rng(3)
log_conc = np.log10(1e-4 / 3.0 ** np.arange(10))                 # 100 uM down to 5 nM
estimates, errors = [], []
for _ in range(500):
    y = activity_left(log_conc, 6.0) + rng.normal(0, 0.06, size=log_conc.size)
    fit, covariance = curve_fit(lambda x, p: activity_left(x, p), log_conc, y, p0=[5.5])
    estimates.append(fit[0]); errors.append(np.sqrt(covariance[0, 0]))
print(f"500 repetitions of the experiment: mean fitted pIC50 {np.mean(estimates):.3f}; spread of the fits (standard deviation) {np.std(estimates):.3f}; typical standard error from the curve fit {np.median(errors):.3f}")
'''

FLOOR = '''
# How many direct-inhibition measurements of the training file lie at or below the pIC50 of 4, the weakest effect the project's label rule distinguishes?
from cypmode.challenge.tdi_targets import ASSAY_FLOOR

floor_rows = []
for enzyme in ("CYP3A4", "CYP2D6"):
    column = train[f"{enzyme}_pIC50_direct_inhibition"].dropna()
    floor_rows.append({"enzyme": enzyme, "measurements": len(column), "below the floor of 4": int((column < ASSAY_FLOOR).sum()), "equal to 4": int((column == ASSAY_FLOOR).sum()),
                 "smallest value": column.min(), "largest value": column.max()})
share_below = {r["enzyme"]: r["below the floor of 4"] / r["measurements"] for r in floor_rows}
show_table(pd.DataFrame(floor_rows), floatfmt=".2f")
'''

SHIFT = '''
# The potency shift that a time-dependent inhibitor shows after a pre-incubation of t_pre minutes, from k_obs([I]) = ln 2 / t_pre, with the constants of the figure: K_I = 10 uM, k_inact = 0.8 per min.
for t_pre in (5.0, 15.0, 30.0, 60.0):
    target = np.log(2) / t_pre                                  # the inactivation rate that halves the activity in t_pre minutes
    ic50_pre = K_I * target / (K_INACT - target)                # [I] at which k_inact [I] / (K_I + [I]) = target
    ic50_direct = K_I                                           # without a head start the inhibitor only competes (reversible part), its IC50 is of the order of K_I at best
    print(f"pre-incubation {t_pre:>4.0f} min: IC50 {ic50_pre:6.3f} uM against about {ic50_direct:.0f} uM without pre-incubation; shift in pIC50 units {np.log10(ic50_direct / ic50_pre):.2f}")
'''

REFERENCE_DISTANCES = '''
# The iron-to-ligand distances in the crystal structures of CYP3A4 with a bound drug-like ligand that the project collected (data/structure_track/inputs/cyp3a4_reference_ligands.json):
# the distance from the heme iron to the closest atom of each ligand.
import json

records = json.loads((ROOT / "data/structure_track/inputs/cyp3a4_reference_ligands.json").read_text())["ligands"]
distance = pd.Series([r["min_fe_distance"] for r in records])
bins = pd.cut(distance, [0, 2.6, 3.5, 5, 8, 100], labels=["below 2.6 A (bonded)", "2.6 to 3.5 A", "3.5 to 5 A", "5 to 8 A", "above 8 A"])
table = bins.value_counts().sort_index().rename("ligands").reset_index().rename(columns={"index": "distance from the iron to the closest ligand atom"})
n_bonded = int((distance < 2.6).sum())
print(f"{len(records)} ligands in crystal structures; {n_bonded} have an atom within bonding distance of the iron")
show_table(table)
'''

ALERTS = '''
# The structural alerts of the project (cypmode/challenge/feature_store.py) against the real labels: of the CYP3A4 training compounds that were labelled, which share is a time-dependent
# inhibitor when the compound carries the alert, and when it does not?
from rdkit import Chem

from cypmode.challenge.feature_store import _ALERT_SMARTS
from cypmode.challenge.tdi_targets import training_view

view = training_view(train, "CYP3A4")
labelled = view[view["population"] == "both_arms"].join(train["SMILES"])
molecules = [Chem.MolFromSmiles(s) for s in labelled["SMILES"]]
alerts = ["methylenedioxyphenyl", "furan", "thiophene", "aminothiazole", "terminal_alkyne", "hydrazine", "cyclopropylamine", "tertiary_alkylamine", "primary_arylamine", "thiocarbonyl", "catechol"]
alert_rows = []
for name in alerts:
    pattern = Chem.MolFromSmarts(_ALERT_SMARTS[name])
    has = np.array([m.HasSubstructMatch(pattern) for m in molecules])
    positive = labelled["label"].astype(bool).to_numpy()
    alert_rows.append({"alert": name, "compounds with it": int(has.sum()), "share positive with it": positive[has].mean() if has.any() else float("nan"), "share positive without it": positive[~has].mean()})
alert_table = pd.DataFrame(alert_rows).sort_values("compounds with it", ascending=False).set_index("alert")
alert_share, alert_n, base_rate = alert_table["share positive with it"], alert_table["compounds with it"], labelled["label"].astype(bool).mean()
show_table(alert_table.reset_index(), floatfmt=".3f")
'''

CELLS = [
    md("""
## 1.6 Time-dependent inhibition: the kinetics of switching an enzyme off

> **Field:** Biochemistry · Mathematics (kinetics)

A **mechanism-based inactivator** is a molecule that the enzyme itself converts, in its normal catalytic cycle, into a species that reacts with the enzyme and disables it. Its kinetic scheme has two steps. The
inhibitor $I$ first binds reversibly, $E + I \\rightleftharpoons EI$, with dissociation constant $K_I = k_{\\mathrm{off}}/k_{\\mathrm{on}}$ (a larger $K_I$ means weaker binding). The complex then converts, at the rate
constant $k_{\\mathrm{inact}}$, into a dead enzyme $E\\!-\\!I^*$ that does not recover:

$$E + I \\;\\underset{k_{\\mathrm{off}}}{\\overset{k_{\\mathrm{on}}}{\\rightleftharpoons}}\\; EI \\;\\overset{k_{\\mathrm{inact}}}{\\longrightarrow}\\; E\\!-\\!I^* .$$

The enzyme that can still work is the free enzyme plus the reversible complex, $E_{\\mathrm{act}} = E + EI$. If the first step is fast compared with the second (the usual case), the two forms stay in
equilibrium, and the share of the active enzyme that sits in the complex is $[EI]/E_{\\mathrm{act}} = [I]/(K_I + [I])$. Only the complex is inactivated, at the rate $k_{\\mathrm{inact}}[EI]$, so that, with $t$ the time since the inhibitor was added,

$$\\frac{dE_{\\mathrm{act}}}{dt} = -k_{\\mathrm{inact}}\\,[EI] = -\\underbrace{\\frac{k_{\\mathrm{inact}}\\,[I]}{K_I + [I]}}_{k_{\\mathrm{obs}}}\\;E_{\\mathrm{act}} \\quad\\Longrightarrow\\quad E_{\\mathrm{act}}(t) = E_0\\,e^{-k_{\\mathrm{obs}}\\,t}.$$

The same exponential decay as in section 1.1 appears, now for the activity of the enzyme, with an **observed inactivation rate** $k_{\\mathrm{obs}}$ that depends on the inhibitor concentration in a saturating way
(compare section 1.2): it is proportional to $[I]$ when $[I] \\ll K_I$, reaches half of its maximum $k_{\\mathrm{inact}}$ at $[I] = K_I$, and cannot exceed $k_{\\mathrm{inact}}$. The half-life of the activity under
a given $[I]$ is $\\ln 2 / k_{\\mathrm{obs}}$. The ratio $k_{\\mathrm{inact}}/K_I$ is the efficiency of the inactivator, the analogue of $V_{\\max}/K_M$.

The experiment that measures these constants has a standard form. The enzyme is mixed with several concentrations of the inhibitor and the cofactor NADPH, which the enzyme needs to turn the inhibitor over;
samples are taken after different pre-incubation times and diluted into a mixture containing a substrate, so that the inhibitor no longer competes, and the remaining activity is measured. A plot of
$\\ln(\\mathrm{activity})$ against pre-incubation time is a straight line for each concentration, with slope $-k_{\\mathrm{obs}}$; a plot of $k_{\\mathrm{obs}}$ against $[I]$ is the saturating curve from which $k_{\\mathrm{inact}}$ and $K_I$
are fitted. The criteria that distinguish true mechanism-based inactivation from slow reversible binding (loss of activity that depends on time, on the concentration and on NADPH, that is not restored by dialysis, and
that is slowed by a competing substrate) go back to Silverman (1995). The cell integrates the three differential equations of the scheme and compares the observed rate with the closed form.
"""),
    code(TDI_SIMULATION),
    md("""
The slopes of the simulated activity agree with the closed form, and the half-life of the activity falls from {{=rows[0]['half-life of the activity (min)']:.1f}} minutes at {{=rows[0]['[I] (uM)']:.0f}} micromolar to {{=rows[-1]['half-life of the activity (min)']:.1f}} minutes at {{=rows[-1]['[I] (uM)']:.0f}} micromolar, which is the
signature of a time-dependent inhibitor: the more inhibitor, the faster the enzyme is lost, up to the limit set by $k_{\\mathrm{inact}}$. The figure shows the progress curves and the saturation of $k_{\\mathrm{obs}}$ for the same rate constants.
"""),
    figure("kin_tdi_progress"),
    md("""
## 1.7 How potency is measured: dose-response, the Hill equation, the IC50 and the pIC50 scale

> **Field:** Pharmacology · Statistics (curve fitting)

An inhibition experiment measures the activity of the enzyme at a series of inhibitor concentrations, usually in three-fold or ten-fold steps, and expresses it as a fraction of the activity without inhibitor.
The result is a **dose-response curve** that falls from 1 to 0 as the concentration grows. A single molecule of inhibitor binding one site gives, from the law of mass action, a fraction of free enzyme of
$1/(1 + [I]/K)$ where $K$ is the dissociation constant. Hill (1910), studying oxygen binding to haemoglobin, generalised this to $n$ molecules binding together, $E + nI \\rightleftharpoons EI_n$ with
$[E][I]^n = K^n\\,[EI_n]$. The fraction of enzyme that is free is $[E]/([E] + [EI_n]) = 1/(1 + ([I]/K)^n)$, and in practice the exponent is treated as a fitted number, the **Hill slope** $h$. Renaming the constant $K$ as the
concentration that halves the activity, the IC50, gives the **Hill equation** for the activity left, written $y$ (1 with no inhibitor, 0 with the enzyme fully blocked):

$$y = \\frac{1}{1 + ([I]/\\mathrm{IC}_{50})^{h}}.$$

At $[I] = \\mathrm{IC}_{50}$ the activity is exactly one half, whatever $h$; the Hill slope only sets how steeply the curve falls around that point (the figure draws three values).

Potencies of different compounds span many orders of magnitude, so they are quoted on a logarithmic scale. The **pIC50** is the negative base-10 logarithm of the IC50 expressed in mol/L:

$$\\mathrm{pIC}_{50} = -\\log_{10}\\bigl(\\mathrm{IC}_{50}/(1\\ \\mathrm{mol/L})\\bigr).$$

An IC50 of 1 micromolar ($10^{-6}$ mol/L) is a pIC50 of 6, 10 nanomolar is 8, 100 micromolar is 4. One unit of pIC50 is a ten-fold difference in potency, and a larger number means a stronger compound. This is the number
the challenge reports for every compound (section 0.3).
"""),
    figure("kin_dose_response"),
    md("""
The IC50 is not read off the data but fitted: the parameters of the Hill equation are chosen to minimise the squared distance to the measurements (nonlinear least squares), and the uncertainty of the fit gives a
standard error of the pIC50. The next cell simulates one such experiment 500 times, with measurement noise, and shows that the spread of the fitted values is what the standard error of a single fit predicts. That is
why the data files carry a **standard deviation** for every pIC50, the typical size of the error of that estimate (section 2.5).
"""),
    code(DOSE_RESPONSE),
    md("""
**The floor.** A dose-response curve can only be fitted well when the activity actually falls within the range of concentrations tested. A compound that does not inhibit at the highest concentration tested has an
IC50 above it, and its potency cannot be measured. The challenge's label rule (section 1.8) therefore treats every direct pIC50 below 4, that is an IC50 above 100 micromolar, as 4: the weakest effect it
distinguishes. The file nevertheless holds fitted values far below 4, and the cell counts them.
"""),
    code(FLOOR),
    md("""
{{=share_below['CYP3A4']:.0%}} of the direct measurements of CYP3A4 and {{=share_below['CYP2D6']:.0%}} of those of CYP2D6 are below 4 (the smallest are near 2,
an IC50 of about ten millimolar): compounds that hardly inhibited at all, whose fitted value is an extrapolation beyond the tested range and is correspondingly uncertain. The label rule of section 1.8 therefore takes the
larger of the direct measurement and 4, so that a compound that was not measurably active in the first experiment is compared against the floor and not against an extrapolated, much weaker value.
"""),
md("""
## 1.8 The two assay arms of the challenge and the shift between them

> **Field:** Biochemistry · Pharmacology

The challenge measures every compound twice. In the **direct inhibition** arm the enzyme, the compound and a substrate are mixed together and the activity is read at once; the result is a pIC50, called
$d$ here. In the **time-dependent inhibition arm** the enzyme and the compound (with NADPH) are left together for a pre-incubation period before the substrate is added; the result is a second pIC50, called
$t$. A compound that only competes reversibly has the same potency in both experiments, because letting enzyme and compound sit together changes nothing. A mechanism-based inactivator is more potent in the second,
because the pre-incubation lets it destroy the enzyme before the substrate has a chance to compete: the second curve lies to the left of the first, as in the figure.

The size of the **shift** between the two pIC50 values follows from section 1.6. After a pre-incubation of $t_{\\mathrm{pre}}$ the activity is $e^{-k_{\\mathrm{obs}} t_{\\mathrm{pre}}}$, which equals one half when $k_{\\mathrm{obs}} = \\ln 2 / t_{\\mathrm{pre}}$. Setting the
saturating expression for $k_{\\mathrm{obs}}$ equal to this value (with the inhibitor concentration $[I]$ and the constant $K_I$ of section 1.6) and solving for the concentration gives the IC50 after pre-incubation:

$$\\frac{k_{\\mathrm{inact}}\\,\\mathrm{IC}_{50}}{K_I + \\mathrm{IC}_{50}} = \\frac{\\ln 2}{t_{\\mathrm{pre}}} \\quad\\Longrightarrow\\quad \\mathrm{IC}_{50}(t_{\\mathrm{pre}}) = K_I\\,\\frac{\\ln 2 / t_{\\mathrm{pre}}}{k_{\\mathrm{inact}} - \\ln 2 / t_{\\mathrm{pre}}}.$$

This is finite only if $k_{\\mathrm{inact}} > \\ln 2/t_{\\mathrm{pre}}$: an inactivator that is too slow for the length of the pre-incubation does not show a shift. The cell evaluates it for the constants of the figure.
"""),
    code(SHIFT),
    md("""
The difference between the two pIC50 values, $s = t - d$, is the **shift**. A pure reversible inhibitor has $s \\approx 0$; the inactivator above has a shift of more than one pIC50 unit for pre-incubations of
fifteen minutes or longer. Since a pIC50 is a logarithm, a shift of $\\log_{10} 2 = 0.301$ means that the compound became exactly twice as potent after the head start, as $10^{0.301} = 2$. The challenge defines a
compound as time-dependent when its shift exceeds that value, measured against the larger of $d$ and the floor 4, $s = t - \\max(d, 4) > \\log_{10} 2$; section 2.3 shows how this rule was recovered from the data and
that it reproduces every label of the file.
"""),
    figure("kin_assay_arms"),
    md("""
## 1.9 Heme chemistry: Type I and Type II binding, and why a nitrogen binds the iron

> **Field:** Chemistry · Biochemistry

The iron at the centre of the heme can hold six ligands: four nitrogens of the porphyrin ring in a plane, the sulfur of the axial cysteine below it, and one position above the ring. In the resting enzyme
that last position is occupied by a water molecule. Two ways in which a molecule can bind in the pocket were named by the spectra they produce when added to a suspension of the enzyme (Schenkman and
co-workers, 1967): the absorption of light by the heme changes in a way that depends on what happens at the iron.

**Type I** molecules sit in the pocket near the heme and displace the water without binding the iron. The iron's electronic state changes from the low-spin form that the enzyme has in the resting state to a
high-spin form, and the difference spectrum shows a peak near 390 nanometres and a trough near 420. Most substrates are of this kind. **Type II** molecules carry a nitrogen atom that binds the iron directly and
takes the place of the water; the iron stays low-spin, and the difference spectrum has a peak near 430 nanometres and a trough near 390 to 410. The bond is a **coordinate bond**: both of its two electrons come from the
nitrogen, in the form of a **lone pair**, a pair of electrons that the nitrogen holds but does not share in any other bond. The iron, which can accept electron pairs, uses it. The nitrogen of an **aromatic** ring (a flat ring whose electrons are shared all the way round it, so that its bonds are neither single nor double, section 2.1; an imidazole, a triazole, a pyridine or a thiazole is such a ring) is well suited: its lone pair lies in the plane of the ring and points away from it, where nothing hinders the approach to the iron, and the ring itself is flat and
fits between the walls of the pocket. The iron-nitrogen distance in solved structures is about 2.0 to 2.3 angstrom (Poulos and Johnson, 2005).

A Type II molecule binds more tightly than it would by contact alone, which is why it is often a potent inhibitor, and the binding is a property of a recognisable group of atoms, which is the reason a
screen for such groups (section 0.2, Part V) is possible at all. The figure shows both kinds in real crystal structures of CYP3A4, and the cell lists how the ligands of the crystal structures collected by the
project are distributed in distance from the iron.
"""),
    figure("bio_type1_type2"),
    code(REFERENCE_DISTANCES),
    md("""
Most of the ligands of the crystal structures of CYP3A4 that the project collected ({{=n_bonded}} of {{=len(records)}}) have an atom within bonding distance of the iron. That is a feature of which complexes are solved and
deposited, not a statement about molecules in general: inhibitors that bind the iron have been studied intensively, so the structures of the Protein Data Bank over-represent the Type II geometry. The consequence for this project is twofold. A
prior or a benchmark built from these structures sees mostly Type II poses (Part XI returns to this when it uses them), and a molecule that carries a group able to bind the iron is not certain to use it. The motif screen of section 0.2
and Part V asks only whether a molecule carries such a group; whether it uses it is a question about a three-dimensional pose, and Part XI is about predicting that. The four drugs of section 0.2 show both outcomes with one chemistry:
"""),
    figure("bio_motif_screen"),
    md("""
## 1.10 Reactive metabolites and structural alerts

> **Field:** Chemistry · Pharmacology (toxicology)

Why does the pre-incubation make some molecules more potent? In the mechanism-based case the enzyme turns the molecule into something that reacts, and the products of the P450 cycle are strongly oxidising
(section 1.3). Four routes are textbook (Kalgutkar and co-workers, 2007; Orr and co-workers, 2012):

- **A metabolic-intermediate complex.** Many amines are oxidised, in steps ending in a nitroso compound (a nitrogen bound to an oxygen), whose nitrogen then binds the iron of the reduced heme so tightly that the
  complex is, for practical purposes, permanent. This is the route of tertiary alkylamines such as those of macrolide antibiotics.
- **A carbene complex.** The methylenedioxyphenyl group (a benzene ring with an oxygen-carbon-oxygen bridge) is oxidised at the bridge carbon to a carbene, a carbon with a free electron pair that binds the iron.
- **A reactive electrophile.** Thiophenes and furans are oxidised to epoxides, sulfoxides and other species with an electron-poor carbon that attacks nucleophilic groups of the protein (an amino acid side chain, for
  example) and attaches to it.
- **A modified heme.** Terminal alkynes and some other groups are oxidised to species that alkylate the porphyrin itself.

A **structural alert** is a substructure known to be connected with such bioactivation. It is a statistical warning and not a rule: most molecules that carry an alert are not time-dependent inhibitors, because
whether the reactive species forms depends on how readily the rest of the molecule is metabolised, and on whether the reactive species finds its target before it is quenched. The project encodes such
substructures as SMARTS patterns (section 0.2) and counts their occurrences in each molecule (the `tdi_alerts` block of the feature store, section 3.4). The cell checks the idea on the real labels: for each
of eleven of the alerts, the share of the labelled CYP3A4 compounds that are time-dependent inhibitors, among those that carry the alert and among those that do not.
"""),
    code(ALERTS),
    md("""
The project's feature store defines {{=len(_ALERT_SMARTS)}} alerts in all. The share of time-dependent inhibitors among all the labelled compounds is about {{=base_rate:.2f}}. A compound with the
methylenedioxyphenyl group is a time-dependent inhibitor in {{=alert_share['methylenedioxyphenyl']:.2f}} of {{=alert_n['methylenedioxyphenyl']}} cases, with a thiophene
in {{=alert_share['thiophene']:.2f}} of {{=alert_n['thiophene']}}, with a tertiary alkylamine in {{=alert_share['tertiary_alkylamine']:.2f}} of
{{=alert_n['tertiary_alkylamine']}}, and with a furan in {{=alert_share['furan']:.2f}} of {{=alert_n['furan']}}. The alerts are therefore
informative, some strongly, but none is a classifier on its own: most compounds that carry an alert are not time-dependent inhibitors, and rare alerts are carried by so few compounds (one by none) that their share cannot be estimated. That is the reason the alerts
enter the model as one block of features among many, and the reason section 5.2 and Part VII look at what happens when an alert is refined (the tertiary-amine alert had to exclude sulfonamides, enamines and
guanidines, which look like amines to a pattern but do not behave like them).
"""),
]

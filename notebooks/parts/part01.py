# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part I (first half): biology and chemistry foundations. How fast a drug leaves, enzymes, cytochrome P450, interactions, reversible inhibition."""

from notebooks.notebook_builder import code, figure, md

STYLE = '''
# The figures drawn by the cells of this part use the project's dark figure style.
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(ROOT / "docs/figures"))
from style import apply, tidy

theme = apply("dark")
'''

ELIMINATION = '''
# One dose, first-order elimination: C(t) = C0 exp(-k t). The elimination rate constant of the example is chosen so that the half-life is four hours.
half_life_h = 4.0
k = np.log(2) / half_life_h                   # per hour
t = np.linspace(0, 24, 241)
C0 = 1.0                                      # relative concentration right after the dose
with_inhibitor_k = k / 2                      # an inhibitor that halves the clearance halves k
print(f"k = {k:.4f} per hour; half-life {np.log(2) / k:.1f} h; with the inhibitor k = {with_inhibitor_k:.4f} per hour and half-life {np.log(2) / with_inhibitor_k:.1f} h")

# Repeated dosing every 12 hours: the concentration after n doses is the sum of the decaying remains of every earlier dose.
tau = 12.0
grid = np.linspace(0, 120, 1201)
def after_doses(rate):
    c = np.zeros_like(grid)
    for dose_time in np.arange(0, 120, tau):
        c += np.where(grid >= dose_time, C0 * np.exp(-rate * (grid - dose_time)), 0.0)
    return c
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.8))
left.plot(t, C0 * np.exp(-k * t), color=theme["blue"], lw=2.4, label="normal clearance")
left.plot(t, C0 * np.exp(-with_inhibitor_k * t), color=theme["red"], lw=2.4, label="clearance halved by an inhibitor")
left.axhline(0.5, color=theme["muted"], ls=":", lw=1)
left.set(xlabel="time after one dose (h)", ylabel="relative concentration C(t) / C0", title="one dose")
left.legend(frameon=False)
tidy(left, theme)
normal, inhibited = after_doses(k), after_doses(with_inhibitor_k)
right.plot(grid, normal, color=theme["blue"], lw=1.8)
right.plot(grid, inhibited, color=theme["red"], lw=1.8)
right.set(xlabel="time (h)", ylabel="relative concentration", title="a dose every 12 hours")
tidy(right, theme)
fig.tight_layout()
average_last = [x[grid >= 108].mean() for x in (normal, inhibited)]
print(f"mean concentration over the last dosing interval: {average_last[0]:.2f} without and {average_last[1]:.2f} with the inhibitor, a ratio of {average_last[1] / average_last[0]:.2f}")
'''

MICHAELIS = '''
# The scheme E + S <-> ES -> E + P integrated as differential equations, and compared with the Michaelis-Menten rate.
from scipy.integrate import solve_ivp

k1, k_1, kcat, E_total = 5.0, 40.0, 10.0, 0.01     # per uM per s, per s, per s, uM
KM = (k_1 + kcat) / k1
Vmax = kcat * E_total
def scheme(_, y, S0):
    E, S, ES, P = y
    bind, unbind, turn = k1 * E * S, k_1 * ES, kcat * ES
    return [-bind + unbind + turn, -bind + unbind, bind - unbind - turn, turn]
rows = []
for S0 in (1.0, 5.0, 10.0, 50.0, 200.0):
    run = solve_ivp(scheme, (0, 60), [E_total, S0, 0.0, 0.0], args=(S0,), method="LSODA", dense_output=True, rtol=1e-9, atol=1e-12)
    rate = kcat * run.sol(0.5)[2]                  # rate of product formation after the initial transient (E << S, so S has hardly changed)
    rows.append({"S0 (uM)": S0, "rate from the full scheme (uM/s)": rate, "Vmax S / (KM + S) (uM/s)": Vmax * S0 / (KM + S0)})
import pandas as pd

table = pd.DataFrame(rows)
table["relative difference"] = table.iloc[:, 1] / table.iloc[:, 2] - 1
print(f"KM = (k-1 + kcat) / k1 = {KM:.1f} uM, Vmax = kcat [E]total = {Vmax:.3f} uM/s")
show_table(table, floatfmt=".4f")
'''

SEQUENCE = '''
# The conserved heme-binding signature of the P450 family (F..G...C.G) in the sequence of CYP3A4, and the cysteine that binds the iron, found by the project's own code.
import re

from cypmode.validation.build_input import CYS_MOTIF_PATTERN, find_coordinating_cysteine
from cypmode.validation.constants import CYP3A4_SEQUENCE

match = re.search(CYS_MOTIF_PATTERN, CYP3A4_SEQUENCE)
position = find_coordinating_cysteine(CYP3A4_SEQUENCE)
print(f"CYP3A4 (UniProt P08684) has {len(CYP3A4_SEQUENCE)} amino acids; the signature {match.group(0)} starts at residue {match.start() + 1}; the cysteine that binds the iron is residue {position} ({CYP3A4_SEQUENCE[position - 1]}).")
'''

AUC_RATIO = '''
# The increase in exposure (AUC ratio) caused by a competitive inhibitor, for an enzyme that handles a fraction f_m of the drug's clearance.
from itertools import product

ratio = lambda fm, I_over_Ki: 1 / (fm / (1 + I_over_Ki) + (1 - fm))
rows = [{"[I] / Ki": x, **{f"f_m = {fm}": ratio(fm, x) for fm in (0.5, 0.9, 0.99)}} for x in (0.1, 1, 3, 10, 30, 100)]
show_table(pd.DataFrame(rows), floatfmt=".2f")
'''

TURNOVER = '''
# Time-dependent inhibition and enzyme turnover: the activity left at steady state is 1 / (1 + k_obs / k_deg). For an illustrative enzyme half-life of three days:
k_deg = np.log(2) / (3 * 24)                      # per hour
rows = [{"k_obs (per hour)": kobs, "activity left at steady state": 1 / (1 + kobs / k_deg), "fold increase in exposure if the enzyme clears all the drug": 1 + kobs / k_deg}
        for kobs in (0.001, 0.005, 0.02, 0.1)]
show_table(pd.DataFrame(rows), floatfmt=".3f")
'''

CHENG = '''
# The IC50 of a competitive inhibitor depends on the substrate concentration used: IC50 = Ki (1 + [S] / KM) (Cheng and Prusoff). With Ki = 1 uM and KM = 10 uM:
Ki, KM_assay = 1.0, 10.0
rows = [{"[S] / KM": s, "IC50 (uM)": Ki * (1 + s)} for s in (0.1, 0.5, 1.0, 5.0)]
show_table(pd.DataFrame(rows), floatfmt=".2f")
'''

CELLS = [
    md("""
# Part I: Biology and chemistry
"""),
    code(STYLE, hide_input=True),
    md("""
## 1.1 Drugs in the body: how fast a drug leaves

> **Field:** Pharmacology · Mathematics (differential equations)

Section 0.1 named the four stages of a drug's life in the body. This section puts one number on the last two, because every later argument about inhibition rests on it: how fast a drug
leaves the body.

Suppose a single dose has put a drug into the blood at time $t = 0$, giving a concentration $C_0$ (the mass of drug per volume of blood plasma). The liver removes the drug at a rate that is,
to a good approximation, proportional to how much drug is there: with twice as much drug present, twice as much is removed in each minute. (This holds while the enzymes are far from
saturated; section 1.2 says when it stops holding.) Write $C(t)$ for the concentration at time $t$, and $k$ for the constant of proportionality, called the **elimination rate constant**
and measured in 1/hour. The statement "the concentration falls at a rate proportional to itself" is the **differential equation**

$$\\frac{dC}{dt} = -k\\,C .$$

The left side is the rate of change of the concentration: the change of $C$ in a tiny time step $dt$, divided by that step. The minus sign says that the concentration falls. To solve the
equation, move everything that contains $C$ to the left and everything that contains $t$ to the right, $dC / C = -k\\,dt$, and add up (integrate) both sides from time $0$ to time $t$:

$$\\int_{C_0}^{C(t)} \\frac{dC'}{C'} = -k \\int_0^t dt' \\quad\\Longrightarrow\\quad \\ln C(t) - \\ln C_0 = -k\\,t \\quad\\Longrightarrow\\quad C(t) = C_0\\,e^{-k t}.$$

The symbol $\\ln$ is the natural logarithm, the inverse of the exponential function $e^{x}$ where $e = 2.718\\ldots$, and the primes on $C'$ and $t'$ only mark that these are the variables being
added up, not the end points. The concentration therefore decays exponentially. The **half-life** $t_{1/2}$ is the time it takes the concentration to halve. Setting $C = C_0 / 2$ in the solution
gives $e^{-k\\,t_{1/2}} = 1/2$, that is

$$t_{1/2} = \\frac{\\ln 2}{k}.$$

Where does $k$ come from? The drug is spread through a volume $V$ (the volume of fluid in which it distributes), and the body removes it at a rate $\\mathrm{CL}\\,C$, where $\\mathrm{CL}$ is the
**clearance**: the volume of blood that is completely cleared of the drug in one unit of time (units: litres per hour). The amount of drug in the body is $V\\,C$, so $V\\,dC/dt = -\\mathrm{CL}\\,C$, and
comparing with the equation above gives

$$k = \\frac{\\mathrm{CL}}{V}.$$

The clearance of most drugs is, to a large extent, the work of the CYP enzymes of the liver. If an inhibitor lowers the clearance by a factor $f$, then $k$ falls to $k / f$ and the half-life
grows to $f\\,t_{1/2}$: the drug stays longer. The cell below shows the effect for an example drug with a half-life of four hours, whose clearance is halved.
"""),
    code(ELIMINATION),
    md("""
The curves show what the equations promise. With the clearance halved, the half-life doubles. The right panel shows the consequence that matters for patients, who take a drug repeatedly.
When one dose is taken every $\\tau$ hours, the amount of drug entering the body in each interval is a fraction $F$ of the dose $D$ (the fraction that is absorbed). After many doses the body
reaches a **steady state** in which the same amount leaves in each interval as enters. The amount that leaves in an interval is the clearance times the **area under the concentration curve**
(the integral of the concentration over time, written $\\mathrm{AUC}$) over that interval, which is $\\mathrm{CL}\\,\\bar C\\,\\tau$ when $\\bar C$ is the average concentration, so

$$F\\,D = \\mathrm{CL}\\,\\bar C\\,\\tau \\quad\\Longrightarrow\\quad \\bar C = \\frac{F\\,D}{\\mathrm{CL}\\,\\tau}.$$

The average concentration is inversely proportional to the clearance: halving the clearance doubles the average concentration at steady state, as the cell printed. A dose that was safe
alone becomes twice as strong when another drug inhibits the enzyme that removes it. How much stronger depends on how strongly the other drug inhibits the enzyme, which is what the rest
of this part is about.
"""),
    md("""
## 1.2 Enzymes: what a catalyst does, and the Michaelis-Menten equation

> **Field:** Biochemistry · Mathematics (rate equations)

An enzyme $E$ binds the molecule it acts on, the **substrate** $S$, to form a complex $ES$. The complex either falls apart again or converts the substrate into a product $P$ and releases the enzyme
unchanged, ready for another substrate molecule:

$$E + S \\;\\underset{k_{-1}}{\\overset{k_{1}}{\\rightleftharpoons}}\\; ES \\;\\overset{k_{\\mathrm{cat}}}{\\longrightarrow}\\; E + P .$$

Each arrow carries a **rate constant**: $k_1$ for binding (units: 1/(concentration x time)), $k_{-1}$ for falling apart and $k_{\\mathrm{cat}}$ for the chemical step (both 1/time). Square brackets
denote concentrations, and $t$ is the time. By the law of mass action the complex forms at the rate $k_1 [E][S]$ and disappears at the rate $(k_{-1} + k_{\\mathrm{cat}})[ES]$. After a very short time the
concentration of the complex stops changing, because it forms as fast as it disappears (the **steady-state assumption** of Briggs and Haldane, 1925):

$$\\frac{d[ES]}{dt} = 0 \\quad\\Longrightarrow\\quad k_1 [E][S] = (k_{-1} + k_{\\mathrm{cat}})\\,[ES].$$

The free enzyme is not known directly, but the total amount of enzyme is fixed, $[E]_T = [E] + [ES]$. Replacing $[E]$ by $[E]_T - [ES]$ and solving for $[ES]$ gives

$$[ES] = \\frac{[E]_T\\,[S]}{K_M + [S]}, \\qquad K_M = \\frac{k_{-1} + k_{\\mathrm{cat}}}{k_1}.$$

The constant $K_M$ is called the **Michaelis constant**; it has the units of a concentration. The rate $v$ at which product appears is $k_{\\mathrm{cat}}[ES]$, so

$$v = \\frac{V_{\\max}\\,[S]}{K_M + [S]}, \\qquad V_{\\max} = k_{\\mathrm{cat}}\\,[E]_T .$$

This is the **Michaelis-Menten equation** (Michaelis and Menten, 1913, derived there from a different assumption and re-derived in this form by Briggs and Haldane). Two limits explain its shape.
When $[S] \\ll K_M$ the denominator is almost $K_M$ and $v \\approx (V_{\\max}/K_M)[S]$: the rate is proportional to the substrate concentration, which is the first-order behaviour that section 1.1
assumed, and the ratio $V_{\\max}/K_M$ is the **intrinsic clearance** of the enzyme. When $[S] \\gg K_M$ the rate approaches $V_{\\max}$ whatever the concentration: every enzyme molecule is busy and the
enzyme is **saturated**. At $[S] = K_M$ the rate is exactly $V_{\\max}/2$. The figure shows the curve; its inset shows the older way of plotting the same data, $1/v$ against $1/[S]$ (Lineweaver and Burk),
which turns the curve into a straight line with intercept $1/V_{\\max}$ and slope $K_M/V_{\\max}$.
"""),
    figure("kin_michaelis_menten"),
    md("""
The steady-state assumption is an approximation, and the next cell checks it. It integrates the three differential equations of the scheme numerically (for a made-up enzyme with
$k_1 = 5$ per micromolar per second, $k_{-1} = 40$ per second, $k_{\\mathrm{cat}} = 10$ per second and 0.01 micromolar of enzyme) and compares the rate of product formation, taken half a second
after mixing, with the Michaelis-Menten equation.
"""),
    code(MICHAELIS),
    md("""
The two agree to within a fraction of a percent at every substrate concentration, because the enzyme is present in a much smaller amount than the substrate. That condition is the one under which the
equation is valid, and it holds in a typical inhibition assay, where the enzyme is present at nanomolar and the tested compound at micromolar concentrations.
"""),
    md("""
## 1.3 Cytochrome P450: the heme, the iron and the catalytic cycle

> **Field:** Biochemistry · Chemistry (the iron of the heme)

The name of the family comes from a pigment (P) first seen in liver cell fragments in 1958 and characterised by Omura and Sato (1962, 1964): it absorbs light of wavelength 450 nanometres (hence 450) when
its iron is reduced and bound to carbon monoxide. The pigment turned out to be a protein with a heme at its centre. A P450 enzyme is a **monooxygenase**: it takes one of the two oxygen atoms of an oxygen
molecule and inserts it into a substrate $\\mathrm{RH}$ (a carbon-hydrogen bond becomes a carbon-hydroxyl bond), reducing the other to water, using two electrons supplied by a second protein that
receives them from NADPH:

$$\\mathrm{RH} + \\mathrm{O_2} + \\mathrm{NADPH} + \\mathrm{H^+} \\;\\longrightarrow\\; \\mathrm{ROH} + \\mathrm{H_2O} + \\mathrm{NADP^+}.$$

(NADPH is a small molecule that carries electrons in cells; $\\mathrm{H^+}$ is a proton.) The heme (iron protoporphyrin IX) is a flat ring of carbon and nitrogen atoms, the porphyrin, with an iron atom in the
middle held by its four nitrogens. The iron has two more places to bind, above and below the ring. Below the ring a sulfur atom of a cysteine of the protein binds to it, the **axial cysteine**; above the ring
is the site where oxygen, and everything this project is concerned with, binds. The cysteine is part of a stretch of amino acids that is nearly the same in all P450 enzymes, a signature that the project's code uses
to find it in any sequence:
"""),
    code(SEQUENCE),
    md("""
**The catalytic cycle.** The enzyme performs the reaction in eight steps, drawn in the figure around the real heme of a CYP3A4 crystal structure. In the resting state the iron holds a water molecule as its sixth
ligand (1). The substrate enters the pocket and pushes the water out (2). The iron receives an electron (3) and binds an oxygen molecule (4); a second electron (5) and two protons (6) split the oxygen: one atom leaves
as water, the other stays on the iron as a highly reactive species called Compound I. Compound I takes a hydrogen atom from the substrate (7), and the resulting radical picks up the hydroxyl from the iron
(8, "oxygen rebound"); the product leaves and a water molecule returns. The same cycle explains how an inhibitor can act. A molecule that competes with the substrate for the pocket slows step 2. A molecule that
binds the iron itself keeps oxygen from step 4. A molecule that the enzyme converts, in steps 6 to 8, into something reactive that attaches to the heme or to the protein switches the enzyme off for good:
that is the time-dependent inhibition this project predicts (section 1.6).
"""),
    figure("bio_p450_cycle"),
    md("""
**The two enzymes of this project.** Humans have 57 CYP enzymes; a handful of them do most of the work of drug metabolism. **CYP3A4** has a large, flexible active site and accepts molecules of very
different sizes and shapes: crystal structures with different ligands show a pocket that adapts its volume to the guest (Williams and co-workers, 2004, with metyrapone and progesterone, and the
cryo-EM preprint of section 0.2). **CYP2D6** has a narrower pocket with two negatively charged residues near the active site (an aspartate at position 301 and a glutamate at position 216; Rowland and
co-workers, 2006), and it prefers molecules with a positively charged (basic) nitrogen, which it can hold electrostatically. The difference between the two pockets is one reason a model that works for
one enzyme need not work for the other, and the project's results for the two enzymes differ accordingly (Part VIII).
"""),
    md("""
## 1.4 Drug-drug interactions and why inhibition matters clinically

> **Field:** Pharmacology

Section 1.1 showed that a lower clearance means a longer half-life and a higher average concentration. This section turns that into a number that tells how large an interaction can be, in terms of the
quantities that the assays of this project measure.

Let a drug be removed by two routes: the enzyme in question, with clearance $\\mathrm{CL}_E$, and everything else, with clearance $\\mathrm{CL}_{\\mathrm{other}}$. The total clearance is the sum. The
**fraction metabolised** by the enzyme is $f_m = \\mathrm{CL}_E / (\\mathrm{CL}_E + \\mathrm{CL}_{\\mathrm{other}})$. After a single dose the **area under the concentration curve** ($\\mathrm{AUC}$, the integral of
$C(t)$ over time) is $F\\,D/\\mathrm{CL}$, so exposure is inversely proportional to the total clearance. At concentrations far below $K_M$ the clearance of the enzyme is its intrinsic clearance
$V_{\\max}/K_M$ (section 1.2). A competitive inhibitor at concentration $[I]$ multiplies the apparent $K_M$ by $1 + [I]/K_i$ (derived in section 1.5), so the clearance of the enzyme falls to
$\\mathrm{CL}_E/(1 + [I]/K_i)$. For a constant concentration of the inhibitor at the enzyme, dividing the exposure with the inhibitor by the exposure without it gives the **ratio of the areas under the curve**:

$$\\frac{\\mathrm{AUC}_{\\mathrm{with\\ inhibitor}}}{\\mathrm{AUC}} = \\frac{1}{\\dfrac{f_m}{1 + [I]/K_i} + (1 - f_m)} .$$

If the enzyme removes all of the drug ($f_m = 1$), the ratio is $1 + [I]/K_i$: an inhibitor at ten times its $K_i$ raises the exposure elevenfold. If the enzyme removes only half ($f_m = 0.5$), the
other route caps the effect at a factor of two however strong the inhibitor is. The cell tabulates the relation.
"""),
    code(AUC_RATIO),
    md("""
Regulators have made this reasoning routine: drug developers are expected to measure the inhibition of the main CYP enzymes in vitro and to compare the concentration of the inhibitor that reaches the liver with
its $K_i$ or its **IC50**, the concentration that halves the enzyme's activity (section 1.5), as the FDA's 2020 guidance on in vitro interaction studies and the European Medicines Agency's 2012 guideline on drug interactions ask. Ketoconazole, a strong inhibitor of CYP3A4, and ritonavir, which
is given in small doses with other HIV drugs precisely because it inhibits CYP3A4 and so raises their levels, are standard examples.

For a time-dependent inhibitor the argument changes in a way that explains why the interaction can last for days. The body makes new enzyme at a constant rate $k_{\\mathrm{syn}}$ and breaks it down at a
rate $k_{\\mathrm{deg}}\\,E$, which gives a steady-state enzyme amount $E_0 = k_{\\mathrm{syn}}/k_{\\mathrm{deg}}$. An inhibitor that inactivates the enzyme at the rate $k_{\\mathrm{obs}}\\,E$ (section 1.6 derives
$k_{\\mathrm{obs}}$) adds a second loss, so that $dE/dt = k_{\\mathrm{syn}} - (k_{\\mathrm{deg}} + k_{\\mathrm{obs}})E$ and the new steady state is

$$E_{\\mathrm{ss}} = \\frac{k_{\\mathrm{syn}}}{k_{\\mathrm{deg}} + k_{\\mathrm{obs}}} = \\frac{E_0}{1 + k_{\\mathrm{obs}}/k_{\\mathrm{deg}}} .$$

The activity falls by the factor $1 + k_{\\mathrm{obs}}/k_{\\mathrm{deg}}$ and, because the enzyme is only replaced at the rate $k_{\\mathrm{syn}}$, it takes about the enzyme's own half-life (days) to come back
after the last dose. The cell shows how large the effect becomes when the inactivation rate is compared with the turnover of an enzyme with a half-life of three days (an illustrative value).
"""),
    code(TURNOVER),
    md("""
The conclusion for this project is practical. An inhibitor is judged by how potent it is (the lower its IC50 or $K_i$, the larger the factor above) and, for a time-dependent inhibitor, also by how fast it
inactivates the enzyme. Predicting from a molecule's structure whether it is a time-dependent inhibitor is therefore predicting a property that changes the size and the duration of an interaction.
"""),
    md("""
## 1.5 Reversible inhibition: competitive, non-competitive and mixed

> **Field:** Biochemistry · Mathematics (rate equations)

A reversible inhibitor $I$ binds the enzyme and lets go again, with no chemical change. It can bind the free enzyme, forming $EI$, with the dissociation constant $K_i$ (the concentration of $I$ at which half of
the enzyme is bound in the absence of substrate); it can bind the complex $ES$, forming $ESI$, with the dissociation constant $K_i'$; or both. The rates below are derived under the same steady-state
assumption as in section 1.2, with the added fast equilibria $[E][I] = K_i [EI]$ and $[ES][I] = K_i' [ESI]$. Conservation of enzyme now reads $[E]_T = [E] + [ES] + [EI] + [ESI]$. Define the two factors

$$\\alpha = 1 + \\frac{[I]}{K_i}, \\qquad \\alpha' = 1 + \\frac{[I]}{K_i'} .$$

Writing every enzyme form in terms of $[E]$ and solving for the rate $v = k_{\\mathrm{cat}}[ES]$ (the steps are those of section 1.2 with the extra terms collected in $\\alpha$ and $\\alpha'$) gives

$$v = \\frac{V_{\\max}\\,[S]}{\\alpha\\,K_M + \\alpha'\\,[S]} = \\frac{V_{\\max}/\\alpha'\\;[S]}{(\\alpha/\\alpha')\\,K_M + [S]} .$$

The second form is again a Michaelis-Menten equation, with an apparent maximum rate $V_{\\max}/\\alpha'$ and an apparent Michaelis constant $(\\alpha/\\alpha')K_M$. The type of inhibition is the pattern of how these two change:

- **Competitive**: the inhibitor binds only the free enzyme, so $K_i' \\to \\infty$ and $\\alpha' = 1$. The apparent $K_M$ grows by $\\alpha$ and $V_{\\max}$ is unchanged: enough substrate wins the competition. This is the case of
  section 1.4 and of molecules that sit in the substrate pocket.
- **Non-competitive**: the inhibitor binds the enzyme and the complex equally well, $K_i = K_i'$, so $\\alpha = \\alpha'$. $K_M$ is unchanged and $V_{\\max}$ falls by $\\alpha$.
- **Uncompetitive**: the inhibitor binds only the complex, $\\alpha = 1$. Both apparent constants fall by the same factor $\\alpha'$.
- **Mixed**: any other case, with $K_i \\ne K_i'$ and both finite.

The figure draws the four cases for one choice of numbers.
"""),
    figure("kin_inhibition_types"),
    md("""
**The IC50 and the dependence on the substrate.** The experiments of this project do not measure $K_i$. They measure the concentration of the inhibitor that halves the rate, the **IC50**, at one fixed
substrate concentration. For a competitive inhibitor the rate is $v_0\\,\\dfrac{1}{1 + [I]/\\bigl(K_i(1 + [S]/K_M)\\bigr)}$ (setting $\\alpha' = 1$ in the equation above and dividing by the rate $v_0$ without inhibitor, which
is $V_{\\max}[S]/(K_M + [S])$). The rate is halved when $[I] = K_i\\,(1 + [S]/K_M)$, so

$$\\mathrm{IC}_{50} = K_i\\left(1 + \\frac{[S]}{K_M}\\right) \\qquad\\text{(Cheng and Prusoff, 1973)}.$$

The IC50 is not a property of the inhibitor alone: it grows with the substrate concentration of the assay. Only when $[S] \\ll K_M$ does it equal $K_i$. The cell shows the dependence for $K_i = 1$ micromolar.
"""),
    code(CHENG),
    md("""
For the purposes of this project the IC50, measured in the same conditions for every compound, is a fair measure to compare compounds with one another, and that is how section 1.7 uses it.
"""),
]

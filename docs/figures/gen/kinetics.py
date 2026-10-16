"""Kinetics figures, drawn from the equations (no data files): Michaelis-Menten, reversible inhibition, time-dependent inhibition,
dose-response and the two assay arms.

Time-dependent inhibition is integrated from its differential equations:
    E + I <-> EI   (K_I = k_off / k_on)       EI -> E-I*   (k_inact)
The activity left after pre-incubation is E + EI (the free enzyme and the reversible complex); E-I* is dead.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.integrate import solve_ivp

from catalog import figure
from style import apply, tidy

GROUP = "kinetics"
K_ON, K_OFF, K_INACT = 50.0, 500.0, 0.8  # per uM per min, per min, per min  ->  K_I = 10 uM
K_I = K_OFF / K_ON


def mm(s, vmax=1.0, km=10.0):
    return vmax * s / (km + s)


@figure("kin_michaelis_menten", GROUP, "Michaelis-Menten saturation and its double-reciprocal form",
        "Reaction rate v against substrate concentration [S]: v = Vmax [S] / (Km + [S]). The rate approaches Vmax; Km is the concentration that gives Vmax/2. "
        "The inset plots 1/v against 1/[S] (Lineweaver-Burk), where the same data fall on a straight line with slope Km/Vmax and intercept 1/Vmax.",
        section="1.2")
def michaelis_menten(theme_name):
    th = apply(theme_name)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    s = np.linspace(0, 100, 400)
    ax.plot(s, mm(s), color=th["blue"], lw=2.4)
    ax.axhline(1.0, color=th["muted"], ls="--", lw=1)
    ax.axhline(0.5, color=th["muted"], ls=":", lw=1)
    ax.axvline(10, color=th["muted"], ls=":", lw=1)
    ax.text(98, 1.02, r"$V_{max}$", ha="right", va="bottom", color=th["muted"])
    ax.text(98, 0.52, r"$V_{max}/2$", ha="right", va="bottom", color=th["muted"])
    ax.text(11, 0.04, r"$K_m$", color=th["muted"])
    ax.set(xlabel="substrate concentration [S] (uM)", ylabel="rate v (relative)", xlim=(0, 100), ylim=(0, 1.15))
    tidy(ax, th)
    inset = ax.inset_axes([0.5, 0.12, 0.43, 0.42])
    inv = np.linspace(0.0, 0.5, 50)
    inset.plot(inv, 1 + 10 * inv, color=th["red"], lw=2)
    inset.set(xlabel="1/[S]", ylabel="1/v")
    inset.tick_params(labelsize=8)
    tidy(inset, th)
    fig.tight_layout()
    return fig


@figure("kin_inhibition_types", GROUP, "Reversible inhibition: competitive, non-competitive, uncompetitive and mixed",
        "The same enzyme without inhibitor (solid) and with a fixed inhibitor concentration (dashed). With alpha = 1 + [I]/Ki and alpha' = 1 + [I]/Ki', "
        "the apparent constants are Vmax/alpha' and Km alpha/alpha'. Competitive (alpha > 1, alpha' = 1): Km rises, Vmax unchanged. "
        "Non-competitive (alpha = alpha'): Vmax falls, Km unchanged. Uncompetitive (alpha = 1, alpha' > 1): both fall by the same factor. "
        "Mixed (alpha > alpha' > 1): Km rises and Vmax falls.",
        section="1.5")
def inhibition_types(theme_name):
    th = apply(theme_name)
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.7), sharey=True)
    s = np.linspace(0, 100, 300)
    cases = [("Competitive", 3.0, 1.0), ("Non-competitive", 3.0, 3.0), ("Uncompetitive", 1.0, 3.0), ("Mixed", 4.0, 2.0)]
    for ax, (name, alpha, alpha_prime) in zip(axes, cases, strict=True):
        ax.plot(s, mm(s), color=th["blue"], lw=2.2, label="no inhibitor")
        ax.plot(s, mm(s, 1.0 / alpha_prime, 10.0 * alpha / alpha_prime), color=th["red"], lw=2.2, ls="--", label="with inhibitor")
        ax.set_title(f"{name}\n" + rf"$\alpha$ = {alpha:g}, $\alpha'$ = {alpha_prime:g}", fontsize=10)
        ax.set_xlabel("[S] (uM)")
        tidy(ax, th)
    axes[0].set_ylabel("rate v (relative)")
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def integrate(inhibitor_um: float, t_max: float = 12.0, n: int = 240):
    def rhs(_, y):
        e, ei, dead = y
        bind = K_ON * inhibitor_um * e - K_OFF * ei
        return [-bind, bind - K_INACT * ei, K_INACT * ei]

    t = np.linspace(0, t_max, n)
    sol = solve_ivp(rhs, (0, t_max), [1.0, 0.0, 0.0], t_eval=t, method="LSODA", rtol=1e-9, atol=1e-12)
    return t, sol.y[0] + sol.y[1]  # activity left = free enzyme + reversible complex


def k_obs_of(inhibitor_um):
    return K_INACT * inhibitor_um / (K_I + inhibitor_um)


def tdi_animation(theme_name: str) -> Path:
    """A looping gif of the progress curves appearing as the inhibitor concentration rises (frames drawn with the light theme)."""
    th = apply(theme_name)
    out = Path(__file__).resolve().parents[1] / GROUP / f"kin_tdi_progress-anim-{theme_name}.gif"
    concentrations = [0, 2, 5, 10, 20, 50, 100]
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(concentrations)))
    curves = [integrate(float(c)) for c in concentrations]
    frames = []
    for upto in range(1, len(concentrations) + 1):
        for fraction in (0.35, 0.7, 1.0):
            fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
            for i in range(upto):
                t, activity = curves[i]
                last = i == upto - 1
                stop = int(len(t) * (fraction if last else 1.0))
                ax.semilogy(t[:stop], np.clip(activity[:stop], 1e-3, 1.0), color=colors[i], lw=2.2, label=f"{concentrations[i]} uM")
            ax.set(xlim=(0, 12), ylim=(1e-3, 1.3), xlabel="pre-incubation time (min)", ylabel="activity left", title=r"$E + I \rightleftharpoons EI \rightarrow E\text{-}I^*$")
            ax.legend(title="[I]", frameon=False, fontsize=7, loc="lower left", ncol=2)
            tidy(ax, th)
            fig.canvas.draw()
            frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3]))
            plt.close(fig)
    frames[0].save(out, save_all=True, append_images=frames[1:] + [frames[-1]] * 6, duration=140, loop=0)
    return out


@figure("kin_tdi_progress", GROUP, "Time-dependent inhibition: progress curves and the observed rate",
        "Left: enzyme activity left after pre-incubation with inhibitor, integrated from E + I <-> EI -> E-I*; each straight segment on the log scale has slope -k_obs. "
        "Right: k_obs rises with inhibitor concentration and saturates at k_inact: k_obs = k_inact [I] / (K_I + [I]). K_I is the concentration giving k_inact/2.",
        section="1.6", animation=tdi_animation)
def tdi_progress(theme_name):
    th = apply(theme_name)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    concentrations = [0, 2, 5, 10, 20, 50, 100]
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(concentrations)))
    for conc, color in zip(concentrations, colors, strict=True):
        t, activity = integrate(float(conc))
        a.semilogy(t, np.clip(activity, 1e-3, 1.0), color=color, lw=2, label=f"{conc} uM")
    a.set(xlabel="pre-incubation time (min)", ylabel="activity left (fraction)", ylim=(1e-3, 1.3))
    a.legend(title="[I]", frameon=False, fontsize=8, ncol=2, loc="lower left")
    tidy(a, th)
    x = np.linspace(0, 120, 300)
    b.plot(x, k_obs_of(x), color=th["red"], lw=2.4)
    for conc, color in zip(concentrations[1:], colors[1:], strict=True):
        t, activity = integrate(float(conc))
        window = (t > 2) & (activity > 2e-3)
        slope = -np.polyfit(t[window], np.log(activity[window]), 1)[0]
        b.plot([conc], [slope], "o", color=color, ms=6)
    b.axhline(K_INACT, color=th["muted"], ls="--", lw=1)
    b.axhline(K_INACT / 2, color=th["muted"], ls=":", lw=1)
    b.axvline(K_I, color=th["muted"], ls=":", lw=1)
    b.text(118, K_INACT + 0.015, r"$k_{inact}$", ha="right", color=th["muted"])
    b.text(K_I + 2, 0.02, r"$K_I$", color=th["muted"])
    b.set(xlabel="inhibitor concentration [I] (uM)", ylabel=r"observed rate $k_{obs}$ (1/min)", xlim=(0, 120), ylim=(0, 0.95))
    tidy(b, th)
    fig.tight_layout()
    return fig


def hill(x, ic50, slope):
    return 1.0 / (1.0 + (x / ic50) ** slope)


@figure("kin_dose_response", GROUP, "Dose-response, the Hill equation and the pIC50 scale",
        "Fraction of activity left against inhibitor concentration on a log axis: y = 1 / (1 + ([I]/IC50)^h). The curve passes 50% at the IC50; the Hill slope h sets how steep it is. "
        "pIC50 = -log10(IC50 in mol/L), so an IC50 of 1 uM is pIC50 6 and each unit is a ten-fold change in potency.",
        section="1.7")
def dose_response(theme_name):
    th = apply(theme_name)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.logspace(-3, 3, 400)
    for slope, color in ((0.6, th["teal"]), (1.0, th["blue"]), (2.0, th["red"])):
        a.semilogx(x, hill(x, 1.0, slope), color=color, lw=2.3, label=f"h = {slope}")
    a.axhline(0.5, color=th["muted"], ls=":", lw=1)
    a.axvline(1.0, color=th["muted"], ls=":", lw=1)
    a.text(1.1, 0.03, "IC50", color=th["muted"])
    a.set(xlabel="inhibitor concentration (uM)", ylabel="activity left (fraction)")
    a.legend(frameon=False)
    tidy(a, th)
    ic50_um = np.array([100, 10, 1, 0.1, 0.01])
    pic50 = -np.log10(ic50_um * 1e-6)
    b.barh([f"{v:g} uM" for v in ic50_um], pic50, color=th["blue"])
    for i, v in enumerate(pic50):
        b.text(v + 0.08, i, f"{v:.0f}", va="center")
    b.set(xlabel="pIC50", xlim=(0, 9.5))
    tidy(b, th, grid=False)
    fig.tight_layout()
    return fig


@figure("kin_assay_arms", GROUP, "The two assay arms and the shift",
        "The same compound measured twice: directly (no pre-incubation, d) and under the time-dependent condition (pre-incubated, t). "
        "A time-dependent inhibitor becomes more potent when it has time to act, so the second curve sits to the left; the distance between the two pIC50 values is the shift. "
        "The challenge calls a compound positive when t - max(d, 4) exceeds log10 2 (a two-fold gain in potency).",
        section="1.8")
def assay_arms(theme_name):
    th = apply(theme_name)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.logspace(-3, 3, 500)
    d_ic50, shift = 3.0, 0.9  # uM; pIC50 units
    t_ic50 = d_ic50 / 10 ** shift
    ax.semilogx(x, hill(x, d_ic50, 1.0), color=th["blue"], lw=2.4, label="direct (d)")
    ax.semilogx(x, hill(x, t_ic50, 1.0), color=th["red"], lw=2.4, label="time-dependent (t)")
    for v, color in ((d_ic50, th["blue"]), (t_ic50, th["red"])):
        ax.axvline(v, color=color, ls=":", lw=1)
    ax.annotate("", xy=(t_ic50, 0.5), xytext=(d_ic50, 0.5), arrowprops={"arrowstyle": "<->", "color": th["text"], "lw": 1.6})
    ax.text(np.sqrt(d_ic50 * t_ic50), 0.55, "shift s = t - d", ha="center", color=th["text"])
    ax.set(xlabel="inhibitor concentration (uM)", ylabel="activity left (fraction)")
    ax.legend(frameon=False, loc="lower left")
    tidy(ax, th)
    fig.tight_layout()
    return fig

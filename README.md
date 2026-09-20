# CYPMode

Mode-aware reinterpretation of TDC's binary CYP450 inhibition benchmarks.

## Problem

[Therapeutics Data Commons](https://tdc.hub.biolib.com) ships binary
inhibitor/non-inhibitor labels for the five major cytochrome P450 isoforms
(`CYP1A2_Veith`, `CYP2C9_Veith`, `CYP2C19_Veith`, `CYP2D6_Veith`,
`CYP3A4_Veith`, from Veith et al., *Nat. Biotechnol.* 2009), and every
ADMET model trained on them inherits the same blind spot: the label says
*whether* a compound inhibits a CYP, not *how*. Pharmacologically, that
"how" is the difference between Type I inhibition (reversible, substrate
in the active site, the heme's axial water stays put) and Type II
inhibition (a nitrogen lone pair coordinates directly to the heme iron,
displacing that water) — a much tighter, often far more clinically
significant interaction (de Groot, *Drug Discov. Today*, 2006, 11:601-606).
A binary "1" collapses both into the same bit.

Checked this against the data itself, not just the framing: across all
five isoforms, only 26-31% of TDC's labeled inhibitors carry any
heme-coordinating motif at all (`cypmode/data/panel.py`,
`summarize_panel()` — CYP1A2 28.1%, CYP2C9 30.0%, CYP2C19 28.2%, CYP2D6
26.5%, CYP3A4 30.6%, n=2,514-5,829 labeled inhibitors per isoform). Most
of the panel's "inhibitor" label describes something a motif screen
wouldn't expect to coordinate the heme directly — which is exactly the
information a plain binary label discards.

This project is scoped as a direct extension of a newly solved CYP3A4
cryo-EM structure paper (ketoconazole, ritonavir, azamulin, and
vardenafil bound), not a general re-annotation of the whole panel by
assertion: `cypmode/validation/structures.py` checks the motif screen's
calls against physics-based structure prediction (Boltz-2) for exactly
the four compounds that paper solved structures for, and no others —
full-panel physics-based validation across ~62,000 compounds would take
on the order of 2,000 GPU-hours on this hardware, which isn't a rerun of
that paper's method, it's a different, unvalidated claim.

## What's here

- `cypmode/metrics/motifs.py` — a fast RDKit/SMARTS screen for the classic
  heme-coordinating substructures (imidazole, triazole, pyridine,
  thiazole), in the structural-alert tradition of Leach, *Molecular
  Modelling: Principles and Applications* (2001, ch. 12, section 12.2).
  Seconds for the whole panel, not a mechanism predictor on its own.
- `cypmode/data/panel.py` — loads and mechanism-annotates TDC's five-isoform
  CYP inhibition panel.
- `cypmode/validation/structures.py` — parses Boltz-2's predicted structures
  (heme explicitly modeled and covalently bonded to the real axial
  cysteine, not left out) and measures the heme iron-to-ligand-nitrogen
  distance for the four reference compounds, to check the motif screen
  against physics rather than against itself.
- `app.py` — a Streamlit tool: paste a compound, get the motif call and,
  for the four reference compounds, the structural check.
- `docs/figures/generate_validation_figure.py` — regenerates the Fe-N bar
  chart below (light and dark variants) from
  `data/boltz_test/validation_summary.json`; needs `pip install matplotlib`,
  not otherwise a project dependency.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows (cmd/PowerShell): .venv\Scripts\activate
pip install -e .
```

The five-isoform panel is cached locally under `data/tdc_cache/`, so the
default install (RDKit + pandas + Streamlit) is enough to run
`cypmode.data.panel` and the app's motif screen without a network call.
To refresh that cache from source instead, install the `tdc` extra plus
`PyTDC` itself as a separate, `--no-deps` step (PyTDC's own dependency tree
pulls in packages that fail to build on Windows and aren't needed for this
project's usage of it):

```bash
pip install -e ".[tdc]"
pip install --no-deps "PyTDC==1.1.14"
```

`pip install -e ".[validate]"` adds Biopython, needed only to parse the
Boltz-2 structures in `cypmode/validation/structures.py`.

## Run the app

```bash
streamlit run app.py
```

## Try it online

*Hosted demo link pending — `requirements.txt` is prepped for Hugging Face
Spaces (Streamlit SDK) or Streamlit Community Cloud; `app.py`'s Structural
validation tab reads the checked-in
`data/boltz_test/validation_summary.json` rather than needing a local
Boltz-2 run, so it works identically hosted or local.*

## Validation against real structures

Boltz-2 ([jwohlwend/boltz](https://github.com/jwohlwend/boltz), MIT) is
used to predict CYP3A4-ligand complexes for the four compounds a recent
cryo-EM paper (co-authored by Pat Walters) solved structures for. The
first attempt at this left the heme cofactor out of the model entirely —
caught before drawing any conclusion from it, because an apo-protein
docking pose can't confirm or refute heme coordination if there's no
heme in the structure. The corrected input explicitly includes heme
(PDB ligand code `HEM`) as a third chain, covalently bonded
(`constraints: bond`) to Cys442 — the real axial thiolate ligand,
located by finding the conserved P450 signature motif
`FxxGxxxCxG` in the UniProt P08684 sequence, not assumed from memory. No
constraint is placed on the *inhibitor's* position relative to the heme,
so a short predicted iron-to-ligand-nitrogen distance is a genuine,
unforced structural prediction, not something the input forced to
happen.

Results, from `cypmode/validation/structures.py`'s `summarize_compound()`
run against the real local output in `data/boltz_test/out/`:

| Compound | Motif call | Fe-N distance | Coordinated | `affinity_probability_binary` | `confidence_score` |
|---|---|---|---|---|---|
| Ketoconazole | Type II (imidazole) | 2.05 Å | yes | 0.86 | 0.92 |
| Ritonavir | Type II (thiazole) | 2.19 Å | yes | 0.40 | 0.92 |
| Azamulin | Type II (triazole) | 13.41 Å | no | 0.73 | 0.93 |
| Vardenafil | Type II (imidazole_Nsub) | 4.84 Å | no | 0.42 | 0.93 |

![Bar chart of heme iron to closest ligand nitrogen distance for ketoconazole, ritonavir, azamulin, and vardenafil, with a dashed line at the 2.6 Å dative-coordination cutoff. Ketoconazole (2.05 Å) and ritonavir (2.19 Å) fall below the cutoff and are colored as coordinated; azamulin (13.41 Å) and vardenafil (4.84 Å) fall above it and are colored as not coordinated.](docs/figures/fe_n_validation-light.svg#gh-light-mode-only)
![Bar chart of heme iron to closest ligand nitrogen distance for ketoconazole, ritonavir, azamulin, and vardenafil, with a dashed line at the 2.6 Å dative-coordination cutoff. Ketoconazole (2.05 Å) and ritonavir (2.19 Å) fall below the cutoff and are colored as coordinated; azamulin (13.41 Å) and vardenafil (4.84 Å) fall above it and are colored as not coordinated.](docs/figures/fe_n_validation-dark.svg#gh-dark-mode-only)

Two of the motif screen's four calls hold up structurally, and the other
two turn out to be informative in different ways, not simple failures:

- **Ketoconazole and ritonavir** land inside dative-bond range (2.0-2.3 Å
  is the range solved Type II CYP structures show — Poulos & Johnson,
  ch. 3 in Ortiz de Montellano (ed.), *Cytochrome P450*, 3rd ed., 2005),
  with no constraint forcing that outcome. Ritonavir's is the compound the
  thiazole motif was added for (see git history) — its coordination here
  confirms that fix structurally, not just by substructure match. One
  honest wrinkle: Boltz-2's own `affinity_probability_binary` for
  ritonavir (0.40) is lower than ketoconazole's (0.86) despite both
  coordinating the heme at similar distance and confidence — the
  structural (geometric) and the learned affinity-probability signals
  don't fully agree here, and this project doesn't have an explanation
  for that beyond noting it.
- **Azamulin** doesn't coordinate (13.41 Å — nowhere near the pocket in
  this pose), despite matching the triazole motif. Checked why instead of
  calling it a false negative: according to PubMed, azamulin is an
  established mechanism-based (suicide-substrate) inactivator of CYP3A4,
  not a reversible Type II coordinator — CYP3A4 metabolizes it into a
  reactive intermediate that covalently modifies the enzyme, a
  turnover-dependent process a static co-folding prediction can't
  represent (Lim et al., *Drug Metab Dispos*, 2005,
  [10.1124/dmd.104.003475](https://doi.org/10.1124/dmd.104.003475); Sevrioukova,
  *Int J Mol Sci*, 2019,
  [10.3390/ijms20174245](https://doi.org/10.3390/ijms20174245)). The motif
  screen's "Type II" call is misleading here in a specific, now-documented
  way: real inhibition, wrong mechanism category. One caveat this project
  can't rule out: the cryo-EM paper's own announcement states that current
  co-folding models struggle with CYP3A4 specifically because of
  conformational plasticity and F/G-loop remodeling not captured in the
  crystal structures those models were trained on (Walters et al., LinkedIn,
  2026-09; preprint, "Structural Plasticity and Ligand Promiscuity of CYP3A4
  Revealed by Cryo-EM," bioRxiv). Azamulin's large Fe-N distance is
  consistent with its documented suicide-substrate mechanism, but a
  co-folding failure of exactly the kind that announcement describes is a
  live alternative (or compounding) explanation this project has no way to
  rule out from a single static prediction — flagged here rather than
  claimed as resolved.
- **Vardenafil** doesn't coordinate either (4.84 Å), resolving the open
  question below in favor of false positive: its `imidazole_Nsub` match
  sits inside a fused pyrazolo-pyrimidinone core, and the structural check
  confirms that's not behaving like a discrete coordinating azole.

### Independent cross-check

[Walters, `cyp-heatmap`](https://github.com/PatWalters/cyp-heatmap) (pushed
2026-09-20) fingerprints 122 CYP3A4 structures with
[ProLIF](https://prolif.readthedocs.io), a completely different pipeline from
this project's: real/modeled structures rather than Boltz-2 co-folding,
interaction-type fingerprinting rather than a raw Fe-N distance. Its manifest
happens to include the same four compounds under this project's exact
mechanism question. Its `HEM601.A_MetalAcceptor` column (metal-coordination
contact with the heme iron) from `CYP3A4_heatmap_rich_data.csv`:

| Compound | `MetalAcceptor` (cyp-heatmap) | Fe-N distance (this project) | Agree? |
|---|---|---|---|
| Ketoconazole (`ketaconazole-a`/`-b`) | 1 | 2.05 Å — coordinated | yes |
| Ritonavir | 1 | 2.19 Å — coordinated | yes |
| Azamulin | 0 | 13.41 Å — not coordinated | yes |
| Vardenafil (`vardenafil21`) | 0 (`HBAcceptor`=1 instead) | 4.84 Å — not coordinated | yes |

Two independent methods agree on all four, including the H-bond-not-metal
distinction cyp-heatmap draws for vardenafil, consistent with this project's
4.84 Å being real proximity without dative coordination. Neither pipeline
validates the other's numeric details (a contact fingerprint isn't a
distance, and this project's structures are predicted, not deposited), but
the binary coordination call lines up across two unrelated approaches built
independently, days apart, from the same underlying cryo-EM release.

## Scope and limitations

- The motif screen is a substructure alert, not a mechanism predictor: a
  match means "capable of heme coordination," not "confirmed to
  coordinate in this isoform's active site." Confirmed above for
  vardenafil specifically — its `imidazole_Nsub` match, inside a fused
  ring system rather than a discrete azole, is a real false positive, not
  a hypothetical one.
- The screen only covers direct heme-iron coordination. It has nothing to
  say about mechanism-based (quasi-irreversible) inactivation or other
  non-coordinating inhibition modes — azamulin, above, is a confirmed
  example: real CYP3A4 inhibitor, "Type II" motif match, no structural
  coordination, because its actual mechanism is suicide inactivation, a
  category this screen was never built to detect.
- Structural validation is restricted to the four compounds with
  independently solved reference structures, by design — see Problem,
  above, on why panel-wide physics-based validation isn't attempted here.
- A short Fe-ligand-N distance is evidence of a geometrically plausible
  coordination pose, not proof of the biological mechanism — Boltz-2 has
  no notion of catalytic turnover or covalent chemistry, so it can
  confirm a static coordination geometry but, as azamulin shows, silence
  on that front doesn't rule out a real, different mechanism.

## License

MIT.

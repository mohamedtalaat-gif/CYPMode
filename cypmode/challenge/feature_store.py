"""Every fingerprint/descriptor block, computed once for all compounds (training
and blinded test) and cached, so any candidate model can assemble its inputs by
block name instead of re-deriving features per experiment.

Blocks are registered in ``BLOCKS`` (name -> function(list[Mol]) -> 2D float32
array); a candidate selects blocks by name. Nothing here reads a label, so a
block can never leak one. Fold-dependent features (KNN read-across) are NOT
blocks: they depend on the training rows of each task and are built by the
runner per task.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import (
    AllChem,
    Descriptors,
    MACCSkeys,
    rdMolDescriptors,
    rdReducedGraphs,
)

from cypmode import provenance
from cypmode.metrics.amines import has_trialkylamine
from cypmode.metrics.motifs import TYPE_II_MOTIFS

RDLogger.DisableLog("rdApp.*")

_MOTIF_PATTERNS = {name: Chem.MolFromSmarts(smarts) for name, smarts in TYPE_II_MOTIFS.items()}


def _folded_counts(fingerprint, n_bits: int) -> np.ndarray:
    out = np.zeros(n_bits, dtype=np.float32)
    for bit, count in fingerprint.GetNonzeroElements().items():
        out[bit % n_bits] += count
    return out


def _stack(function: Callable, mols: Sequence) -> np.ndarray:
    return np.stack([function(m) for m in mols]).astype(np.float32)


def _bitvector(m, radius: int, n_bits: int) -> np.ndarray:
    out = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits), out)
    return out


def _descriptors(mols) -> np.ndarray:
    values = np.array([[f(m) for _, f in Descriptors.descList] for m in mols], dtype=np.float64)
    values[~np.isfinite(values)] = np.nan
    return np.nan_to_num(np.clip(values, -1e6, 1e6)).astype(np.float32)



# Mechanism-based-inactivation / reactive-metabolite structural alerts (after Orr et al., J Med Chem
# 2012, "Mechanism-based inactivation of cytochrome P450 enzymes: structural alerts and
# quantitative structure-activity relationships", and Kalgutkar et al., Curr Drug Metab 2007). The
# TDI label is a potency SHIFT after pre-incubation, so substructures that generate reactive
# metabolites are the textbook cause; counts (not flags) because several copies raise the odds.
_ALERT_SMARTS = {
    "methylenedioxyphenyl": "c1ccc2OCOc2c1",
    "furan": "c1ccoc1",
    "thiophene": "c1ccsc1",
    "aminothiazole": "[NX3]c1nccs1",
    "terminal_alkyne": "[CX2]#[CH]",
    "alkyne": "C#C",
    "hydrazine": "[NX3][NX3]",
    "primary_arylamine": "[NH2]c",
    "secondary_arylamine": "[NH;!$(NC=O)](c)[#6]",
    "tertiary_alkylamine": "[NX3;H0;!$(NC=[O,S]);!$(N-a);!$(N=*)]([CX4])([CX4])[CX4]",
    "cyclic_tertiary_amine": "[NX3;H0;R;!$(NC=[O,S]);!$(N-a)]([#6])([#6])[#6]",
    "cyclopropylamine": "[NX3]C1CC1",
    "cyclopropyl": "C1CC1",
    "terminal_alkene": "[CH2]=[CH]",
    "isothiocyanate": "N=C=S",
    "thiocarbonyl": "C(=S)",
    "catechol": "c([OH])c[OH]",
    "hydroquinone": "[OH]c1ccc([OH])cc1",
    "phenol": "c[OH]",
    "nitroaromatic": "c[N+](=O)[O-]",
    "alkyl_halide": "[CX4][Cl,Br,I]",
    "epoxide": "C1OC1",
    "michael_acceptor": "C=CC(=O)",
    "thioether": "[SX2]([#6])[#6]",
    "thiol": "[SH]",
    "hydroxylamine": "[NX3][OH]",
    "benzylic_amine": "c[CH2][NX3;!$(NC=O)]",
    "aryl_piperazine": "c[NX3]1CC[NX3]CC1",
    "methoxy_aryl": "c[OX2][CH3]",
    "n_methyl_amide_or_amine": "[NX3][CH3]",
}
_ALERT_PATTERNS = {name: Chem.MolFromSmarts(smarts) for name, smarts in _ALERT_SMARTS.items()}
assert all(p is not None for p in _ALERT_PATTERNS.values()), "a TDI-alert SMARTS failed to parse"


def _alerts(m) -> np.ndarray:
    counts = [float(len(m.GetSubstructMatches(p))) for p in _ALERT_PATTERNS.values()]
    ring_sizes = [len(r) for r in m.GetRingInfo().AtomRings()]
    return np.array([*counts, float(any(size >= 12 for size in ring_sizes)), float(max(ring_sizes, default=0))])


# Mechanism-informed context for the bioactivation alerts. Medicinal-chemistry practice for avoiding
# time-dependent inhibition (Leach et al., "Tactics to Avoid Inhibition of Cytochrome P450s", in Meanwell (ed.),
# Tactics in Contemporary Drug Design, Springer 2014) is to slow the metabolism that creates the reactive
# intermediate: lower lipophilicity, put a blocking group (fluorine) at the site, redirect metabolism elsewhere,
# or remove the alerting substructure. So an alert is a risk in proportion to how readily the molecule is
# metabolized and how unprotected the alert is; flat yes/no alert flags (the "tdi_alerts" block) cannot say that.
_BIOACTIVATION = (
    "methylenedioxyphenyl", "furan", "thiophene", "aminothiazole", "terminal_alkyne", "hydrazine", "cyclopropylamine",
    "tertiary_alkylamine", "cyclic_tertiary_amine", "secondary_arylamine", "primary_arylamine", "thiocarbonyl", "catechol",
    "hydroquinone", "benzylic_amine", "aryl_piperazine",
)
_LABILE_PATTERNS = [Chem.MolFromSmarts(x) for x in ("[NX3][CH3]", "c[OX2][CH3]", "[CH2,CH3]c", "[CH2,CH3]C=C", "[CH2][OX2]c")]
_FLUORINE = Chem.MolFromSmarts("[F]")
_CF3 = Chem.MolFromSmarts("C(F)(F)F")


def _alert_context(m) -> np.ndarray:
    from rdkit.Chem import Crippen, Descriptors

    n_heavy = max(m.GetNumHeavyAtoms(), 1)
    matches = [set(atoms) for name in _BIOACTIVATION for atoms in m.GetSubstructMatches(_ALERT_PATTERNS[name])]
    classes = sum(1 for name in _BIOACTIVATION if m.HasSubstructMatch(_ALERT_PATTERNS[name]))
    any_alert = float(bool(matches))
    log_p = Crippen.MolLogP(m)
    weight = Descriptors.MolWt(m) / 100.0
    labile = float(sum(len(m.GetSubstructMatches(p)) for p in _LABILE_PATTERNS))
    fluorines = {a[0] for a in m.GetSubstructMatches(_FLUORINE)}
    distance = Chem.GetDistanceMatrix(m)
    alert_atoms = set().union(*matches) if matches else set()
    near_fluorine = float(sum(1 for f in fluorines if alert_atoms and min(distance[f, a] for a in alert_atoms) <= 3))
    crowding = float(np.mean([np.sum(distance[list(atoms)].min(axis=0) <= 3) / n_heavy for atoms in matches])) if matches else 0.0
    return np.array(
        [
            classes, len(matches), log_p, any_alert * log_p, len(matches) * log_p, weight, any_alert * weight,
            near_fluorine, float(len(fluorines)), float(len(m.GetSubstructMatches(_CF3))), near_fluorine / (1.0 + len(matches)),
            labile, any_alert * labile, crowding, any_alert * rdMolDescriptors.CalcNumAromaticRings(m),
        ],
        dtype=float,
    )


BLOCKS: dict[str, Callable[[Sequence], np.ndarray]] = {
    "morgan_bits": lambda mols: _stack(lambda m: _bitvector(m, 2, 2048), mols),
    "morgan_c2": lambda mols: _stack(lambda m: _folded_counts(AllChem.GetHashedMorganFingerprint(m, 2, nBits=2048), 2048), mols),
    "morgan_c3": lambda mols: _stack(lambda m: _folded_counts(AllChem.GetHashedMorganFingerprint(m, 3, nBits=2048), 2048), mols),
    "fmorgan_c2": lambda mols: _stack(
        lambda m: _folded_counts(AllChem.GetHashedMorganFingerprint(m, 2, nBits=1024, useFeatures=True), 1024), mols
    ),
    "atompair_c": lambda mols: _stack(lambda m: _folded_counts(rdMolDescriptors.GetHashedAtomPairFingerprint(m, nBits=2048), 2048), mols),
    "torsion_c": lambda mols: _stack(
        lambda m: _folded_counts(rdMolDescriptors.GetHashedTopologicalTorsionFingerprint(m, nBits=1024), 1024), mols
    ),
    "maccs": lambda mols: _stack(lambda m: np.array(list(MACCSkeys.GenMACCSKeys(m))), mols),
    "avalon": lambda mols: _stack(lambda m: np.array(list(pyAvalonTools.GetAvalonFP(m, nBits=1024))), mols),
    "erg": lambda mols: _stack(lambda m: np.asarray(rdReducedGraphs.GetErGFingerprint(m)), mols),
    "rdkit_desc": _descriptors,
    "motif": lambda mols: _stack(
        lambda m: np.array([float(m.HasSubstructMatch(p)) for p in _MOTIF_PATTERNS.values()] + [float(any(m.HasSubstructMatch(p) for p in _MOTIF_PATTERNS.values()))]),
        mols,
    ),
    "tdi_alerts": lambda mols: _stack(_alerts, mols),
    "tdi_alert_context": lambda mols: _stack(_alert_context, mols),
    "amine": lambda mols: _stack(lambda m: np.array([float(has_trialkylamine(Chem.MolToSmiles(m)))]), mols),
}


def _external_activity(mols):
    from cypmode.challenge.external_activity import external_activity_block

    return external_activity_block(mols)


BLOCKS["ext_activity"] = _external_activity  # label-free, trained on external panels only


def _physics(mols):
    from cypmode.challenge.physics_block import load_physics_features

    return load_physics_features([Chem.MolToSmiles(m) for m in mols])  # read-only cache lookup, never imports tblite


BLOCKS["physics"] = _physics  # GFN2-xTB atom/site/pharmacophore block; needs `physics_block --extract` first
def _chemeleon(mols):
    from cypmode.challenge.frozen_embeddings import load_chemeleon_pca

    return load_chemeleon_pca([Chem.MolToSmiles(m) for m in mols])  # read-only cache lookup, never imports torch


BLOCKS["chemeleon"] = _chemeleon  # frozen CheMeleon embeddings, 64 principal components; needs `frozen_embeddings --extract`
def _physics_b(mols):
    from cypmode.challenge.physics_tierb import load_tierb_features

    return load_tierb_features([Chem.MolToSmiles(m) for m in mols])  # read-only cache lookup, never imports tblite


BLOCKS["physics_b"] = _physics_b  # tier-B physics (Boltzmann conformers, relaxed hydrogen abstraction); needs `physics_tierb --extract`
def _basic_centre(mols):
    from cypmode.challenge.basic_centre import load_basic_centre_features

    return load_basic_centre_features([Chem.MolToSmiles(m) for m in mols])  # read-only cache lookup, never imports Dimorphite-DL


BLOCKS["basic_centre"] = _basic_centre  # protonation states, basic-centre classes and charge-to-ring geometry; needs `basic_centre --extract`
def _monroe(mols):
    from cypmode.challenge.monroe_embeddings import load_monroe_pca

    return load_monroe_pca([Chem.MolToSmiles(m) for m in mols])  # read-only cache lookup, never imports torch


def _ext_cont(mols):
    from cypmode.challenge.external_continuous import load_ext_continuous

    return load_ext_continuous([Chem.MolToSmiles(m) for m in mols])  # read-only cache lookup


def _ext_tox(mols):
    from cypmode.challenge.external_continuous import load_ext_tox21

    return load_ext_tox21([Chem.MolToSmiles(m) for m in mols])  # read-only cache lookup


BLOCKS["ext_tox"] = _ext_tox  # the Tox21 CYP screen (P450-Glo qHTS, luciferase inhibitors removed), predicted by models trained on Tox21 compounds only; needs `external_continuous --download-tox21 --extract`
def _ext_adme(mols):
    from cypmode.challenge.openadmet_surrogates import load_openadmet_surrogates

    return load_openadmet_surrogates([Chem.MolToSmiles(m) for m in mols])  # read-only cache lookup


BLOCKS["ext_adme"] = _ext_adme  # logD and mouse brain protein binding predicted by OpenADMET's surrogate Chemprop models (trained on ExpansionRx); needs `openadmet_surrogates --extract`
def _atomjepa(mols):
    from cypmode.challenge.atomjepa_embeddings import load_atomjepa_pca

    return load_atomjepa_pca([Chem.MolToSmiles(m) for m in mols])  # read-only cache lookup, never imports torch


BLOCKS["atomjepa"] = _atomjepa  # frozen Atom-JEPA embeddings of all 8 blocks, 64 principal components; needs `atomjepa_embeddings --extract`
BLOCKS["ext_cont"] = _ext_cont  # continuous responses and potencies of the NCATS qHTS panel (AID 1851), predicted by models trained on external compounds only; needs `external_continuous --extract`
BLOCKS["monroe"] = _monroe  # frozen Monroe (GRIT) embeddings, 64 principal components; needs `monroe_embeddings --extract`
OPT_IN_BLOCKS = ("ext_activity", "physics", "chemeleon", "physics_b", "basic_centre", "monroe", "ext_cont", "ext_tox", "ext_adme", "atomjepa")


class FeatureStore:
    """All blocks for an ordered list of compounds; row ``i`` is compound ``i`` everywhere."""

    def __init__(self, smiles: Sequence[str], blocks: dict[str, np.ndarray]):
        self.smiles = list(smiles)
        self.blocks = blocks
        for name, matrix in blocks.items():
            if matrix.shape[0] != len(self.smiles):
                raise ValueError(f"block {name!r} has {matrix.shape[0]} rows for {len(self.smiles)} compounds")

    @classmethod
    def build(cls, smiles: Sequence[str], names: Sequence[str] | None = None) -> "FeatureStore":
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        bad = [s for s, m in zip(smiles, mols, strict=True) if m is None]
        if bad:
            raise ValueError(f"{len(bad)} unparseable SMILES, e.g. {bad[0]!r}")
        return cls(smiles, {name: BLOCKS[name](mols) for name in (names or [n for n in BLOCKS if n not in OPT_IN_BLOCKS])})

    @classmethod
    def load_or_build(cls, path: str | Path, smiles: Sequence[str], names: Sequence[str] | None = None) -> "FeatureStore":
        """Load cached blocks; compute (and add to the cache) only the requested blocks that are missing.
        A cache built for a different compound list is discarded.
        """
        path = Path(path)
        wanted = list(names or [n for n in BLOCKS if n not in OPT_IN_BLOCKS])
        cached: dict[str, np.ndarray] = {}
        if path.exists():
            z = np.load(path, allow_pickle=True)
            if list(z["smiles"]) == list(smiles):
                cached = {n: z[n] for n in z.files if n != "smiles"}
        missing = [n for n in wanted if n not in cached]
        if missing:
            cached.update(cls.build(smiles, missing).blocks)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, smiles=np.array(list(smiles)), **cached)
            provenance.stamp(path)
        return cls(smiles, {n: cached[n] for n in wanted})

    def matrix(self, block_names: Sequence[str], rows: np.ndarray | None = None) -> np.ndarray:
        missing = [b for b in block_names if b not in self.blocks]
        if missing:
            raise KeyError(f"unknown blocks {missing}; available: {sorted(self.blocks)}")
        stacked = np.hstack([self.blocks[b] for b in block_names])
        return stacked if rows is None else stacked[np.asarray(rows)]

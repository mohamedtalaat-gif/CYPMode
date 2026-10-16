"""Pre-registered tests of the candidates that use the official auxiliary readings and the two-part form of the label (see ``aux_blocks``), and of the candidates
whose predictions come from the torch environment (TabICL, the multitask network, Chemprop and CheMeleon).

    python -m cypmode.challenge.aux_experiments --experiment screen --isoform CYP3A4 --repeats 3 --run

Experiments
  screen      predicted single-concentration screen (log2 fold change of the four enzymes) as features of the deep LightGBM, feature set A and A+fps
  emax        predicted Emax (both arms, four enzymes) as features
  pxr         predicted PXR-induction pEC50 and Emax (the same compound library, measured for about 1,900 CYP training compounds) as features
  gate_shift  P(t > 4.301) x P(t - d > log10 2), with the diagnostic parts; the gate reads every compound with a TDI-arm curve
  nodirect    the compounds without a direct curve used through what their TDI-arm pIC50 implies (certain negatives; soft labels), CYP3A4
  basic_centre  the protonation and charge-to-ring block (``basic_centre``) added to feature sets A and A+fps
  ext_cont    the continuous responses and potencies of the NCATS qHTS panel (``external_continuous``) added to the feature sets that carry the binary external block
  atomjepa    the frozen Atom-JEPA embeddings (``atomjepa_embeddings``: 8 blocks, 64 principal components) alone and added to feature sets A and A+fps
  ext_adme    the logD and mouse brain protein binding predicted by OpenADMET's surrogate Chemprop models (``openadmet_surrogates``) added to feature sets A and A+fps
  ext_tox     the Tox21 CYP-screen readings (luciferase inhibitors removed) added to the same sets, as a separate test with its own gate
  tabicl, tabpfn, atomjepa_head, tabicl_monroe, tabicl_chemeleon, tabpfn_chemeleon, mlp, mlp_aux, chemprop   the cached torch-side candidates added to the default stack

Each is judged on the nested similarity x prevalence x confident-rows yardstick with z0 in {1.0, 1.96} (``addition_experiment``): the added candidates are
validated alone (best single, where the addition is a feature block that can be switched off) and inside the assembled stack. GATE: corrected resampled t with
p < 0.05 and a positive mean AUC difference at both z0. A group that fails is recorded as a measured negative result and does not enter the final model.
"""

import argparse

from cypmode.challenge.addition_experiment import evaluate_addition
from cypmode.challenge.aux_blocks import aux_candidates, gate_shift_candidates, nodirect_candidates
from cypmode.challenge.cached_candidates import GROUPS as CACHED
from cypmode.challenge.nested_runners import Candidate, atomjepa_candidates, basic_centre_candidates, ext_cont_candidates, ext_adme_candidates, ext_tox_candidates

NESTED_ROOT = "data/tdi_track/outputs/nested"
ALONE = {
    "screen": [("lgbm_A_deep", "scrblk_A_deep"), ("lgbm_A+fps_deep", "scrblk_A+fps_deep")],
    "emax": [("lgbm_A_deep", "emxblk_A_deep"), ("lgbm_A+fps_deep", "emxblk_A+fps_deep")],
    "gate_shift": [("lgbm_A_deep", "gs_A_deep"), ("lgbm_A+fps_deep", "gs_A+fps_deep")],
    "basic_centre": [("lgbm_A_deep", "lgbm_A+bc_deep"), ("lgbm_A+fps_deep", "lgbm_A+fps+bc_deep")],
    "nodirect": [("lgbm_A_deep", "ndsoft_A_deep"), ("lgbm_A+fps_deep", "ndsoft_A+fps_deep")],
    "pxr": [("lgbm_A_deep", "pxrblk_A_deep"), ("lgbm_A+fps_deep", "pxrblk_A+fps_deep")],
    "ext_cont": [("lgbm_A+ext_deep", "lgbm_A+ext+xc_deep"), ("lgbm_A+fps+ext_deep", "lgbm_A+fps+ext+xc_deep")],
    "ext_tox": [("lgbm_A+ext_deep", "lgbm_A+ext+xt_deep"), ("lgbm_A+fps+ext_deep", "lgbm_A+fps+ext+xt_deep")],
    "ext_adme": [("lgbm_A_deep", "lgbm_A+ad_deep"), ("lgbm_A+fps_deep", "lgbm_A+fps+ad_deep")],
    "atomjepa": [("lgbm_A_deep", "lgbm_A+aj_deep"), ("lgbm_A+fps_deep", "lgbm_A+fps+aj_deep")],
}
DIAGNOSTICS = {"atomjepa": ("lgbm_aj_deep",), "nodirect": ("ndneg_A_deep", "ndneg_A+fps_deep"), "gate_shift": ("tgate_A_deep", "tshift_A_deep", "tgate_A+fps_deep", "tshift_A+fps_deep"), "tabicl": ("tabicl_svd",), "tabpfn": ("tabpfn_svd",), "atomjepa_head": ("atomjepa_head",), "chemprop_foundation": ("chemeleon_minimol_mt_shift", "chemeleon_surrogate_mt_shift"), "tabicl_monroe": ("tabicl_monroe",), "tabicl_chemeleon": ("tabicl_chemeleon",), "tabpfn_chemeleon": ("tabpfn_chemeleon",), "mlp": ("mlp_multitask_label", "mlp_multitask_shift"), "mlp_aux": ("mlp_multitask_aux_label", "mlp_multitask_aux_shift"), "chemprop": ("chemprop_mt_shift", "chemeleon_mt_shift")}


def candidates_of(experiment: str) -> list[Candidate]:
    if experiment == "screen":
        return aux_candidates(("screen",))
    if experiment == "emax":
        return aux_candidates(("emax",))
    if experiment == "pxr":
        return aux_candidates(("pxr",))
    if experiment == "gate_shift":
        return gate_shift_candidates()
    if experiment == "basic_centre":
        return basic_centre_candidates()
    if experiment == "ext_cont":
        return ext_cont_candidates()
    if experiment == "ext_tox":
        return ext_tox_candidates()
    if experiment == "ext_adme":
        return ext_adme_candidates()
    if experiment == "atomjepa":
        return atomjepa_candidates()
    if experiment == "nodirect":
        return nodirect_candidates()
    return CACHED[experiment]()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=["screen", "emax", "pxr", "gate_shift", "basic_centre", "nodirect", "ext_cont", "ext_tox", "ext_adme", "atomjepa", *CACHED])
    parser.add_argument("--isoform", required=True, choices=["CYP3A4", "CYP2D6"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--run", action="store_true", help="fill the prediction cache for the added candidates first (not for the cached torch-side candidates)")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--outer", type=int, default=5)
    parser.add_argument("--root", default=NESTED_ROOT)
    args = parser.parse_args()
    experiment = args.experiment
    evaluate_addition(args.isoform, candidates_of(experiment), args.repeats, args.n_jobs, args.run and experiment not in CACHED, ALONE.get(experiment, ()), DIAGNOSTICS.get(experiment, ()), outer=args.outer, root=args.root)


if __name__ == "__main__":
    main()

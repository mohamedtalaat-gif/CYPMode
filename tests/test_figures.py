import sys
from pathlib import Path

import numpy as np
import pytest

FIGURES = Path("docs/figures")
sys.path[:0] = [str(FIGURES), str(FIGURES.parent.parent)]

from catalog import REGISTRY, Figure, caption_text, figure  # noqa: E402

pytestmark = pytest.mark.skipif(not Path("data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv").exists(), reason="the training data is not in this checkout")


def test_a_figure_id_can_only_be_registered_once():
    figure("test_only_unique", "test", "t", "c")(lambda theme: None)
    try:
        with pytest.raises(ValueError):
            figure("test_only_unique", "test", "t", "c")(lambda theme: None)
    finally:
        REGISTRY.pop("test_only_unique", None)


def test_a_caption_can_be_computed_from_the_data():
    fig = Figure("x", "g", "t", lambda: "from data", lambda theme: None)
    assert caption_text(fig) == "from data"
    assert caption_text(Figure("y", "g", "t", "plain", lambda theme: None)) == "plain"


def test_files_of_a_figure_are_attributed_to_its_id():
    from build_all import figure_id_of

    assert figure_id_of(Path("kin_x-light.svg")) == "kin_x"
    assert figure_id_of(Path("kin_x-dark.svg")) == "kin_x"
    assert figure_id_of(Path("kin_x-anim-light.gif")) == "kin_x"
    assert figure_id_of(Path("kin_x-anim-dark.gif")) == "kin_x"
    assert figure_id_of(Path("kin_x-light.png")) == "kin_x"
    assert figure_id_of(Path("kin_x-dark.png")) == "kin_x"


@pytest.mark.parametrize("isoform", ["CYP3A4", "CYP2D6"])
def test_the_label_rule_drawn_in_the_figure_reproduces_every_label(isoform):
    from gen.data import LOG2, labelled_arms

    arms = labelled_arms(isoform)
    assert len(arms) > 1000
    assert ((arms["shift"] > LOG2) == arms["label"]).all()


def test_the_integrated_inactivation_curves_follow_the_closed_form_rate():
    from gen.kinetics import K_I, K_INACT, integrate, k_obs_of

    for inhibitor in (5.0, 20.0, 100.0):
        t, activity = integrate(inhibitor)
        window = (t > 2) & (activity > 2e-3)
        slope = -np.polyfit(t[window], np.log(activity[window]), 1)[0]
        assert slope == pytest.approx(k_obs_of(inhibitor), rel=0.03)
    assert k_obs_of(K_I) == pytest.approx(K_INACT / 2)


def test_the_memory_formula_gives_the_size_the_runtime_refused():
    from gen.structure import allocation_gib

    assert allocation_gib(592) == pytest.approx(9.27, abs=0.005)  # bench-YNV: 503 residues + 46 ligand atoms + 43 heme atoms; the log says 9.27 GiB
    assert allocation_gib(592, 128) < allocation_gib(592) / 4

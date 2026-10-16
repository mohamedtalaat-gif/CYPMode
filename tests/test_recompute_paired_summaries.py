import numpy as np
import pandas as pd

from cypmode.challenge.recompute_paired_summaries import recompute


def test_recompute_writes_one_paired_table_per_raw_table_without_refitting(tmp_path):
    rng = np.random.default_rng(0)
    rows = [{"isoform": iso, "repeat": r, "fold": f, "variant": v, "mcc": 0.3 + rng.normal(scale=0.03)} for iso in ("CYP3A4", "CYP2D6") for r in range(5) for f in range(5) for v in ("M3", "M3+X", "M3+Y")]
    pd.DataFrame(rows).to_csv(tmp_path / "demo_significance_raw.csv", index=False)
    (written,) = recompute(tmp_path)
    out = pd.read_csv(written)
    assert written.name == "demo_significance_paired.csv" and len(out) == 6 and set(out.isoform) == {"CYP3A4", "CYP2D6"}
    assert list(out.columns) == ["isoform", "group1", "group2", "meandiff", "p-adj", "lower", "upper", "reject"]

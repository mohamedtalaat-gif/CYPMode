"""Shared pieces of the audit: check/section containers, deterministic Markdown helpers, shared context.

Everything an audit section prints is computed when the audit runs; the only prose is wording around the
computed values. Output must be byte-identical between two runs on the same state of the repository, so no
timestamps, durations, absolute temp paths or unordered iteration appear in the report.
"""

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PASS, FAIL, INFO, SKIP = "PASS", "FAIL", "INFO", "SKIP"
ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = ROOT / "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
TEST_CSV = ROOT / "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv"
STORE_NPZ = ROOT / "data/tdi_track/outputs/caches/feature_store.npz"
NESTED_ROOT = ROOT / "data/tdi_track/outputs/nested"
SNAPSHOTS = ROOT / "data/tdi_track/inputs/leaderboard_snapshots"
LOGS = ROOT / "data/structure_track/outputs/cyp3a4/logs"
ISOFORMS = ("CYP3A4", "CYP2D6")


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""


@dataclass
class Section:
    title: str
    body: list[str] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, ok: bool | None, detail: str = "") -> None:
        self.checks.append(Check(name, INFO if ok is None else PASS if ok else FAIL, detail))


def fmt(x, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.{digits}f}"


def md_table(df: pd.DataFrame, digits: int = 4) -> str:
    if df.empty:
        return "_(empty)_"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(v, digits) if isinstance(v, (float, np.floating, int, np.integer)) and not isinstance(v, bool) else str(v) for v in row) + " |")
    return "\n".join(lines)


def sha256(path: Path, n: int = 12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:n]


def run(cmd: list[str], cwd: Path = ROOT) -> str:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True).stdout


class Context:
    """Shared, lazily built objects and the metrics the limitations log reads."""

    def __init__(self, n_jobs: int = 2):
        self.n_jobs = n_jobs
        self.metrics: dict[str, float | int | str] = {}
        self._cache: dict = {}

    def once(self, key, build):
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    @property
    def train(self) -> pd.DataFrame:
        return self.once("train", lambda: pd.read_csv(TRAIN_CSV))

    @property
    def test(self) -> pd.DataFrame:
        return self.once("test", lambda: pd.read_csv(TEST_CSV))

    def runner(self, iso: str):
        """(Context for the nested runners, plan) rebuilt from the raw files for ``iso``."""
        from cypmode.challenge.nested_runners import build_context_and_plan

        return self.once(("runner", iso), lambda: build_context_and_plan(str(TRAIN_CSV), str(TEST_CSV), iso, str(STORE_NPZ), 3, 5, 4, 0, self.n_jobs))

    def cache(self, iso: str):
        from cypmode.challenge.nested import NestedPlan, PredictionCache

        def build():
            plan = NestedPlan.load(NESTED_ROOT / iso / "plan.npz")
            return plan, PredictionCache(NESTED_ROOT / iso / "cache", plan)

        return self.once(("cache", iso), build)

    def lgbm_family(self, iso: str) -> list[str]:
        from cypmode.challenge.nested_runners import default_candidates

        plan, cache = self.cache(iso)
        return [c.name for c in default_candidates() if not cache.missing(c.name, False)]

    def weight_fn(self, iso: str, z0: float | None = None):
        """Evaluation weights: similarity to the test set, optionally restricted to confident rows."""
        from cypmode.challenge.label_confidence import confident_weights
        from cypmode.challenge.validation_matching import SimilarityMatcher

        rctx, plan = self.runner(iso)
        matcher = self.once(("matcher", iso), lambda: SimilarityMatcher(rctx.store.blocks["morgan_bits"], plan.external))
        return matcher if z0 is None else confident_weights(matcher, rctx.confidence, z0)

    def prevalence(self, iso: str) -> float:
        from cypmode.challenge.leaderboard_prior import estimate_from_csv

        return float(estimate_from_csv(SNAPSHOTS / f"interim_{iso}_2026-10-04.csv")["prevalence"])

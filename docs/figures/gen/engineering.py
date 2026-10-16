"""Figures of the engineering behind the results: the job queue as a timeline, what the machine did while it ran, and the size of the test suite."""

import re
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from catalog import figure
from style import apply, tidy

GROUP = "engineering"
ROOT = Path(__file__).resolve().parents[3]
STAGE_LOG = "queue/logs/queue_stage.txt"
YEAR = 2026
MIN_BAR_DAYS = 0.01  # a run shorter than about 15 minutes is drawn this wide so that it stays visible
EVENT = re.compile(r"^(\d\d-\d\d \d\d:\d\d:\d\d) (RUN|DONE|RETRY|FAILED|RUNNER STARTED)\s*(\S*)(?:.*?\| swap used ([\d.]+)([MG]))?")
SWAP = re.compile(r"^(\d\d-\d\d \d\d:\d\d:\d\d) .*\| swap used ([\d.]+)([MG])")


def stamp(text: str) -> datetime:
    return datetime.strptime(f"{YEAR}-{text}", "%Y-%m-%d %H:%M:%S")


def queue_events():
    """(runs, swap): the job runs as a table (job, start, end, outcome) and the swap used over time (GB), both read from the runner's own log."""
    runs, open_runs, swap = [], {}, []
    for line in (ROOT / STAGE_LOG).read_text().splitlines():
        s = SWAP.match(line)
        if s:
            swap.append((stamp(s.group(1)), float(s.group(2)) / (1024 if s.group(3) == "M" else 1)))
        m = EVENT.match(line)
        if not m:
            continue
        when, kind, job = stamp(m.group(1)), m.group(2), m.group(3)
        if kind == "RUNNER STARTED":
            open_runs.clear()  # a job still open here was killed with the runner
        elif kind == "RUN":
            open_runs[job] = when  # a repeated RUN line is the same job being re-checked (deferred), so the latest start counts
        elif job in open_runs:
            runs.append({"job": job, "start": open_runs.pop(job), "end": when, "outcome": {"DONE": "done", "RETRY": "retry", "FAILED": "failed"}[kind]})
    return pd.DataFrame(runs), pd.DataFrame(swap, columns=["time", "swap_gb"])


def queue_caption() -> str:
    runs, swap = queue_events()
    hours = (runs["end"] - runs["start"]).dt.total_seconds() / 3600
    longest = runs.assign(hours=hours).sort_values("hours", ascending=False).head(3)
    names = ", ".join(f"{r.job} ({r.hours:.1f} h)" for r in longest.itertuples())
    counts = runs["outcome"].value_counts().to_dict()
    return (f"Each bar is one run of one job of the queue, from the runner's own log ({runs['job'].nunique()} jobs, {len(runs)} runs: "
            f"{counts.get('done', 0)} finished, {counts.get('retry', 0)} stopped and retried, {counts.get('failed', 0)} failed after their attempts). "
            f"The runs add up to {hours.sum():.0f} hours of wall-clock time between {runs['start'].min():%d %b} and {runs['end'].max():%d %b} (this includes any time the machine was asleep during a run); the longest are {names}. "
            f"The line shows the swap in use at every runner event (peak {swap['swap_gb'].max():.1f} GB on a 16 GB machine).")


@figure("eng_queue_timeline", GROUP, "The job queue as a timeline", queue_caption, data=(STAGE_LOG,), section="12.2")
def queue_timeline(theme_name):
    th = apply(theme_name)
    runs, swap = queue_events()
    order = runs.groupby("job")["start"].min().sort_values().index.tolist()
    row = {job: i for i, job in enumerate(order)}
    colors = {"done": th["blue"], "retry": th["orange"], "failed": th["red"]}
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11.5, 0.17 * len(order) + 3.6), sharex=True, gridspec_kw={"height_ratios": [len(order) * 0.17 + 1, 2.4]})
    for r in runs.itertuples():
        ax.barh(row[r.job], max(mdates.date2num(r.end) - mdates.date2num(r.start), MIN_BAR_DAYS), left=mdates.date2num(r.start), height=0.7, color=colors[r.outcome])
    ax.set_yticks(range(len(order)), order, fontsize=6.5)
    ax.invert_yaxis()
    ax.legend(handles=[Patch(color=color, label=f"run ended: {outcome}") for outcome, color in colors.items()], frameon=False, fontsize=8, loc="lower left")
    tidy(ax, th)
    bx.plot(swap["time"], swap["swap_gb"], color=th["purple"], lw=1.4)
    bx.set(ylabel="swap used (GB)")
    bx.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    bx.xaxis.set_major_locator(mdates.DayLocator())
    tidy(bx, th)
    fig.tight_layout()
    return fig


SLEEP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \S+ Sleep\s+Entering Sleep state due to '([^']+)'")


def power_log() -> str:
    return subprocess.run(["pmset", "-g", "log"], capture_output=True, text=True, check=False).stdout


def power_log_start(text: str) -> str:
    return min(re.findall(r"^(\d{4}-\d\d-\d\d) ", text, re.M))


def power_events(since: str = "2026-10-08") -> pd.DataFrame:
    text = power_log()
    rows = [(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), m.group(2)) for m in map(SLEEP.match, text.splitlines()) if m]
    frame = pd.DataFrame(rows, columns=["time", "cause"])
    return frame[frame["time"] >= pd.Timestamp(since)].reset_index(drop=True)


def power_caption() -> str:
    frame = power_events()
    if frame.empty:
        return "The power log of this machine holds no sleep events for the period."
    counts = frame["cause"].value_counts()
    parts = ", ".join(f"{n} x {cause}" for cause, n in counts.items())
    return (f"Every time the machine entered sleep between {frame['time'].min():%d %b} and {frame['time'].max():%d %b}, by the cause its own power log gives: {parts}. "
            f"The power log itself begins on {power_log_start(power_log())}, so nothing is known about earlier days. "
            f"Read from `pmset -g log` when the figure is built, so it describes the machine that ran the queue; it records what happened, not why.")


@figure("eng_power_events", GROUP, "What the machine did while the queue ran", power_caption, data=("pmset -g log",), section="12.3")
def power_events_figure(theme_name):
    th = apply(theme_name)
    frame = power_events()
    runs, _ = queue_events()
    fig, ax = plt.subplots(figsize=(11.5, 2.8 + 0.5 * max(1, frame["cause"].nunique())))
    causes = list(frame["cause"].value_counts().index) or ["no sleep events"]
    palette = [th["red"], th["orange"], th["blue"], th["purple"], th["teal"], th["gold"]]
    ax.barh([0], [mdates.date2num(runs["end"].max()) - mdates.date2num(runs["start"].min())], left=mdates.date2num(runs["start"].min()), height=0.9, color=th["panel"], zorder=0)
    ax.text(mdates.date2num(runs["start"].min()), -0.75, "queue active", fontsize=8, color=th["muted"], va="center")
    for i, cause in enumerate(causes):
        times = frame.loc[frame["cause"] == cause, "time"]
        ax.plot(times, np.full(len(times), i + 1), "|", ms=14, mew=2, color=palette[i % len(palette)])
    ax.set_yticks(range(1, len(causes) + 1), causes)
    ax.set_ylim(len(causes) + 0.7, -1.1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    tidy(ax, th)
    fig.tight_layout()
    return fig


def test_counts() -> pd.DataFrame:
    rows = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        rows.append({"module": path.stem.removeprefix("test_"), "tests": len(re.findall(r"^\s*def test_", path.read_text(), re.M))})
    return pd.DataFrame(rows).sort_values("tests", ascending=False)


def tests_caption() -> str:
    frame = test_counts()
    return (f"Test functions per test module of the repository: {int(frame['tests'].sum())} test functions in {len(frame)} modules (parametrized cases count once). "
            f"The audit runs the whole suite and the full report twice and requires the two reports to be byte-identical.")


@figure("eng_tests_by_module", GROUP, "The test suite by module", tests_caption, data=("tests/test_*.py",), section="12.5")
def tests_by_module(theme_name):
    th = apply(theme_name)
    frame = test_counts().head(30)
    fig, ax = plt.subplots(figsize=(8.6, 0.24 * len(frame) + 1.2))
    ax.barh(frame["module"][::-1], frame["tests"][::-1], color=th["blue"])
    ax.bar_label(ax.containers[0], fontsize=7, padding=2)
    ax.set(xlabel="test functions")
    ax.tick_params(axis="y", labelsize=7)
    tidy(ax, th, grid=False)
    fig.tight_layout()
    return fig

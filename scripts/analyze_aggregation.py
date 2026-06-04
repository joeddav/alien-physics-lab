#!/usr/bin/env python
"""Adaptive-aggregation analysis for a GRPO run's logged completions.

Core question for the *diverse* (varying-noise) task: does the policy run MORE
experiments on NOISIER worlds? GRPO only ever sees an outcome reward, so adaptive
aggregation — if it emerges — is an implicit behavior we have to read off the rollouts.

Two data paths, auto-detected per parquet:
  • EXACT  — if the run logged the diagnostic columns (``n_experiments``,
    ``world_noise``, ``world_gravity`` via ``measurement_reward``'s ``log_extra``),
    use them directly. Gives a clean corr(world_noise, n_experiments).
  • PARSE  — older runs without those columns: count ``<tool_call>`` blocks whose
    name is an experiment (drop_ball / pendulum_period; calculator excluded) from the
    rendered completion string. ``world_noise`` is then unknown (constant-noise runs
    have a single value anyway), so only the n_experiments trajectory/histogram is shown.

Usage:
    python scripts/analyze_aggregation.py out/grpo-rl-varnoise-g16
    python scripts/analyze_aggregation.py out/grpo-rl-varnoise-g16 --md report.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import pandas as pd

_TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_EXPERIMENT_TOOLS = {"drop_ball", "pendulum_period"}


def count_experiments(completion: str) -> int:
    """#experiment tool calls (drop_ball/pendulum_period) in a rendered completion."""
    n = 0
    for blk in _TOOLCALL_RE.findall(completion or ""):
        try:
            name = json.loads(blk).get("name")
        except (json.JSONDecodeError, AttributeError):
            # be lenient: grab the first "name": "..." if JSON is malformed
            m = re.search(r'"name"\s*:\s*"([^"]+)"', blk)
            name = m.group(1) if m else None
        if name in _EXPERIMENT_TOOLS:
            n += 1
    return n


def load(run_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(run_dir, "completions", "*.parquet")))
    if not files:
        raise SystemExit(f"no completions parquet under {run_dir}/completions/")
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        if "n_experiments" not in df.columns:
            df["n_experiments"] = df["completion"].map(count_experiments)
            df["_source"] = "parsed"
        else:
            df["_source"] = "exact"
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out


def fmt(x: float, p: int = 3) -> str:
    return "nan" if pd.isna(x) else f"{x:.{p}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--md", default=None, help="Also write a markdown summary here.")
    args = ap.parse_args()

    df = load(args.run_dir)
    src = df["_source"].iloc[0]
    steps = sorted(df["step"].unique())
    n_steps = len(steps)
    has_noise = "world_noise" in df.columns and df["world_noise"].notna().any()

    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out(f"# Aggregation analysis — {os.path.basename(args.run_dir.rstrip('/'))}")
    out(f"source={src}  parquet_steps={n_steps} ({steps[0]}..{steps[-1]})  rollouts={len(df)}")
    out()

    # Trajectory: bin steps into thirds (early/mid/late) for a quick before/after.
    df = df.sort_values("step")
    thirds = pd.qcut(df["step"].rank(method="first"), 3, labels=["early", "mid", "late"])
    out("## n_experiments over training (thirds of logged steps)")
    out("| phase | mean n_exp | median | %>=3 | physics | measurement |")
    out("|---|---|---|---|---|---|")
    for ph in ["early", "mid", "late"]:
        d = df[thirds == ph]
        phys = d["physics_reward"].mean() if "physics_reward" in d else float("nan")
        meas = d["reward_measurement"].mean() if "reward_measurement" in d else (
            d["measurement_reward"].mean() if "measurement_reward" in d else float("nan"))
        pct3 = (d["n_experiments"] >= 3).mean() * 100
        out(f"| {ph} | {fmt(d['n_experiments'].mean(),2)} | {fmt(d['n_experiments'].median(),1)} "
            f"| {fmt(pct3,1)}% | {fmt(phys)} | {fmt(meas)} |")
    out()

    # Histogram over the late third (steady-state behavior).
    late = df[thirds == "late"]
    out("## n_experiments distribution (late third)")
    vc = late["n_experiments"].value_counts(normalize=True).sort_index()
    out("| n_exp | " + " | ".join(str(int(k)) for k in vc.index) + " |")
    out("|---" * (len(vc) + 1) + "|")
    out("| frac | " + " | ".join(f"{v*100:.1f}%" for v in vc.values) + " |")
    out()

    # The headline: does aggregation track hidden noise?
    if has_noise:
        d = df.dropna(subset=["world_noise"]).copy()
        pear = d["world_noise"].corr(d["n_experiments"], method="pearson")
        spear = d["world_noise"].corr(d["n_experiments"], method="spearman")
        out("## Adaptive aggregation: corr(world_noise, n_experiments)")
        out(f"Pearson r = {fmt(pear)}   Spearman ρ = {fmt(spear)}   (over all {len(d)} rollouts)")
        out()
        # Late-third only (after the policy has learned) — the cleaner signal.
        dl = late.dropna(subset=["world_noise"])
        if len(dl) > 10:
            out(f"Late-third only: Pearson r = {fmt(dl['world_noise'].corr(dl['n_experiments']))}   "
                f"Spearman ρ = {fmt(dl['world_noise'].corr(dl['n_experiments'], method='spearman'))} "
                f"(n={len(dl)})")
            out()
        # Binned: noise quartile -> behavior. The adaptive hypothesis predicts n_exp ↑ with noise.
        d["noise_q"] = pd.qcut(d["world_noise"], 4, labels=["Q1(low)", "Q2", "Q3", "Q4(high)"])
        out("## Behavior by hidden-noise quartile (all rollouts)")
        out("| noise quartile | noise range | mean n_exp | %>=3 | physics_reward |")
        out("|---|---|---|---|---|")
        for q in ["Q1(low)", "Q2", "Q3", "Q4(high)"]:
            dd = d[d["noise_q"] == q]
            if not len(dd):
                continue
            rng = f"{dd['world_noise'].min():.3f}–{dd['world_noise'].max():.3f}"
            phys = dd["physics_reward"].mean() if "physics_reward" in dd else float("nan")
            pct3 = (dd["n_experiments"] >= 3).mean() * 100
            out(f"| {q} | {rng} | {fmt(dd['n_experiments'].mean(),2)} | {fmt(pct3,1)}% | {fmt(phys)} |")
        out()
        out("Interpretation: a clear positive trend (mean n_exp rising Q1→Q4, ρ>0) is direct "
            "evidence the policy learned to *adaptively aggregate* — the whole point of the "
            "varying-noise task. Flat across quartiles ⇒ it picked one fixed procedure regardless "
            "of noise (the toy-task failure mode the diversity was meant to break).")
    else:
        out("## Adaptive aggregation")
        out("`world_noise` not logged for this run (pre-diagnostic-column run, or constant-noise) "
            "→ per-world noise unknown, so corr(noise, n_exp) is unavailable here. The chained run "
            "(launched with the diagnostic columns) reports it exactly. Above: the n_exp trajectory "
            "still shows whether the policy escapes the n=2 plateau on average.")

    if args.md:
        with open(args.md, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n[wrote {args.md}]")


if __name__ == "__main__":
    main()

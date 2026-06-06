#!/usr/bin/env python
"""Benchmark an OpenAI model (default gpt-4o-mini) on the alien-physics-lab tasks via
native function-calling. Validates whether the NEW targets (diameter, spin) are well-posed
and how hard they are, BEFORE investing in GRPO training.

Per target the model gets a target-aware briefing + only the relevant tools, runs a
multi-turn tool loop, and ends by writing \\boxed{<number>}; we parse that and score it
with env.score_value (relative error, success <=3%). Reports solve-rate / median rel-err /
mean tool calls per target.

Usage:
  source ~/.bash_profile   # OPENAI_API_KEY
  python scripts/bench_openai.py --model gpt-4o-mini --episodes 10 --targets gravity,diameter,spin
"""
from __future__ import annotations
import argparse, json, math, os, random, re, statistics as st, time, urllib.error, urllib.request

from alien_physics_lab.env import AlienPhysicsLab
from alien_physics_lab.world import EARTH_GRAVITY_M_S2, WorldParams

CHAT_URL = "https://api.openai.com/v1/chat/completions"
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# ---- tool JSON schemas (OpenAI function-calling) ----
def _fn(name, desc, props=None, required=None):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props or {}, "required": required or []}}}

CALC = _fn("calculator", "Evaluate one arithmetic expression. Supports + - * / ** , sqrt, abs, "
           "sin, cos, tan, asin, acos, atan, radians, degrees, and the constants pi, e. "
           "Angles for trig are in radians.", {"expression": {"type": "string"}}, ["expression"])
TOOLS = {
    "drop_ball": _fn("drop_ball", "Drop a ball of mass_kg from height_m; returns the measured fall time (s).",
                     {"mass_kg": {"type": "number"}, "height_m": {"type": "number"}}, ["mass_kg", "height_m"]),
    "pendulum_period": _fn("pendulum_period", "Measure the period (s) of a simple pendulum of length_m.",
                           {"length_m": {"type": "number"}}, ["length_m"]),
    "measure_horizon_dip": _fn("measure_horizon_dip", "Climb to height_m (0.1-1000) and measure how far below "
                               "level the horizon appears (the dip angle, in degrees).",
                               {"height_m": {"type": "number"}}, ["height_m"]),
    "foucault_pendulum": _fn("foucault_pendulum", "Measure the precession rate (deg/hour) of a Foucault "
                             "pendulum's swing plane at this lab."),
    "star_tracker": _fn("star_tracker", "Observe the night sky; returns the altitude (deg) above the horizon "
                        "of the celestial pole (the fixed point the stars rotate around)."),
    "calculator": CALC,
}

# ---- per-target setup: tools, briefing, world sampler ----
def _earth():
    return dict(world_diameter_m=1.2742e7, world_spin_rad_s=7.2921159e-5, lab_latitude_deg=35.0)

def make_world(target, rng, noise):
    g = rng.uniform(0.4, 2.2) * EARTH_GRAVITY_M_S2
    p = _earth()
    if target == "diameter":
        p["world_diameter_m"] = math.exp(rng.uniform(math.log(3e5), math.log(1.2742e7)))
    elif target == "spin":
        p["world_spin_rad_s"] = math.exp(rng.uniform(math.log(2e-5), math.log(2e-4)))
        p["lab_latitude_deg"] = rng.uniform(15.0, 75.0)
    return WorldParams(gravity_m_s2=g, measurement_noise=noise, seed=rng.randint(1, 10**9), **p)

BRIEF = {
"gravity": (
"You are in an alien physics lab. Infer the lab's effective gravity g (m/s^2) by experiment. "
"From a drop: g = 2*height / time^2. From a pendulum: g = 4*pi^2*length / period^2. "
"Measurements are noisy (~3%), so repeat experiments and average. "
"When confident, give your final answer as a boxed number in m/s^2, e.g. \\boxed{9.81}."),
"diameter": (
"You are in an alien physics lab on an unknown planet. Infer the planet's DIAMETER (in meters) by experiment. "
"Use measure_horizon_dip(height_m): from height h the horizon dips below level by angle alpha, where "
"cos(alpha) = R/(R+h) and R is the planet radius. For small angles R = 2*h/alpha^2 with ALPHA IN RADIANS "
"(use radians() in the calculator: e.g. if dip is 0.5 degrees, alpha = radians(0.5)). Then DIAMETER = 2*R. "
"Measurements are noisy (~3%), so repeat and average. Final answer: boxed diameter in meters, e.g. \\boxed{1.27e7}."),
"spin": (
"You are in an alien physics lab on an unknown planet. Infer the LENGTH OF THE PLANET'S DAY in HOURS "
"(one full rotation) by experiment. Two facts: (1) the altitude of the celestial pole above the horizon "
"(from star_tracker) EQUALS your latitude phi; (2) a Foucault pendulum's swing plane precesses at rate "
"omega_prec = Omega*sin(phi), where Omega is the planet's rotation rate (rad/s). So measure the precession "
"rate and the latitude, then Omega = omega_prec / sin(phi), and the day length = 2*pi/Omega. "
"Watch units: foucault_pendulum returns deg/hour; convert to rad/s with radians() and /3600; sin() takes "
"radians. Measurements are noisy (~3%), repeat and average. Final answer: boxed day length in hours, e.g. \\boxed{23.9}."),
}
TARGET_TOOLS = {
    "gravity": ["drop_ball", "pendulum_period", "calculator"],
    "diameter": ["measure_horizon_dip", "calculator"],
    "spin": ["foucault_pendulum", "star_tracker", "calculator"],
}

def _post(body, api_key, timeout, retries=4):
    req = urllib.request.Request(CHAT_URL, data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):  # transient -> retry
                last = e
            else:
                raise
        except (TimeoutError, urllib.error.URLError, ConnectionError) as e:
            last = e
        time.sleep(2 * (attempt + 1))  # linear backoff
    raise last

def parse_boxed(text):
    if not text:
        return None
    idx = text.rfind("\\boxed")
    seg = text[idx:] if idx != -1 else text
    nums = _NUM_RE.findall(seg)
    try:
        return float(nums[0]) if nums else None
    except ValueError:
        return None

def run_episode(lab, target, model, api_key, max_turns, timeout):
    tools = [TOOLS[t] for t in TARGET_TOOLS[target]]
    msgs = [{"role": "system", "content": BRIEF[target]}, {"role": "user", "content": "Begin the experiment."}]
    for _ in range(max_turns):
        try:
            resp = _post({"model": model, "messages": msgs, "tools": tools,
                          "tool_choice": "auto", "temperature": 0.2}, api_key, timeout)
        except urllib.error.HTTPError as e:
            return None, f"http {e.code}: {e.read()[:200]!r}"
        m = resp["choices"][0]["message"]
        msgs.append(m)
        tcs = m.get("tool_calls")
        if not tcs:
            return parse_boxed(m.get("content") or ""), "ok"
        for tc in tcs:
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
                out = lab.call_tool(tc["function"]["name"], **args)
            except Exception as e:  # noqa: BLE001
                out = {"error": str(e)}
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(out)})
    return None, "max_turns"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--targets", default="gravity,diameter,spin")
    ap.add_argument("--noise", type=float, default=0.03)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set (source ~/.bash_profile)")

    print(f"model={args.model} episodes={args.episodes} noise={args.noise} max_turns={args.max_turns}\n")
    summary = {}
    for target in args.targets.split(","):
        rng = random.Random(args.seed * 1000 + hash(target) % 1000)
        relerrs, successes, ncalls, fails = [], 0, [], 0
        for ep in range(args.episodes):
            lab = AlienPhysicsLab(world=make_world(target, rng, args.noise), target=target)
            pred, status = run_episode(lab, target, args.model, api_key, args.max_turns, args.timeout)
            res = lab.score_value(pred)
            ok = res.success
            successes += int(ok)
            ncalls.append(lab.tool_calls)
            if res.relative_error is not None:
                relerrs.append(res.relative_error)
            else:
                fails += 1
            print(f"  [{target:8s}] ep{ep:02d} pred={pred} true={res.true_gravity_m_s2:.4g} "
                  f"relerr={'%.1f%%'%(res.relative_error*100) if res.relative_error is not None else 'NA':>7} "
                  f"{'OK' if ok else '..'} tools={lab.tool_calls} ({status})")
        med = st.median(relerrs) if relerrs else float('nan')
        summary[target] = (successes / args.episodes, med, st.mean(ncalls), fails)
        print()
    print("=== SUMMARY (success = rel_err <= 3%) ===")
    print(f"{'target':10s} {'solve_rate':>10s} {'median_relerr':>14s} {'mean_tools':>11s} {'no_answer':>10s}")
    for t, (sr, med, mc, fa) in summary.items():
        print(f"{t:10s} {sr*100:>9.0f}% {('%.1f%%'%(med*100)) if med==med else 'NA':>14s} {mc:>11.1f} {fa:>10d}")

if __name__ == "__main__":
    main()

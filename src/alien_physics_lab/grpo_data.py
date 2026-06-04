"""Dataset construction for GRPO training on the alien physics lab.

Each dataset row describes ONE hidden world (one distinct seed). TRL repeats
each row ``num_generations`` times to form a GRPO group, and every repeat calls
``AlienPhysicsGRPOEnv.reset(**row)`` with the *same* seed. Because
``AlienPhysicsLab.random_gravity_lab`` is fully seeded (gravity draw + the
measurement-noise RNG), all rollouts in a group share the same hidden gravity
and the same frozen noise stream. So two rollouts issuing the identical tool
call get the identical noisy reading: measurement noise becomes part of the
*task*, not per-rollout luck, and the only thing varying within a group is the
policy's behavior — exactly what GRPO's ``(reward - mean) / std`` advantage
needs. (The RNG still advances *within* an episode, so repeat-and-average across
multiple drops in a single rollout is still rewarded.)

Do NOT pre-duplicate rows here; TRL handles within-group repetition.
"""

from __future__ import annotations

import math
import random

from datasets import Dataset

from alien_physics_lab.world import EARTH_GRAVITY_M_S2

# Broadened world distribution vs the env defaults, per the handoff's
# reward-hacking notes (wider gravity breaks fixed priors; higher noise rewards
# repeated measurement). Kept in sync with grpo_env defaults.
GRAVITY_MIN = 0.4 * EARTH_GRAVITY_M_S2
GRAVITY_MAX = 2.2 * EARTH_GRAVITY_M_S2
MEASUREMENT_NOISE = 0.03
MAX_TOOL_CALLS = 5

# --- Procedural-diversity knobs (all default OFF; see grpo_env.reset) ---
# Knob 1 (per-world required precision): the success tolerance is drawn as a MULTIPLE of
# the world's measurement noise, NOT an absolute value. Rationale: g is recovered from a
# time measurement (g=2h/t^2 or 4*pi^2*L/T^2), so rel_err(g) ~= 2*noise and averaging k
# trials shrinks it ~1/sqrt(k). A multiplier in [0.8, 3.0] of noise therefore spans
# ~1 measurement (loose: tol ~= 3*noise) to ~8 measurements (tight: tol ~= 0.8*noise),
# and every world is reachable within the rollout budget by construction (an absolute
# range would make the tight end structurally unwinnable at higher noise).
PRECISION_TOL_MIN_MULT = 0.8
PRECISION_TOL_MAX_MULT = 3.0

# Knob 2 (available tools): per-world experiment subset, uniform over these three.
# calculator is always available and is never listed here.
TOOL_SUBSETS = ("drop_ball,pendulum_period", "drop_ball", "pendulum_period")

# Knob 3 (prompt paraphrase): number of scenario-framing templates; MUST match
# len(grpo_env.SCENARIO_INTROS). Index 0 reproduces today's briefing.
N_TEMPLATES = 4

# Per-knob decorrelated RNG seeds (distinct large primes + small offsets) so each draw is
# independent of the gravity draw and of each other. NOTE on the existing RNGs: the gravity
# draw and the per-measurement noise STREAM are BOTH random.Random(seed) (env.py) — the SAME
# seed integer; only the per-world noise MAGNITUDE below uses random.Random(seed*7919+13).
# The new knobs use fresh, mutually-decorrelated multipliers.
_NOISE_MULT_RNG = (7919, 13)
_TOL_RNG = (15485863, 7)
_TOOLS_RNG = (32452843, 17)
_TMPL_RNG = (49979687, 23)

# Disjoint seed ranges so the held-out eval split never overlaps training worlds.
TRAIN_SEED_START = 0
EVAL_SEED_START = 1_000_000

SYSTEM_PROMPT = (
    "You are a careful experimental physicist working in an alien physics lab. "
    "Use the available experiment tools to infer the lab's effective gravity, "
    "then call submit_answer with your best estimate."
)
USER_PROMPT = "Begin the experiment."


def make_dataset(
    n_rows: int,
    *,
    seed_start: int = TRAIN_SEED_START,
    gravity_min: float = GRAVITY_MIN,
    gravity_max: float = GRAVITY_MAX,
    measurement_noise: float = MEASUREMENT_NOISE,
    noise_min: float | None = None,
    noise_max: float | None = None,
    max_tool_calls: int = MAX_TOOL_CALLS,
    vary_precision: bool = False,
    vary_tools: bool = False,
    vary_prompt: bool = False,
) -> Dataset:
    """Build a dataset of ``n_rows`` distinct hidden worlds (one seed each).

    If ``noise_min``/``noise_max`` are given, each world draws its OWN hidden
    measurement noise log-uniformly in that range (reproducibly, from its seed),
    so the optimal number of measurements VARIES across worlds and the agent must
    adaptively decide how much to aggregate — breaking the single-procedure toy.
    Otherwise all worlds use the constant ``measurement_noise``.

    Three procedural-diversity knobs (all default OFF -> today's behavior; see
    grpo_env.reset for how each is consumed). Every per-world value is a deterministic
    function of the row seed (its own decorrelated RNG), so all rollouts of a GRPO group
    share it and the common-random-number contract holds:
      * ``vary_precision``: draw a per-world ``success_tolerance`` = noise * mult, mult
        log-uniform in [PRECISION_TOL_MIN_MULT, PRECISION_TOL_MAX_MULT]. Stated in the
        briefing; tighter worlds demand more aggregation but stay winnable in budget.
      * ``vary_tools``: draw a per-world ``available_tools`` subset (calculator always on).
      * ``vary_prompt``: draw a per-world ``template_idx`` (scenario-framing paraphrase).

    The ``prompt`` column is a fixed stub; ``reset`` appends the per-world briefing.
    All other columns are splatted into ``reset`` as kwargs.
    """
    vary_noise = noise_min is not None and noise_max is not None
    rows = []
    for i in range(n_rows):
        seed = seed_start + i
        if vary_noise:
            # Per-world noise MAGNITUDE (the noise STREAM is Random(seed) in env.py).
            rng = random.Random(seed * _NOISE_MULT_RNG[0] + _NOISE_MULT_RNG[1])
            noise = math.exp(rng.uniform(math.log(noise_min), math.log(noise_max)))
        else:
            noise = measurement_noise

        if vary_precision:
            rng_tol = random.Random(seed * _TOL_RNG[0] + _TOL_RNG[1])
            mult = math.exp(
                rng_tol.uniform(math.log(PRECISION_TOL_MIN_MULT), math.log(PRECISION_TOL_MAX_MULT))
            )
            success_tolerance: float | None = noise * mult
        else:
            success_tolerance = None

        if vary_tools:
            rng_tools = random.Random(seed * _TOOLS_RNG[0] + _TOOLS_RNG[1])
            available_tools: str | None = TOOL_SUBSETS[rng_tools.randrange(len(TOOL_SUBSETS))]
        else:
            available_tools = None

        if vary_prompt:
            rng_tmpl = random.Random(seed * _TMPL_RNG[0] + _TMPL_RNG[1])
            template_idx = rng_tmpl.randrange(N_TEMPLATES)
        else:
            template_idx = 0

        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT},
                ],
                "seed": seed,
                "gravity_min": gravity_min,
                "gravity_max": gravity_max,
                "measurement_noise": noise,
                "max_tool_calls": max_tool_calls,
                "success_tolerance": success_tolerance,
                "available_tools": available_tools,
                "template_idx": template_idx,
            }
        )
    return Dataset.from_list(rows)


def make_splits(
    n_train: int,
    n_eval: int,
    **kwargs: float,
) -> tuple[Dataset, Dataset]:
    """Return (train, eval) datasets over disjoint seed ranges."""
    train = make_dataset(n_train, seed_start=TRAIN_SEED_START, **kwargs)
    eval_ds = make_dataset(n_eval, seed_start=EVAL_SEED_START, **kwargs)
    return train, eval_ds

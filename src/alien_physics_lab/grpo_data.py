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
) -> Dataset:
    """Build a dataset of ``n_rows`` distinct hidden worlds (one seed each).

    If ``noise_min``/``noise_max`` are given, each world draws its OWN hidden
    measurement noise log-uniformly in that range (reproducibly, from its seed),
    so the optimal number of measurements VARIES across worlds and the agent must
    adaptively decide how much to aggregate — breaking the single-procedure toy.
    Otherwise all worlds use the constant ``measurement_noise``.

    The ``prompt`` column is a fixed stub; ``reset`` appends the per-world briefing.
    All other columns are splatted into ``reset`` as kwargs.
    """
    vary = noise_min is not None and noise_max is not None
    rows = []
    for i in range(n_rows):
        seed = seed_start + i
        if vary:
            # Decorrelated from the gravity/measurement RNG so noise != f(gravity).
            rng = random.Random(seed * 7919 + 13)
            noise = math.exp(rng.uniform(math.log(noise_min), math.log(noise_max)))
        else:
            noise = measurement_noise
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

"""TRL v1.x multi-turn GRPO environment wrapper for the alien physics lab.

This adapts :class:`alien_physics_lab.env.AlienPhysicsLab` to TRL's
``environment_factory`` interface (TRL >= 1.0 / requires transformers >= 5.2):

- The class is passed *as a class* to ``GRPOTrainer(environment_factory=...)``.
  TRL creates one instance per rollout via a zero-arg ``__init__``.
- ``reset(**row)`` receives the full dataset row splatted as kwargs and returns
  a string that TRL appends to the last prompt message.
- Every public bound method (not ``reset``, not ``_``-prefixed) becomes a tool.
  TRL builds the JSON schema from type hints + Google-style ``Args:`` docstrings
  via ``transformers.get_json_schema``.
- A tool that raises has its exception caught by TRL and fed back to the model
  as ``{"error": "..."}`` — we use that as the episode-end signal for
  ``submit_answer``.

Reward is read after the episode by :func:`physics_reward` / :func:`validity_reward`,
which receive ``environments=`` (the list of per-rollout instances) from TRL.
"""

from __future__ import annotations

import json
from typing import Any

from alien_physics_lab.env import AlienPhysicsLab
from alien_physics_lab.world import EARTH_GRAVITY_M_S2

# Default world distribution for training. Broadened vs the env default
# (0.5g-2.0g, noise 0.005) per the handoff's reward-hacking notes: a wider
# gravity range breaks fixed-prior shortcuts, and higher noise genuinely
# rewards repeated measurement + aggregation. The dataset normally overrides
# these via the row kwargs; the defaults here are only a safety net.
DEFAULT_GRAVITY_MIN = 0.4 * EARTH_GRAVITY_M_S2
DEFAULT_GRAVITY_MAX = 2.2 * EARTH_GRAVITY_M_S2
DEFAULT_MEASUREMENT_NOISE = 0.03
DEFAULT_MAX_TOOL_CALLS = 5

SUCCESS_BONUS = 0.25
VALIDITY_BONUS = 0.1

# Explicit FLAT incentive to take MULTIPLE measurements (so noise can be averaged out):
# a flat bonus for 2 or more experiments, nothing for 0 or 1. "Experiment" =
# drop_ball / pendulum_period (the budget-consuming calls); calculator is free and does
# not count. Gated on submitting so it can't be farmed by measuring without answering.
MEASUREMENT_BONUS = 0.15
MEASUREMENT_MIN_EXPERIMENTS = 2


class AlienPhysicsGRPOEnv:
    """One alien-physics-lab episode, exposed to TRL as a stateful tool environment."""

    def __init__(self) -> None:
        self._lab: AlienPhysicsLab | None = None
        self.reward: float = 0.0
        self.submitted: bool = False
        self.last_result: dict[str, Any] | None = None

    # ------------------------------------------------------------------ reset
    def reset(
        self,
        *,
        seed: int | None = None,
        gravity_min: float = DEFAULT_GRAVITY_MIN,
        gravity_max: float = DEFAULT_GRAVITY_MAX,
        measurement_noise: float = DEFAULT_MEASUREMENT_NOISE,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
        **_ignored: Any,
    ) -> str:
        """Start a fresh hidden world for this rollout and return the briefing.

        ``**_ignored`` swallows the other dataset columns TRL splats in (e.g.
        ``prompt``). The returned string is appended to the last prompt message.
        """
        self._lab = AlienPhysicsLab.random_gravity_lab(
            seed=seed,
            gravity_min=gravity_min,
            gravity_max=gravity_max,
            measurement_noise=measurement_noise,
            max_tool_calls=max_tool_calls,
        )
        self.reward = 0.0
        self.submitted = False
        self.last_result = None

        briefing = self._lab.instructions()
        # The base instructions end with a "Final answer must be JSON ..." line
        # that is specific to the JSON-protocol runner. In tool mode the model
        # finishes via submit_answer, so drop that line and point at the tool.
        marker = "Final answer must be JSON"
        idx = briefing.find(marker)
        if idx != -1:
            briefing = briefing[:idx].rstrip()
        briefing += (
            "\n\nWhen you have determined the lab's effective gravity, call "
            "submit_answer(gravity_m_s2=<your estimate>) to record your final "
            "answer and end the episode. submit_answer does not consume the "
            "experiment budget. Measurements may vary slightly between trials, "
            "so repeat experiments when precision matters."
        )
        return "\n\n" + briefing

    # ------------------------------------------------------------------ tools
    def drop_ball(self, mass_kg: float, height_m: float) -> str:
        """Drop a ball from a height and measure how long it takes to land.

        Args:
            mass_kg: Mass of the ball in kilograms (positive, up to 10000).
            height_m: Drop height in meters (between 0.1 and 1000).
        """
        return self._call("drop_ball", mass_kg=mass_kg, height_m=height_m)

    def pendulum_period(self, length_m: float) -> str:
        """Measure the swing period of a simple pendulum of a given length.

        Args:
            length_m: Pendulum length in meters (between 0.05 and 100).
        """
        return self._call("pendulum_period", length_m=length_m)

    def calculator(self, expression: str) -> str:
        """Evaluate one arithmetic expression. Free: does not consume budget.

        Args:
            expression: A single arithmetic expression, e.g. "2*10/(1.5**2)".
                No variables, assignments, or semicolons.
        """
        return self._call("calculator", expression=expression)

    def submit_answer(self, gravity_m_s2: float) -> str:
        """Record your final estimate of effective gravity and end the episode.

        Args:
            gravity_m_s2: Best estimate of the lab's effective gravity in m/s^2.
        """
        assert self._lab is not None, "reset() must run before submit_answer()"
        result = self._lab.score_answer({"gravity_m_s2": float(gravity_m_s2)})
        self.reward = float(result.score)
        self.submitted = True
        self.last_result = {
            "score": result.score,
            "success": result.success,
            "relative_error": result.relative_error,
            "true_g": result.true_gravity_m_s2,
            "pred_g": result.predicted_gravity_m_s2,
            "tool_calls": result.tool_calls,
        }
        # Raising is TRL's documented episode-end signal: the reward is already
        # stored, and the exception is fed back so the model stops calling tools.
        raise _EpisodeComplete(
            "Final answer recorded. The episode is complete; stop calling tools."
        )

    # ---------------------------------------------------------------- helpers
    def _call(self, name: str, **kwargs: Any) -> str:
        assert self._lab is not None, "reset() must run before any tool call"
        return json.dumps(self._lab.call_tool(name, **kwargs), sort_keys=True)


class _EpisodeComplete(RuntimeError):
    """Raised by submit_answer to terminate a rollout via TRL's tool-error path."""


# --------------------------------------------------------------------- rewards
def physics_reward(environments: list[AlienPhysicsGRPOEnv], **_kwargs: Any) -> list[float]:
    """Accuracy reward: the lab's continuous score plus a success bonus.

    The lab score is ``max(0, 1 - relative_error / 0.12)`` in [0, 1]; we add a
    bonus when the answer is within the success tolerance (relative_error <=
    0.03). Continuous (not binary) so early policies still get a usable gradient
    — every rollout in a GRPO group shares the same hidden world (same gravity +
    same frozen noise seed), so reward differences reflect only policy quality.
    """
    rewards: list[float] = []
    for env in environments:
        r = float(env.reward)
        if env.last_result and env.last_result.get("success"):
            r += SUCCESS_BONUS
        rewards.append(r)
    return rewards


def validity_reward(environments: list[AlienPhysicsGRPOEnv], **_kwargs: Any) -> list[float]:
    """Small shaping reward: submitted an answer *after* running >=1 experiment.

    Discourages the degenerate policies of never submitting and of submitting a
    prior without experimenting. Kept small so accuracy dominates.
    """
    out: list[float] = []
    for env in environments:
        ran_experiment = env._lab is not None and env._lab.tool_calls > 0
        out.append(VALIDITY_BONUS if (env.submitted and ran_experiment) else 0.0)
    return out


def measurement_reward(environments: list[AlienPhysicsGRPOEnv], **_kwargs: Any) -> list[float]:
    """Flat bonus for taking >= MEASUREMENT_MIN_EXPERIMENTS experiments (and submitting).

    Explicitly incentivizes running multiple measurements so the model can aggregate
    away observation noise. Flat: no extra reward beyond the threshold.
    """
    out: list[float] = []
    for env in environments:
        n_experiments = env._lab.tool_calls if env._lab is not None else 0
        earned = env.submitted and n_experiments >= MEASUREMENT_MIN_EXPERIMENTS
        out.append(MEASUREMENT_BONUS if earned else 0.0)
    return out

"""TRL v1.x multi-turn GRPO environment wrapper for the alien physics lab.

Adapts :class:`alien_physics_lab.env.AlienPhysicsLab` to TRL's ``environment_factory``:
the class is passed (not an instance) to ``GRPOTrainer(environment_factory=...)``; TRL
creates one instance per rollout, splats the dataset row into ``reset(**row)``, and
exposes the public (non-``_``, non-``reset``) methods as tools built from type hints +
Google-style docstrings.

Answer protocol: there is NO ``submit_answer`` tool. The agent runs experiments and, when
done, simply STOPS calling tools and writes its final answer as ``\\boxed{<number>}`` in a
normal message — which is how TRL's multi-turn loop naturally terminates (no tool call in
the latest turn). The reward functions parse that boxed value from the final completion
text and score it against the hidden gravity. This avoids the submit-and-loop pathology of
a tool that raises to "end" the episode.

Per-rollout reward values are exposed two ways: ``log_extra`` adds them as columns to TRL's
wandb completions table, and (if ``GRPO_REWARD_DUMP`` is set) the raw arrays are appended
per step to a JSONL for offline distribution plotting.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from alien_physics_lab.env import AlienPhysicsLab
from alien_physics_lab.world import EARTH_GRAVITY_M_S2

# Broadened world distribution vs the env defaults (handoff reward-hacking notes).
DEFAULT_GRAVITY_MIN = 0.4 * EARTH_GRAVITY_M_S2
DEFAULT_GRAVITY_MAX = 2.2 * EARTH_GRAVITY_M_S2
DEFAULT_MEASUREMENT_NOISE = 0.03
DEFAULT_MAX_TOOL_CALLS = 5

SUCCESS_BONUS = 0.25
VALIDITY_BONUS = 0.1
# Flat bonus for taking >= N experiments (see [[user request]]). Overridable at runtime
# via train_grpo's --measurement-bonus (sets this module global before training).
MEASUREMENT_BONUS = 0.15
MEASUREMENT_MIN_EXPERIMENTS = 2

# Optional raw per-step reward dump (set by train_grpo to out/<run>/reward_dist.jsonl).
_REWARD_DUMP = os.environ.get("GRPO_REWARD_DUMP")

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _final_answer_text(completion: Any) -> str:
    """Text of the final assistant turn (where the boxed answer lives)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):  # multimodal content parts
                    return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        # fallback: any assistant-ish text
        return " ".join(
            m.get("content", "") for m in completion
            if isinstance(m, dict) and isinstance(m.get("content"), str)
        )
    return ""


def parse_boxed_gravity(text: str) -> float | None:
    """Extract the number from the LAST ``\\boxed{...}`` in `text` (after </think> if present).

    Robust to LaTeX/units inside the box (e.g. ``\\boxed{14.7\\,\\text{m/s}^2}``) by taking
    the first number following the box marker. Returns None if there is no boxed value.
    """
    if not text:
        return None
    tail = text.rsplit("</think>", 1)[-1] if "</think>" in text else text
    idx = tail.rfind("\\boxed")
    if idx == -1:
        idx = text.rfind("\\boxed")
        tail = text
    if idx == -1:
        return None
    nums = _NUM_RE.findall(tail[idx + len("\\boxed"):])
    if not nums:
        return None
    try:
        return float(nums[0])
    except ValueError:
        return None


def _dump_rewards(name: str, values: list[float], trainer_state: Any) -> None:
    if not _REWARD_DUMP:
        return
    step = getattr(trainer_state, "global_step", None) if trainer_state is not None else None
    try:
        with open(_REWARD_DUMP, "a") as f:
            f.write(json.dumps({"step": step, "reward": name, "values": [round(float(v), 6) for v in values]}) + "\n")
    except OSError:
        pass


class AlienPhysicsGRPOEnv:
    """One alien-physics-lab episode, exposed to TRL as a stateful tool environment."""

    def __init__(self) -> None:
        self._lab: AlienPhysicsLab | None = None

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
        """Start a fresh hidden world for this rollout and return the briefing."""
        self._lab = AlienPhysicsLab.random_gravity_lab(
            seed=seed,
            gravity_min=gravity_min,
            gravity_max=gravity_max,
            measurement_noise=measurement_noise,
            max_tool_calls=max_tool_calls,
        )
        briefing = self._lab.instructions()
        # Drop the original JSON-final-answer directive; we use a boxed answer instead.
        marker = "Final answer must be JSON"
        idx = briefing.find(marker)
        if idx != -1:
            briefing = briefing[:idx].rstrip()
        briefing += (
            "\n\nRun experiments with the tools to determine the lab's effective gravity. "
            "Measurements may vary slightly between trials, so repeat experiments when "
            "precision matters. When you are confident, STOP calling tools and give your "
            "final answer as a boxed number in m/s^2 on its own, e.g. \\boxed{14.7}. Do "
            "not call a tool for the final answer."
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

    # ---------------------------------------------------------------- helpers
    def _call(self, name: str, **kwargs: Any) -> str:
        assert self._lab is not None, "reset() must run before any tool call"
        return json.dumps(self._lab.call_tool(name, **kwargs), sort_keys=True)


# --------------------------------------------------------------------- rewards
def physics_reward(completions, environments, log_extra=None, **kwargs) -> list[float]:
    """Accuracy reward from the boxed final answer: env score + success bonus.

    Parses ``\\boxed{g}`` from each rollout's final text and scores it against the hidden
    gravity. No boxed value -> 0.0 (teaches the model to actually conclude with the format).
    """
    out: list[float] = []
    for completion, env in zip(completions, environments):
        g = parse_boxed_gravity(_final_answer_text(completion))
        if g is None or env._lab is None:
            out.append(0.0)
            continue
        result = env._lab.score_answer({"gravity_m_s2": float(g)})
        out.append(float(result.score) + (SUCCESS_BONUS if result.success else 0.0))
    if log_extra is not None:
        log_extra("reward_physics", out)
    _dump_rewards("physics", out, kwargs.get("trainer_state"))
    return out


def validity_reward(completions, environments, log_extra=None, **kwargs) -> list[float]:
    """Small shaping reward: produced a parseable boxed answer AND ran >=1 experiment."""
    out: list[float] = []
    for completion, env in zip(completions, environments):
        answered = parse_boxed_gravity(_final_answer_text(completion)) is not None
        ran = env._lab is not None and env._lab.tool_calls > 0
        out.append(VALIDITY_BONUS if (answered and ran) else 0.0)
    if log_extra is not None:
        log_extra("reward_validity", out)
    _dump_rewards("validity", out, kwargs.get("trainer_state"))
    return out


def measurement_reward(completions, environments, log_extra=None, **kwargs) -> list[float]:
    """Flat bonus for taking >= MEASUREMENT_MIN_EXPERIMENTS experiments (and answering).

    "Experiment" = drop_ball / pendulum_period (calculator is free and excluded). Gated on a
    parseable answer so it can't be farmed by measuring without ever answering.
    """
    out: list[float] = []
    for completion, env in zip(completions, environments):
        answered = parse_boxed_gravity(_final_answer_text(completion)) is not None
        n_experiments = env._lab.tool_calls if env._lab is not None else 0
        earned = answered and n_experiments >= MEASUREMENT_MIN_EXPERIMENTS
        out.append(MEASUREMENT_BONUS if earned else 0.0)
    if log_extra is not None:
        log_extra("reward_measurement", out)
    _dump_rewards("measurement", out, kwargs.get("trainer_state"))
    return out

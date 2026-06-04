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

# Knob 3 (prompt paraphrase): scenario-framing variants. ONLY the intro paragraph
# changes between templates; the "Available tools:" list, the precision clause, and the
# boxed-answer directive are byte-identical across all of them (so every reward fn that
# parses \boxed is unaffected and there is no place to leak a hint about g or the noise).
# Index 0 is the canonical wording and MUST equal the intro that env.instructions() emits,
# so template 0 reproduces today's briefing byte-for-byte. Keep these strictly
# semantics-preserving: infer THIS site's effective gravity in m/s^2, nothing given,
# experiment to find it. No real-body names, no gravity values, no mention of noise.
CANONICAL_INTRO = (
    "You are in an alien physics lab. Your task is to infer the lab's "
    "effective gravity in m/s^2 by running experiments. Nothing about the "
    "lab's gravity is given to you in advance: the only way to determine it "
    "is to run experiments and reason from their measurements."
)
SCENARIO_INTROS = [
    CANONICAL_INTRO,
    (
        "You are aboard a derelict survey probe that has just powered up over an "
        "uncharted moon. The probe's local effective gravity in m/s^2 is recorded "
        "nowhere: the only way to determine it is to run experiments and reason from "
        "their measurements."
    ),
    (
        "You are operating a sealed gravimetry calibration bay whose local effective "
        "gravity in m/s^2 has been wiped from the logs. The only way to recover it is to "
        "run experiments and reason from their measurements."
    ),
    (
        "You are a field physicist on an expedition to an unexplored exoplanet. The "
        "site's effective gravity in m/s^2 is unknown and given to you nowhere: the only "
        "way to determine it is to run experiments and reason from their measurements."
    ),
]

SUCCESS_BONUS = 0.25
VALIDITY_BONUS = 0.1
# Measurement reward: saturating ("geometric") reward crediting the 2nd-AND-LATER experiment
# (0 for 0 or 1). Rationale: a reward for the *1st* measurement shrank the lazy-vs-diligent
# advantage gap and made the policy DRIFT DOWN to ~2 experiments + collapse entropy; gating
# at >=2 restores the "you must aggregate" cliff that the earlier flat reward had.
#   reward(n) = MEASUREMENT_REWARD_CAP * (1 - MEASUREMENT_DECAY ** max(0, n - 1))
# n<=1 -> 0; n=2 -> CAP*(1-decay); marginal decays geometrically, sustaining a push toward
# more measurements and tapering toward CAP around n~8-12 with decay=0.8. NOISE-FREE => a
# clean, luck-free signal in the GRPO advantage (best paired with scale_rewards="none" so it
# isn't divided away by the group std). n = #drop_ball + #pendulum_period (calculator
# excluded). CAP/DECAY tunable via train_grpo (--measurement-bonus / --measurement-decay).
MEASUREMENT_REWARD_CAP = 0.5
MEASUREMENT_DECAY = 0.8

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
        self._available: set[str] | None = None  # knob 2: usable experiments (None = all)
        self._template_idx: int = 0  # knob 3: scenario-framing template

    # ------------------------------------------------------------------ reset
    def reset(
        self,
        *,
        seed: int | None = None,
        gravity_min: float = DEFAULT_GRAVITY_MIN,
        gravity_max: float = DEFAULT_GRAVITY_MAX,
        measurement_noise: float = DEFAULT_MEASUREMENT_NOISE,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
        success_tolerance: float | None = None,
        available_tools: str | None = None,
        template_idx: int = 0,
        **_ignored: Any,
    ) -> str:
        """Start a fresh hidden world for this rollout and return the briefing.

        Three optional procedural-diversity knobs, all default OFF (= today's exact
        briefing, byte-for-byte):
          * ``success_tolerance`` (knob 1): per-world required precision. Stated in the
            briefing and used by ``score_answer`` (with a proportional zero-reward band).
          * ``available_tools`` (knob 2): comma-separated usable experiments; the rest are
            soft-disabled (omitted from the briefing; calling one returns an in-band error
            without touching the lab) — calculator is always available.
          * ``template_idx`` (knob 3): which scenario-framing intro to render.
        Every value is a deterministic function of the row seed (drawn in
        ``grpo_data.make_dataset``), so all rollouts in a GRPO group share it (CRN intact).
        """
        self._lab = AlienPhysicsLab.random_gravity_lab(
            seed=seed,
            gravity_min=gravity_min,
            gravity_max=gravity_max,
            measurement_noise=measurement_noise,
            max_tool_calls=max_tool_calls,
            success_tolerance=success_tolerance,
        )
        # Knob 2: resolve which experiments are usable (calculator always on). None = all.
        if available_tools is None:
            self._available = None
        else:
            self._available = {t.strip() for t in available_tools.split(",") if t.strip()}
            self._available.add("calculator")
        self._template_idx = template_idx

        # Tool-list section from instructions() (lists only the available experiments);
        # strip its JSON-answer tail since we use a boxed answer.
        full = self._lab.instructions(available=self._available)
        marker = "Final answer must be JSON"
        idx = full.find(marker)
        body = full[:idx].rstrip() if idx != -1 else full.rstrip()
        # Knob 3: swap ONLY the intro paragraph for the chosen scenario framing (index 0
        # is canonical -> unchanged). Tool list / precision clause / boxed directive stay.
        if 0 < template_idx < len(SCENARIO_INTROS):
            body = body.replace(CANONICAL_INTRO, SCENARIO_INTROS[template_idx], 1)
        # Knob 1: state the required precision as its own paragraph ("" when the knob is
        # OFF, so the assembled string is byte-identical to before).
        if success_tolerance is not None:
            precision_clause = (
                f"\n\nYour estimate must be within {self._lab.success_tolerance * 100:.1f}% "
                "of the true value to count as a success."
            )
        else:
            precision_clause = ""
        boxed_directive = (
            "\n\nRun experiments with the tools to determine the lab's effective gravity. "
            "Measurements may vary slightly between trials, so repeat experiments when "
            "precision matters. When you are confident, STOP calling tools and give your "
            "final answer as a boxed number in m/s^2 on its own, e.g. \\boxed{14.7}. Do "
            "not call a tool for the final answer."
        )
        return "\n\n" + body + precision_clause + boxed_directive

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
        # Knob 2 soft-disable: an experiment not available in this world returns an in-band
        # error WITHOUT touching the lab — so it never advances the frozen noise stream or
        # increments the experiment counter (lab.tool_calls). calculator is always allowed.
        if self._available is not None and name not in self._available:
            return json.dumps(
                {
                    "error": (
                        f"{name} is not available in this lab. "
                        f"Available experiments: {', '.join(sorted(self._available))}."
                    )
                },
                sort_keys=True,
            )
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
    """Saturating ("geometric") reward in the number of experiments (and answering).

    reward(n) = CAP * (1 - DECAY**max(0, n-1)): 0 for n<=1, then monotonically increasing,
    concave, tapering toward CAP (~n=8-12 at decay 0.8) — mirroring the diminishing variance
    reduction of averaging more noisy measurements, while the >=2 gate keeps a single
    measurement unrewarded. n = #drop_ball + #pendulum_period (calculator excluded). Gated on
    a parseable answer so it can't be farmed by measuring without ever answering.
    """
    out: list[float] = []
    n_exps: list[float] = []
    world_noises: list[float] = []
    world_gravities: list[float] = []
    world_tolerances: list[float] = []
    world_tools: list[str] = []
    template_idxs: list[float] = []
    for completion, env in zip(completions, environments):
        answered = parse_boxed_gravity(_final_answer_text(completion)) is not None
        n_experiments = env._lab.tool_calls if env._lab is not None else 0
        beyond_first = max(0, n_experiments - 1)  # credit the 2nd+ measurement only
        bonus = MEASUREMENT_REWARD_CAP * (1.0 - MEASUREMENT_DECAY ** beyond_first)
        out.append(bonus if answered else 0.0)
        # Diagnostics for the adaptive-behavior analysis: per-rollout experiment count
        # alongside the (group-constant) hidden world parameters and the three diversity
        # knobs, so we can ask offline whether the policy adapts — aggregates more on
        # noisier/tighter-tolerance worlds, changes procedure per available tools, is
        # robust across prompt templates. Constant within a group, logged per-rollout to
        # land as parquet columns. Purely additive — does NOT affect the reward value.
        n_exps.append(float(n_experiments))
        if env._lab is not None:
            world_noises.append(float(env._lab.world.measurement_noise))
            world_gravities.append(float(env._lab.world.effective_gravity_m_s2))
            world_tolerances.append(float(env._lab.success_tolerance))
        else:
            world_noises.append(float("nan"))
            world_gravities.append(float("nan"))
            world_tolerances.append(float("nan"))
        avail = getattr(env, "_available", None)
        world_tools.append(",".join(sorted(avail - {"calculator"})) if avail else "all")
        template_idxs.append(float(getattr(env, "_template_idx", 0)))
    if log_extra is not None:
        log_extra("reward_measurement", out)
        log_extra("n_experiments", n_exps)
        log_extra("world_noise", world_noises)
        log_extra("world_gravity", world_gravities)
        log_extra("world_tolerance", world_tolerances)
        log_extra("world_tools", world_tools)
        log_extra("template_idx", template_idxs)
    _dump_rewards("measurement", out, kwargs.get("trainer_state"))
    return out

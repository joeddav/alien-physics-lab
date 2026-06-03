from __future__ import annotations

from dataclasses import dataclass, field
import ast
import json
import math
import random
from typing import Any

from alien_physics_lab.world import EARTH_GRAVITY_M_S2, WorldParams


@dataclass(frozen=True)
class LabResult:
    score: float
    success: bool
    relative_error: float | None
    true_gravity_m_s2: float
    predicted_gravity_m_s2: float | None
    tool_calls: int
    calculator_calls: int
    message: str


@dataclass
class AlienPhysicsLab:
    world: WorldParams
    max_tool_calls: int = 5
    success_tolerance: float = 0.03
    reward_zero_tolerance: float = 0.12
    transcript: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.world.seed)

    @classmethod
    def random_gravity_lab(
        cls,
        *,
        seed: int | None = None,
        gravity_min: float = 0.5 * EARTH_GRAVITY_M_S2,
        gravity_max: float = 2.0 * EARTH_GRAVITY_M_S2,
        measurement_noise: float = 0.005,
        max_tool_calls: int = 5,
    ) -> "AlienPhysicsLab":
        rng = random.Random(seed)
        world = WorldParams(
            gravity_m_s2=rng.uniform(gravity_min, gravity_max),
            measurement_noise=measurement_noise,
            seed=seed,
        )
        return cls(world=world, max_tool_calls=max_tool_calls)

    @property
    def tool_calls(self) -> int:
        return sum(
            1
            for event in self.transcript
            if event["type"] == "tool_call" and event.get("counts_toward_budget", True)
        )

    @property
    def calculator_calls(self) -> int:
        return sum(1 for event in self.transcript if event.get("tool") == "calculator")

    def instructions(self) -> str:
        public = self.world.public_summary()
        return (
            "You are in an alien physics lab. Your task is to infer the lab's "
            "effective gravity in m/s^2. You may run experiments, but each tool "
            f"call consumes budget. Budget: {self.max_tool_calls} calls.\n\n"
            f"Public world parameters: {json.dumps(public, sort_keys=True)}\n\n"
            "Available tools:\n"
            "- drop_ball(mass_kg: positive number, height_m: 0.1 to 1000): "
            "returns lab measurements from a falling-ball trial.\n"
            "- pendulum_period(length_m: 0.05 to 100): returns a lab "
            "measurement from a pendulum trial.\n"
            "- calculator(expression: string) for arithmetic. Calculator calls "
            "do not consume the experiment budget.\n\n"
            "Final answer must be JSON with a numeric gravity_m_s2 field, e.g. "
            '{"gravity_m_s2": 14.715}.'
        )

    def call_tool(self, name: str, **kwargs: Any) -> dict[str, Any]:
        counts_toward_budget = name != "calculator"
        if counts_toward_budget and self.tool_calls >= self.max_tool_calls:
            return {"error": "tool budget exhausted", "remaining_tool_calls": 0}

        if name == "drop_ball":
            result = self._drop_ball(**kwargs)
        elif name == "pendulum_period":
            result = self._pendulum_period(**kwargs)
        elif name == "calculator":
            result = self._calculator(**kwargs)
        else:
            result = {"error": f"unknown tool: {name}"}

        self.transcript.append(
            {
                "type": "tool_call",
                "tool": name,
                "args": dict(kwargs),
                "result": result,
                "counts_toward_budget": counts_toward_budget,
            }
        )
        result["remaining_tool_calls"] = self.max_tool_calls - self.tool_calls
        return result

    def _drop_ball(self, *, mass_kg: float, height_m: float) -> dict[str, Any]:
        mass_kg = _require_range("mass_kg", mass_kg, 0.001, 10_000.0)
        height_m = _require_range("height_m", height_m, 0.1, 1000.0)

        gravity = self.world.effective_gravity_m_s2
        if self.world.atmosphere_drag <= 0:
            time_s = math.sqrt(2.0 * height_m / gravity)
        else:
            time_s, _impact_speed_m_s = self._simulate_drag_drop(
                mass_kg=mass_kg,
                height_m=height_m,
                gravity=gravity,
            )

        return {
            "mass_kg": mass_kg,
            "height_m": height_m,
            "measured_time_s": self._measure(time_s),
        }

    def _pendulum_period(self, *, length_m: float) -> dict[str, Any]:
        length_m = _require_range("length_m", length_m, 0.05, 100.0)
        period_s = 2.0 * math.pi * math.sqrt(length_m / self.world.effective_gravity_m_s2)
        return {
            "length_m": length_m,
            "measured_period_s": self._measure(period_s),
        }

    def _calculator(self, *, expression: str) -> dict[str, Any]:
        if not isinstance(expression, str):
            raise ValueError("expression must be a string")
        if len(expression) > 200:
            raise ValueError("expression must be at most 200 characters")

        value = _safe_calculate(expression)
        return {
            "expression": expression,
            "value": round(value, 10),
        }

    def _simulate_drag_drop(
        self,
        *,
        mass_kg: float,
        height_m: float,
        gravity: float,
    ) -> tuple[float, float]:
        # The drag knob is intentionally dimensionless and simple for playground
        # use. It is not meant as a high-fidelity atmospheric model.
        dt = 0.001
        elapsed = 0.0
        distance = 0.0
        velocity = 0.0
        drag = self.world.atmosphere_drag

        while distance < height_m and elapsed < 10_000.0:
            acceleration = gravity - (drag / mass_kg) * velocity * abs(velocity)
            velocity = max(0.0, velocity + acceleration * dt)
            distance += velocity * dt
            elapsed += dt

        return elapsed, velocity

    def _measure(self, value: float) -> float:
        sigma = abs(value) * self.world.measurement_noise
        noisy = value + self._rng.gauss(0.0, sigma)
        return round(noisy, 6)

    def score_answer(self, answer: str | dict[str, Any]) -> LabResult:
        parsed = _parse_answer(answer)
        true_g = self.world.effective_gravity_m_s2

        if parsed is None:
            return LabResult(
                score=0.0,
                success=False,
                relative_error=None,
                true_gravity_m_s2=true_g,
                predicted_gravity_m_s2=None,
                tool_calls=self.tool_calls,
                calculator_calls=self.calculator_calls,
                message="answer was not valid JSON",
            )

        predicted = parsed.get("gravity_m_s2")
        if predicted is None and "gravity_multiplier" in parsed:
            predicted = float(parsed["gravity_multiplier"]) * EARTH_GRAVITY_M_S2

        if not isinstance(predicted, int | float) or not math.isfinite(float(predicted)):
            return LabResult(
                score=0.0,
                success=False,
                relative_error=None,
                true_gravity_m_s2=true_g,
                predicted_gravity_m_s2=None,
                tool_calls=self.tool_calls,
                calculator_calls=self.calculator_calls,
                message="answer did not contain finite gravity_m_s2",
            )

        predicted_g = float(predicted)
        relative_error = abs(predicted_g - true_g) / true_g
        score = max(0.0, 1.0 - relative_error / self.reward_zero_tolerance)
        success = relative_error <= self.success_tolerance

        return LabResult(
            score=round(score, 4),
            success=success,
            relative_error=relative_error,
            true_gravity_m_s2=true_g,
            predicted_gravity_m_s2=predicted_g,
            tool_calls=self.tool_calls,
            calculator_calls=self.calculator_calls,
            message="ok" if success else "prediction outside success tolerance",
        )


def _require_range(name: str, value: Any, low: float, high: float) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _parse_answer(answer: str | dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(answer, dict):
        return answer

    try:
        parsed = json.loads(answer)
    except json.JSONDecodeError:
        start = answer.find("{")
        end = answer.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(answer[start : end + 1])
        except json.JSONDecodeError:
            return None

    return parsed if isinstance(parsed, dict) else None


def _safe_calculate(expression: str) -> float:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("calculator accepts one arithmetic expression, not statements") from exc

    value = _eval_ast(tree.body)
    if not math.isfinite(value):
        raise ValueError("calculator result must be finite")
    return value


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)

    if isinstance(node, ast.Name):
        constants = {"pi": math.pi, "e": math.e}
        if node.id in constants:
            return constants[node.id]
        raise ValueError(f"unknown calculator name: {node.id}")

    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand

    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            if abs(right) > 10:
                raise ValueError("power exponent is too large")
            return left**right

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or len(node.args) != 1 or node.keywords:
            raise ValueError("calculator supports one-argument functions only")
        value = _eval_ast(node.args[0])
        if node.func.id == "sqrt":
            return math.sqrt(value)
        if node.func.id == "abs":
            return abs(value)

    raise ValueError(f"unsupported calculator expression: {ast.dump(node)}")

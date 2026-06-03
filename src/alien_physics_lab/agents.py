from __future__ import annotations

from dataclasses import asdict
import statistics

from alien_physics_lab.env import AlienPhysicsLab, LabResult


def gravity_from_drop(height_m: float, time_s: float) -> float:
    return 2.0 * height_m / (time_s**2)


def gravity_from_pendulum(length_m: float, period_s: float) -> float:
    return 4.0 * 3.141592653589793**2 * length_m / (period_s**2)


def run_heuristic_agent(lab: AlienPhysicsLab) -> LabResult:
    estimates: list[float] = []

    for height_m in (10.0, 40.0, 160.0):
        observation = lab.call_tool("drop_ball", mass_kg=1.0, height_m=height_m)
        estimates.append(gravity_from_drop(observation["height_m"], observation["measured_time_s"]))

    observation = lab.call_tool("pendulum_period", length_m=5.0)
    estimates.append(gravity_from_pendulum(observation["length_m"], observation["measured_period_s"]))

    answer = {"gravity_m_s2": statistics.fmean(estimates)}
    return lab.score_answer(answer)


def result_to_dict(result: LabResult) -> dict[str, object]:
    data = asdict(result)
    if result.relative_error is not None:
        data["relative_error"] = round(result.relative_error, 6)
    return data

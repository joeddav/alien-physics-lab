import math
import unittest

from alien_physics_lab.agents import gravity_from_drop, run_heuristic_agent
from alien_physics_lab.env import AlienPhysicsLab
from alien_physics_lab.openai_runner import _build_prompt
from alien_physics_lab.world import EARTH_GRAVITY_M_S2, WorldParams


class AlienPhysicsLabTest(unittest.TestCase):
    def test_direct_gravity_world_param_wins(self) -> None:
        world = WorldParams(gravity_m_s2=1.5 * EARTH_GRAVITY_M_S2)
        self.assertAlmostEqual(world.effective_gravity_m_s2, 1.5 * EARTH_GRAVITY_M_S2)

    def test_drop_ball_no_noise_matches_closed_form(self) -> None:
        lab = AlienPhysicsLab(
            world=WorldParams(gravity_m_s2=12.0, measurement_noise=0.0, seed=1),
            max_tool_calls=1,
        )
        observation = lab.call_tool("drop_ball", mass_kg=1.0, height_m=24.0)
        self.assertAlmostEqual(observation["measured_time_s"], 2.0)
        self.assertAlmostEqual(gravity_from_drop(24.0, observation["measured_time_s"]), 12.0)
        self.assertNotIn("measured_impact_speed_m_s", observation)

    def test_pendulum_no_noise_matches_closed_form(self) -> None:
        lab = AlienPhysicsLab(
            world=WorldParams(gravity_m_s2=9.0, measurement_noise=0.0, seed=1),
            max_tool_calls=1,
        )
        observation = lab.call_tool("pendulum_period", length_m=9.0)
        self.assertAlmostEqual(observation["measured_period_s"], 2.0 * math.pi, places=6)

    def test_heuristic_recovers_noisy_gravity(self) -> None:
        lab = AlienPhysicsLab(
            world=WorldParams(gravity_m_s2=1.5 * EARTH_GRAVITY_M_S2, measurement_noise=0.005, seed=2),
            max_tool_calls=5,
        )
        result = run_heuristic_agent(lab)
        self.assertTrue(result.success, result)
        self.assertLessEqual(result.tool_calls, 5)

    def test_calculator_does_not_consume_experiment_budget(self) -> None:
        lab = AlienPhysicsLab(
            world=WorldParams(gravity_m_s2=10.0, measurement_noise=0.0, seed=1),
            max_tool_calls=1,
        )
        observation = lab.call_tool("calculator", expression="2*10/(2**2)")
        self.assertEqual(observation["value"], 5.0)
        self.assertEqual(lab.tool_calls, 0)
        self.assertEqual(lab.calculator_calls, 1)

        lab.call_tool("drop_ball", mass_kg=1.0, height_m=20.0)
        self.assertEqual(lab.tool_calls, 1)
        second_experiment = lab.call_tool("drop_ball", mass_kg=1.0, height_m=20.0)
        self.assertEqual(second_experiment["error"], "tool budget exhausted")

    def test_calculator_rejects_statements(self) -> None:
        lab = AlienPhysicsLab(world=WorldParams(gravity_m_s2=10.0, seed=1))
        with self.assertRaises(ValueError):
            lab.call_tool("calculator", expression="x = 2; x")

    def test_prompt_uses_measurement_language_without_revealing_noise(self) -> None:
        lab = AlienPhysicsLab.random_gravity_lab(seed=1, measurement_noise=0.1)
        prompt = _build_prompt(lab, []).lower()
        self.assertIn("measurement", prompt)
        self.assertIn("measured_time_s", prompt)
        self.assertIn("calculator tool is available", prompt)
        self.assertNotIn("must use the calculator", prompt)
        self.assertNotIn("will be rejected", prompt)
        self.assertNotIn("measurement_noise", prompt)
        self.assertNotIn("noise", prompt)
        self.assertNotIn("noisy", prompt)

    def test_random_gravity_lab_is_reproducible(self) -> None:
        a = AlienPhysicsLab.random_gravity_lab(seed=10)
        b = AlienPhysicsLab.random_gravity_lab(seed=10)
        self.assertEqual(a.world.effective_gravity_m_s2, b.world.effective_gravity_m_s2)


if __name__ == "__main__":
    unittest.main()

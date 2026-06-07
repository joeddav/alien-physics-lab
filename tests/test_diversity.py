"""Tests for the three procedural-diversity knobs (per-world precision, available
tools, prompt paraphrase) added to the GRPO env. The load-bearing guarantee is that
all knobs default OFF == today's exact behavior (byte-for-byte briefing), and that
each knob is CRN-safe (a deterministic function of the row seed)."""

import json
import math
import unittest

from alien_physics_lab.env import REWARD_ZERO_RATIO, AlienPhysicsLab
from alien_physics_lab.grpo_data import (
    PRECISION_TOL_MAX_MULT,
    PRECISION_TOL_MIN_MULT,
    TOOL_SUBSETS,
    make_dataset,
)
from alien_physics_lab.grpo_env import SCENARIO_INTROS, AlienPhysicsGRPOEnv
from alien_physics_lab.world import WorldParams

# The exact instructions() text before the diversity refactor — locks byte-identity of
# the full-tool listing (the OFF path must reproduce this).
ORIGINAL_INSTRUCTIONS = (
    "You are in an alien physics lab. Your task is to infer the lab's "
    "effective gravity in m/s^2 by running experiments. Nothing about the "
    "lab's gravity is given to you in advance: the only way to determine it "
    "is to run experiments and reason from their measurements.\n\n"
    "Available tools:\n"
    "- drop_ball(mass_kg: positive number, height_m: 0.1 to 1000): "
    "returns lab measurements from a falling-ball trial.\n"
    "- pendulum_period(length_m: 0.05 to 100): returns a lab "
    "measurement from a pendulum trial.\n"
    "- calculator(expression: string) for arithmetic.\n\n"
    "Final answer must be JSON with a numeric gravity_m_s2 field, e.g. "
    '{"gravity_m_s2": 14.715}.'
)


class InstructionsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.lab = AlienPhysicsLab(world=WorldParams(gravity_m_s2=10.0, seed=1))

    def test_full_listing_byte_identical(self) -> None:
        self.assertEqual(self.lab.instructions(), ORIGINAL_INSTRUCTIONS)
        self.assertEqual(self.lab.instructions(available=None), ORIGINAL_INSTRUCTIONS)
        # explicit full set must also reproduce it exactly
        self.assertEqual(
            self.lab.instructions(available={"drop_ball", "pendulum_period", "calculator"}),
            ORIGINAL_INSTRUCTIONS,
        )

    def test_subset_omits_disabled_experiment(self) -> None:
        text = self.lab.instructions(available={"pendulum_period", "calculator"})
        self.assertNotIn("drop_ball(", text)
        self.assertIn("pendulum_period(", text)
        self.assertIn("calculator(", text)
        text2 = self.lab.instructions(available={"drop_ball", "calculator"})
        self.assertIn("drop_ball(", text2)
        self.assertNotIn("pendulum_period(", text2)


class ResetOffPathTest(unittest.TestCase):
    """All-knobs-OFF reset must be byte-identical regardless of how OFF is expressed."""

    def test_off_paths_agree_and_have_no_diversity_text(self) -> None:
        env = AlienPhysicsGRPOEnv()
        none_path = env.reset(seed=7)
        explicit = AlienPhysicsGRPOEnv().reset(
            seed=7, available_tools="drop_ball,pendulum_period", template_idx=0, success_tolerance=None
        )
        self.assertEqual(none_path, explicit)
        # Structure of today's briefing is preserved.
        self.assertTrue(none_path.startswith("\n\nYou are in an alien physics lab."))
        self.assertIn("- drop_ball(", none_path)
        self.assertIn("- pendulum_period(", none_path)
        self.assertIn("\\boxed{14.7}", none_path)
        # No knob text leaks in the OFF path.
        self.assertNotIn("within", none_path)  # precision clause absent
        self.assertNotIn("not available in this lab", none_path)


class PrecisionKnobTest(unittest.TestCase):
    def test_tolerance_flows_into_briefing_and_scoring(self) -> None:
        env = AlienPhysicsGRPOEnv()
        briefing = env.reset(seed=3, success_tolerance=0.05)
        self.assertIn("within 5.0%", briefing)
        self.assertAlmostEqual(env._lab.success_tolerance, 0.05)
        self.assertAlmostEqual(env._lab.reward_zero_tolerance, 0.05 * REWARD_ZERO_RATIO)
        # success boundary tracks the per-world tolerance
        true_g = env._lab.world.effective_gravity_m_s2
        ok = env._lab.score_answer({"gravity_m_s2": true_g * 1.04})  # 4% < 5%
        bad = env._lab.score_answer({"gravity_m_s2": true_g * 1.06})  # 6% > 5%
        self.assertTrue(ok.success)
        self.assertFalse(bad.success)

    def test_off_keeps_default_tolerance(self) -> None:
        env = AlienPhysicsGRPOEnv()
        env.reset(seed=3)
        self.assertAlmostEqual(env._lab.success_tolerance, 0.03)
        self.assertAlmostEqual(env._lab.reward_zero_tolerance, 0.12)


class ToolsKnobTest(unittest.TestCase):
    def test_disabled_call_errors_without_touching_lab(self) -> None:
        env = AlienPhysicsGRPOEnv()
        env.reset(seed=11, available_tools="pendulum_period")
        out = json.loads(env.drop_ball(mass_kg=1.0, height_m=20.0))
        self.assertIn("error", out)
        self.assertIn("not available", out["error"])
        self.assertEqual(env._lab.tool_calls, 0)  # disabled call never reached the lab

    def test_soft_disable_does_not_perturb_noise_stream(self) -> None:
        # A disabled call between two real measurements must not advance the frozen
        # noise RNG: the second real reading must match a run without the disabled call.
        a = AlienPhysicsGRPOEnv()
        a.reset(seed=21, available_tools="pendulum_period", measurement_noise=0.05)
        a_r1 = json.loads(a.pendulum_period(length_m=1.0))["measured_period_s"]
        json.loads(a.drop_ball(mass_kg=1.0, height_m=10.0))  # disabled -> error
        a_r2 = json.loads(a.pendulum_period(length_m=1.0))["measured_period_s"]

        b = AlienPhysicsGRPOEnv()
        b.reset(seed=21, available_tools="pendulum_period", measurement_noise=0.05)
        b_r1 = json.loads(b.pendulum_period(length_m=1.0))["measured_period_s"]
        b_r2 = json.loads(b.pendulum_period(length_m=1.0))["measured_period_s"]

        self.assertEqual(a_r1, b_r1)
        self.assertEqual(a_r2, b_r2)
        self.assertEqual(a._lab.tool_calls, 2)

    def test_briefing_lists_only_available(self) -> None:
        env = AlienPhysicsGRPOEnv()
        briefing = env.reset(seed=11, available_tools="pendulum_period")
        self.assertNotIn("- drop_ball(", briefing)
        self.assertIn("- pendulum_period(", briefing)

    def test_calculator_always_available(self) -> None:
        env = AlienPhysicsGRPOEnv()
        env.reset(seed=11, available_tools="pendulum_period")
        out = json.loads(env.calculator(expression="2*3"))
        self.assertEqual(out["value"], 6.0)


class PromptKnobTest(unittest.TestCase):
    def test_template_swaps_intro_only(self) -> None:
        env0 = AlienPhysicsGRPOEnv()
        b0 = env0.reset(seed=9, template_idx=0)
        env2 = AlienPhysicsGRPOEnv()
        b2 = env2.reset(seed=9, template_idx=2)
        self.assertNotEqual(b0, b2)
        # Intro differs, but the tool list + boxed directive are invariant.
        self.assertIn(SCENARIO_INTROS[2], b2)
        self.assertNotIn(SCENARIO_INTROS[2], b0)
        for invariant in ("- drop_ball(", "- pendulum_period(", "\\boxed{14.7}", "Do not call a tool"):
            self.assertIn(invariant, b0)
            self.assertIn(invariant, b2)


class DatasetKnobTest(unittest.TestCase):
    def test_off_columns_are_neutral(self) -> None:
        ds = make_dataset(4, seed_start=0)
        for row in ds:
            self.assertIsNone(row["success_tolerance"])
            self.assertIsNone(row["available_tools"])
            self.assertEqual(row["template_idx"], 0)

    def test_knobs_deterministic_and_in_range(self) -> None:
        kw = dict(vary_precision=True, vary_tools=True, vary_prompt=True,
                  measurement_noise=0.05)
        a = make_dataset(16, seed_start=0, **kw)
        b = make_dataset(16, seed_start=0, **kw)
        for ra, rb in zip(a, b):
            self.assertEqual(ra["success_tolerance"], rb["success_tolerance"])  # CRN
            self.assertEqual(ra["available_tools"], rb["available_tools"])
            self.assertEqual(ra["template_idx"], rb["template_idx"])
            # precision is a multiple of the world noise, within the configured band
            self.assertGreaterEqual(ra["success_tolerance"], 0.05 * PRECISION_TOL_MIN_MULT - 1e-9)
            self.assertLessEqual(ra["success_tolerance"], 0.05 * PRECISION_TOL_MAX_MULT + 1e-9)
            self.assertIn(ra["available_tools"], TOOL_SUBSETS)
            self.assertIn(ra["template_idx"], range(len(SCENARIO_INTROS)))

    def test_precision_couples_to_per_world_noise(self) -> None:
        # With varying noise, the drawn tolerance must scale with each world's own noise.
        ds = make_dataset(32, seed_start=0, vary_precision=True, noise_min=0.02, noise_max=0.15)
        for row in ds:
            mult = row["success_tolerance"] / row["measurement_noise"]
            self.assertGreaterEqual(mult, PRECISION_TOL_MIN_MULT - 1e-9)
            self.assertLessEqual(mult, PRECISION_TOL_MAX_MULT + 1e-9)


class DiameterTaskTest(unittest.TestCase):
    def test_dataset_draws_diameter_worlds(self) -> None:
        from alien_physics_lab.grpo_data import DIAMETER_MAX, DIAMETER_MIN, make_dataset
        a = make_dataset(8, seed_start=0, target="diameter")
        b = make_dataset(8, seed_start=0, target="diameter")
        for ra, rb in zip(a, b):
            self.assertEqual(ra["target"], "diameter")
            self.assertEqual(ra["available_tools"], "measure_horizon_dip")
            self.assertEqual(ra["world_diameter_m"], rb["world_diameter_m"])  # CRN-deterministic
            self.assertGreaterEqual(ra["world_diameter_m"], DIAMETER_MIN - 1)
            self.assertLessEqual(ra["world_diameter_m"], DIAMETER_MAX + 1)

    def test_reset_recovers_diameter_and_soft_disables_gravity_tools(self) -> None:
        env = AlienPhysicsGRPOEnv()
        briefing = env.reset(seed=5, target="diameter", world_diameter_m=8.0e6,
                             available_tools="measure_horizon_dip", measurement_noise=0.0)
        self.assertEqual(env._target, "diameter")
        self.assertEqual(env._lab.target, "diameter")
        self.assertIn("diameter", briefing.lower())
        self.assertIn("measure_horizon_dip", briefing)
        # recover diameter from the (noise-free) dip via R = 2h/alpha^2
        dip = json.loads(env.measure_horizon_dip(height_m=500.0))["measured_dip_deg"]
        alpha = math.radians(dip)
        d_est = 2 * (2 * 500.0 / alpha**2)
        self.assertTrue(env._lab.score_value(d_est).success)
        # gravity tools soft-disabled on a diameter world
        out = json.loads(env.drop_ball(mass_kg=1.0, height_m=20.0))
        self.assertIn("error", out)
        self.assertEqual(env._lab.tool_calls, 1)  # only the dip counted

    def test_gravity_world_soft_disables_horizon_dip(self) -> None:
        env = AlienPhysicsGRPOEnv()
        env.reset(seed=7)  # gravity (default)
        out = json.loads(env.measure_horizon_dip(height_m=500.0))
        self.assertIn("error", out)
        # drop_ball still works on a gravity world
        ok = json.loads(env.drop_ball(mass_kg=1.0, height_m=20.0))
        self.assertIn("measured_time_s", ok)

    def test_score_value_targets_diameter(self) -> None:
        env = AlienPhysicsGRPOEnv()
        env.reset(seed=5, target="diameter", world_diameter_m=8.0e6, available_tools="measure_horizon_dip")
        self.assertTrue(env._lab.score_value(8.0e6 * 1.02).success)   # 2% < 3%
        self.assertFalse(env._lab.score_value(8.0e6 * 1.05).success)  # 5% > 3%


if __name__ == "__main__":
    unittest.main()

# Alien Physics Lab

A tiny RLVR-style playground for active experimental reasoning.

The agent is dropped into an alien physics lab with hidden world parameters and a
small budget of experiment tools. It must run experiments, infer the hidden
physics, and submit a strict final answer that can be scored automatically.

This first version keeps the task deliberately narrow: recover the lab's
effective gravity from noisy measurements.

## World Parameters

The environment accepts these init params:

- `gravity_m_s2`: direct effective gravity at the lab. If omitted, it is computed
  from world mass, diameter, spin, and lab latitude.
- `world_diameter_m`
- `world_spin_rad_s`
- `lab_latitude_deg`
- `world_mass_kg`
- `atmosphere_drag`
- `measurement_noise`
- `seed`

For the initial benchmark, `gravity_m_s2` is randomized and
`atmosphere_drag = 0.0`. Measurement noise defaults to low but non-zero.

## Tools

Current experiment tools:

- `drop_ball(mass_kg, height_m)`: returns measured time to hit ground.
- `pendulum_period(length_m)`: returns period of a simple pendulum.
- `calculator(expression)`: evaluates safe arithmetic expressions. This is a
  helper tool and does not consume the experiment budget.

The environment tracks a fixed tool budget and exposes a verifier that scores
the final answer against hidden state.

## Quickstart

```bash
uv sync --no-editable
.venv/bin/python -m unittest discover -s tests
.venv/bin/alien-lab play --seed 1
.venv/bin/alien-lab eval-heuristic --episodes 20
```

## OpenAI Runner

The runner uses the Responses API with a simple JSON action protocol:

```json
{"action": "tool", "tool": "drop_ball", "args": {"mass_kg": 1, "height_m": 10}}
```

or:

```json
{"action": "final", "gravity_m_s2": 14.7}
```

Run:

```bash
source ~/.bash_profile
.venv/bin/alien-lab eval-openai --model gpt-5-nano --episodes 5
```

The runner defaults to `gpt-5-nano`, the current low-latency/cost small GPT-5
model ID listed in the OpenAI model docs. If another model is available on your
account, pass it with `--model`.

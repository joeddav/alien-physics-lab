# Alien Physics Lab Handoff

This note summarizes the conversation and current project state before moving
work onto the RunPod GPU box.

## Original Idea

The project is a small RLVR-style playground based on an "alien physics lab".
An agent is dropped into an alien environment with a small set of experiment
tools. It must actively run experiments, infer which physical parameter differs
from Earth, and submit a final answer that can be scored automatically.

The first concrete task is deliberately simple: infer effective gravity. The
initial example was a world with gravity around `1.5g`, where the agent can call
something like `drop_ball(mass, height)` and observe how long the ball takes to
hit the ground.

The broader hidden world parameters discussed were:

- gravity strength
- world diameter
- world spin rate
- lab placement on the world
- world mass
- atmosphere thickness / drag

The current code exposes those as initialization parameters, but only the direct
gravity recovery task is currently benchmarked.

## Environment Design Decisions

The current environment is intentionally tiny and deterministic except for
measurement noise.

- `WorldParams` stores hidden alien-world knobs.
- `AlienPhysicsLab` provides tool calls and scoring.
- The default randomized task samples direct effective gravity from
  `0.5g` to `2.0g`.
- The public prompt shows world diameter, spin, lab latitude, and world mass.
- Direct `gravity_m_s2` and `measurement_noise` are hidden.
- The model must submit JSON containing `gravity_m_s2`.
- Experiment tools consume a small budget.
- Calculator calls do not consume experiment budget.
- The harness no longer requires calculator use.

The prompt uses neutral measurement language. It does not explicitly tell the
model that observations are noisy, and it does not include the strings
`noise`, `noisy`, or `measurement_noise`. The idea is that a sufficiently good
experimentalist should infer that measurements may vary and should repeat
experiments when precision matters. This is especially relevant for future GRPO:
the reward can teach agents that repeated measurement and aggregation are useful
without the prompt giving the policy an obvious hint.

## Current Tools

The lab currently exposes:

- `drop_ball(mass_kg, height_m)`: returns `measured_time_s`.
- `pendulum_period(length_m)`: returns `measured_period_s`.
- `calculator(expression)`: evaluates safe arithmetic expressions and is free.

Important change: `drop_ball` used to return measured impact speed too, but that
was removed because it made the gravity inference task too direct.

## Scoring

The scorer compares predicted gravity to hidden effective gravity.

- Success tolerance: relative error <= `0.03`.
- Reward scale: `score = max(0, 1 - relative_error / 0.12)`.
- The `/ 0.12` term is just a reward-shaping knob: estimates 12% or more off
  receive zero reward, and estimates closer than that get linearly increasing
  partial credit.

## Model Evaluation Notes

The current pattern is that models often do not repeat measurements enough, even
when observation noise is substantial.

`gpt-5-nano`, with calculator optional, hidden 10% measurement noise, and impact
speed removed:

- 20 episodes
- success: `4/20`
- mean tool calls: about `1.65`
- mean calculator calls: about `1.1`
- most successes looked like lucky one-shot estimates rather than robust
  aggregation

`gpt-5.5`, high reasoning, hidden 10% measurement noise:

- 3 episodes
- success: `0/3`
- mean tool calls: about `1.67`
- mean calculator calls: about `1.67`
- ran slowly enough that it should not be stopped early

Local Apple Silicon experiments:

- Hugging Face/PyTorch on MPS for `Qwen/Qwen2.5-3B-Instruct` crashed with an
  Apple MPS temporary array size error.
- CPU worked but was slow.
- MLX worked much better locally.
- Cached or tested local models included:
  - `Qwen/Qwen2.5-0.5B-Instruct`
  - `Qwen/Qwen2.5-3B-Instruct`
  - `Qwen/Qwen3-4B-Base`
  - `mlx-community/Phi-4-mini-reasoning-4bit`
  - `mlx-community/Qwen3-4B-Instruct-2507-4bit`

`Qwen/Qwen2.5-3B-Instruct` through MLX:

- 20 episodes at 10% hidden noise
- success: `4/20`
- mean tool calls: `1.0`
- mean calculator calls: `1.0`

`mlx-community/Qwen3-4B-Instruct-2507-4bit`:

- 20 episodes at 10% hidden noise
- success: `1/20`
- mean tool calls: `2.0`
- mean calculator calls: `0.0`
- often took two measurements and then answered close to Earth gravity

## Training Discussion

For real GRPO work, the 4-bit MLX inference model is not the best artifact. It
is fine for local inference and maybe lightweight adapter experiments, but a
clean policy-gradient run should use an official Hugging Face training format,
ideally the BF16 checkpoint if the GPU box can handle it.

Expected behavior if we train GRPO on this one toy problem:

- With a narrow gravity-only distribution, the model may quickly learn brittle
  shortcuts.
- If measurement noise is high enough, the model should be rewarded for taking
  repeated measurements and aggregating.
- The easiest reward hacking risk is learning a default prior or exploiting a
  fixed distribution rather than doing active experimentation.
- To reduce that risk, randomize gravity broadly, vary seeds, keep tool outputs
  minimal, and add held-out distributions.
- The next meaningful version should probably include multiple tool families
  and hidden parameter combinations where one tool alone is insufficient.

## Current Repository State

Package name: `alien-physics-lab`

Useful commands:

```bash
uv sync --no-editable
.venv/bin/python -m unittest discover -s tests
.venv/bin/alien-lab play --seed 1
.venv/bin/alien-lab eval-heuristic --episodes 20
.venv/bin/alien-lab eval-openai --model gpt-5-nano --episodes 5
```

The OpenAI runner expects an API key in the shell environment. On the local
machine, the user indicated that the key is available through the bash profile.

Current unit tests pass locally:

```text
8 tests OK
```

## RunPod Transfer Notes

The target RunPod pod id is `lrlz9ubjhr9c8s`. It was initially unable to start
because the host did not have enough free GPUs. A retry loop was attempted until
the pod eventually reported `RUNNING`.

Do not commit pod environment details or credentials. The RunPod status command
can print environment fields, so any future automation should sanitize output
before logging.

The first smoke test on the pod should be:

```bash
cd /workspace
git clone <private-github-repo-url>
cd alien-physics-lab
uv sync --no-editable
.venv/bin/python -m unittest discover -s tests
.venv/bin/alien-lab play --seed 1
```

If `uv` is not installed on the pod, install or bootstrap it there before trying
to run the package.

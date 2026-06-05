# Alien Physics Lab

A small RLVR-style playground for **active experimental reasoning**: an agent is dropped
into an alien lab whose physics it cannot see, runs experiments to probe it, reasons from
the (noisy) measurements, and submits a single answer that is scored automatically. The
reward is *verifiable* — it comes from the hidden world state, not a learned judge — which
makes the task a clean target for RL.

The current task is deliberately narrow: **recover the lab's effective gravity** `g`
(m/s²) from noisy experiments. Nothing about `g` is given in advance — the only way to
determine it is to experiment and reason. An episode ends when the agent stops calling
tools and writes its final answer as a boxed number, e.g. `\boxed{14.7}`; a parser reads
that value and scores it against the hidden gravity.

Two layers live in this repo:

1. **Environment** (`src/alien_physics_lab/`) — the lab, its tools, and the verifier.
2. **GRPO training** (`scripts/train_grpo.py`) — multi-turn RL (TRL) that trains a model
   to do the experimental reasoning. This is where most of the action is; see
   [Training](#grpo-training) below.

## How the task works

**The world.** Each episode draws a fresh hidden world from a seed. The live task varies
the effective gravity `g` (uniform over ~0.4–2.2× Earth) and applies multiplicative
measurement noise (Gaussian, σ = value × noise) to every reading. The full `WorldParams`
dataclass also supports `atmosphere_drag`, `world_spin_rad_s`, `lab_latitude_deg`,
`world_mass_kg`, and `world_diameter_m`, but those are left at defaults in the current
task. None of the world parameters are shown to the agent — it must infer `g` purely by
experiment. There is **no tool budget**; rollout length is bounded by the training loop.

**The tools.**

- `drop_ball(mass_kg, height_m)` — drop a ball and measure the (noisy) time to land.
  In free fall `g = 2·height / t²`.
- `pendulum_period(length_m)` — measure a simple pendulum's (noisy) period.
  `g = 4·π²·length / T²`.
- `calculator(expression)` — safe arithmetic; not counted as an experiment.

Because each reading is noisy, a single measurement gives ~2·noise relative error in `g`;
averaging `k` independent trials shrinks it ~1/√k. So the core skill is **aggregating
noisy measurements** to the precision the task needs.

**Scoring.** The verifier computes relative error of the boxed answer vs. the hidden `g`:
`score = max(0, 1 − rel_err / 0.12)`, with a success flag at `rel_err ≤ 0.03` (a +0.25
bonus during training). Both tolerances become per-world and are *stated in the prompt*
when the precision knob is enabled (see below).

## Repo layout

| Path | What |
|---|---|
| `src/alien_physics_lab/world.py` | `WorldParams` (hidden state) + effective-gravity model |
| `src/alien_physics_lab/env.py` | `AlienPhysicsLab` — tools, simulation, `score_answer` verifier |
| `src/alien_physics_lab/agents.py` | heuristic baseline agent |
| `src/alien_physics_lab/grpo_env.py` | TRL multi-turn wrapper + reward functions |
| `src/alien_physics_lab/grpo_data.py` | procedural episode/dataset generation |
| `scripts/train_grpo.py` | GRPO training entrypoint |
| `scripts/analyze_aggregation.py` | offline analysis of logged rollouts |
| `docs/grpo_training.md` | full training runbook (stack pins, gotchas) |
| `docs/results/YYYY-MM-DD-grpo.md` | dated experiment logs |

## Quickstart (environment)

The project venv stays dependency-free (just the env + CLI):

```bash
uv sync --no-editable
.venv/bin/python -m unittest discover -s tests
.venv/bin/alien-lab play --seed 1                # step through an episode by hand
.venv/bin/alien-lab eval-heuristic --episodes 20 # run the heuristic baseline
```

## GRPO training

Multi-turn GRPO (TRL v1) trains a model to run experiments and box an answer. Training has
its **own venv** (`/workspace/trl-grpo-venv`) so the project venv stays clean, and must be
launched with the Blackwell/SM120 environment shim:

```bash
export HF_HOME=/workspace/.cache/huggingface
export LD_LIBRARY_PATH=/workspace/trl-grpo-venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export VLLM_LOGGING_LEVEL=WARN TRL_EXPERIMENTAL_SILENCE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ~8-step smoke test (validates the full loop end to end)
/workspace/trl-grpo-venv/bin/python scripts/train_grpo.py --preset smoke

# a real run
/workspace/trl-grpo-venv/bin/python scripts/train_grpo.py \
  --preset real --run-name my-run --max-steps 200 --lr 2e-6 --wandb
```

**Model & setup.** Default model is **Qwen/Qwen3-1.7B** (instruction-tuned, thinking ON —
the experimental reasoning lives in the thinking trace). Full bf16 with colocated vLLM
rollouts on a single GPU. The stack is a pinned, fragile nightly (torch cu129 + vLLM 0.22
nightly on SM120); the exact pins, the CUDA-13 shim, and the critical gotchas
(sleep-mode off, token-level importance sampling, etc.) are documented in
[`docs/grpo_training.md`](docs/grpo_training.md). `--lora` trains an adapter instead of
full fine-tuning (for larger models, e.g. Qwen3-4B).

**Reward.** Three components (`grpo_env.py`):
- `physics_reward` — accuracy of the boxed `g` (relative-error score + success bonus).
- `validity_reward` — small bonus for producing a parseable boxed answer *and* running ≥1 experiment.
- `measurement_reward` — geometric reward in the number of experiments, to incentivize
  aggregating multiple noisy measurements rather than answering off one.

**Common knobs** (any preset value can be overridden per run):

```
--lr --beta --num-generations --per-device-batch --grad-accum --max-steps
--max-completion-length --gpu-mem-util --measurement-noise --no-thinking
--lora --wandb --run-name --notes --tags
```

`--beta` is the KL coefficient (TRL defaults to `0.0` = no KL / no reference model; a small
value, e.g. `0.02`, anchors the policy and prevents long-run divergence).

**Procedural diversity** — the base task is effectively one prompt (only the hidden latent
varies), so optional knobs add input/task-structure variety. All default OFF (byte-identical
to the base task) and are per-seed deterministic (so a GRPO group is homogeneous):

- `--noise-min/--noise-max` — per-world measurement noise (log-uniform) instead of a constant.
- `--vary-precision` — per-world required precision, **stated in the prompt** ("within X%")
  and used by scoring.
- `--vary-tools` — per-world subset of available experiments (drop-only / pendulum-only / both).
- `--vary-prompt` — per-world scenario-framing of the briefing.

**Results & analysis.** Each run logs to `out/grpo-<run-name>/` (per-step completions
parquet, raw reward arrays, metric history) and optionally to Weights & Biases. Analyze a
run's behavior — aggregation depth, and whether it adapts to the diversity knobs — with:

```bash
/workspace/trl-grpo-venv/bin/python scripts/analyze_aggregation.py out/grpo-<run-name>
```

Dated experiment write-ups (findings, what worked, what broke) live in
[`docs/results/`](docs/results/); a condensed running summary is in
[`CLAUDE.md`](CLAUDE.md).

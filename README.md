# Alien Physics Lab

**This is an RL training project.** It trains a local language model — **Qwen3-1.7B**, full
fine-tuning — with multi-turn **GRPO** (TRL) to do *active experimental reasoning*. Dropped into
an alien lab whose physics it cannot observe, the model must run noisy experiments, aggregate the
readings, and submit one boxed answer. The reward is **verifiable** (RLVR): a programmatic verifier
scores the answer against the hidden world state — there is **no learned judge** — which makes it a
clean RL signal. The current task is to **recover the lab's effective gravity** `g` purely by
experiment (nothing about `g` is given; an episode ends when the model stops calling tools and
writes e.g. `\boxed{14.7}`).

The headline work is the **training loop and what the policy learns** — see
[Training](#grpo-training) and the dated write-ups in [`docs/results/`](docs/results/). The
environment, tools, and verifier exist to produce the reward.

> **The model runners are baselines, not the training loop.** The OpenAI / Hugging Face / MLX
> runners (`*_runner.py`) and the `alien-lab eval-*` CLI commands exist only to (a) validate a task
> is well-posed before committing GPU — e.g. benchmarking `gpt-4o-mini` on a new task — and (b)
> compare the trained policy against an untrained baseline. **They are not in the training path**:
> `scripts/train_grpo.py` fine-tunes a local model on-GPU via TRL + colocated vLLM and never imports
> a runner. So "calls the OpenAI API" describes a *baseline harness* here — not how the model is trained.

The repo has two layers:

1. **GRPO training** (`scripts/train_grpo.py`, `src/alien_physics_lab/grpo_env.py`, `grpo_data.py`)
   — the multi-turn RL loop, the reward functions, and procedural episode generation. **This is the
   project**; see [Training](#grpo-training).
2. **Environment** (`src/alien_physics_lab/world.py`, `env.py`) — the hidden world, the experiment
   tools, and the verifier that turns an answer into a reward. Plus the heuristic + model-runner
   baselines and the `alien-lab` CLI for hand-stepping and evaluating.

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
| `src/alien_physics_lab/grpo_env.py` | **TRL multi-turn wrapper + reward functions** (training) |
| `src/alien_physics_lab/grpo_data.py` | **procedural episode/dataset generation** (training) |
| `scripts/train_grpo.py` | **GRPO training entrypoint** — the main loop |
| `src/alien_physics_lab/agents.py` | heuristic baseline agent (*not* training) |
| `src/alien_physics_lab/{openai,hf,mlx}_runner.py` | **baseline backends** — run an OpenAI-API / local-HF / MLX model through the env for task validation + comparison; **not used in training** |
| `src/alien_physics_lab/cli.py` | `alien-lab` CLI: `play` + `eval-{heuristic,openai,hf,mlx}` (baseline evaluation only) |
| `scripts/analyze_aggregation.py` | offline analysis of logged rollouts |
| `docs/grpo_training.md` | full training runbook (stack pins, gotchas) |
| `docs/results/YYYY-MM-DD-grpo.md` | dated experiment logs |

## Quickstart — poke at the environment by hand

*(This explores the task/env and the baselines. For the actual point of the repo, see
[GRPO training](#grpo-training) below.)* The project venv stays dependency-free (just the env + CLI):

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

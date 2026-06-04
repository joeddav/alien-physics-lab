# CLAUDE.md — Alien Physics Lab

RLVR-style playground: an agent is dropped in an alien lab with hidden physics, runs
experiments (drop_ball / pendulum_period / calculator) and submits a scored answer as
`\boxed{g}`. Current task: recover effective gravity. (No tool budget, no public world
params — both removed; gravity is inferred purely by experiment.) Two layers:

1. **Environment** (`src/alien_physics_lab/`): `world.py` (hidden `WorldParams`),
   `env.py` (`AlienPhysicsLab` tools + scoring), `agents.py` (heuristic baseline),
   `openai_runner.py`/`hf_runner.py`/`mlx_runner.py` (model runners), `cli.py`.
2. **GRPO training** (added 2026-06-03): TRL v1 multi-turn RL on this env.

## GRPO training quickstart

Separate training venv (the project venv stays dependency-free):
`/workspace/trl-grpo-venv`. Always launch with the SM120 env shim:

```bash
export HF_HOME=/workspace/.cache/huggingface
export LD_LIBRARY_PATH=/workspace/trl-grpo-venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export VLLM_LOGGING_LEVEL=WARN TRL_EXPERIMENTAL_SILENCE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/workspace/trl-grpo-venv/bin/python scripts/train_grpo.py --preset smoke    # ~8-step loop check
/workspace/trl-grpo-venv/bin/python scripts/train_grpo.py --preset real --run-name foo --max-steps 60 --lr 1e-6
```

Model: **Qwen/Qwen3-1.7B** (instruction-tuned hybrid, thinking ON). Full bf16,
colocated vLLM rollouts. Sweep knobs: `--lr --beta --num-generations --max-steps
--measurement-noise --max-completion-length --gpu-mem-util --no-thinking`.

### Procedural diversity (added 2026-06-04)

The base task is "effectively one prompt" (byte-identical briefing every episode; only the
hidden latent varies). Three opt-in knobs add input/task-structure variety — all default OFF
(byte-identical to before), all CRN-safe (each per-world value is a deterministic `f(seed)`):
`--vary-precision` (per-world required precision drawn `~noise·U[0.8,3.0]`, **stated** in the
briefing as "within X%", used by `score_answer`; makes aggregation load-bearing AND
conditionable), `--vary-tools` (per-world subset of drop_ball/pendulum_period via **soft-disable**
— TRL freezes the tool schema, so a disabled call returns an in-band error and is omitted from
the briefing; calculator always on), `--vary-prompt` (per-world scenario-framing intro; only the
intro paragraph varies). `scripts/analyze_aggregation.py <run-dir>` reports whether behavior
adapts to each knob. Design + adversarial review: `docs/results/2026-06-04-grpo.md`.

## Stack (pinned, Blackwell RTX PRO 6000 / SM120)

torch 2.11.0+cu129 · vLLM 0.22.1 nightly · transformers 5.9.0 · TRL 1.5.1 (NO
`[vllm]` extra). Full pins + install commands: `docs/grpo_training.md`.

## Critical gotchas (full detail in docs/grpo_training.md)

- **SM120 / CUDA-13 shim**: vLLM nightly `_C` needs `libcudart.so.13`; torch is
  cu129. Preload `nvidia/cu13/lib` (`LD_LIBRARY_PATH` + in-script `ctypes`). Required
  for the vLLM EngineCore subprocess too.
- **`enforce_eager=True`**: forced via a `vllm.LLM` monkeypatch — avoids SM120
  CUDA-graph issues. (`grad_norm` healthy, generation correct.)
- **vLLM colocate sleep-mode MUST be OFF** (`vllm_enable_sleep_mode=False`, the default):
  with vLLM 0.22 it runs `collective_rpc("reload_weights")` before each generation, which
  reloads the base checkpoint and **clobbers the synced policy weights → frozen policy, no
  RL** (this invalidated the first overnight sweep — see docs/results/2026-06-03). Fit on
  one GPU instead via low `vllm_gpu_memory_utilization` (0.22) + gradient checkpointing.
- **Token-level importance sampling** (`vllm_importance_sampling_mode="token_truncate"`):
  TRL's default `sequence_mask` underflows `exp(Σ logp-diff)` to ~0 on long thinking
  traces → zero gradient. Token-level keeps it healthy.
- **Qwen3, not Qwen3.5**: 3.5 is multimodal + GDN-hybrid (needs `max_num_batched_tokens`
  override) + nightly-only; 3.x is plain dense and Just Works. (`--max-num-batched-tokens
  2096` + the patch are wired up for a future 3.5 attempt.)
- Background launchers: end with `exit $ec` (a trailing `echo` masks the real exit code).

## Results

Per-date result docs in `docs/results/YYYY-MM-DD-grpo.md`. Latest:
`docs/results/2026-06-04-grpo.md` (measurement-reward design + the aggregation/exploration
problem); `docs/results/2026-06-03-grpo.md` (setup + frozen-policy bug fix).

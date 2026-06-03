# CLAUDE.md — Alien Physics Lab

RLVR-style playground: an agent is dropped in an alien lab with hidden physics, runs
experiments under a tool budget, and submits a scored answer. Current task: recover
effective gravity. Two layers:

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

## Stack (pinned, Blackwell RTX PRO 6000 / SM120)

torch 2.11.0+cu129 · vLLM 0.22.1 nightly · transformers 5.9.0 · TRL 1.5.1 (NO
`[vllm]` extra). Full pins + install commands: `docs/grpo_training.md`.

## Critical gotchas (full detail in docs/grpo_training.md)

- **SM120 / CUDA-13 shim**: vLLM nightly `_C` needs `libcudart.so.13`; torch is
  cu129. Preload `nvidia/cu13/lib` (`LD_LIBRARY_PATH` + in-script `ctypes`). Required
  for the vLLM EngineCore subprocess too.
- **`enforce_eager=True`**: forced via a `vllm.LLM` monkeypatch — avoids SM120
  CUDA-graph issues. (`grad_norm` healthy, generation correct.)
- **vLLM colocate sleep-mode** (`vllm_enable_sleep_mode=True`): without it, vLLM holds
  its memory fraction during the optimizer step → OOM on one GPU.
- **Token-level importance sampling** (`vllm_importance_sampling_mode="token_truncate"`):
  TRL's default `sequence_mask` underflows `exp(Σ logp-diff)` to ~0 on long thinking
  traces → zero gradient. Token-level keeps it healthy.
- **Qwen3, not Qwen3.5**: 3.5 is multimodal + GDN-hybrid (needs `max_num_batched_tokens`
  override) + nightly-only; 3.x is plain dense and Just Works. (`--max-num-batched-tokens
  2096` + the patch are wired up for a future 3.5 attempt.)
- Background launchers: end with `exit $ec` (a trailing `echo` masks the real exit code).

## Results

Per-date result docs in `docs/results/YYYY-MM-DD-grpo.md`. Latest:
`docs/results/2026-06-03-grpo.md`.

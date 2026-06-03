# GRPO Training Runbook

Multi-turn GRPO (TRL v1) on the alien physics lab, on a single **NVIDIA RTX PRO 6000
Blackwell (96 GB, SM120)**. Set up 2026-06-03. This is the full runbook; `CLAUDE.md`
has the condensed version.

## Hardware / environment
- 1× RTX PRO 6000 Blackwell, 96 GB, compute capability 12.0 (SM120), driver 580 / CUDA 13.
- CUDA toolkit 12.8 (`nvcc`) on the host; 256-core Xeon; 2 TB RAM.
- `/workspace` is a network filesystem (RunPod `mfs`), 131 TB free — venv + caches live
  here (the `/` overlay is only 30 GB). `HF_HOME=/workspace/.cache/huggingface`.

## Why this exact stack
The "obvious" stable stack is **unsatisfiable**: TRL's `environment_factory` (multi-turn
envs) requires `transformers>=5.2`, and no *stable* vLLM release works with transformers
5.x for the models we want. So:
- **vLLM 0.22.1 nightly** (`wheels.vllm.ai/nightly`) — supports transformers 5.x.
- **transformers 5.9.0**, **TRL 1.5.1** installed **without** the `[vllm]` extra (the
  extra pins `vllm<=0.18` and would drag transformers below 5, recreating the deadlock).
- **torch 2.11.0+cu129** (uv `--torch-backend` maxes at cu129; there is no cu130 value).
- No `flash-attn` (no SM120 wheel) — vLLM uses **flashinfer** + torch SDPA for training.

### Pins (captured 2026-06-03)
torch 2.11.0+cu129 · vllm 0.22.1rc1.dev108+g4454a1869 · transformers 5.9.0 ·
trl 1.5.1 · accelerate 1.13.0 · datasets 4.8.5 · flashinfer-python 0.6.12 · jmespath 1.1.0.
A full freeze is in `requirements-train-lock.txt` (the nightly is a moving target — pin
the build hash from there to reproduce).

## Install
```bash
export UV_CACHE_DIR=/workspace/.cache/uv
uv venv --python 3.12 /workspace/trl-grpo-venv
uv pip install --python /workspace/trl-grpo-venv/bin/python -e /workspace/alien-physics-lab
# vLLM nightly (dictates torch). NOTE: cu129, NOT cu130 — see the CUDA-13 shim below.
uv pip install --python /workspace/trl-grpo-venv/bin/python -U vllm \
    --torch-backend=cu129 --extra-index-url https://wheels.vllm.ai/nightly
uv pip install --python /workspace/trl-grpo-venv/bin/python \
    transformers==5.9.0 'trl==1.5.1' jmespath 'accelerate>=1.4.0' 'datasets>=4.7.0'
# sanity: transformers must stay 5.x, torch must see SM120
/workspace/trl-grpo-venv/bin/python -c "import transformers,torch;assert transformers.__version__.startswith('5.');print(torch.cuda.get_device_capability())"
```

## The Blackwell / SM120 gotchas (all handled in scripts/train_grpo.py)
1. **CUDA-13 libcudart shim.** The vLLM nightly's `_C` is built for CUDA 13 and links
   `libcudart.so.13`, but uv can only give us cu129 torch (`libcudart.so.12`). The CUDA-13
   runtime (`libcudart.so.13`, `libnvrtc.so.13`) ships in the `nvidia/cu13/lib` wheel
   dir — preload it. The script does this in-process (`_ensure_cuda13_runtime`, `ctypes`
   `RTLD_GLOBAL`), AND you must export `LD_LIBRARY_PATH=.../nvidia/cu13/lib` so vLLM's
   **EngineCore subprocess** (spawned separately) finds it too. Benign mismatch: vLLM's
   `_C` just needs the CUDA-13 *runtime*; torch keeps using its cu12.9 math libs. A
   `Failed to get device capability: SM 12.x requires CUDA >= 12.9` warning is expected
   and non-fatal (generation is correct).
2. **`enforce_eager=True`.** Forced onto every `vllm.LLM(...)` via a monkeypatch
   (`_patch_vllm_llm`) since TRL exposes no passthrough. Avoids SM120 CUDA-graph capture
   issues. Pass `--no-enforce-eager` to try graphs once verified.
3. **vLLM sleep-mode** (`vllm_enable_sleep_mode=True`). Colocate on one GPU: vLLM offloads
   weights+KV to CPU during the optimizer step and wakes for generation, so vLLM and the
   full-bf16 optimizer time-share VRAM. Without it → OOM (vLLM holds ~0.45·96 GB the whole
   step). We also run at `vllm_gpu_memory_utilization=0.30` + `gradient_checkpointing` +
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
4. **Token-level importance sampling** (`vllm_importance_sampling_mode="token_truncate"`).
   TRL's default `sequence_mask` computes `exp(Σ_t (logp_HF − logp_vLLM))` over the whole
   sequence; on 3k+ token thinking traces that underflows to ~1e-26, scaling the loss to 0
   (`grad_norm≈1e-10`). Token-level keeps per-token ratios ~1 (clamped at cap=3), so the
   correction works and gradients flow (`grad_norm≈2-3`).

## Model
**Qwen/Qwen3-1.7B** — instruction-tuned hybrid, thinking ON (the task rewards active
experimental reasoning, which lives in the thinking trace). Full bf16. We deliberately
avoid Qwen3.5 (multimodal vision tower + Gated-DeltaNet hybrid needing
`max_num_batched_tokens=2096` + nightly-only loading). A future 3.5 attempt: add
`--max-num-batched-tokens 2096` (the monkeypatch force-overrides TRL's hardcoded 4096).

## How it maps to the env
- `src/alien_physics_lab/grpo_env.py`: `AlienPhysicsGRPOEnv` (TRL `environment_factory`):
  `reset(**row)` samples a hidden world and returns the briefing; tools `drop_ball`,
  `pendulum_period`, `calculator`, `submit_answer` (the last stores reward and raises to
  end the episode). Rewards: `physics_reward` (env score + success bonus) and
  `validity_reward` (submitted after experimenting).
- `src/alien_physics_lab/grpo_data.py`: one hidden world per row (distinct seed). TRL
  repeats each row `num_generations` times → every rollout in a group shares the SAME
  gravity + frozen noise seed, so within-group reward variance reflects only policy
  quality (the GRPO advantage is meaningful) while noise still varies *within* an episode
  (so repeat-and-average is rewarded). Broadened gravity 0.4g–2.2g, noise 0.03, disjoint
  held-out eval seeds.

## Running
Launch (note `exit $ec` so background task exit codes are truthful — a trailing `echo`
masks them):
```bash
export HF_HOME=/workspace/.cache/huggingface
export LD_LIBRARY_PATH=/workspace/trl-grpo-venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export VLLM_LOGGING_LEVEL=WARN TRL_EXPERIMENTAL_SILENCE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/workspace/trl-grpo-venv/bin/python scripts/train_grpo.py --preset smoke
```
Each run writes `out/grpo-<run-name>/log_history.json` (full TRL metric history) and the
final model. Read reward trends from `rewards/physics_reward/mean` over steps.

#!/usr/bin/env python
"""GRPO training entrypoint for the alien physics lab (TRL v1.x, multi-turn).

Usage:
    python scripts/train_grpo.py --preset smoke                 # fast loop validation
    python scripts/train_grpo.py --preset real --max-steps 120  # a real/sweep run

Default model is Qwen/Qwen3-1.7B (instruction-tuned hybrid) with THINKING ENABLED
-- the task rewards active experimental reasoning, which lives in the thinking
trace. Presets set sensible batch/length/step defaults; any hyperparameter can be
overridden per-run via the flags below (used by the sweep). Pass --no-thinking to
isolate the bare tool-use loop.

Blackwell/SM120 notes:
  * vLLM nightly is CUDA-13; torch is cu129 -> we preload the cu13 libcudart shim.
  * enforce_eager avoids SM120 CUDA-graph issues (forced via a vllm.LLM patch).
  * colocate uses vLLM sleep-mode so vLLM + the full-bf16 optimizer time-share VRAM.
  * token-level importance sampling avoids gradient underflow on long traces.
"""

from __future__ import annotations

import argparse
import json
import os


def _ensure_cuda13_runtime() -> None:
    """Preload the CUDA-13 runtime vLLM's _C extension links against.

    The vLLM nightly is built for CUDA 13 (NEEDED libcudart.so.13), but uv's
    --torch-backend tops out at cu129, so torch ships libcudart.so.12. The
    CUDA-13 runtime libs ride along in the nvidia/cu13 wheel; preloading them
    RTLD_GLOBAL lets vllm._C resolve libcudart.so.13 against a cu129 torch (the
    documented SM120 shim). Safe no-op if the libs are absent or already loaded.
    """
    import ctypes
    import glob
    import os as _os
    import sysconfig

    libdir = _os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cu13", "lib")
    for so in sorted(glob.glob(_os.path.join(libdir, "*.so*"))):
        try:
            ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


def _patch_vllm_llm(**overrides) -> None:
    """Force keyword overrides onto every ``vllm.LLM(...)`` construction.

    TRL 1.5.1's colocate path (trl/generation/vllm_generation.py) builds the
    engine via ``vllm.LLM(...)`` with a HARDCODED ``max_num_batched_tokens=4096``
    and no ``enforce_eager``. Neither is exposed through GRPOConfig, so we patch
    the constructor. Overrides are FORCED (``kw.update``, not ``setdefault``)
    because TRL passes some of these explicitly. Used to:
      - force ``enforce_eager=True`` (avoids SM120 CUDA-graph capture issues),
      - force ``max_num_batched_tokens=2096`` for GDN-hybrid models (e.g. Qwen3.5).
    """
    import vllm

    _orig = vllm.LLM.__init__

    def _patched(self, *args, **kwargs):  # noqa: ANN001
        kwargs.update(overrides)
        return _orig(self, *args, **kwargs)

    vllm.LLM.__init__ = _patched


def build_config(preset: str, *, thinking: bool, output_dir: str, overrides: dict | None = None):
    from trl import GRPOConfig

    kwargs: dict[str, object] = dict(
        output_dir=output_dir,
        use_vllm=True,
        vllm_mode="colocate",
        # MUST stay False: with vLLM 0.22, sleep-mode runs collective_rpc("reload_weights")
        # before each generation, which reloads the ORIGINAL checkpoint and clobbers the
        # policy weights sync_weights just pushed -> the policy is frozen at base and no RL
        # happens. We fit on one GPU instead via a low vLLM mem fraction + grad checkpointing.
        vllm_enable_sleep_mode=False,
        vllm_importance_sampling_correction=True,
        # Token-level (not the default "sequence_mask"): exp(sum of per-token
        # vLLM<->HF logp diffs) underflows to ~0 on long thinking traces and zeros
        # the gradient. Per-token ratios stay ~1 (clamped at the cap).
        vllm_importance_sampling_mode="token_truncate",
        beta=0.0,  # KL off -> no reference model loaded (saves a model + a forward)
        scale_rewards="group",
        num_iterations=1,
        learning_rate=1e-6,
        bf16=True,
        # Exploration: GRPO needs within-group rollout DIVERSITY for non-zero
        # advantage, so keep temperature ~1.0 (NOT Qwen3's inference-time 0.6).
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        chat_template_kwargs={"enable_thinking": thinking},
        log_completions=True,
        report_to="none",
        gradient_checkpointing=True,
        vllm_gpu_memory_utilization=0.22,
    )

    if preset == "smoke":
        kwargs.update(
            num_generations=4,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            vllm_max_model_length=6144,
            max_completion_length=4096 if thinking else 1536,
            max_tool_calling_iterations=10,
            max_steps=8,
            logging_steps=1,
            save_strategy="no",
        )
    else:  # "real"
        # Memory profile matched to the validated smoke (per-backward = 4 seqs x 4096
        # tokens): keep per_device_train_batch_size=4 and reach the group size of 8 via
        # gradient accumulation (peak activation memory tracks the micro-batch, not the
        # group). With sleep-mode OFF (required — see above), vLLM holds its fraction the
        # whole step, so a low gpu fraction (0.22, ~21 GB) leaves headroom for the full-bf16
        # backward. per_device=8 x 6144 OOMs on backward (~3x this profile); don't raise blindly.
        kwargs.update(
            num_generations=8,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            vllm_gpu_memory_utilization=0.22,
            vllm_max_model_length=6144,
            max_completion_length=4096 if thinking else 3072,
            max_tool_calling_iterations=12,
            num_train_epochs=1,
            logging_steps=2,
            save_strategy="no",  # sweep runs save only the final model (save_model below)
        )

    if overrides:
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return GRPOConfig(**kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", choices=("smoke", "real"), default="smoke")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--run-name", default=None, help="Names the output dir (out/grpo-<run-name>).")
    ap.add_argument("--thinking", dest="thinking", action="store_true", default=True)
    ap.add_argument("--no-thinking", dest="thinking", action="store_false")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--wandb", action="store_true", help="Log to Weights & Biases (report_to=wandb).")
    ap.add_argument("--wandb-project", default="alien-physics-grpo")
    # vLLM engine overrides (forced into vllm.LLM via monkeypatch).
    ap.add_argument(
        "--max-num-batched-tokens", type=int, default=None,
        help="Force vLLM scheduler cap (set 2096 for GDN-hybrid models like Qwen3.5).",
    )
    ap.add_argument("--enforce-eager", dest="enforce_eager", action="store_true", default=True)
    ap.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    # Sleep-mode MUST stay off (default False): its reload_weights workaround clobbers the
    # synced policy weights every step, freezing the policy at base. --sleep-mode to re-enable
    # (only safe if the vLLM/TRL reload_weights interaction is fixed).
    ap.add_argument("--sleep-mode", dest="sleep_mode", action="store_true", default=False)
    ap.add_argument("--no-sleep-mode", dest="sleep_mode", action="store_false")
    # Per-run hyperparameter overrides (None -> keep preset value). Used by the sweep.
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None, help="KL coefficient (0 = off).")
    ap.add_argument("--num-generations", type=int, default=None, help="GRPO group size.")
    ap.add_argument("--per-device-batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--max-completion-length", type=int, default=None)
    ap.add_argument("--gpu-mem-util", type=float, default=None)
    ap.add_argument("--measurement-noise", type=float, default=None, help="Hidden world noise (default 0.03).")
    ap.add_argument("--measurement-bonus", type=float, default=None,
                    help="Override the measurement-reward CAP (asymptote of the geometric reward; default 0.5).")
    ap.add_argument("--measurement-decay", type=float, default=None,
                    help="Geometric decay r for the measurement reward (default 0.7; lower => saturates sooner).")
    ap.add_argument("--save-final", dest="save_final", action="store_true", default=True)
    ap.add_argument("--no-save-final", dest="save_final", action="store_false")
    ap.add_argument("--train-rows", type=int, default=None)
    ap.add_argument("--eval-rows", type=int, default=None)
    args = ap.parse_args()

    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
    if args.wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    output_dir = args.output_dir or f"out/grpo-{args.run_name or args.preset}"
    os.makedirs(output_dir, exist_ok=True)
    # Raw per-step per-rollout reward arrays for offline distribution plotting (read at
    # grpo_env import time below; also surfaced per-rollout in the wandb completions table).
    os.environ["GRPO_REWARD_DUMP"] = os.path.join(output_dir, "reward_dist.jsonl")

    _ensure_cuda13_runtime()

    vllm_overrides: dict[str, object] = {}
    if args.enforce_eager:
        vllm_overrides["enforce_eager"] = True
    if args.max_num_batched_tokens:
        vllm_overrides["max_num_batched_tokens"] = args.max_num_batched_tokens
    if vllm_overrides:
        _patch_vllm_llm(**vllm_overrides)

    from trl import GRPOTrainer

    from alien_physics_lab.grpo_data import make_splits
    from alien_physics_lab import grpo_env as _grpo_env
    from alien_physics_lab.grpo_env import (
        AlienPhysicsGRPOEnv,
        measurement_reward,
        physics_reward,
        validity_reward,
    )

    if args.measurement_bonus is not None:
        _grpo_env.MEASUREMENT_REWARD_CAP = args.measurement_bonus
        print(f"[train_grpo] measurement reward cap -> {args.measurement_bonus}")
    if args.measurement_decay is not None:
        _grpo_env.MEASUREMENT_DECAY = args.measurement_decay
        print(f"[train_grpo] measurement decay -> {args.measurement_decay}")

    overrides = {
        "learning_rate": args.lr,
        "beta": args.beta,
        "num_generations": args.num_generations,
        "per_device_train_batch_size": args.per_device_batch,
        "gradient_accumulation_steps": args.grad_accum,
        "max_steps": args.max_steps,
        "max_completion_length": args.max_completion_length,
        "vllm_gpu_memory_utilization": args.gpu_mem_util,
        "vllm_enable_sleep_mode": args.sleep_mode,
        "report_to": "wandb" if args.wandb else None,
        "run_name": args.run_name,
    }
    cfg = build_config(args.preset, thinking=args.thinking, output_dir=output_dir, overrides=overrides)

    # GRPO requires the global generation batch to be divisible by the group size.
    eff = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    assert eff % cfg.num_generations == 0, (
        f"per_device_train_batch_size*grad_accum ({eff}) must be divisible by "
        f"num_generations ({cfg.num_generations})"
    )

    if args.preset == "smoke":
        n_train, n_eval = args.train_rows or 16, args.eval_rows or 4
    else:
        n_train, n_eval = args.train_rows or 8192, args.eval_rows or 64
    ds_kwargs: dict[str, float] = {}
    if args.measurement_noise is not None:
        ds_kwargs["measurement_noise"] = args.measurement_noise
    train_ds, eval_ds = make_splits(n_train, n_eval, **ds_kwargs)

    print(
        f"[train_grpo] model={args.model} preset={args.preset} run={args.run_name} "
        f"thinking={args.thinking} lr={cfg.learning_rate} beta={cfg.beta} "
        f"G={cfg.num_generations} bsz={cfg.per_device_train_batch_size}x{cfg.gradient_accumulation_steps} "
        f"max_steps={cfg.max_steps} max_completion_length={cfg.max_completion_length} "
        f"noise={ds_kwargs.get('measurement_noise', 0.03)} train_rows={n_train}"
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[physics_reward, validity_reward, measurement_reward],
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        environment_factory=AlienPhysicsGRPOEnv,
    )
    trainer.train()

    # Persist the final model + full metric history for downstream documentation.
    if args.save_final:
        trainer.save_model(output_dir)
    try:
        with open(os.path.join(output_dir, "log_history.json"), "w") as f:
            json.dump(trainer.state.log_history, f, indent=2, default=str)
        print(f"[train_grpo] wrote {output_dir}/log_history.json ({len(trainer.state.log_history)} entries)")
    except Exception as exc:  # noqa: BLE001
        print(f"[train_grpo] WARN: could not dump log_history: {exc}")


if __name__ == "__main__":
    main()

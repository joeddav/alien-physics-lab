#!/usr/bin/env bash
# Overnight LONG run: --vary-tools for 1000 steps to see whether the still-climbing,
# non-tapering reward trajectory eventually plateaus / how far aggregation + accuracy go.
# Identical config to rl-vary-tools-g8 (G8, lr 2e-6, noise 0.08, completion 4096) just
# 5x the steps. --no-save-final (unattended overnight; avoid any end-of-run disk crash;
# analysis is via the per-step completions parquet, not the checkpoint).
set -uo pipefail
cd /workspace/alien-physics-lab
export HF_HOME=/workspace/.cache/huggingface/
export LD_LIBRARY_PATH=/workspace/trl-grpo-venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export VLLM_LOGGING_LEVEL=WARN TRL_EXPERIMENTAL_SILENCE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "START $(date +%H:%M:%S)"
/workspace/trl-grpo-venv/bin/python scripts/train_grpo.py \
  --preset real --run-name rl-vary-tools-g8-1k \
  --vary-tools --measurement-noise 0.08 \
  --lr 2e-6 --num-generations 8 --per-device-batch 4 --grad-accum 2 \
  --max-completion-length 4096 --max-steps 1000 --no-save-final \
  --wandb --tags "vary-tools,diverse-task,g8,lr2e-6,1.7b,long-1k" \
  --notes "OVERNIGHT 1000-step continuation of rl-vary-tools-g8 (same config). The 200-step run's reward was still climbing with no taper; this tests where it plateaus and how far aggregation/accuracy go. Watch for late entropy collapse or reward-hacking."
ec=$?
echo "DONE ec=$ec $(date +%H:%M:%S)"
exit $ec

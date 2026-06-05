#!/usr/bin/env bash
# STABILIZED long run. The previous 1000-step run learned well (aggregation -> n~9,
# reward ~1.2 by ~step 300) then DIVERGED (~step 380): entropy ran away upward, the policy
# emitted garbage (physics->0, clip 0.75). Cause: beta=0 (no KL anchor) + sustained lr 2e-6
# => unbounded policy drift over a long run (linear lr-decay alone didn't prevent it).
# Fix: add a KL penalty (beta 0.02) to anchor the policy to the base. KL loads a reference
# model, so make room: per_device 2 x grad_accum 4 (= G8, half the activation memory) and a
# lower vLLM fraction (0.18). Same task/lr otherwise. Watch entropy in BOTH directions.
set -uo pipefail
cd /workspace/alien-physics-lab
export HF_HOME=/workspace/.cache/huggingface/
export LD_LIBRARY_PATH=/workspace/trl-grpo-venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export VLLM_LOGGING_LEVEL=WARN TRL_EXPERIMENTAL_SILENCE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "START $(date +%H:%M:%S)"
/workspace/trl-grpo-venv/bin/python scripts/train_grpo.py \
  --preset real --run-name rl-vary-tools-g8-kl \
  --vary-tools --measurement-noise 0.08 \
  --lr 2e-6 --beta 0.02 --num-generations 8 --per-device-batch 2 --grad-accum 4 \
  --gpu-mem-util 0.18 --max-completion-length 4096 --max-steps 1000 --no-save-final \
  --wandb --tags "vary-tools,g8,lr2e-6,kl0.02,long-1k,stabilized" \
  --notes "Stabilized re-run of the 1000-step vary-tools after it diverged (~step 380: entropy runaway, physics->0). Added KL beta=0.02 to anchor the policy (loads a ref model -> per_device 2/ga 4 + vLLM 0.18 for memory). Q: does KL hold the gains (n~9, reward ~1.2) WITHOUT diverging, and without over-suppressing aggregation toward the base's n~1-2?"
ec=$?
echo "DONE ec=$ec $(date +%H:%M:%S)"
exit $ec

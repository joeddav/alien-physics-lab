#!/usr/bin/env bash
# beta-DOWN probe. beta=0.02 prevented divergence but over-anchored: reward plateaued
# ~0.63 and aggregation eroded to ~3.5 (KL slowly pulling toward the base's n~1-2), vs the
# no-KL run's ~1.0 reward / n~9 before it blew up. Try beta=0.01 (half): weaker anchor ->
# more room to learn/aggregate, hopefully still enough to prevent the catastrophic runaway.
# Otherwise identical to the beta=0.02 run (G8, lr 2e-6, per_device 2/ga 4, vLLM 0.18 for
# the ref model, noise 0.08, completion 4096, 1000 steps). KEY: does reward recover toward
# ~1.0 / aggregation toward n~9, AND does entropy stay bounded (no runaway like beta=0)?
set -uo pipefail
cd /workspace/alien-physics-lab
export HF_HOME=/workspace/.cache/huggingface/
export LD_LIBRARY_PATH=/workspace/trl-grpo-venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export VLLM_LOGGING_LEVEL=WARN TRL_EXPERIMENTAL_SILENCE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "START $(date +%H:%M:%S)"
/workspace/trl-grpo-venv/bin/python scripts/train_grpo.py \
  --preset real --run-name rl-vary-tools-g8-kl01 \
  --vary-tools --measurement-noise 0.08 \
  --lr 2e-6 --beta 0.01 --num-generations 8 --per-device-batch 2 --grad-accum 4 \
  --gpu-mem-util 0.18 --max-completion-length 4096 --max-steps 1000 --no-save-final \
  --wandb --tags "vary-tools,g8,lr2e-6,kl0.01,long-1k,beta-down" \
  --notes "beta-down probe: beta=0.01 (half of the 0.02 run that plateaued ~0.63 / aggregation eroded to 3.5). Same config otherwise. Does the weaker anchor recover reward (~1.0)/aggregation (n~9) while still preventing the beta=0 entropy runaway (which hit ~step 380)? Watch entropy in BOTH directions."
ec=$?
echo "DONE ec=$ec $(date +%H:%M:%S)"
exit $ec

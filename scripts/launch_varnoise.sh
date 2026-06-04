#!/usr/bin/env bash
# Relaunch of the per-world varying-noise diverse task. Identical config to the
# proven `rl-lowlr-g16-noise08` run (per_device=4, ga=4, G16, lr 1e-6, completion
# 4096, vllm 0.22) EXCEPT for the varying hidden noise. The prior attempt OOM'd
# only because it used completion 5120; 4096 is the memory-safe, proven value.
set -uo pipefail
cd /workspace/alien-physics-lab
export HF_HOME=/workspace/.cache/huggingface/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # cheap anti-fragmentation headroom

echo "START $(date +%H:%M:%S)"
/workspace/trl-grpo-venv/bin/python scripts/train_grpo.py \
  --preset real --run-name rl-varnoise-g16 \
  --noise-min 0.02 --noise-max 0.15 \
  --lr 1e-6 --num-generations 16 --per-device-batch 4 --grad-accum 4 \
  --max-completion-length 4096 --max-steps 200 \
  --wandb --tags "varying-noise,diverse-task,g16,lowlr,1.7b" \
  --notes "Diverse task: per-world hidden noise log-uniform [0.02,0.15] so the optimal #measurements varies and the agent must adaptively aggregate. Same config as rl-lowlr-g16-noise08 (G16, lr1e-6, completion4096) for a controlled comparison vs constant noise 0.08. Relaunch after a 5120-completion OOM."
ec=$?
echo "DONE ec=$ec $(date +%H:%M:%S)"
exit $ec

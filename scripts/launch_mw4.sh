#!/usr/bin/env bash
# MULTIPLE WORLDS PER STEP. Until now every optimizer step trained on ONE world
# (per_device*grad_accum = num_generations = 8 -> 1 group/step), so each update was
# dominated by one world's idiosyncrasies (and the reward curve oscillated wildly).
# Fix: grad_accum 4->16 with per_device 2, G8 -> generation batch 32 -> 4 DISTINCT
# worlds x 8 generations per optimizer step (TRL RepeatSampler: generation_batch_size/
# num_generations = 4 unique prompts). grad_accum is sequential so memory is unchanged
# (per_device still 2 + KL ref model, ~75 GiB). Each step ~4x more rollouts -> lower-
# variance gradient + smoother reward; fewer total steps for similar world-coverage.
# Same task/anchor as the sweep winner: vary-tools, beta=0.005, lr 2e-6, noise 0.08.
set -uo pipefail
cd /workspace/alien-physics-lab
export HF_HOME=/workspace/.cache/huggingface/
export LD_LIBRARY_PATH=/workspace/trl-grpo-venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export VLLM_LOGGING_LEVEL=WARN TRL_EXPERIMENTAL_SILENCE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "START $(date +%H:%M:%S)"
/workspace/trl-grpo-venv/bin/python scripts/train_grpo.py \
  --preset real --run-name rl-mw4-kl005 \
  --vary-tools --measurement-noise 0.08 \
  --lr 2e-6 --beta 0.005 --num-generations 8 --per-device-batch 2 --grad-accum 16 \
  --gpu-mem-util 0.18 --max-completion-length 4096 --max-steps 250 --no-save-final \
  --wandb --tags "vary-tools,g8,kl0.005,4worlds-per-step,lr2e-6" \
  --notes "4 WORLDS PER OPTIMIZER STEP (grad_accum 16, per_device 2, G8 -> gen_batch 32 -> 4 distinct worlds x 8 gens). vs all prior runs which were 1 world/step. Same anchor/task as the beta=0.005 sweep winner. Expect: smoother train/reward (between-world variance cut ~4x) + lower-variance gradient. 250 steps (lower-variance updates; ~1000 world-exposures, comparable to the 1000-step single-world runs)."
ec=$?
echo "DONE ec=$ec $(date +%H:%M:%S)"
exit $ec

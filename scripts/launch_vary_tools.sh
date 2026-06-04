#!/usr/bin/env bash
# First procedural-diversity training run: knob 2 (--vary-tools) in isolation.
# Per-world tool subset {drop_ball | pendulum_period | both} via soft-disable; the
# model must adapt its procedure to which experiment(s) are available. Same proven
# memory-safe config as rl-lowlr-g16 / rl-varnoise-g16 (G16, lr 1e-6, completion 4096,
# constant noise 0.08), so any change is attributable to the tool-availability diversity.
set -uo pipefail
cd /workspace/alien-physics-lab
export HF_HOME=/workspace/.cache/huggingface/
export LD_LIBRARY_PATH=/workspace/trl-grpo-venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export VLLM_LOGGING_LEVEL=WARN TRL_EXPERIMENTAL_SILENCE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "START $(date +%H:%M:%S)"
/workspace/trl-grpo-venv/bin/python scripts/train_grpo.py \
  --preset real --run-name rl-vary-tools-g16 \
  --vary-tools --measurement-noise 0.08 \
  --lr 1e-6 --num-generations 16 --per-device-batch 4 --grad-accum 4 \
  --max-completion-length 4096 --max-steps 200 --no-save-final \
  --wandb --tags "vary-tools,diverse-task,g16,lowlr,1.7b" \
  --notes "First diversity-knob run: --vary-tools (per-world subset drop/pendulum/both, soft-disabled, calculator always on). Isolates knob 2 on the proven lowlr-g16 config (G16, lr 1e-6, completion 4096, constant noise 0.08). Q: does the policy adapt its procedure to the available tools, and does accuracy hold on single-tool worlds vs both-tool worlds? Analyze via scripts/analyze_aggregation.py (world_tools breakdown)."
ec=$?
echo "DONE ec=$ec $(date +%H:%M:%S)"
exit $ec

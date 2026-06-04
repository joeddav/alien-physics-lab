#!/usr/bin/env bash
# Procedural-diversity run: knob 2 (--vary-tools), 1/3 each {drop|pendulum|both}.
# G8 (num_generations=8 via per_device=4 x grad_accum=2) with a SLIGHT lr bump to 2e-6
# (2x the 1e-6 baseline; well below the 1e-5 that caused entropy collapse earlier — G8
# is noisier per step, so keep the bump modest). Constant noise 0.08, completion 4096.
set -uo pipefail
cd /workspace/alien-physics-lab
export HF_HOME=/workspace/.cache/huggingface/
export LD_LIBRARY_PATH=/workspace/trl-grpo-venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export VLLM_LOGGING_LEVEL=WARN TRL_EXPERIMENTAL_SILENCE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "START $(date +%H:%M:%S)"
/workspace/trl-grpo-venv/bin/python scripts/train_grpo.py \
  --preset real --run-name rl-vary-tools-g8 \
  --vary-tools --measurement-noise 0.08 \
  --lr 2e-6 --num-generations 8 --per-device-batch 4 --grad-accum 2 \
  --max-completion-length 4096 --max-steps 200 --no-save-final \
  --wandb --tags "vary-tools,diverse-task,g8,lr2e-6,1.7b" \
  --notes "Knob 2 (--vary-tools, 1/3 each drop/pendulum/both, soft-disabled, calculator always on). G8 + slight lr bump to 2e-6 (vs 1e-6 baseline; below the 1e-5 collapse). NOTE: TRL freezes the tool schema, so the model is always SHOWN all 3 tools; per-world variation lives in the briefing's 'Available tools' list + the in-band error on a disabled call. Q: does the policy learn to avoid disabled tools (tools/failure_frequency down) and keep accuracy on single-tool worlds? Analyze via scripts/analyze_aggregation.py world_tools breakdown."
ec=$?
echo "DONE ec=$ec $(date +%H:%M:%S)"
exit $ec

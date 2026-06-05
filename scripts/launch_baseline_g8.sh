#!/usr/bin/env bash
# ABLATION: isolate whether the n=2 -> 3.5 aggregation jump in rl-vary-tools-g8 came from
# the lr bump + G8, or from --vary-tools. This is the SAME config (G8, lr 2e-6, noise 0.08,
# completion 4096) with NO diversity knobs. If n_experiments climbs to ~3.5 here too, the
# lr/G8 broke the plateau (not the knob); if it stays ~2, --vary-tools was responsible.
set -uo pipefail
cd /workspace/alien-physics-lab
export HF_HOME=/workspace/.cache/huggingface/
export LD_LIBRARY_PATH=/workspace/trl-grpo-venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export VLLM_LOGGING_LEVEL=WARN TRL_EXPERIMENTAL_SILENCE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "START $(date +%H:%M:%S)"
/workspace/trl-grpo-venv/bin/python scripts/train_grpo.py \
  --preset real --run-name rl-baseline-g8-lr2e6 \
  --measurement-noise 0.08 \
  --lr 2e-6 --num-generations 8 --per-device-batch 4 --grad-accum 2 \
  --max-completion-length 4096 --max-steps 200 --no-save-final \
  --wandb --tags "baseline,ablation,g8,lr2e-6,1.7b" \
  --notes "ABLATION (no diversity knobs): G8, lr 2e-6, noise 0.08 — identical to rl-vary-tools-g8 minus --vary-tools. Isolates the cause of the n=2->3.5 aggregation jump: lr/G8 vs the knob. Compare n_experiments trajectory to vary-tools (3.5) and the lr-1e-6 runs (~2.1)."
ec=$?
echo "DONE ec=$ec $(date +%H:%M:%S)"
exit $ec

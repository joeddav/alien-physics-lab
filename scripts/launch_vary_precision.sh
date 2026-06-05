#!/usr/bin/env bash
# Procedural-diversity run: knob 1 (--vary-precision) in ISOLATION — the clean test of
# whether a per-world STATED required precision drives adaptive aggregation. Tolerance is
# drawn per world as noise*U[0.8,3.0] (constant noise 0.08 here -> "within 6.4%..24%"), so
# the required #measurements (k ~= 4/mult^2) spans ~1..7. Hypothesis: n_experiments should
# RISE as the stated tolerance tightens -> Spearman rho(world_tolerance, n_exp) < 0
# (analyze_aggregation.py reports it). Same G8/lr-2e-6 config as rl-vary-tools-g8.
set -uo pipefail
cd /workspace/alien-physics-lab
export HF_HOME=/workspace/.cache/huggingface/
export LD_LIBRARY_PATH=/workspace/trl-grpo-venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export VLLM_LOGGING_LEVEL=WARN TRL_EXPERIMENTAL_SILENCE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "START $(date +%H:%M:%S)"
/workspace/trl-grpo-venv/bin/python scripts/train_grpo.py \
  --preset real --run-name rl-vary-precision-g8 \
  --vary-precision --measurement-noise 0.08 \
  --lr 2e-6 --num-generations 8 --per-device-batch 4 --grad-accum 2 \
  --max-completion-length 4096 --max-steps 200 --no-save-final \
  --wandb --tags "vary-precision,diverse-task,g8,lr2e-6,1.7b" \
  --notes "Knob 1 (--vary-precision) isolated: per-world required precision drawn noise*U[0.8,3.0], STATED in the briefing ('within X%'), used by score_answer (proportional zero band). Same config as rl-vary-tools-g8 (G8, lr 2e-6, noise 0.08, completion 4096). KEY TEST: does n_experiments rise as the stated tolerance tightens? Check Spearman rho(world_tolerance, n_exp) (expect <0) + the tolerance-quartile table in analyze_aggregation.py."
ec=$?
echo "DONE ec=$ec $(date +%H:%M:%S)"
exit $ec

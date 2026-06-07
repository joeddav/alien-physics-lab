#!/usr/bin/env bash
# First REAL diameter-task training run. New target: infer the planet's diameter via the
# horizon-dip tool (g-independent geometry). Built on the validated multi-world batching
# (4 worlds/step via grad_accum 16) + the sweep-winning anchor (beta 0.005) + safe lr 2e-6.
# Spin deferred. The diameter task was benchmark-validated (gpt-4o-mini solves it) + unit
# tested + smoke-tested end-to-end before this run.
set -uo pipefail
cd /workspace/alien-physics-lab
export HF_HOME=/workspace/.cache/huggingface/
export LD_LIBRARY_PATH=/workspace/trl-grpo-venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export VLLM_LOGGING_LEVEL=WARN TRL_EXPERIMENTAL_SILENCE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "START $(date +%H:%M:%S)"
/workspace/trl-grpo-venv/bin/python scripts/train_grpo.py \
  --preset real --run-name rl-diameter-mw4-lr4e6 \
  --target diameter --measurement-noise 0.08 \
  --lr 4e-6 --beta 0.005 --num-generations 8 --per-device-batch 2 --grad-accum 16 \
  --gpu-mem-util 0.18 --max-completion-length 4096 --max-steps 250 --no-save-final \
  --wandb --tags "diameter,4worlds,kl0.005,lr4e-6,horizon-dip" \
  --notes "FIRST diameter-task run. Infer planet diameter (m) via measure_horizon_dip (R=2h/alpha^2, alpha in radians, diameter=2R). Multi-world (4/step), beta 0.005, lr 2e-6. Q: does the 1.7B learn the dip task (deg->rad conversion is the #1 failure mode) and aggregate to beat the noise? gpt-4o-mini got 60% solve / 1.0% median; under-averaging was its limiter."
ec=$?
echo "DONE ec=$ec $(date +%H:%M:%S)"
exit $ec

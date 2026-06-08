#!/usr/bin/env bash
# Diameter task, beta=0.0075 (was 0.005). The first diameter run (rl-diameter-mw4-lr4e6,
# beta 0.005) trained cleanly to accuracy~0.55 and learned to aggregate, but at ~step 160
# entropy ran away (0.55->1.68 over ~12 steps, grad_norm 0.6->1.9, kl 0.56->0.91) and
# accuracy eroded off its ~0.68 peak. gravity@beta0.005 was stable, but diameter pushes the
# call-frequency to ~12 (vs ~6 for gravity) -> longer completions -> more entropy the 0.005
# KL anchor can't hold. beta=0.0075 is the documented next probe (06-06: 0=diverge, 0.02=
# over-anchor, 0.01=sweet spot). Single-variable change; everything else matches the prior run.
set -uo pipefail
cd /workspace/alien-physics-lab
export HF_HOME=/workspace/.cache/huggingface/
export LD_LIBRARY_PATH=/workspace/trl-grpo-venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export VLLM_LOGGING_LEVEL=WARN TRL_EXPERIMENTAL_SILENCE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "START $(date +%H:%M:%S)"
/workspace/trl-grpo-venv/bin/python scripts/train_grpo.py \
  --preset real --run-name rl-diameter-mw4-beta0075 \
  --target diameter --measurement-noise 0.08 \
  --lr 4e-6 --beta 0.0075 --num-generations 8 --per-device-batch 2 --grad-accum 16 \
  --gpu-mem-util 0.18 --max-completion-length 4096 --max-steps 250 --no-save-final \
  --wandb --tags "diameter,4worlds,kl0.0075,lr4e-6,horizon-dip" \
  --notes "Diameter task, beta 0.0075 — BISECTS the bracket: beta=0.005 learned to aggregate (n->12, accuracy peak 0.68) then entropy ran away ~step 160; beta=0.01 was stable but OVER-ANCHORED (n stuck at 1, accuracy flat ~0.29, never explored aggregation). Q: does 0.0075 explore aggregation AND stay stable through step 250? Same recipe (lr 4e-6, 4 worlds/step) otherwise."
ec=$?
echo "DONE ec=$ec $(date +%H:%M:%S)"
exit $ec

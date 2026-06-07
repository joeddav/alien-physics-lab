#!/usr/bin/env bash
# Diameter task, beta=0.01 (was 0.005). The first diameter run (rl-diameter-mw4-lr4e6,
# beta 0.005) trained cleanly to accuracy~0.55 and learned to aggregate, but at ~step 160
# entropy ran away (0.55->1.68 over ~12 steps, grad_norm 0.6->1.9, kl 0.56->0.91) and
# accuracy eroded off its ~0.68 peak. gravity@beta0.005 was stable, but diameter pushes the
# call-frequency to ~12 (vs ~6 for gravity) -> longer completions -> more entropy the 0.005
# KL anchor can't hold. beta=0.01 is the documented next probe (06-06: 0=diverge, 0.02=
# over-anchor, 0.01=sweet spot). Single-variable change; everything else matches the prior run.
set -uo pipefail
cd /workspace/alien-physics-lab
export HF_HOME=/workspace/.cache/huggingface/
export LD_LIBRARY_PATH=/workspace/trl-grpo-venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export VLLM_LOGGING_LEVEL=WARN TRL_EXPERIMENTAL_SILENCE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "START $(date +%H:%M:%S)"
/workspace/trl-grpo-venv/bin/python scripts/train_grpo.py \
  --preset real --run-name rl-diameter-mw4-beta01 \
  --target diameter --measurement-noise 0.08 \
  --lr 4e-6 --beta 0.01 --num-generations 8 --per-device-batch 2 --grad-accum 16 \
  --gpu-mem-util 0.18 --max-completion-length 4096 --max-steps 250 --no-save-final \
  --wandb --tags "diameter,4worlds,kl0.01,lr4e-6,horizon-dip" \
  --notes "Diameter task, beta 0.01 (up from 0.005). Prior diameter run diverged via entropy runaway at ~step 160 after peaking at accuracy~0.68; beta=0.005 KL anchor couldn't hold the call-freq-12 long completions. Q: does beta=0.01 hold the ~0.55 accuracy plateau through step 250 without the entropy/grad_norm/kl runaway? Single-variable change from rl-diameter-mw4-lr4e6."
ec=$?
echo "DONE ec=$ec $(date +%H:%M:%S)"
exit $ec

#!/usr/bin/env bash
# Round-2 per-task fine-tune (step 3) for the 3 target RoboCasa tasks, feeding the
# RoboCasa-retrained LAM (step 2).  crank profile: LoRA r64/alpha128, vla_ce x10,
# 8000 steps, n_demos 400.  One adapter per task.  Single GPU, run sequentially.
#
#   bash run_round2_pertask.sh <gpu_id> [lam_ckpt] [run_root] [profile]
# profile: crank (default) | pertask
set -euo pipefail

GPU="${1:-0}"
LAM="${2:-/root/weights/univla-latent-action-model/lam-stage-2.ckpt}"
ROOT="${3:-/root/runs/r2_pertask_v30}"
PROFILE="${4:-crank}"
PY=/root/conda/envs/univla/bin/python
HERE="$(cd "$(dirname "$0")" && pwd)"
TASKS=(CloseToasterOvenDoor OpenDrawer TurnOnMicrowave)

case "$PROFILE" in
  crank)   ARGS=(--use_lora --lora_rank 64 --lora_alpha 128 --vla_ce_weight 10 --decoder_loss_weight 1 --max_steps 8000) ;;
  pertask) ARGS=(--use_lora --lora_rank 32                  --vla_ce_weight 1  --decoder_loss_weight 5 --max_steps 6000) ;;
  *) echo "unknown profile $PROFILE"; exit 1 ;;
esac

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH=/root/dev/LaWAM/UniVLA:"$HERE"

for T in "${TASKS[@]}"; do
  echo "======== $T ($PROFILE) @ $(date '+%F %T') ========"
  "$PY" "$HERE/finetune_robocasa_r2.py" \
    --vision_variant v1 --tasks "$T" --n_demos 400 \
    --batch_size 8 --num_workers 6 --warmup 100 --save_every 2000 --log_every 50 \
    "${ARGS[@]}" \
    --run_dir "$ROOT/$T" \
    --vla_path /root/weights/univla-7b \
    --lam_path "$LAM"
done
echo "ALL DONE -> $ROOT"

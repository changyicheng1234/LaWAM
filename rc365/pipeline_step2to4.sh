#!/usr/bin/env bash
# Unattended driver: wait for step-2 LAM training to finish, then run step-3
# (3 per-task round-2 fine-tunes in parallel on GPU 0/1/2) then step-4 eval.
#
#   nohup bash pipeline_step2to4.sh > /root/runs/pipeline.log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LAM_RUN=/root/runs/lam_robocasa_v30
NEW_LAM=/root/weights/univla-latent-action-model/lam-stage-2-robocasa-v30.ckpt
R2_ROOT=/root/runs/r2_pertask_v30
MIRROR=/mnt/data/changyicheng/rc365/rc365_runs
PROFILE="${PROFILE:-crank}"
TASKS=(CloseToasterOvenDoor OpenDrawer TurnOnMicrowave)
say(){ echo "[$(date '+%F %T')] $*"; }
mkdir -p "$R2_ROOT"

# ---- 1. wait for LAM ----
say "waiting for step-2 LAM ($LAM_RUN/train.log :: DONE) ..."
while ! grep -q '^DONE$' "$LAM_RUN/train.log" 2>/dev/null; do
  if ! pgrep -f lam_finetune_lerobot.py >/dev/null && ! grep -q '^DONE$' "$LAM_RUN/train.log" 2>/dev/null; then
    say "LAM process gone but no DONE marker -- aborting"; exit 1
  fi
  sleep 120
done
say "LAM done."
cp -v "$LAM_RUN/last.ckpt" "$NEW_LAM"
mkdir -p "$MIRROR/lam_robocasa_v30" && cp "$LAM_RUN/last.ckpt" "$MIRROR/lam_robocasa_v30/last.ckpt" \
  && say "mirrored LAM ckpt" || say "mirror LAM failed (non-fatal)"

# ---- 2. step-3: 3 per-task runs, parallel, gpu 0/1/2 ----
case "$PROFILE" in
  crank)   ARGS=(--use_lora --lora_rank 64 --lora_alpha 128 --vla_ce_weight 10 --decoder_loss_weight 1 --max_steps 8000) ;;
  pertask) ARGS=(--use_lora --lora_rank 32 --vla_ce_weight 1  --decoder_loss_weight 5 --max_steps 6000) ;;
  *) say "bad PROFILE=$PROFILE"; exit 1 ;;
esac
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/root/dev/LaWAM/UniVLA:"$HERE"
say "launching step-3 per-task ($PROFILE), LAM=$NEW_LAM"
for i in "${!TASKS[@]}"; do
  T="${TASKS[$i]}"
  ( CUDA_VISIBLE_DEVICES="$i" /root/conda/envs/univla/bin/python "$HERE/finetune_robocasa_r2.py" \
      --vision_variant v1 --tasks "$T" --n_demos 400 \
      --batch_size 8 --num_workers 6 --warmup 100 --save_every 2000 --log_every 50 \
      "${ARGS[@]}" --run_dir "$R2_ROOT/$T" \
      --vla_path /root/weights/univla-7b --lam_path "$NEW_LAM" ) \
    > "$R2_ROOT/console_${T}.log" 2>&1 &
done
wait
say "step-3 runs finished"
mkdir -p "$MIRROR/r2_pertask_v30" && cp -r "$R2_ROOT"/* "$MIRROR/r2_pertask_v30/" 2>/dev/null \
  && say "mirrored r2 adapters" || say "mirror r2 failed (non-fatal)"

# ---- 3. step-4 eval (serve on gpu3, sim on gpu4) ----
say "starting step-4 eval"
bash "$HERE/run_eval_r2.sh" 3 4 "$R2_ROOT" 25 || say "eval had errors"
say "PIPELINE COMPLETE -> $R2_ROOT"

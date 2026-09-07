#!/bin/bash
# S4/S5: 对某个 variant 跑 round-1 eval. 用法: bash run_eval.sh v1 [n_episodes] [gpu]
set -u
VAR=${1:?v1|v2}
NEP=${2:-5}
GPU=${3:-5}
PORT=$((5600 + GPU))
ROOT=/mnt/workspace/changyicheng/work/univla
RUN=/opt/rc365_runs/round1/$VAR
OUT=/opt/rc365_eval/round1_$VAR
mkdir -p "$OUT"

echo "[server] starting on GPU$GPU port $PORT ..."
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$ROOT/UniVLA \
  /opt/miniconda3/envs/univla/bin/python $ROOT/rc365/serve_policy.py \
  --decoder_path $RUN/action_decoder_last.pt --proprio_stats $RUN/proprio_stats.npz \
  --vision_variant $VAR --port $PORT --device cuda:0 > $ROOT/logs/srv_${VAR}.log 2>&1 &
SRV=$!
until grep -q "policy server ready" $ROOT/logs/srv_${VAR}.log 2>/dev/null; do
  sleep 5; kill -0 $SRV 2>/dev/null || { echo "server died"; tail -20 $ROOT/logs/srv_${VAR}.log; exit 1; }
done
echo "[server] ready (pid $SRV)"

source /root/conda/etc/profile.d/conda.sh && conda activate rc365
python $ROOT/rc365/eval_robocasa.py --tasks NavigateKitchen,CloseToasterOvenDoor \
  --n_episodes $NEP --max_steps 350 --port $PORT --out_dir $OUT --save_video_n 2 --egl_device $GPU 2>&1 \
  | grep -vE "^\[robosuite|pygame|Warning: robosuite|No private macro|To make robosuite|mimicgen|^\s*$"

kill $SRV 2>/dev/null
echo "=== $VAR results ==="; cat $OUT/results.json

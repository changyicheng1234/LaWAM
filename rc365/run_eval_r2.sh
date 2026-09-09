#!/usr/bin/env bash
# Step 4: eval the 3 round-2 per-task adapters. serve_policy_r2.py (univla env)
# on one GPU + eval_robocasa.py (rc365 env, EGL sim) on another, exec_horizon 8.
#
#   bash run_eval_r2.sh <serve_gpu> <sim_gpu> [run_root] [n_episodes]
# Deliberately not `set -e`: one task failing must not skip the others.
set -uo pipefail

SERVE_GPU="${1:-0}"
SIM_GPU="${2:-1}"
ROOT="${3:-/root/runs/r2_pertask_v30}"
NEPS="${4:-25}"
PORT=5599
HERE="$(cd "$(dirname "$0")" && pwd)"
PY_UNIVLA=/root/conda/envs/univla/bin/python
PY_RC365=/root/conda/envs/rc365/bin/python
OUT="${ROOT}/eval_$(date +%m%d_%H%M)"
mkdir -p "$OUT"
TASKS=(CloseToasterOvenDoor OpenDrawer TurnOnMicrowave)

for T in "${TASKS[@]}"; do
  RUN="$ROOT/$T/v1"
  [ -f "$RUN/action_decoder_last.pt" ] || { echo "skip $T (no decoder)"; continue; }
  echo "======== eval $T @ $(date '+%F %T') ========"

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$SERVE_GPU" \
  PYTHONPATH=/root/dev/LaWAM/UniVLA:"$HERE" \
  "$PY_UNIVLA" "$HERE/serve_policy_r2.py" \
    --vision_variant v1 --exec_horizon 8 --port "$PORT" \
    --vla_path /root/weights/univla-7b \
    --lora_adapter "$RUN/lora_adapter" \
    --decoder_path "$RUN/action_decoder_last.pt" \
    --proprio_stats "$RUN/proprio_stats.npz" > "$OUT/${T}_serve.log" 2>&1 &
  SERVE_PID=$!
  # wait for readiness
  for _ in $(seq 1 120); do grep -q "policy server ready" "$OUT/${T}_serve.log" && break; sleep 2; done

  PYTHONPATH="$HERE" "$PY_RC365" "$HERE/eval_robocasa.py" \
    --tasks "$T" --n_episodes "$NEPS" --max_steps 350 --port "$PORT" \
    --egl_device "$SIM_GPU" --out_dir "$OUT/$T" 2>&1 | tee "$OUT/${T}_eval.log" || true

  kill "$SERVE_PID" 2>/dev/null || true
  wait "$SERVE_PID" 2>/dev/null || true
done

echo "=== summary ==="
for T in "${TASKS[@]}"; do
  [ -f "$OUT/$T/results.json" ] && python3 -c "import json,sys;r=json.load(open('$OUT/$T/results.json'));print('$T', r.get('$T'))"
done
echo "eval out -> $OUT"

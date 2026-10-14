#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAWINAIR_DIR="${ROOT_DIR}/methods/RawinAir"
TRAIN_PY="${RAWINAIR_DIR}/train_rawinair.py"
EVAL_PY="${RAWINAIR_DIR}/eval_rawinair.py"

# ===== Data =====
NC_PATH="${NC_PATH:-${ROOT_DIR}/data/BTHSAData/dataset_bthsa.nc}"
STATIONS_PATH="${STATIONS_PATH:-${ROOT_DIR}/data/BTHSAData/stations_bthsa.csv}"

# ===== Run Output =====
RUN_NAME="${RUN_NAME:-rawinair_bthsa_3x110}"
RUN_ROOT="${RUN_ROOT:-${RAWINAIR_DIR}/runs/${RUN_NAME}}"
mkdir -p "${RUN_ROOT}"
export RUN_ROOT
export T_PRED

# ===== Experiment Setup =====
RUNS="${RUNS:-1}"
SEEDS_CSV="${SEEDS_CSV:-42,123,456}"
T_HIST="${T_HIST:-12}"
T_PRED="${T_PRED:-12}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-3}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
DROPOUT="${DROPOUT:-0.1}"
RADIUS_KM="${RADIUS_KM:-200.0}"
SCHEDULER="${SCHEDULER:-plateau}"
BEST_METRIC="${BEST_METRIC:-pm25_rmse}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-0}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
DEVICE="${DEVICE:-cuda}"
ENABLE_DYNAMIC_STD_GATE="${ENABLE_DYNAMIC_STD_GATE:-false}"
DYNAMIC_GATE_HIDDEN_DIM="${DYNAMIC_GATE_HIDDEN_DIM:-32}"
DYNAMIC_GATE_TEMPERATURE="${DYNAMIC_GATE_TEMPERATURE:-1.0}"
DYNAMIC_GATE_BLEND_INIT="${DYNAMIC_GATE_BLEND_INIT:--3.0}"
ENABLE_WIND_DIRECTION_GRAPH="${ENABLE_WIND_DIRECTION_GRAPH:-false}"
WIND_DIRECTION_BLEND_INIT="${WIND_DIRECTION_BLEND_INIT:--2.0}"
WIND_DIRECTION_SCALE_INIT="${WIND_DIRECTION_SCALE_INIT:-2.0}"
ENABLE_WIND_HISTORY_GATE="${ENABLE_WIND_HISTORY_GATE:-false}"
WIND_HISTORY_GATE_HIDDEN_DIM="${WIND_HISTORY_GATE_HIDDEN_DIM:-16}"
WIND_HISTORY_GATE_INIT_BIAS="${WIND_HISTORY_GATE_INIT_BIAS:-2.0}"
WIND_HISTORY_GATE_RESIDUAL_INIT="${WIND_HISTORY_GATE_RESIDUAL_INIT:-0.0}"
ENABLE_LIGHT_DYNAMIC_GRAPH="${ENABLE_LIGHT_DYNAMIC_GRAPH:-false}"
LIGHT_DYNAMIC_HIDDEN_DIM="${LIGHT_DYNAMIC_HIDDEN_DIM:-16}"
LIGHT_DYNAMIC_BLEND_INIT="${LIGHT_DYNAMIC_BLEND_INIT:--3.0}"
LIGHT_DYNAMIC_SCALE_INIT="${LIGHT_DYNAMIC_SCALE_INIT:-1.0}"
ENABLE_ADAPTIVE_PRIOR_EMBEDDING="${ENABLE_ADAPTIVE_PRIOR_EMBEDDING:-false}"
NODE_EMB_DIM="${NODE_EMB_DIM:-8}"
TIME_EMB_DIM="${TIME_EMB_DIM:-8}"
MAX_TIME_STEPS="${MAX_TIME_STEPS:-256}"
ADAPTIVE_PRIOR_BLEND_INIT="${ADAPTIVE_PRIOR_BLEND_INIT:--2.0}"
ENABLE_STRUCTURED_PROP_BIAS="${ENABLE_STRUCTURED_PROP_BIAS:-false}"
STRUCTURED_PROP_DIM="${STRUCTURED_PROP_DIM:-8}"
STRUCTURED_PROP_SCALE_INIT="${STRUCTURED_PROP_SCALE_INIT:-1.0}"
ENABLE_DUAL_BRANCH_FUSION="${ENABLE_DUAL_BRANCH_FUSION:-false}"
DUAL_BRANCH_FUSION_INIT="${DUAL_BRANCH_FUSION_INIT:--2.0}"
ENABLE_DUAL_BRANCH_RESIDUAL="${ENABLE_DUAL_BRANCH_RESIDUAL:-false}"
RESIDUAL_HIDDEN_DIM="${RESIDUAL_HIDDEN_DIM:-16}"
RESIDUAL_BLEND_INIT="${RESIDUAL_BLEND_INIT:--3.0}"
RESIDUAL_SCALE_INIT="${RESIDUAL_SCALE_INIT:-1.0}"
TRAIN_YEARS="${TRAIN_YEARS:-2016,2017,2018,2019}"
VAL_YEARS="${VAL_YEARS:-2020,2021}"
TEST_YEARS="${TEST_YEARS:-2022,2023}"

readarray -t SEED_LIST < <(printf '%s\n' "${SEEDS_CSV}" | tr ',' '\n')

if [[ "${#SEED_LIST[@]}" -lt "${RUNS}" ]]; then
  echo "SEEDS_CSV has ${#SEED_LIST[@]} seeds but RUNS=${RUNS}" >&2
  exit 1
fi

echo "ROOT_DIR=${ROOT_DIR}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "NC_PATH=${NC_PATH}"
echo "STATIONS_PATH=${STATIONS_PATH}"
echo "BEST_METRIC=${BEST_METRIC}"
echo "DEVICE=${DEVICE}"
echo "ENABLE_DYNAMIC_STD_GATE=${ENABLE_DYNAMIC_STD_GATE}"
echo "ENABLE_WIND_DIRECTION_GRAPH=${ENABLE_WIND_DIRECTION_GRAPH}"
echo "ENABLE_WIND_HISTORY_GATE=${ENABLE_WIND_HISTORY_GATE}"
echo "WIND_HISTORY_GATE_RESIDUAL_INIT=${WIND_HISTORY_GATE_RESIDUAL_INIT}"
echo "ENABLE_LIGHT_DYNAMIC_GRAPH=${ENABLE_LIGHT_DYNAMIC_GRAPH}"
echo "ENABLE_ADAPTIVE_PRIOR_EMBEDDING=${ENABLE_ADAPTIVE_PRIOR_EMBEDDING}"
echo "ENABLE_STRUCTURED_PROP_BIAS=${ENABLE_STRUCTURED_PROP_BIAS}"
echo "ENABLE_DUAL_BRANCH_FUSION=${ENABLE_DUAL_BRANCH_FUSION}"
echo "ENABLE_DUAL_BRANCH_RESIDUAL=${ENABLE_DUAL_BRANCH_RESIDUAL}"

for ((idx=1; idx<=RUNS; idx++)); do
  SEED="${SEED_LIST[idx-1]}"
  DIR="${RUN_ROOT}/run${idx}"
  mkdir -p "${DIR}"
  echo "===== run ${idx}/${RUNS} seed=${SEED} ====="

  EXTRA_ARGS=()
  if [[ "${ENABLE_DYNAMIC_STD_GATE}" == "true" ]]; then
    EXTRA_ARGS+=(
      --enable_dynamic_std_gate
      --dynamic_gate_hidden_dim "${DYNAMIC_GATE_HIDDEN_DIM}"
      --dynamic_gate_temperature "${DYNAMIC_GATE_TEMPERATURE}"
      --dynamic_gate_blend_init "${DYNAMIC_GATE_BLEND_INIT}"
    )
  fi
  if [[ "${ENABLE_WIND_DIRECTION_GRAPH}" == "true" ]]; then
    EXTRA_ARGS+=(
      --enable_wind_direction_graph
      --wind_direction_blend_init "${WIND_DIRECTION_BLEND_INIT}"
      --wind_direction_scale_init "${WIND_DIRECTION_SCALE_INIT}"
    )
  fi
  if [[ "${ENABLE_WIND_HISTORY_GATE}" == "true" ]]; then
    EXTRA_ARGS+=(
      --enable_wind_history_gate
      --wind_history_gate_hidden_dim "${WIND_HISTORY_GATE_HIDDEN_DIM}"
      --wind_history_gate_init_bias "${WIND_HISTORY_GATE_INIT_BIAS}"
      --wind_history_gate_residual_init "${WIND_HISTORY_GATE_RESIDUAL_INIT}"
    )
  fi
  if [[ "${ENABLE_LIGHT_DYNAMIC_GRAPH}" == "true" ]]; then
    EXTRA_ARGS+=(
      --enable_light_dynamic_graph
      --light_dynamic_hidden_dim "${LIGHT_DYNAMIC_HIDDEN_DIM}"
      --light_dynamic_blend_init "${LIGHT_DYNAMIC_BLEND_INIT}"
      --light_dynamic_scale_init "${LIGHT_DYNAMIC_SCALE_INIT}"
    )
  fi
  if [[ "${ENABLE_ADAPTIVE_PRIOR_EMBEDDING}" == "true" ]]; then
    EXTRA_ARGS+=(
      --enable_adaptive_prior_embedding
      --node_emb_dim "${NODE_EMB_DIM}"
      --time_emb_dim "${TIME_EMB_DIM}"
      --max_time_steps "${MAX_TIME_STEPS}"
      --adaptive_prior_blend_init "${ADAPTIVE_PRIOR_BLEND_INIT}"
    )
  fi
  if [[ "${ENABLE_STRUCTURED_PROP_BIAS}" == "true" ]]; then
    EXTRA_ARGS+=(
      --enable_structured_prop_bias
      --structured_prop_dim "${STRUCTURED_PROP_DIM}"
      --structured_prop_scale_init "${STRUCTURED_PROP_SCALE_INIT}"
    )
  fi
  if [[ "${ENABLE_DUAL_BRANCH_FUSION}" == "true" ]]; then
    EXTRA_ARGS+=(
      --enable_dual_branch_fusion
      --dual_branch_fusion_init "${DUAL_BRANCH_FUSION_INIT}"
    )
  fi
  if [[ "${ENABLE_DUAL_BRANCH_RESIDUAL}" == "true" ]]; then
    EXTRA_ARGS+=(
      --enable_dual_branch_residual
      --residual_hidden_dim "${RESIDUAL_HIDDEN_DIM}"
      --residual_blend_init "${RESIDUAL_BLEND_INIT}"
      --residual_scale_init "${RESIDUAL_SCALE_INIT}"
    )
  fi

  python3 "${TRAIN_PY}" \
    --nc "${NC_PATH}" \
    --stations "${STATIONS_PATH}" \
    --run_dir "${DIR}" \
    --t_hist "${T_HIST}" \
    --t_pred "${T_PRED}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    --lr "${LR}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --dropout "${DROPOUT}" \
    --radius_km "${RADIUS_KM}" \
    --seed "${SEED}" \
    --scheduler "${SCHEDULER}" \
    --best_metric "${BEST_METRIC}" \
    --early_stop_patience "${EARLY_STOP_PATIENCE}" \
    --early_stop_min_delta "${EARLY_STOP_MIN_DELTA}" \
    --train_years "${TRAIN_YEARS}" \
    --val_years "${VAL_YEARS}" \
    --test_years "${TEST_YEARS}" \
    --amp \
    --amp_dtype "${AMP_DTYPE}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${DIR}/train.log"

  python3 "${EVAL_PY}" \
    --ckpt "${DIR}/ckpt_best.pt" \
    --nc "${NC_PATH}" \
    --stations "${STATIONS_PATH}" \
    --t_hist "${T_HIST}" \
    --t_pred "${T_PRED}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    --test_years "${TEST_YEARS}" \
    --amp \
    --amp_dtype "${AMP_DTYPE}" \
    --out "${DIR}/eval.json" \
    2>&1 | tee "${DIR}/eval.log"
done

python3 - <<'PY'
import json
import os
from pathlib import Path
import numpy as np

run_root = Path(os.environ["RUN_ROOT"])
horizon_len = int(os.environ.get("T_PRED", "12"))
runs = sorted([p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("run")])
if not runs:
    raise SystemExit("No run directories found")

rows = []
for run_dir in runs:
    eval_path = run_dir / "eval.json"
    if not eval_path.exists():
        raise FileNotFoundError(eval_path)
    d = json.loads(eval_path.read_text())
    row = {
        "run": run_dir.name,
        "mae": d.get("mae", np.nan),
        "rmse": d.get("rmse", np.nan),
        "smape": d.get("smape", np.nan),
        "pm25_mae": d.get("PM2.5_mae", np.nan),
        "pm25_rmse": d.get("PM2.5_rmse", np.nan),
        "pm25_smape": d.get("PM2.5_smape", np.nan),
        "o3_mae": d.get("O3_mae", np.nan),
        "o3_rmse": d.get("O3_rmse", np.nan),
        "o3_smape": d.get("O3_smape", np.nan),
    }
    for prefix in ["PM2.5", "O3"]:
        for metric in ["mae_t", "rmse_t", "smape_t"]:
            arr = d.get(f"{prefix}_{metric}", None)
            if arr:
                key_prefix = "pm25" if prefix == "PM2.5" else "o3"
                metric_name = metric.replace("_t", "")
                max_h = min(len(arr), horizon_len)
                for h in [3, 6, 9, 12, 24]:
                    if h <= max_h:
                        row[f"{key_prefix}_{metric_name}_{h}"] = arr[h - 1]
                row[f"{key_prefix}_{metric_name}_avg"] = float(np.mean(arr[:max_h]))
    rows.append(row)

def mean_std(key):
    vals = np.array([r.get(key, np.nan) for r in rows], dtype=float)
    return float(np.nanmean(vals)), float(np.nanstd(vals))

def selected_horizons(max_h):
    base = [3, 6, 9, 12, 24]
    hs = [h for h in base if h <= max_h]
    if max_h >= 1 and max_h not in hs:
        hs.append(max_h)
    return hs

def build_horizon_line(row, prefix, label):
    parts = []
    for h in selected_horizons(horizon_len):
        key = f"{prefix}_{h}"
        if key in row:
            parts.append(f"h{h}={row[key]:.4f}")
    avg_key = f"{prefix}_avg"
    if avg_key in row:
        parts.append(f"avg{horizon_len}={row[avg_key]:.4f}")
    return f"  PM2.5 horizon {label}: " + " ".join(parts) if parts else None

summary_lines = []
summary_lines.append(f"Results for {len(rows)} runs")
summary_lines.append("")
for row in rows:
    summary_lines.append(
        f"{row['run']}: PM2.5 mae={row['pm25_mae']:.4f} rmse={row['pm25_rmse']:.4f} smape={row['pm25_smape']:.4f}"
    )
    for prefix, label in [
        ("pm25_rmse", "rmse"),
        ("pm25_mae", "mae"),
        ("pm25_smape", "smape"),
    ]:
        line = build_horizon_line(row, prefix, label)
        if line:
            summary_lines.append(line)
summary_lines.append("")
for key, label in [
    ("pm25_mae", "PM2.5 MAE"),
    ("pm25_rmse", "PM2.5 RMSE"),
    ("pm25_smape", "PM2.5 SMAPE"),
    ("pm25_mae_avg", f"PM2.5 MAE avg{horizon_len}"),
    ("pm25_rmse_avg", f"PM2.5 RMSE avg{horizon_len}"),
    ("pm25_smape_avg", f"PM2.5 SMAPE avg{horizon_len}"),
]:
    mean, std = mean_std(key)
    if not np.isnan(mean):
        summary_lines.append(f"{label}: mean={mean:.4f} std={std:.4f}")

summary_path = run_root / "summary_3runs.txt"
summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
print("\n".join(summary_lines))
print(f"summary saved: {summary_path}")
PY

echo "all done"

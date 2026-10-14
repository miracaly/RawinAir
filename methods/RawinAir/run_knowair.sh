#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAWINAIR_DIR="${ROOT_DIR}/methods/RawinAir"
TRAIN_PY="${RAWINAIR_DIR}/train_rawinair_knowair.py"
EVAL_PY="${RAWINAIR_DIR}/eval_rawinair_knowair.py"
SUMMARY_PY="${RAWINAIR_DIR}/summarize_knowair_runs.py"

NPY_PATH="${NPY_PATH:-${ROOT_DIR}/data/KnowAirData/KnowAir.npy}"
CITY_PATH="${CITY_PATH:-${ROOT_DIR}/data/KnowAirData/city.txt}"

RUN_NAME="${RUN_NAME:-rawinair_knowair}"
RUN_ROOT="${RUN_ROOT:-${RAWINAIR_DIR}/runs/${RUN_NAME}}"
mkdir -p "${RUN_ROOT}"
export RUN_ROOT
export T_PRED

RUNS="${RUNS:-1}"
SEEDS_CSV="${SEEDS_CSV:-42,123,456}"
T_HIST="${T_HIST:-24}"
T_PRED="${T_PRED:-24}"
EPOCHS="${EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-3}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
DROPOUT="${DROPOUT:-0.1}"
RADIUS_KM="${RADIUS_KM:-300.0}"
SCHEDULER="${SCHEDULER:-plateau}"
BEST_METRIC="${BEST_METRIC:-pm25_rmse}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-0}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
DEVICE="${DEVICE:-cuda}"
MET_FEATURE_SETUP="${MET_FEATURE_SETUP:-standard}"
WIND_SOURCE="${WIND_SOURCE:-100m}"
DISABLE_DERIVED_FEATURES="${DISABLE_DERIVED_FEATURES:-false}"
ENABLE_DYNAMIC_STD_GATE="${ENABLE_DYNAMIC_STD_GATE:-false}"
DYNAMIC_GATE_HIDDEN_DIM="${DYNAMIC_GATE_HIDDEN_DIM:-32}"
DYNAMIC_GATE_TEMPERATURE="${DYNAMIC_GATE_TEMPERATURE:-1.0}"
DYNAMIC_GATE_BLEND_INIT="${DYNAMIC_GATE_BLEND_INIT:--3.0}"
ENABLE_WIND_DIRECTION_GRAPH="${ENABLE_WIND_DIRECTION_GRAPH:-true}"
WIND_DIRECTION_BLEND_INIT="${WIND_DIRECTION_BLEND_INIT:--2.0}"
WIND_DIRECTION_SCALE_INIT="${WIND_DIRECTION_SCALE_INIT:-3.0}"
TRANSPORT_DIRECTION="${TRANSPORT_DIRECTION:-current}"
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
MAX_TIME_STEPS="${MAX_TIME_STEPS:-512}"
ADAPTIVE_PRIOR_BLEND_INIT="${ADAPTIVE_PRIOR_BLEND_INIT:--2.0}"
ENABLE_STRUCTURED_PROP_BIAS="${ENABLE_STRUCTURED_PROP_BIAS:-true}"
STRUCTURED_PROP_DIM="${STRUCTURED_PROP_DIM:-8}"
STRUCTURED_PROP_SCALE_INIT="${STRUCTURED_PROP_SCALE_INIT:-1.2}"
ENABLE_DUAL_BRANCH_FUSION="${ENABLE_DUAL_BRANCH_FUSION:-false}"
DUAL_BRANCH_FUSION_INIT="${DUAL_BRANCH_FUSION_INIT:--2.0}"
ENABLE_DUAL_BRANCH_RESIDUAL="${ENABLE_DUAL_BRANCH_RESIDUAL:-false}"
RESIDUAL_HIDDEN_DIM="${RESIDUAL_HIDDEN_DIM:-16}"
RESIDUAL_BLEND_INIT="${RESIDUAL_BLEND_INIT:--3.0}"
RESIDUAL_SCALE_INIT="${RESIDUAL_SCALE_INIT:-1.0}"
TRAIN_YEARS="${TRAIN_YEARS:-2015,2016}"
VAL_YEARS="${VAL_YEARS:-2017}"
TEST_YEARS="${TEST_YEARS:-2018}"

readarray -t SEED_LIST < <(printf '%s\n' "${SEEDS_CSV}" | tr ',' '\n')
if [[ "${#SEED_LIST[@]}" -lt "${RUNS}" ]]; then
  echo "SEEDS_CSV has ${#SEED_LIST[@]} seeds but RUNS=${RUNS}" >&2
  exit 1
fi

echo "ROOT_DIR=${ROOT_DIR}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "NPY_PATH=${NPY_PATH}"
echo "CITY_PATH=${CITY_PATH}"
echo "BEST_METRIC=${BEST_METRIC}"
echo "DEVICE=${DEVICE}"
echo "MET_FEATURE_SETUP=${MET_FEATURE_SETUP}"
echo "WIND_SOURCE=${WIND_SOURCE}"
if [[ "${MET_FEATURE_SETUP}" == "uv100" && "${WIND_SOURCE}" != "100m" ]]; then
  echo "MET_FEATURE_SETUP=uv100 requires WIND_SOURCE=100m" >&2
  exit 1
fi
echo "ENABLE_DYNAMIC_STD_GATE=${ENABLE_DYNAMIC_STD_GATE}"
echo "ENABLE_WIND_DIRECTION_GRAPH=${ENABLE_WIND_DIRECTION_GRAPH}"
echo "TRANSPORT_DIRECTION=${TRANSPORT_DIRECTION}"
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
  if [[ "${DISABLE_DERIVED_FEATURES}" == "true" ]]; then
    EXTRA_ARGS+=(--disable_derived_features)
  fi
  if [[ "${ENABLE_DYNAMIC_STD_GATE}" == "true" ]]; then
    EXTRA_ARGS+=(--enable_dynamic_std_gate --dynamic_gate_hidden_dim "${DYNAMIC_GATE_HIDDEN_DIM}" --dynamic_gate_temperature "${DYNAMIC_GATE_TEMPERATURE}" --dynamic_gate_blend_init "${DYNAMIC_GATE_BLEND_INIT}")
  fi
  if [[ "${ENABLE_WIND_DIRECTION_GRAPH}" == "true" ]]; then
    EXTRA_ARGS+=(--enable_wind_direction_graph --wind_direction_blend_init "${WIND_DIRECTION_BLEND_INIT}" --wind_direction_scale_init "${WIND_DIRECTION_SCALE_INIT}" --transport_direction "${TRANSPORT_DIRECTION}")
  fi
  if [[ "${ENABLE_WIND_HISTORY_GATE}" == "true" ]]; then
    EXTRA_ARGS+=(--enable_wind_history_gate --wind_history_gate_hidden_dim "${WIND_HISTORY_GATE_HIDDEN_DIM}" --wind_history_gate_init_bias "${WIND_HISTORY_GATE_INIT_BIAS}" --wind_history_gate_residual_init "${WIND_HISTORY_GATE_RESIDUAL_INIT}")
  fi
  if [[ "${ENABLE_LIGHT_DYNAMIC_GRAPH}" == "true" ]]; then
    EXTRA_ARGS+=(--enable_light_dynamic_graph --light_dynamic_hidden_dim "${LIGHT_DYNAMIC_HIDDEN_DIM}" --light_dynamic_blend_init "${LIGHT_DYNAMIC_BLEND_INIT}" --light_dynamic_scale_init "${LIGHT_DYNAMIC_SCALE_INIT}")
  fi
  if [[ "${ENABLE_ADAPTIVE_PRIOR_EMBEDDING}" == "true" ]]; then
    EXTRA_ARGS+=(--enable_adaptive_prior_embedding --node_emb_dim "${NODE_EMB_DIM}" --time_emb_dim "${TIME_EMB_DIM}" --max_time_steps "${MAX_TIME_STEPS}" --adaptive_prior_blend_init "${ADAPTIVE_PRIOR_BLEND_INIT}")
  fi
  if [[ "${ENABLE_STRUCTURED_PROP_BIAS}" == "true" ]]; then
    EXTRA_ARGS+=(--enable_structured_prop_bias --structured_prop_dim "${STRUCTURED_PROP_DIM}" --structured_prop_scale_init "${STRUCTURED_PROP_SCALE_INIT}")
  fi
  if [[ "${ENABLE_DUAL_BRANCH_FUSION}" == "true" ]]; then
    EXTRA_ARGS+=(--enable_dual_branch_fusion --dual_branch_fusion_init "${DUAL_BRANCH_FUSION_INIT}")
  fi
  if [[ "${ENABLE_DUAL_BRANCH_RESIDUAL}" == "true" ]]; then
    EXTRA_ARGS+=(--enable_dual_branch_residual --residual_hidden_dim "${RESIDUAL_HIDDEN_DIM}" --residual_blend_init "${RESIDUAL_BLEND_INIT}" --residual_scale_init "${RESIDUAL_SCALE_INIT}")
  fi

  python3 "${TRAIN_PY}" \
    --npy "${NPY_PATH}" \
    --city "${CITY_PATH}" \
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
    --met_feature_setup "${MET_FEATURE_SETUP}" \
    --wind_source "${WIND_SOURCE}" \
    --amp \
    --amp_dtype "${AMP_DTYPE}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${DIR}/train.log"

  python3 "${EVAL_PY}" \
    --ckpt "${DIR}/ckpt_best.pt" \
    --npy "${NPY_PATH}" \
    --city "${CITY_PATH}" \
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

python3 "${SUMMARY_PY}"

echo "all done"

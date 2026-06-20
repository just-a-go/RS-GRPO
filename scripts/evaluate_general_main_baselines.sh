#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# Run general video QA evaluation for main-table baseline rows in order.
# Usage:
#   DATASET_NAME=eval_nextgqa_mixed_server bash scripts/evaluate_general_main_baselines.sh
#   DATASET_NAME=eval_star_mixed_server bash scripts/evaluate_general_main_baselines.sh
#
# If your local JSON names are nextgqa_val_mixed.json / STAR_mixed.json, set:
#   DATASET_NAME=nextgqa_val_mixed
#   DATASET_NAME=STAR_mixed

export PRIVATE_DATA_ROOT="${PRIVATE_DATA_ROOT:-$PROJECT_ROOT}"
export EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-$PRIVATE_DATA_ROOT}"
export PYTHONPATH="$PRIVATE_DATA_ROOT/src:$PRIVATE_DATA_ROOT/qwen-vl-utils/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,5}"

DATASET_NAME="${DATASET_NAME:?Set DATASET_NAME, e.g. eval_nextgqa_mixed_server or eval_star_mixed_server}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MODEL_ROOT="$PRIVATE_DATA_ROOT/Qwen"

cd "$PRIVATE_DATA_ROOT"

DATASET_PATH="$PRIVATE_DATA_ROOT/data/evaluation/$DATASET_NAME.json"
if [[ ! -f "$DATASET_PATH" ]]; then
  DATASET_PATH="$PRIVATE_DATA_ROOT/evaluation/$DATASET_NAME.json"
fi

echo "[check] PRIVATE_DATA_ROOT=$PRIVATE_DATA_ROOT"
echo "[check] DATASET_NAME=$DATASET_NAME"
echo "[check] DATASET_PATH=$DATASET_PATH"
test -f "$DATASET_PATH"

echo "[1/5] Qwen2.5-VL-7B Zero-Shot"
python src/eval/eval_general_zeroshot.py \
  --model_name "$MODEL_ROOT/Qwen2.5-VL-7B-Instruct" \
  --dataset_name "$DATASET_NAME" \
  --batch_size "$BATCH_SIZE"

echo "[2/5] Qwen2.5-VL-7B CoT"
python src/eval/eval_general_videor1.py \
  --model_name "$MODEL_ROOT/Qwen2.5-VL-7B-Instruct" \
  --dataset_name "$DATASET_NAME" \
  --batch_size "$BATCH_SIZE"

echo "[3/5] Qwen2.5-VL-7B SFT"
python src/eval/eval_general_videor1.py \
  --model_name "$MODEL_ROOT/Qwen2.5-VL-7B-COT-SFT" \
  --dataset_name "$DATASET_NAME" \
  --batch_size "$BATCH_SIZE"

echo "[4/5] Video-R1"
python src/eval/eval_general_videor1.py \
  --model_name "$MODEL_ROOT/Qwen2.5-VL-Video-R1-7B" \
  --dataset_name "$DATASET_NAME" \
  --batch_size "$BATCH_SIZE"

echo "[5/5] VideoChat-R1"
python src/eval/eval_general_videor1.py \
  --model_name "$MODEL_ROOT/Qwen2.5-VL-VideoChat-R1_7B" \
  --dataset_name "$DATASET_NAME" \
  --batch_size "$BATCH_SIZE"

echo "[done] Metrics are under $EVAL_OUTPUT_ROOT/logs/$DATASET_NAME/test"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# Run CLEVRER_cf evaluation for the main-table baseline rows in order.
# Expected model layout:
#   $PRIVATE_DATA_ROOT/Qwen/Qwen2.5-VL-7B-Instruct
#   $PRIVATE_DATA_ROOT/Qwen/Qwen2.5-VL-7B-COT-SFT
#   $PRIVATE_DATA_ROOT/Qwen/Qwen2.5-VL-Video-R1-7B
#   $PRIVATE_DATA_ROOT/Qwen/Qwen2.5-VL-VideoChat-R1_7B

export PRIVATE_DATA_ROOT="${PRIVATE_DATA_ROOT:-$PROJECT_ROOT}"
export EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-$PRIVATE_DATA_ROOT}"
export CLEVRER_VAL_PATH="${CLEVRER_VAL_PATH:-$PRIVATE_DATA_ROOT/data/CLEVRER/clevrer_counterfactual_val.json}"
export PYTHONPATH="$PRIVATE_DATA_ROOT/src:$PRIVATE_DATA_ROOT/qwen-vl-utils/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,5}"

BATCH_SIZE="${BATCH_SIZE:-4}"
MODEL_ROOT="$PRIVATE_DATA_ROOT/Qwen"

cd "$PRIVATE_DATA_ROOT"

echo "[check] PRIVATE_DATA_ROOT=$PRIVATE_DATA_ROOT"
echo "[check] CLEVRER_VAL_PATH=$CLEVRER_VAL_PATH"
test -f "$CLEVRER_VAL_PATH"

echo "[1/5] Qwen2.5-VL-7B Zero-Shot"
python src/eval/eval_clevrer_zeroshot.py \
  --model_name "$MODEL_ROOT/Qwen2.5-VL-7B-Instruct" \
  --batch_size "$BATCH_SIZE"

echo "[2/5] Qwen2.5-VL-7B CoT"
python src/eval/eval_clevrer.py \
  --model_name "$MODEL_ROOT/Qwen2.5-VL-7B-Instruct" \
  --batch_size "$BATCH_SIZE"

echo "[3/5] Qwen2.5-VL-7B SFT"
python src/eval/eval_clevrer.py \
  --model_name "$MODEL_ROOT/Qwen2.5-VL-7B-COT-SFT" \
  --batch_size "$BATCH_SIZE"

echo "[4/5] Video-R1"
python src/eval/eval_clevrer.py \
  --model_name "$MODEL_ROOT/Qwen2.5-VL-Video-R1-7B" \
  --batch_size "$BATCH_SIZE"

echo "[5/5] VideoChat-R1"
python src/eval/eval_clevrer.py \
  --model_name "$MODEL_ROOT/Qwen2.5-VL-VideoChat-R1_7B" \
  --batch_size "$BATCH_SIZE"

echo "[done] Metrics are under $EVAL_OUTPUT_ROOT/logs/clevrer_counterfactual_val/test"

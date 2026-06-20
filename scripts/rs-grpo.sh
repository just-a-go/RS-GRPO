#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# Training method:
#   grpo, tw_grpo, rs_grpo, dapo, limr, dr_grpo, gspo
export LOSS_TYPE=${LOSS_TYPE:-rs_grpo}

# RS-GRPO learnability L(q), used only by rs_grpo:
#   full, wo_contrast, wo_unsolvedness, wo_best_quality, contrast_only
export RS_LEARNABILITY_DEFINITION=${RS_LEARNABILITY_DEFINITION:-full}

# RS-GRPO group-weight mapping Omega(q), used only by rs_grpo:
#   hard_filter, binary_up, amplify_only, conservative
export RS_GROUP_WEIGHTING_RULE=${RS_GROUP_WEIGHTING_RULE:-conservative}

# RS-GRPO parameter ablations.
export RS_LEARNABLE_LAMBDA=${RS_LEARNABLE_LAMBDA:-0.15}
export RS_LEARNABLE_TAU=${RS_LEARNABLE_TAU:-0.5}

# Reward source for estimating L(q), used only by rs_grpo:
#   accuracy: accuracy reward only
#   total:    accuracy reward + format reward
export RS_REWARD_SOURCE=${RS_REWARD_SOURCE:-${LEARNABILITY_REWARD_SOURCE:-accuracy}}

# TW/RS token weighting. Ignored by grpo/dapo/limr/dr_grpo/gspo.
case "$LOSS_TYPE" in
    tw_grpo|rs_grpo)
        DEFAULT_ALPHA=1.70
        ;;
    *)
        DEFAULT_ALPHA=1.00
        ;;
esac
export ALPHA=${ALPHA:-$DEFAULT_ALPHA}

# Runtime and data paths.
export PRIVATE_DATA_ROOT=${PRIVATE_DATA_ROOT:-$PROJECT_ROOT}
export WANDB_PROJECT=${WANDB_PROJECT:-Qwen2.5-VL-7B-Video-GRPO}
export MODEL_NAME=${MODEL_NAME:-Qwen2.5-VL-7B-Instruct_clevrer_${LOSS_TYPE}}
export WANDB_NAME=${WANDB_NAME:-$MODEL_NAME}
export MODEL_PATH=${MODEL_PATH:-$PRIVATE_DATA_ROOT/Qwen/Qwen2.5-VL-7B-Instruct}
export JSONL_PATH=${JSONL_PATH:-$PRIVATE_DATA_ROOT/data/CLEVRER/clevrer_counterfactual_train.json}
export WANDB_MODE=${WANDB_MODE:-offline}
export DEBUG_MODE=${DEBUG_MODE:-false}
export SAMPLE_MODE=${SAMPLE_MODE:-true}
export LOG_PATH=${LOG_PATH:-$PRIVATE_DATA_ROOT/$WANDB_NAME/debug.log}

# Hardware and distributed launch.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export NPROC_PER_NODE=${NPROC_PER_NODE:-2}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-12533}

# Common training ablations.
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
export MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-2048}
export NUM_GENERATIONS=${NUM_GENERATIONS:-8}
export TRAIN_SAMPLES=${TRAIN_SAMPLES:-2000}
export QUESTION_TYPE=${QUESTION_TYPE:-mixed}
export LEARNING_RATE=${LEARNING_RATE:-1e-6}
export BETA=${BETA:-0.00}
export PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
export NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
export SAVE_STEPS=${SAVE_STEPS:-100}
export MAX_GRAD_NORM=${MAX_GRAD_NORM:-20}
export FREEZE_VISION_MODULES=${FREEZE_VISION_MODULES:-true}
export REWARD_FUNCS=${REWARD_FUNCS:-"accuracy format"}
export DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-scripts/zero3_offload.json}
export REPORT_TO=${REPORT_TO:-wandb}

# Optional method-specific ablations.
EXTRA_ARGS=()
if [ -n "${EPSILON_LOW:-}" ]; then
    EXTRA_ARGS+=(--method_epsilon_low "$EPSILON_LOW")
fi
if [ -n "${EPSILON_HIGH:-}" ]; then
    EXTRA_ARGS+=(--method_epsilon_high "$EPSILON_HIGH")
fi
if [ -n "${DR_GRPO_CONSTANT_NORMALIZER:-}" ]; then
    EXTRA_ARGS+=(--dr_grpo_constant_normalizer "$DR_GRPO_CONSTANT_NORMALIZER")
fi
if [ -n "${LIMR_SCORES_PATH:-}" ]; then
    EXTRA_ARGS+=(--limr_scores_path "$LIMR_SCORES_PATH")
fi
if [ -n "${LIMR_ID_FIELD:-}" ]; then
    EXTRA_ARGS+=(--limr_id_field "$LIMR_ID_FIELD")
fi

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nproc_per_node="$NPROC_PER_NODE" \
    --nnodes="${NNODES:-1}" \
    --node_rank="${NODE_RANK:-0}" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    src/open_r1/grpo.py \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --output_dir "$PRIVATE_DATA_ROOT/$MODEL_NAME" \
    --model_name_or_path "$MODEL_PATH" \
    --dataset_name xxx \
    --jsonl_path "$JSONL_PATH" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --reward_funcs $REWARD_FUNCS \
    --learning_rate "$LEARNING_RATE" \
    --beta "$BETA" \
    --alpha "$ALPHA" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --logging_steps "${LOGGING_STEPS:-1}" \
    --question_type "$QUESTION_TYPE" \
    --train_samples "$TRAIN_SAMPLES" \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed "${DATA_SEED:-42}" \
    --report_to "$REPORT_TO" \
    --gradient_checkpointing true \
    --attn_implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}" \
    --freeze_vision_modules "$FREEZE_VISION_MODULES" \
    --loss_type "$LOSS_TYPE" \
    --learnability_reward_source "$RS_REWARD_SOURCE" \
    --rs_learnability_definition "$RS_LEARNABILITY_DEFINITION" \
    --rs_group_weighting_rule "$RS_GROUP_WEIGHTING_RULE" \
    --rs_learnable_lambda "$RS_LEARNABLE_LAMBDA" \
    --rs_learnable_tau "$RS_LEARNABLE_TAU" \
    --limr_threshold "${LIMR_THRESHOLD:-0.0}" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --run_name "$WANDB_NAME" \
    --save_steps "$SAVE_STEPS" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --save_only_model true \
    --num_generations "$NUM_GENERATIONS" \
    "${EXTRA_ARGS[@]}"

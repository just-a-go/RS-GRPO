# RS-GRPO: Reward-Separable GRPO for Video Reasoning

This repository contains the code for **Reward-Separable Group Relative Policy Optimization (RS-GRPO)**, a lightweight group-level reweighting extension of TW-GRPO for multi-answer video reasoning.

RS-GRPO keeps the TW-GRPO token-level objective and adds an accuracy-driven prompt-group weight. The implementation also includes switches for GRPO, TW-GRPO, RS-GRPO, DAPO-style filtering, LIMR-style data selection, Dr.GRPO, and GSPO-style sequence-level optimization.

This codebase is adapted from TW-GRPO and Open-R1-style training code. The repository is organized as a code-only reproduction package; paper figures, local logs, checkpoints, and datasets are not included.

## Repository Structure

```text
RS-GRPO/
+-- src/
|   +-- open_r1/                 # training entry and trainer implementation
|   +-- eval/                    # video QA evaluation scripts
+-- scripts/
|   +-- rs-grpo.sh               # unified training script
|   +-- evaluate_*.sh            # evaluation helpers
|   +-- zero*.json / zero*.yaml  # DeepSpeed configs
+-- configs/                     # accelerate/deepspeed configs
+-- data/
|   +-- CLEVRER/                 # CLEVRER placeholders
|   +-- NExTQA/                  # NExT-QA/NExT-GQA video placeholders
|   +-- STAR/                    # STAR video placeholders
|   +-- evaluation/              # evaluation JSON placeholders
|   +-- question_answer_inverse/ # QAI construction scripts
+-- example/                     # QAI tutorial
+-- qwen-vl-utils/               # local Qwen-VL utility package
+-- setup.py
+-- LICENSE
```

Large assets are intentionally excluded: model checkpoints, logs, generated build files, real dataset JSON files, videos, paper figures, and baseline presentation images.

## 1. Environment

```bash
git clone https://github.com/just-a-go/RS-GRPO.git
cd RS-GRPO

conda create -n rs-grpo python=3.10 -y
conda activate rs-grpo

pip install -e ".[dev]"
pip install flash_attn --no-build-isolation
pip install decord

cd qwen-vl-utils
pip install -e .
cd ..
```

The training scripts assume a Linux server with CUDA, PyTorch, Transformers, TRL, Accelerate, and DeepSpeed.

## 2. Model Backbone

Download Qwen2.5-VL-7B-Instruct:

```bash
mkdir -p Qwen
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir Qwen/Qwen2.5-VL-7B-Instruct \
  --resume-download
```

Optional baseline checkpoints for reproducing comparison rows:

```bash
huggingface-cli download Video-R1/Video-R1-7B \
  --local-dir Qwen/Qwen2.5-VL-Video-R1-7B \
  --resume-download
```

Place other baseline checkpoints under `Qwen/` only if you want to reproduce all comparison rows. They are not required for training RS-GRPO.

## 3. Dataset Preparation

Only the folder structure is included in this repository. Download the original datasets separately and place files under the paths expected by the scripts.

### CLEVRER Counterfactual

Expected structure:

```text
data/CLEVRER/
+-- clevrer_counterfactual_train.json
+-- clevrer_counterfactual_val.json
+-- train_video/
+-- validation_video/
```

Download CLEVRER videos from the official CLEVRER release and put training videos in `train_video/` and validation videos in `validation_video/`.

### NExT-GQA-Mixed and STAR-Mixed

Expected evaluation structure:

```text
data/evaluation/
+-- eval_nextgqa_mixed_server.json
+-- eval_star_mixed_server.json
```

Suggested video structure:

```text
data/NExTQA/
+-- videos/
|   +-- <video_id>.mp4

data/STAR/
+-- <video_file>.mp4
```

The evaluation code reads the video path from each sample's `video` field. Therefore, the JSON paths must point to your local video files. You can either use absolute paths, create symlinks matching the paths in the JSON files, or rewrite the JSON `video` fields to the suggested local structure above.

If you construct mixed multi-answer data yourself, use the QAI scripts:

```bash
python data/question_answer_inverse/convert_nextgqa.py
python data/question_answer_inverse/convert_star.py
```

Check `example/tutorial/qai_tutorial.md` for the QAI data construction idea.

## 4. Training

All training methods are selected with one script by changing environment variables.

Set paths first:

```bash
export PRIVATE_DATA_ROOT=/path/to/RS-GRPO
export MODEL_PATH=$PRIVATE_DATA_ROOT/Qwen/Qwen2.5-VL-7B-Instruct
export JSONL_PATH=$PRIVATE_DATA_ROOT/data/CLEVRER/clevrer_counterfactual_train.json
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
```

Run the default RS-GRPO setting:

```bash
LOSS_TYPE=rs_grpo \
MODEL_NAME=Qwen2.5-VL-7B-Instruct_clevrer_rs_grpo \
bash scripts/rs-grpo.sh
```

Available methods:

```bash
LOSS_TYPE=grpo      bash scripts/rs-grpo.sh
LOSS_TYPE=tw_grpo   bash scripts/rs-grpo.sh
LOSS_TYPE=rs_grpo   bash scripts/rs-grpo.sh
LOSS_TYPE=dapo      bash scripts/rs-grpo.sh
LOSS_TYPE=limr      bash scripts/rs-grpo.sh
LOSS_TYPE=dr_grpo   bash scripts/rs-grpo.sh
LOSS_TYPE=gspo      bash scripts/rs-grpo.sh
```

DAPO, LIMR, Dr.GRPO, and GSPO are implemented on top of the GRPO branch and do not use TW token weighting. TW-GRPO and RS-GRPO use token-level weighting.

### RS-GRPO Ablations

Learnability definition:

```bash
RS_LEARNABILITY_DEFINITION=full
RS_LEARNABILITY_DEFINITION=wo_contrast
RS_LEARNABILITY_DEFINITION=wo_unsolvedness
RS_LEARNABILITY_DEFINITION=wo_best_quality
RS_LEARNABILITY_DEFINITION=contrast_only
```

Group weighting rule:

```bash
RS_GROUP_WEIGHTING_RULE=hard_filter
RS_GROUP_WEIGHTING_RULE=binary_up
RS_GROUP_WEIGHTING_RULE=amplify_only
RS_GROUP_WEIGHTING_RULE=conservative
```

Reward source for estimating group learnability:

```bash
RS_REWARD_SOURCE=accuracy  # accuracy reward only
RS_REWARD_SOURCE=total     # accuracy + format reward
```

Hyperparameter ablations:

```bash
RS_LEARNABLE_LAMBDA=0.15
RS_LEARNABLE_TAU=0.5
MAX_COMPLETION_LENGTH=2048
NUM_GENERATIONS=8
```

Example:

```bash
LOSS_TYPE=rs_grpo \
RS_LEARNABILITY_DEFINITION=full \
RS_GROUP_WEIGHTING_RULE=conservative \
RS_REWARD_SOURCE=accuracy \
RS_LEARNABLE_LAMBDA=0.15 \
RS_LEARNABLE_TAU=0.5 \
MODEL_NAME=Qwen2.5-VL-7B-Instruct_clevrer_rs_full \
bash scripts/rs-grpo.sh
```

## 5. Evaluation

### CLEVRER

```bash
export PRIVATE_DATA_ROOT=/path/to/RS-GRPO
export EVAL_OUTPUT_ROOT=$PRIVATE_DATA_ROOT
export CLEVRER_VAL_PATH=$PRIVATE_DATA_ROOT/data/CLEVRER/clevrer_counterfactual_val.json
export CUDA_VISIBLE_DEVICES=0,1
export BATCH_SIZE=4

bash scripts/evaluate_clevrer_main_baselines.sh
```

To evaluate a trained RS-GRPO checkpoint:

```bash
python src/eval/eval_clevrer.py \
  --model_name /path/to/checkpoint \
  --batch_size 4
```

### NExT-GQA-Mixed

```bash
export PRIVATE_DATA_ROOT=/path/to/RS-GRPO
export EVAL_OUTPUT_ROOT=$PRIVATE_DATA_ROOT
export CUDA_VISIBLE_DEVICES=0,1
export BATCH_SIZE=4
export DATASET_NAME=eval_nextgqa_mixed_server

bash scripts/evaluate_general_main_baselines.sh
```

Evaluate an RS-GRPO checkpoint:

```bash
python src/eval/eval_general_videor1.py \
  --model_name /path/to/checkpoint \
  --dataset_name eval_nextgqa_mixed_server \
  --batch_size 4
```

### STAR-Mixed

```bash
export PRIVATE_DATA_ROOT=/path/to/RS-GRPO
export EVAL_OUTPUT_ROOT=$PRIVATE_DATA_ROOT
export CUDA_VISIBLE_DEVICES=0,1
export BATCH_SIZE=2
export DATASET_NAME=eval_star_mixed_server

bash scripts/evaluate_general_main_baselines.sh
```

Evaluate an RS-GRPO checkpoint:

```bash
python src/eval/eval_general_videor1.py \
  --model_name /path/to/checkpoint \
  --dataset_name eval_star_mixed_server \
  --batch_size 2
```

Evaluation logs are written to:

```text
logs/<dataset_name>/test/<model_name>/
```

## 6. Notes for Reproduction

- Use the same base model, dataset split, group size, maximum completion length, and checkpoint selection when comparing methods.
- `evaluate_general_main_baselines.sh` evaluates baseline models only. It does not evaluate RS-GRPO or TW-GRPO checkpoints.
- For NExT-GQA-Mixed and STAR-Mixed, use checkpoints trained on the corresponding training data if you want in-domain results. A CLEVRER-trained checkpoint is a cross-dataset transfer setting.
- The default script uses `WANDB_MODE=offline`; change it if you want online logging.
- If your server has a different directory layout, set `PRIVATE_DATA_ROOT`, `MODEL_PATH`, and `JSONL_PATH` explicitly before launching training.

## 7. Citation

If this repository helps your research, please cite the RS-GRPO paper once available.

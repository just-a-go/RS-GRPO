# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import math
from functools import partial

import torch
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration
from transformers import AutoConfig

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config

class ProcessLogger:
    def __init__(self, prefix=""):
        self.pid = os.getpid()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(
            os.getenv("PRIVATE_DATA_ROOT"),
            os.getenv("WANDB_NAME"),
            "debug_logs"
        )
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{prefix}_{timestamp}_pid{self.pid}.log")
        
    def log(self, message):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

accuracy_logger = ProcessLogger("accuracy")
format_logger = ProcessLogger("format")
description_logger = ProcessLogger("description")

@dataclass
class GRPOScriptArguments(ScriptArguments):

    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    jsonl_path: Optional[str] = field(
        default=None,
        metadata={"help": "json file path"},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False
    loss_type: str = "grpo"
    alpha: float = 1.4
    learnability_reward_source: str = field(
        default="accuracy",
        metadata={
            "help": (
                "Reward source for rs_grpo learnability. "
                "'accuracy' uses only answer correctness; 'total' uses the summed reward, e.g. accuracy + format."
            )
        },
    )
    rs_learnability_definition: str = field(
        default="full",
        metadata={
            "help": (
                "Definition of RS-GRPO learnability L(q). Options: full, wo_contrast, "
                "wo_unsolvedness, wo_best_quality, contrast_only."
            )
        },
    )
    generate_temperature: float = 1.0
    question_type: str = "mixed"
    use_epsilon: bool = False
    use_dynamic_sampling: bool = False
    train_samples: int = 2000
    method_epsilon_low: Optional[float] = None
    method_epsilon_high: Optional[float] = None
    dr_grpo_constant_normalizer: Optional[float] = None
    limr_scores_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional JSON/JSONL file containing LIMR prompt scores. "
                "If omitted, loss_type=limr uses a limr_score/learning_impact field in the dataset when available."
            )
        },
    )
    rs_group_weighting_rule: str = field(
        default="conservative",
        metadata={
            "help": (
                "RS-GRPO group weighting rule. Options: hard_filter, binary_up, "
                "amplify_only, conservative."
            )
        },
    )
    rs_learnable_lambda: float = 0.15
    rs_learnable_tau: float = 0.5
    limr_threshold: float = 0.0
    limr_id_field: Optional[str] = field(
        default=None,
        metadata={"help": "Optional dataset field used as the prompt ID for LIMR score lookup."},
    )

def accuracy_reward(completions, solution, **kwargs):
    """
    Reward function that checks if the completion is correct, supporting partial credit for subset matches.
    Returns a score between 0 and 1, where:
    - 1.0: perfect match
    - 0.0-1.0: partial match (when model answer is a correct subset)
    - 0.0: incorrect match
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        verification_method = "none"
        
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
                verification_method = "symbolic"
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching with partial credit
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r"<answer>(.*?)</answer>", sol, re.DOTALL)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                # Extract answer from content if it has think/answer tags
                content_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
                student_answer = content_match.group(1).strip() if content_match else content.strip()

                # Convert answers to sets for comparison
                ground_truth_set = set(option.strip() for option in ground_truth.replace(' ', '').split(','))
                student_answer_set = set(option.strip() for option in student_answer.replace(' ', '').split(','))

                # Check if student answer is a subset of correct answers
                if student_answer_set.issubset(ground_truth_set):
                    # Calculate partial credit: number of correct answers / total number of correct answers
                    reward = len(student_answer_set) / len(ground_truth_set)
                    verification_method = "string_matching"
                else:
                    reward = 0.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail

        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            if os.getenv("DEBUG_MODE") == "true":
                accuracy_logger.log(f"video path: {kwargs.get('video_path', 'N/A')}")
                accuracy_logger.log(f"Model output: {content}")
                accuracy_logger.log(f"Solution: {sol}")
                accuracy_logger.log(f"Calculated reward: {reward}")
                accuracy_logger.log(f"Verification method: {verification_method}")
                if verification_method == "string_matching":
                    accuracy_logger.log(f"Ground truth set: {ground_truth_set}")
                    accuracy_logger.log(f"Student answer set: {student_answer_set}")
    
    return rewards

def origin_accuracy_reward(completions, solution, **kwargs):
    """
    Reward function that checks if the completion is correct, supporting partial credit for subset matches.
    Returns a score between 0 and 1, where:
    - 1.0: perfect match
    - 0.0-1.0: partial match (when model answer is a correct subset)
    - 0.0: incorrect match
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        verification_method = "none"
        
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
                verification_method = "symbolic"
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching with partial credit
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r"<answer>(.*?)</answer>", sol, re.DOTALL)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                # Extract answer from content if it has think/answer tags
                content_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
                student_answer = content_match.group(1).strip() if content_match else content.strip()

                # Convert answers to sets for comparison
                ground_truth_set = set(option.strip() for option in ground_truth.replace(' ', '').split(','))
                student_answer_set = set(option.strip() for option in student_answer.replace(' ', '').split(','))

                # Check if student answer is a subset of correct answers
                if student_answer_set.issubset(ground_truth_set):
                    # Calculate partial credit: number of correct answers / total number of correct answers
                    reward = len(student_answer_set) / len(ground_truth_set)
                    if reward < 1:
                        reward = 0
                    verification_method = "string_matching"
                else:
                    reward = 0.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail

        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            if os.getenv("DEBUG_MODE") == "true":
                accuracy_logger.log(f"video path: {kwargs.get('video_path', 'N/A')}")
                accuracy_logger.log(f"Model output: {content}")
                accuracy_logger.log(f"Solution: {sol}")
                accuracy_logger.log(f"Calculated reward: {reward}")
                accuracy_logger.log(f"Verification method: {verification_method}")
                if verification_method == "string_matching":
                    accuracy_logger.log(f"Ground truth set: {ground_truth_set}")
                    accuracy_logger.log(f"Student answer set: {student_answer_set}")
    
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>[\s!]*$"
    completion_contents = [completion[0]["content"] for completion in completions]
    
    def check_format(content):
        # Check for duplicate tags
        if content.count("<think>") > 1 or content.count("<answer>") > 1:
            return False
        # Use more lenient pattern matching
        return bool(re.search(pattern, content.strip(), re.DOTALL))
    
    rewards = [1.0 if check_format(content) else 0.0 for content in completion_contents]
    
    if os.getenv("DEBUG_MODE") == "true":
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        for content, reward in zip(completion_contents, rewards):
            log_entry = (
                f"{current_time} | "
                f"Model output: {content} | "
                f"Calculated reward: {reward} | "
                f"Format match: {'Yes' if reward == 1.0 else 'No'}"
            )
            if reward == 0.0:
                think_count = content.count("<think>")
                answer_count = content.count("<answer>")
                log_entry += (
                    f" | Number of <think> tags: {think_count} | "
                    f"Number of <answer> tags: {answer_count}"
                )
            format_logger.log(log_entry)
    return rewards


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "origin_accuracy": origin_accuracy_reward,
    "format": format_reward,
}

from datasets import Dataset, DatasetDict
import json

def create_dataset_from_jsonl_simple(jsonl_path, question_type, train_samples=2000):
    base_dataset = Dataset.from_json(jsonl_path)
    
    # No filtering when question_type="mixed"
    if question_type != "mixed":
        def is_target_choice_type(example):
          # Extract answer from solution tag
          answer = example['solution'].replace('<answer>', '').replace('</answer>', '')
          # Check if answer contains comma to determine if it's multiple choice
          is_multiple = ',' in answer
          # question_type="single" for single choice, "multiple" for multiple choice
          return not is_multiple if question_type == "single" else is_multiple

        base_dataset = base_dataset.filter(is_target_choice_type)
    
    if len(base_dataset) > train_samples:
        base_dataset = base_dataset.shuffle(seed=42).select(range(train_samples))

    return DatasetDict({
      "train": base_dataset
    })


def normalize_loss_type(loss_type):
    normalized = (loss_type or "grpo").lower().replace("-", "_").replace(".", "_")
    aliases = {
        "drgrpo": "dr_grpo",
    }
    return aliases.get(normalized, normalized)


def _example_key(example, idx=None, id_field=None):
    if id_field and id_field in example:
        return str(example[id_field])
    for field_name in ("id", "uid", "qid", "question_id", "sample_id", "video_id"):
        if field_name in example:
            return str(example[field_name])
    if "video" in example and "problem" in example:
        return f"{example['video']}||{example['problem']}"
    if "problem" in example:
        return str(example["problem"])
    return str(idx)


def _extract_limr_score(record):
    for field_name in ("limr_score", "learning_impact", "score", "s"):
        if field_name in record:
            return float(record[field_name])
    raise KeyError("No LIMR score field found.")


def load_limr_scores(path, id_field=None):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            records = [json.loads(line) for line in f if line.strip()]
        else:
            records = json.load(f)

    if isinstance(records, dict):
        score_map = {}
        for key, value in records.items():
            if isinstance(value, dict):
                score_map[str(key)] = _extract_limr_score(value)
            else:
                score_map[str(key)] = float(value)
        return score_map

    score_map = {}
    for idx, record in enumerate(records):
        key = _example_key(record, idx=idx, id_field=id_field)
        score_map[key] = _extract_limr_score(record)
    return score_map


def apply_limr_filter(dataset, split, limr_scores_path=None, threshold=0.0, id_field=None):
    train_dataset = dataset[split]
    score_field = None
    if limr_scores_path:
        score_map = load_limr_scores(limr_scores_path, id_field=id_field)

        def keep_with_score_file(example, idx):
            key = _example_key(example, idx=idx, id_field=id_field)
            return score_map.get(key, -math.inf) > threshold

        filtered = train_dataset.filter(keep_with_score_file, with_indices=True)
    else:
        for field_name in ("limr_score", "learning_impact", "score", "s"):
            if field_name in train_dataset.features:
                score_field = field_name
                break
        if score_field is None:
            print("loss_type=limr selected, but no LIMR score file or score field was found; using the full dataset.")
            return dataset

        def keep_with_dataset_field(example):
            return float(example[score_field]) > threshold

        filtered = train_dataset.filter(keep_with_dataset_field)

    if len(filtered) == 0:
        raise ValueError("LIMR filtering removed all training samples. Lower limr_threshold or check the score keys.")
    print(f"LIMR filtering kept {len(filtered)} / {len(train_dataset)} training samples.")
    dataset[split] = filtered
    return dataset

def main(script_args, training_args, model_args):    
    
    # Add debug log settings
    if os.getenv("DEBUG_MODE") == "true":
        log_dir = os.path.join(training_args.output_dir, "debug_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        os.environ["LOG_PATH"] = log_path
        print(f"Debug logs will be saved to: {log_path}")

    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    
    if not hasattr(model_args, 'train_samples'):
        model_args.train_samples = 2000

    if not hasattr(model_args, 'question_type'):
        model_args.question_type = "mixed"

    if script_args.jsonl_path:
        # # load dataset from jsonl
        print(model_args.question_type)
        dataset = create_dataset_from_jsonl_simple(script_args.jsonl_path, model_args.question_type, model_args.train_samples)
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    if not hasattr(model_args, 'alpha'):
        model_args.alpha = 1.4

    if not hasattr(model_args, 'learnability_reward_source'):
        model_args.learnability_reward_source = "accuracy"

    if not hasattr(model_args, 'rs_learnability_definition'):
        model_args.rs_learnability_definition = "full"

    if not hasattr(model_args, 'rs_group_weighting_rule'):
        model_args.rs_group_weighting_rule = "conservative"

    if not hasattr(model_args, 'rs_learnable_lambda'):
        model_args.rs_learnable_lambda = 0.15

    if not hasattr(model_args, 'rs_learnable_tau'):
        model_args.rs_learnable_tau = 0.5
    
    if not hasattr(model_args, 'generate_temperature'):
        model_args.generate_temperature = 1.0

    model_args.loss_type = normalize_loss_type(model_args.loss_type)

    if model_args.loss_type == "limr":
        dataset = apply_limr_filter(
            dataset,
            script_args.dataset_train_split,
            limr_scores_path=model_args.limr_scores_path,
            threshold=model_args.limr_threshold,
            id_field=model_args.limr_id_field,
        )

    if model_args.question_type == "single":
        QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (select one letter) in <answer> </answer> tags."
        print("single")
    elif model_args.question_type == "multiple":
        QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (select letters separated by ,) in <answer> </answer> tags."
        print("multiple")
    else:  # mixed
        QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (letters separated by , if multiple) in <answer> </answer> tags."

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }
    
    def make_conversation_video(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        # {"type": "video", "video": example["video"]},
                        # {"type": "video", "bytes": open(example["video"],"rb").read()},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
    }
    
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "user", "content": example["problem"]},
            ],
        }

    if "image" in dataset[script_args.dataset_train_split].features:
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
    elif "video" in dataset[script_args.dataset_train_split].features:
        dataset = dataset.map(
            make_conversation_video,
        )
    else:
        dataset = dataset.map(make_conversation)
        # dataset = dataset.remove_columns("messages")
    
    # import pdb; pdb.set_trace()

    trainer_cls = Qwen2VLGRPOTrainer

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        loss_type=model_args.loss_type,
        alpha=model_args.alpha,
        learnability_reward_source=model_args.learnability_reward_source,
        rs_learnability_definition=model_args.rs_learnability_definition,
        rs_group_weighting_rule=model_args.rs_group_weighting_rule,
        rs_learnable_lambda=model_args.rs_learnable_lambda,
        rs_learnable_tau=model_args.rs_learnable_tau,
        epsilon_low=model_args.method_epsilon_low,
        epsilon_high=model_args.method_epsilon_high,
        dr_grpo_constant_normalizer=model_args.dr_grpo_constant_normalizer,
        generate_temperature=model_args.generate_temperature,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    main(script_args, training_args, model_args)

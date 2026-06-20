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

import copy
import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
from datetime import datetime
import math

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url


from qwen_vl_utils import process_vision_info

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

# Learnable weighting for rs_grpo.
# By default, learnability is estimated from the accuracy reward only, so answer
# correctness is separated from format compliance. For reward-source ablations,
# learnability can instead be estimated from the total reward.
LEARNABLE_LAMBDA = 0.15
LEARNABLE_TAU = 0.5
ACCURACY_REWARD_FUNC_NAMES = {"accuracy_reward", "origin_accuracy_reward"}
LEARNABILITY_REWARD_SOURCES = {"accuracy", "total"}
RS_LEARNABILITY_DEFINITIONS = {
    "full",
    "wo_contrast",
    "wo_unsolvedness",
    "wo_best_quality",
    "contrast_only",
}
RS_GROUP_WEIGHTING_RULES = {
    "hard_filter",
    "binary_up",
    "amplify_only",
    "conservative",
}
RS_GRPO_LEARNABILITY_DEFINITIONS = {
    "rs_grpo": "full",
    "rs_grpo_wo_contrast": "wo_contrast",
    "rs_grpo_wo_unsolvedness": "wo_unsolvedness",
    "rs_grpo_wo_best_quality": "wo_best_quality",
    "rs_grpo_contrast_only": "contrast_only",
}
GRPO_BASE_LOSS_TYPES = {"grpo", "limr", "dapo"}
SEQUENCE_LEVEL_LOSS_TYPES = {"gspo"}
DR_GRPO_LOSS_TYPES = {"dr_grpo"}

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

token_logger = ProcessLogger("token")


class Qwen2VLGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        freeze_vision_modules: Optional[bool] = False,
        loss_type: str = "grpo",
        alpha: Optional[float] = 1.4,
        learnability_reward_source: str = "accuracy",
        rs_learnability_definition: str = "full",
        rs_group_weighting_rule: str = "conservative",
        rs_learnable_lambda: float = LEARNABLE_LAMBDA,
        rs_learnable_tau: float = LEARNABLE_TAU,
        epsilon_low: Optional[float] = None,
        epsilon_high: Optional[float] = None,
        dr_grpo_constant_normalizer: Optional[float] = None,
        generate_temperature: Optional[float] = 1.0,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "bfloat16",
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = torch_dtype
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        self.vision_modules_keywords = ["visual"]
        if peft_config is not None:
            def find_all_linear_names(model, multimodal_keywords):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    # LoRA is not applied to the vision modules
                    if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                        continue
                    if isinstance(module, cls):
                        lora_module_names.add(name)
                for m in lora_module_names:  # needed for 16-bit
                    if "embed_tokens" in m:
                        lora_module_names.remove(m)
                return list(lora_module_names)
            target_modules = find_all_linear_names(model, self.vision_modules_keywords)
            peft_config.target_modules = target_modules
            model = get_peft_model(model, peft_config)

        if freeze_vision_modules:
            print("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.vision_modules_keywords):
                    p.requires_grad = False


        self.beta = args.beta
        # Reference model
        if is_deepspeed_zero3_enabled():
            if self.beta > 0:
                if "Qwen2-VL" in model_id:
                    self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                elif "Qwen2.5-VL" in model_id:
                    self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                elif "Aria" in model_id:
                    self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                else:
                    self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = None
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model_id, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.loss_type = _normalize_loss_type(loss_type)
        self.learnability_reward_source = _normalize_learnability_reward_source(learnability_reward_source)
        if self.loss_type in RS_GRPO_LEARNABILITY_DEFINITIONS and self.loss_type != "rs_grpo":
            rs_learnability_definition = RS_GRPO_LEARNABILITY_DEFINITIONS[self.loss_type]
        self.rs_group_weighting_rule = _normalize_rs_group_weighting_rule(rs_group_weighting_rule)
        self.rs_learnability_definition = _normalize_rs_learnability_definition(rs_learnability_definition)
        self.rs_learnable_lambda = rs_learnable_lambda
        self.rs_learnable_tau = rs_learnable_tau
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=generate_temperature,  # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.alpha = alpha
        self.epsilon_low = 0.20 if epsilon_low is None else epsilon_low
        if epsilon_high is None:
            self.epsilon_high = 0.28 if self.loss_type == "dapo" else self.epsilon_low
        else:
            self.epsilon_high = epsilon_high
        self.dr_grpo_constant_normalizer = dr_grpo_constant_normalizer

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the training loss for GRPO.
        Args:
            model: The model to train
            inputs: The inputs to the model
            return_outputs: Whether to return the outputs along with the loss
            num_items_in_batch: Number of items in the batch (new parameter)
        """
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        
        if "image" in inputs[0]:
            images = [x["image"] for x in inputs]
        elif "video" in inputs[0]:
            videos = [x["video"] for x in inputs]
            video_inputs = []
            for (inp_idx, inp) in enumerate(inputs):
                new_inp = inp.copy()
                new_inp['prompt'][0]['content'][0]['text'] = inputs[inp_idx]["video"]
                video_path = inputs[inp_idx]["video"]
                video_inputs.append(process_vision_info(new_inp["prompt"])[0])

        if "image" in inputs[0] or "video" in inputs[0]:
            prompt_inputs = self.processing_class(
                text=prompts_text,
                images=images if "image" in inputs[0] else None,
                videos=video_inputs if "video" in inputs[0] else None,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
        else:
            prompt_inputs = self.processing_class(
                text=prompts_text,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

        if prompt_inputs["attention_mask"].size(1) != prompt_inputs["input_ids"].size(1):
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -prompt_inputs["input_ids"].size(1) :]

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            # prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
            # Generate N times, each generate one with the temp_generation_config , stack the output_ids to prompt_completion_ids, pad the empty places with number 151613
                prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)


        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # import pdb; pdb.set_trace()

        # Get the per-token log probabilities for the completions for the model and the reference model
        def get_per_token_logps(model, input_ids, **kwargs):
            logits = model(input_ids, **kwargs).logits  # (B, L, V)
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
            # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
            per_token_logps = []
            per_logps = []
            for logits_row, input_ids_row in zip(logits, input_ids):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
                per_logps.append(log_probs)
            return torch.stack(per_token_logps), torch.stack(per_logps)

        prompt_inputs.pop("input_ids")
        prompt_inputs.pop("attention_mask")
        # Okay I am assuming that the inputs are Qwen2VL processor
        # and no video for now, repeat the image for each completion
        if "image" in inputs[0]:
            prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["image_grid_thw"] = prompt_inputs["image_grid_thw"].repeat(len(prompt_completion_ids), 1)
        # import pdb; pdb.set_trace()
        
        # XXX if input video
        # image_grid_thw is from image_process_qwen2_vl
        # https://github.com/huggingface/transformers/blob/dd16acb8a3e93b643aa374c9fb80749f5235c1a6/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L414
        # automatic process
        if "video" in inputs[0]:
            prompt_inputs["pixel_values_videos"] = prompt_inputs["pixel_values_videos"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["video_grid_thw"] = prompt_inputs["video_grid_thw"].repeat(len(prompt_completion_ids), 1)
            if "second_per_grid_ts" in prompt_inputs:
                prompt_inputs["second_per_grid_ts"] = prompt_inputs["second_per_grid_ts"] * len(prompt_completion_ids)

        per_token_logps, per_logps = get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]
        per_logps = per_logps[:, prompt_length - 1 :]

        if self.beta > 0:
            with torch.inference_mode():
                if self.ref_model is not None:
                    ref_per_token_logps, ref_per_logps = get_per_token_logps(self.ref_model, prompt_completion_ids, **prompt_inputs)
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps, ref_per_logps = get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
            ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]
        else:
            ref_per_token_logps = per_token_logps.detach()
            ref_per_logps = per_logps.detach()
            
        # Following previous works like R1-V and TRL, we simplify the clipping mechanism which has been shown to work well in practice.
        # For reference, please see:
        # https://github.com/huggingface/trl/issues/2608#issuecomment-2609844003
        old_per_token_logps = per_token_logps.detach()

        # Compute the KL divergence between the model and the reference model
        if self.beta > 0:
            diff = ref_per_token_logps - per_token_logps
            diff = torch.clamp(diff, min=-11.0, max=11.0) 
        else:
            diff = torch.zeros_like(per_token_logps)

        per_token_kl = torch.exp(diff) - (diff) - 1

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]): # true
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Add trainer to reward_kwargs
                reward_kwargs = {
                    key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]
                }
                for key in reward_kwargs:
                    for example in inputs:
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                # Pass per_token_logps, trainer, completion_ids and tag_tokens to the reward function
                reward_kwargs["per_token_logps"] = per_token_logps.detach()
                reward_kwargs["per_logps"] = per_logps.detach()
                reward_kwargs["trainer"] = self
                reward_kwargs["completion_ids"] = completion_ids
                reward_kwargs["tag_tokens"] = getattr(self, 'tag_tokens', None)
                try:
                    reward_kwargs["video_path"] = video_path
                except NameError:
                    reward_kwargs["video_path"] = None
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)

        # Compute grouped-wise rewards
        grouped_rewards = rewards.view(-1, self.num_generations)
        mean_grouped_rewards = grouped_rewards.mean(dim=1)
        std_grouped_rewards = grouped_rewards.std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        centered_advantages = rewards - mean_grouped_rewards
        if self.loss_type == "dr_grpo":
            advantages = centered_advantages
        else:
            advantages = centered_advantages / (std_grouped_rewards + 1e-4)

        # # x - x.detach() allows for preserving gradients from x
        # per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        # per_token_loss = -(per_token_loss - self.beta * per_token_kl) # default 0.04

        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = torch.min(per_token_loss1, per_token_loss2)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()


        group_learnable = group_learnable_omega = None
        dapo_group_keep = None

        token_count = completion_mask.sum(dim=1).clamp_min(1)
        sequence_loss = (per_token_loss * completion_mask).sum(dim=1) / token_count

        if self.loss_type == "dapo":
            dapo_group_keep = compute_dapo_group_filter(
                rewards_per_func,
                rewards,
                self.reward_funcs,
                self.num_generations,
                device=device,
            )
            if dapo_group_keep.any():
                sequence_keep = dapo_group_keep.repeat_interleave(self.num_generations).to(sequence_loss.dtype)
                loss = (sequence_loss * sequence_keep).sum() / sequence_keep.sum().clamp_min(1.0)
            else:
                # Avoid a zero-size loss when a small local batch contains only degenerate groups.
                loss = sequence_loss.mean()
        elif self.loss_type in {"grpo", "limr"}:
            # Original GRPO family loss: normalize each sequence first, then average.
            loss = sequence_loss.mean()
        elif self.loss_type == "dr_grpo":
            normalizer = self.dr_grpo_constant_normalizer or self.max_completion_length or completion_mask.size(1)
            normalizer = max(float(normalizer), 1.0)
            loss = (per_token_loss * completion_mask).sum(dim=1).div(normalizer).mean()
        elif self.loss_type == "gspo":
            log_ratio = per_token_logps - old_per_token_logps
            sequence_log_ratio = (log_ratio * completion_mask).sum(dim=1) / token_count
            sequence_ratio = torch.exp(sequence_log_ratio)
            clipped_sequence_ratio = torch.clamp(sequence_ratio, 1 - self.epsilon_low, 1 + self.epsilon_high)
            sequence_loss1 = sequence_ratio * advantages
            sequence_loss2 = clipped_sequence_ratio * advantages
            sequence_surrogate = torch.min(sequence_loss1, sequence_loss2)
            sequence_kl = (per_token_kl * completion_mask).sum(dim=1) / token_count
            loss = -(sequence_surrogate - self.beta * sequence_kl).mean()
        elif self.loss_type == "tw_grpo" or self.loss_type in RS_GRPO_LEARNABILITY_DEFINITIONS:
            # Focus on reasoning
            token_weights = compute_token_importance_kl_logs_uniform(per_logps, completion_mask, completion_ids, self.num_generations, max_weight=self.alpha)
            weighted_loss = per_token_loss * token_weights
            if self.loss_type == "tw_grpo":
                loss = (weighted_loss * completion_mask).sum() / completion_mask.sum()
            else:
                group_learnable, group_learnable_omega = compute_learnable_weights(
                    rewards_per_func,
                    rewards,
                    self.reward_funcs,
                    self.num_generations,
                    device=device,
                    dtype=weighted_loss.dtype,
                    definition=self.rs_learnability_definition,
                    reward_source=self.learnability_reward_source,
                    weighting_rule=self.rs_group_weighting_rule,
                    learnable_lambda=self.rs_learnable_lambda,
                    learnable_tau=self.rs_learnable_tau,
                )
                batch_size = weighted_loss.size(0) // self.num_generations
                token_length = weighted_loss.size(1)
                grouped_loss = weighted_loss.view(batch_size, self.num_generations, token_length)
                grouped_mask = completion_mask.view(batch_size, self.num_generations, token_length)
                group_loss_sum = (grouped_loss * grouped_mask).sum(dim=(1, 2))

                # Keep the denominator identical to TW-GRPO. When every group
                # has omega=1, this branch exactly reduces to the TW-GRPO loss.
                loss = (group_loss_sum * group_learnable_omega).sum() / completion_mask.sum().clamp_min(1)
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")
        
        # import pdb; pdb.set_trace()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        
        self._metrics["advantages"].append(self.accelerator.gather_for_metrics(advantages).mean().item())
        
        self._metrics["reward_mean"].append(self.accelerator.gather_for_metrics(mean_grouped_rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        if group_learnable is not None:
            self._metrics["sample_learnable"].append(
                self.accelerator.gather_for_metrics(group_learnable).mean().item()
            )
            self._metrics["sample_learnable_omega"].append(
                self.accelerator.gather_for_metrics(group_learnable_omega).mean().item()
            )
        if dapo_group_keep is not None:
            kept_ratio = self.accelerator.gather_for_metrics(dapo_group_keep.float()).mean().item()
            self._metrics["dapo_kept_group_ratio"].append(kept_ratio)

        # import pdb; pdb.set_trace()

        return loss


def _normalize_loss_type(loss_type):
    normalized = (loss_type or "grpo").lower().replace("-", "_").replace(".", "_")
    aliases = {
        "drgrpo": "dr_grpo",
        "dr__grpo": "dr_grpo",
    }
    normalized = aliases.get(normalized, normalized)
    valid_loss_types = (
        GRPO_BASE_LOSS_TYPES
        | SEQUENCE_LEVEL_LOSS_TYPES
        | DR_GRPO_LOSS_TYPES
        | {"tw_grpo"}
        | set(RS_GRPO_LEARNABILITY_DEFINITIONS)
    )
    if normalized not in valid_loss_types:
        raise ValueError(f"Unsupported loss_type: {loss_type}. Expected one of {sorted(valid_loss_types)}.")
    return normalized


def _find_accuracy_reward_index(reward_funcs):
    """Return the reward column that reflects answer correctness."""
    for idx, reward_func in enumerate(reward_funcs):
        reward_func_name = getattr(reward_func, "__name__", "")
        if reward_func_name in ACCURACY_REWARD_FUNC_NAMES or "accuracy" in reward_func_name:
            return idx
    return None


def _normalize_learnability_reward_source(reward_source):
    source = (reward_source or "accuracy").lower()
    if source in {"accuracy_only", "acc"}:
        source = "accuracy"
    elif source in {"total_reward", "all", "sum"}:
        source = "total"
    if source not in LEARNABILITY_REWARD_SOURCES:
        raise ValueError(
            f"Unsupported learnability_reward_source: {reward_source}. "
            f"Expected one of {sorted(LEARNABILITY_REWARD_SOURCES)}."
        )
    return source


def _normalize_rs_learnability_definition(definition):
    definition = (definition or "full").lower().replace("-", "_")
    aliases = {
        "all": "full",
        "default": "full",
        "without_contrast": "wo_contrast",
        "w_o_contrast": "wo_contrast",
        "no_contrast": "wo_contrast",
        "without_unsolvedness": "wo_unsolvedness",
        "w_o_unsolvedness": "wo_unsolvedness",
        "no_unsolvedness": "wo_unsolvedness",
        "without_best_quality": "wo_best_quality",
        "without_quality": "wo_best_quality",
        "w_o_best_quality": "wo_best_quality",
        "no_best_quality": "wo_best_quality",
        "only_contrast": "contrast_only",
    }
    definition = aliases.get(definition, definition)
    if definition not in RS_LEARNABILITY_DEFINITIONS:
        raise ValueError(
            f"Unsupported rs_learnability_definition: {definition}. "
            f"Expected one of {sorted(RS_LEARNABILITY_DEFINITIONS)}."
        )
    return definition


def _normalize_rs_group_weighting_rule(weighting_rule):
    rule = (weighting_rule or "conservative").lower().replace("-", "_")
    aliases = {
        "filter": "hard_filter",
        "hard": "hard_filter",
        "hard_filtering": "hard_filter",
        "binary": "binary_up",
        "binary_up_weighting": "binary_up",
        "up_weighting": "binary_up",
        "amplify": "amplify_only",
        "bounded": "conservative",
        "conservative_bounded": "conservative",
    }
    rule = aliases.get(rule, rule)
    if rule not in RS_GROUP_WEIGHTING_RULES:
        raise ValueError(
            f"Unsupported rs_group_weighting_rule: {weighting_rule}. "
            f"Expected one of {sorted(RS_GROUP_WEIGHTING_RULES)}."
        )
    return rule


def _select_learnability_rewards(
    rewards_per_func,
    rewards,
    reward_funcs,
    reward_source,
    device=None,
):
    """Select the scalar reward used to estimate group learnability."""
    reward_source = _normalize_learnability_reward_source(reward_source)
    if reward_source == "accuracy":
        accuracy_reward_idx = _find_accuracy_reward_index(reward_funcs)
        if accuracy_reward_idx is None:
            raise ValueError("rs_grpo with accuracy learnability requires an accuracy or origin_accuracy reward function.")
        return rewards_per_func[:, accuracy_reward_idx].to(device=device, dtype=torch.float32)
    if reward_source == "total":
        if rewards is None:
            raise ValueError("rs_grpo with total learnability requires the summed reward tensor.")
        return rewards.to(device=device, dtype=torch.float32)
    raise ValueError(f"Unsupported learnability_reward_source: {reward_source}")


def compute_learnable_weights(
    rewards_per_func,
    rewards,
    reward_funcs,
    num_generations,
    device=None,
    dtype=torch.float32,
    definition="full",
    reward_source="accuracy",
    weighting_rule="conservative",
    learnable_lambda=LEARNABLE_LAMBDA,
    learnable_tau=LEARNABLE_TAU,
):
    """Compute conservative group learnability weights for rs_grpo.

    For each prompt group, using the selected reward source:
        contrast = max(R) - min(R)
        unsolved = 1 - mean(R)
        quality = 0.5 + 0.5 * max(R)
        full: learnable = clip(contrast * unsolved * quality / tau, 0, 1)
        wo_contrast: learnable = clip(unsolved * quality / tau, 0, 1)
        wo_unsolvedness: learnable = clip(contrast * quality / tau, 0, 1)
        wo_best_quality: learnable = clip(contrast * unsolved / tau, 0, 1)
        contrast_only: learnable = clip(contrast / tau, 0, 1)
        hard_filter: omega = I[learnable > 0]
        binary_up: omega = 1 + lambda * I[learnable > 0]
        amplify_only: omega = 1 + lambda * learnable
        conservative: omega = 1 + lambda * (2 * learnable - 1)

    This strengthens groups where the model samples meaningfully different
    answer quality and still has room to improve, while mildly down-weighting
    all-wrong, all-correct, or no-contrast groups.
    """
    if num_generations <= 0 or rewards_per_func.size(0) % num_generations != 0:
        raise ValueError("Reward tensor size must be divisible by num_generations for rs_grpo.")

    learnability_rewards = _select_learnability_rewards(
        rewards_per_func,
        rewards,
        reward_funcs,
        reward_source,
        device=device,
    )
    grouped_rewards = learnability_rewards.view(-1, num_generations)
    max_rewards = grouped_rewards.max(dim=1).values
    min_rewards = grouped_rewards.min(dim=1).values
    mean_rewards = grouped_rewards.mean(dim=1)

    reward_contrast = max_rewards - min_rewards
    unsolved = torch.clamp(1.0 - mean_rewards, min=0.0, max=1.0)
    best_quality = 0.5 + 0.5 * max_rewards
    definition = _normalize_rs_learnability_definition(definition)
    if definition == "full":
        learnable_raw = reward_contrast * unsolved * best_quality
    elif definition == "wo_contrast":
        learnable_raw = unsolved * best_quality
    elif definition == "wo_unsolvedness":
        learnable_raw = reward_contrast * best_quality
    elif definition == "wo_best_quality":
        learnable_raw = reward_contrast * unsolved
    elif definition == "contrast_only":
        learnable_raw = reward_contrast
    else:
        raise ValueError(f"Unsupported learnability definition: {definition}")
    if learnable_tau <= 0:
        raise ValueError("rs_learnable_tau must be positive.")
    learnable = torch.clamp(learnable_raw / learnable_tau, min=0.0, max=1.0)
    weighting_rule = _normalize_rs_group_weighting_rule(weighting_rule)
    if weighting_rule == "hard_filter":
        omega = (learnable > 0).to(dtype=torch.float32)
    elif weighting_rule == "binary_up":
        omega = 1.0 + learnable_lambda * (learnable > 0).to(dtype=torch.float32)
    elif weighting_rule == "amplify_only":
        omega = 1.0 + learnable_lambda * learnable
    elif weighting_rule == "conservative":
        omega = 1.0 + learnable_lambda * (2.0 * learnable - 1.0)
    else:
        raise ValueError(f"Unsupported RS group weighting rule: {weighting_rule}")
    return learnable.to(dtype=dtype), omega.to(dtype=dtype)


def compute_dapo_group_filter(
    rewards_per_func,
    rewards,
    reward_funcs,
    num_generations,
    device=None,
):
    """Keep only non-degenerate prompt groups for DAPO-style dynamic sampling."""
    if num_generations <= 0 or rewards_per_func.size(0) % num_generations != 0:
        raise ValueError("Reward tensor size must be divisible by num_generations for DAPO filtering.")
    try:
        filter_rewards = _select_learnability_rewards(
            rewards_per_func,
            rewards,
            reward_funcs,
            "accuracy",
            device=device,
        )
    except ValueError:
        filter_rewards = rewards.to(device=device, dtype=torch.float32)
    grouped_rewards = filter_rewards.view(-1, num_generations)
    num_positive = (grouped_rewards > 0).sum(dim=1)
    return (num_positive > 0) & (num_positive < num_generations)


def compute_token_importance_kl_logs_uniform(per_logps, completion_mask, completion_ids, num_generations, max_weight=1.5):
    # Basic dimension checks
    if not isinstance(per_logps, torch.Tensor) or per_logps.ndim != 3:
        return torch.ones_like(per_logps[..., 0])
        
    total_size, token_length, vocab_size = per_logps.size()
    if total_size % num_generations != 0:
        return torch.ones_like(per_logps[..., 0])
        
    batch_size = total_size // num_generations
    
    # Reshape tensors for group-wise computation
    grouped_logps = per_logps.view(batch_size, num_generations, token_length, vocab_size)
    grouped_masks = completion_mask.view(batch_size, num_generations, token_length)
    grouped_masks = grouped_masks.unsqueeze(-1)  # (batch_size, num_generations, token_length, 1)
    
    # Create uniform distribution for masked positions (log space)
    uniform_logps = torch.full_like(grouped_logps, -math.log(vocab_size))
    
    # Set logps to uniform distribution for positions beyond sequence length
    masked_logps = torch.where(grouped_masks == 1, grouped_logps, uniform_logps)
    
    # Calculate mean distribution for each token position
    mean_logps = masked_logps.mean(dim=1, keepdim=True)  # (batch_size, 1, token_length, vocab_size)
    
    # Calculate KL divergence between each sequence and the mean
    diff = mean_logps - masked_logps
    diff = torch.clamp(diff, min=-11.0, max=11.0)
    token_kl = (torch.exp(diff) - diff - 1).sum(dim=-1)
    token_kl = token_kl.mean(dim=1)
    
    # Apply min-max normalization to KL divergence
    kl_min = token_kl.min(dim=1, keepdim=True)[0]
    kl_max = token_kl.max(dim=1, keepdim=True)[0]
    normalized_kl = (token_kl - kl_min) / (kl_max - kl_min + 1e-8)
    
    # Map normalized KL to weights range [1.0, max_weight]
    token_weights = 1.0 + (max_weight - 1.0) * normalized_kl
    # Repeat weights for each generation
    token_weights = token_weights.repeat_interleave(num_generations, dim=0)
    
    # Apply the completion mask to ensure consistency
    token_weights = token_weights * completion_mask

    # log token_weights
    # for per_logps, completion_id in zip(per_logps, completion_ids):
    #     # token_logger.log(f"per_token_logp: {per_logps}")
    #     token_logger.log(f"completion_id: {completion_id}")
    # token_logger.log(f"token_weights: {normalized_kl}")
    
    return token_weights

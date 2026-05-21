import json
import itertools
import math
import random
import re
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from datasets import load_dataset
import pdb
import logging
from itertools import count


logging.basicConfig(
    filename='qwenvl_sqa_4.log',  
    level=logging.DEBUG,          
    format='[%(asctime)s] %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S' 
)

DEFAULT_RHO_SCHEDULE = (
    0.0, 0.0,
    0.2, 0.2,
    0.4, 0.4,
    0.6, 0.6,
    0.8, 0.8,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
)


def get_epoch_rho(epoch: int, configs) -> float:
    schedule = getattr(configs, "rho_schedule", DEFAULT_RHO_SCHEDULE)
    if not schedule:
        schedule = DEFAULT_RHO_SCHEDULE
    rho = float(schedule[min(epoch, len(schedule) - 1)])
    return max(0.0, min(1.0, rho))


def split_rationale_into_sentences(rationale: str) -> list[str]:
    rationale = (rationale or "").replace("\n", " ").strip()
    if not rationale:
        return []

    sentences = re.findall(r"[^.!?]+(?:[.!?]+|$)", rationale)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def group_steps_to_max(steps: list[str], max_steps: int) -> list[str]:
    if max_steps <= 0:
        return []
    if len(steps) <= max_steps:
        return steps

    grouped_steps = []
    start = 0
    for group_idx in range(max_steps):
        remaining_items = len(steps) - start
        remaining_groups = max_steps - group_idx
        group_size = math.ceil(remaining_items / remaining_groups)
        end = start + group_size
        grouped_steps.append(" ".join(steps[start:end]).strip())
        start = end
    return grouped_steps


def get_dynamic_latent_counts(num_steps: int, rho: float) -> tuple[int, int]:
    if num_steps <= 0:
        return 0, 0
    n_latent_tokens = int(math.ceil(rho * num_steps))
    n_latent_tokens = max(0, min(num_steps, n_latent_tokens))
    return n_latent_tokens, n_latent_tokens


def get_dataset(dataset, tokenizer, processor, max_size=1000000000):

    def tokenize_sample(sample, max_length=3400):
        image = sample["image"]
        pixel_values = sample["pixel_values"]
        image_grid_thw = sample["image_grid_thw"]
        
        processed_question = sample["question"]

        # Tokenize question
        question_tokenized = sample["input_ids"]
        logging.debug(f"step length: {len(sample["steps"])}")
        # Tokenize steps
        steps_tokenized = [
            tokenizer.encode(s + "\n", add_special_tokens=False)
            for s in sample["steps"]
        ]
        sample["answer"] = str(sample["answer"])
        # Tokenize answer
        answer_tokenized = tokenizer.encode(
            "Therefore, the answer is " + sample["answer"], add_special_tokens=False
        ) + [tokenizer.eos_token_id]
        
        # Calculate total sequence length
        total_length = (
            len(question_tokenized)
            + sum(len(step) for step in steps_tokenized)
            + len(answer_tokenized)
        )
        print("question length: ", len(question_tokenized))
        # If total length exceeds max_length, truncate steps_tokenized
        if total_length > max_length:
            # Calculate how much to reduce
            excess_length = total_length - max_length
            # Reduce steps_tokenized
            new_steps_tokenized = []
            current_length = 0
            for step in steps_tokenized:
                if current_length + len(step) <= (sum(len(s) for s in steps_tokenized) - excess_length):
                    new_steps_tokenized.append(step)
                    current_length += len(step)
                else:
                    break
            steps_tokenized = new_steps_tokenized
        # Build the final sample
        sample = {
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "idx": sample["idx"],
        }
        
        return sample

    dataset = dataset.map(lambda example, idx: {"idx": idx}, with_indices=True)
    data = dataset

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            processed_dataset = [
                dataset.map(
                    tokenize_sample, remove_columns=list(dataset.features), num_proc=2
                )
            ]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]

    else:
        dataset = dataset.map(
            tokenize_sample, remove_columns=list(dataset.features), num_proc=2
        )

    return dataset


@dataclass
class MyCollator:

    tokenizer: PreTrainedTokenizerBase
    latent_id: Optional[int] = None
    label_pad_token_id: Optional[int] = -100

    def __call__(self, features, return_tensors=None):

        assert self.tokenizer.padding_side == "right"

        earliest_latent = [
            feature["input_ids"].index(self.latent_id)
            for feature in features
            if self.latent_id in feature["input_ids"]
        ]

        if len(earliest_latent) > 0:  
            latest_earliest_latent = max(earliest_latent)
            for feature in features:
                if self.latent_id in feature["input_ids"]:
                    n_tok_pad = latest_earliest_latent - feature["input_ids"].index(
                        self.latent_id
                    )
                else:
                    n_tok_pad = 0
                feature["position_ids"] = [0] * n_tok_pad + list(
                    range(len(feature["input_ids"]))
                )
                feature["input_ids"] = [
                    self.tokenizer.pad_token_id
                ] * n_tok_pad + feature["input_ids"]
                if "labels" in feature:
                    feature["labels"] = [self.label_pad_token_id] * n_tok_pad + feature[
                        "labels"
                    ]
                feature["attention_mask"] = [0] * n_tok_pad + feature["attention_mask"]

        return_tensors = "pt"

        label_name = "label" if "label" in features[0].keys() else "labels"

        non_label_position_features = [
            {
                k: v
                for k, v in feature.items()
                if k != label_name and k != "position_ids"
            }
            for feature in features
        ]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_label_position_features,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors=return_tensors,
        )

        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        if labels is not None and all(label is None for label in labels):
            labels = None
        position_ids = (
            [feature["position_ids"] for feature in features]
            if "position_ids" in features[0].keys()
            else None
        )
        # we have to pad the labels and position_ids manually as we cannot rely on `tokenizer.pad`

        if labels is not None:
            max_label_length = max(len(l) for l in labels)

            batch["labels"] = [
                label + [self.label_pad_token_id] * (max_label_length - len(label))
                for label in labels
            ]
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

        if position_ids is not None:
            max_pos_length = max(len(l) for l in position_ids)

            batch["position_ids"] = [
                position_id + [0] * (max_pos_length - len(position_id))
                for position_id in position_ids
            ]
            batch["position_ids"] = torch.tensor(
                batch["position_ids"], dtype=torch.int64
            )

        return batch

def get_cot_latent_dataset(
    rho,
    base_dataset,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
    shuffle=False,
):

    n_additional_tokens = 0 if no_special_marker else 2

    def process_dataset(sample):
        n_skip_steps, n_latent_tokens = get_dynamic_latent_counts(
            len(sample["steps_tokenized"]),
            float(rho),
        )

        tokens = (
            sample["question_tokenized"]
            + [latent_id] * n_latent_tokens
            + list(
                itertools.chain.from_iterable(sample["steps_tokenized"][n_skip_steps:])
            )
            + sample["answer_tokenized"]
        )
        
        return {
            "input_ids": tokens,
            "labels": [-100]
            * (
                len(sample["question_tokenized"])
                + n_latent_tokens
            )
            + tokens[
                n_latent_tokens
                + len(sample["question_tokenized"]) :
            ],
            "attention_mask": [1] * len(tokens),
            "idx": sample["idx"],
            "position_ids": list(range(len(tokens))),
            "pixel_values": torch.tensor(sample["pixel_values"]),
            "image_grid_thw": sample["image_grid_thw"]
        }

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            processed_dataset = base_dataset.map(
                process_dataset, remove_columns=list(base_dataset.features), num_proc=2
            )
            if shuffle:
                processed_dataset = processed_dataset.shuffle()
            processed_dataset = [processed_dataset]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]

    else:
        processed_dataset = base_dataset.map(
            process_dataset, remove_columns=list(base_dataset.features), num_proc=2
        )
        if shuffle:
            processed_dataset = processed_dataset.shuffle()
        dataset = processed_dataset

    return dataset

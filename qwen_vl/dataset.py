import itertools
import math
import re
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
import logging


logging.basicConfig(
    filename='qwenvl_sqa_4.log',  
    level=logging.DEBUG,          
    format='[%(asctime)s] %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S' 
)

DEFAULT_RHO_SCHEDULE = (
    0.0, 0.0,
    0.1, 0.1,
    0.2, 0.2,
    0.3, 0.3,
    0.4, 0.4,
    0.5, 0.6,
    0.7, 0.8,
    0.9, 1.0, 
    1.0, 1.0, 
    1.0, 1.0,
)
DEFAULT_LAMBDA_HIDDEN_SCHEDULE = (
    0.0, 0.0,
    0.005, 0.005,
    0.01, 0.01,
    0.015, 0.015,
    0.02, 0.02,
    0.03, 0.03,
    0.02, 0.015,
    0.01, 0.005, 
    0.005, 0.005, 
    0.005, 0.005,
)
DEFAULT_FIXED_MASK_SCHEDULE = (
    0, 0,
    1, 1,
    2, 2,
    3, 3,
    4, 4,
    5, 5,
    6, 6,
    7, 7,
    8, 8, 
    8, 8,
)


@dataclass(frozen=True)
class CurriculumState:
    epoch: int
    mask_mode: str
    rho: Optional[float]
    mask_count: Optional[int]
    train_all_samples: bool = False

    @property
    def uses_fixed_mask(self) -> bool:
        return self.rho is None


def get_epoch_rho(epoch: int, configs) -> float:
    # The curriculum is epoch-indexed so every worker builds the same latent mix
    # without storing per-example state. Values beyond the configured schedule
    # reuse the final entry, which makes resume/extension runs predictable.
    schedule = getattr(configs, "rho_schedule", DEFAULT_RHO_SCHEDULE)
    if not schedule:
        schedule = DEFAULT_RHO_SCHEDULE
    rho = float(schedule[min(epoch, len(schedule) - 1)])
    return max(0.0, min(1.0, rho))


def get_fixed_mask_count(epoch: int, configs) -> int:
    schedule = getattr(configs, "fixed_mask_schedule", DEFAULT_FIXED_MASK_SCHEDULE)
    if not schedule:
        schedule = DEFAULT_FIXED_MASK_SCHEDULE
    mask_count = int(schedule[min(epoch, len(schedule) - 1)])
    max_steps = int(getattr(configs, "fixed_mask_max_steps", 8))
    return max(0, min(mask_count, max_steps))


def should_train_all_samples_fixed_epoch(epoch: int, configs) -> bool:
    schedule = getattr(configs, "fixed_mask_schedule", DEFAULT_FIXED_MASK_SCHEDULE)
    if not schedule:
        schedule = DEFAULT_FIXED_MASK_SCHEDULE

    max_steps = int(getattr(configs, "fixed_mask_max_steps", 8))
    clamped_schedule = [max(0, min(int(mask_count), max_steps)) for mask_count in schedule]
    schedule_idx = min(epoch, len(clamped_schedule) - 1)

    if clamped_schedule[schedule_idx] != max_steps:
        return False

    max_step_epochs = [
        idx for idx, mask_count in enumerate(clamped_schedule)
        if mask_count == max_steps
    ]
    return schedule_idx in max_step_epochs[-2:]


def _reference_mask_count_for_epoch(epoch: int, configs) -> int:
    max_steps = int(getattr(configs, "fixed_mask_max_steps", 8))
    if getattr(configs, "fixed_mask_step_curriculum", False):
        return get_fixed_mask_count(epoch, configs)
    return get_dynamic_latent_counts(max_steps, get_epoch_rho(epoch, configs))[1]


def get_epoch_mask_mode(epoch: int, configs) -> str:
    if not getattr(configs, "prefix_span", False):
        return "prefix"

    if _reference_mask_count_for_epoch(epoch, configs) == 0:
        return "prefix"

    if epoch <= 0:
        return "prefix"

    current_count = _reference_mask_count_for_epoch(epoch, configs)
    previous_count = _reference_mask_count_for_epoch(epoch - 1, configs)
    return "span" if current_count == previous_count else "prefix"


def get_epoch_curriculum_state(epoch: int, configs) -> CurriculumState:
    if getattr(configs, "fixed_mask_step_curriculum", False):
        return CurriculumState(
            epoch=epoch,
            mask_mode=get_epoch_mask_mode(epoch, configs),
            rho=None,
            mask_count=get_fixed_mask_count(epoch, configs),
            train_all_samples=should_train_all_samples_fixed_epoch(epoch, configs),
        )

    rho = get_epoch_rho(epoch, configs)
    return CurriculumState(
        epoch=epoch,
        mask_mode=get_epoch_mask_mode(epoch, configs),
        rho=rho,
        mask_count=None,
    )


def get_epoch_lambda_hidden(epoch: int, configs) -> float:
    # Keep the no-distillation ablation explicit in YAML. A disabled alignment
    # flag wins over any lambda schedule values left in the config for reference.
    if not getattr(configs, "hidden_state_alignment", True):
        return 0.0

    schedule = getattr(configs, "lambda_hidden_schedule", DEFAULT_LAMBDA_HIDDEN_SCHEDULE)
    if not schedule:
        schedule = DEFAULT_LAMBDA_HIDDEN_SCHEDULE
    return max(0.0, float(schedule[min(epoch, len(schedule) - 1)]))


def split_rationale_into_sentences(rationale: str) -> list[str]:
    rationale = (rationale or "").replace("\n", " ").strip()
    if not rationale:
        return []

    # Keep sentence punctuation attached so tokenized teacher boundaries point
    # to the same natural-language units that were visible in the original CoT.
    sentences = re.findall(r"[^.!?]+(?:[.!?]+|$)", rationale)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def group_steps_to_max(steps: list[str], max_steps: int) -> list[str]:
    if max_steps <= 0:
        return []
    if len(steps) <= max_steps:
        return steps

    # Long rationales are compressed into ordered contiguous blocks. This keeps
    # K bounded while preserving the causal prefix each latent step summarizes.
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
    # Student inputs replace a prefix of the explicit CoT. The same count is
    # skipped from labels/text because CE is only applied to unreplaced steps
    # plus the final answer.
    if num_steps <= 0:
        return 0, 0
    n_latent_tokens = int(math.ceil(rho * num_steps))
    n_latent_tokens = max(0, min(num_steps, n_latent_tokens))
    return n_latent_tokens, n_latent_tokens


def sample_span_start(num_steps: int, mask_count: int, epoch: int, sample_idx: int, seed: int) -> int:
    valid_starts = num_steps - mask_count + 1
    if mask_count <= 0 or valid_starts <= 1:
        return 0
    value = (
        int(seed) * 1_000_003
        + int(epoch) * 97_003
        + int(sample_idx) * 1_009
        + int(mask_count) * 101
        + int(num_steps)
    )
    return value % valid_starts


def get_masked_step_indices(
    num_steps: int,
    curriculum_state: CurriculumState,
    sample_idx: int,
    seed: int,
) -> list[int]:
    if num_steps <= 0:
        return []

    if curriculum_state.uses_fixed_mask:
        mask_count = int(curriculum_state.mask_count or 0)
    else:
        mask_count = get_dynamic_latent_counts(num_steps, float(curriculum_state.rho or 0.0))[1]

    mask_count = max(0, min(num_steps, mask_count))
    if mask_count == 0:
        return []

    if curriculum_state.mask_mode == "span":
        start = sample_span_start(
            num_steps,
            mask_count,
            curriculum_state.epoch,
            sample_idx,
            seed,
        )
    else:
        start = 0

    return list(range(start, start + mask_count))


def build_latent_replacement_sequence(
    question_tokenized: list[int],
    steps_tokenized: list[list[int]],
    answer_tokenized: list[int],
    latent_id: int,
    masked_step_indices: list[int],
    label_pad_token_id: int = -100,
) -> tuple[list[int], list[int]]:
    masked_steps = set(masked_step_indices)
    tokens = list(question_tokenized)
    labels = [label_pad_token_id] * len(question_tokenized)

    for step_idx, step_tokens in enumerate(steps_tokenized):
        if step_idx in masked_steps:
            tokens.append(latent_id)
            labels.append(label_pad_token_id)
        else:
            tokens.extend(step_tokens)
            labels.extend(step_tokens)

    tokens.extend(answer_tokenized)
    labels.extend(answer_tokenized)
    return tokens, labels


def get_teacher_sentence_end_positions(
    question_length: int,
    steps_tokenized: list[list[int]],
    masked_step_indices: list[int],
) -> list[int]:
    masked_steps = set(masked_step_indices)
    teacher_sentence_end_positions = []
    cursor = question_length
    for step_idx, step_tokens in enumerate(steps_tokenized):
        cursor += len(step_tokens)
        if step_idx in masked_steps:
            teacher_sentence_end_positions.append(cursor - 1)
    return teacher_sentence_end_positions


def should_keep_for_curriculum(sample, curriculum_state: CurriculumState) -> bool:
    if not curriculum_state.uses_fixed_mask:
        return True
    if curriculum_state.train_all_samples:
        return True
    mask_count = int(curriculum_state.mask_count or 0)
    return mask_count <= 0 or len(sample["steps"]) >= mask_count


def filter_dataset_for_curriculum(base_dataset, curriculum_state: CurriculumState):
    if (
        not curriculum_state.uses_fixed_mask
        or curriculum_state.train_all_samples
        or int(curriculum_state.mask_count or 0) <= 0
    ):
        return base_dataset
    return base_dataset.filter(
        lambda sample: should_keep_for_curriculum(sample, curriculum_state),
        num_proc=1,
        load_from_cache_file=False,
        desc=f"Filtering examples for {curriculum_state.mask_count} masked steps",
    )


def get_dataset(dataset, tokenizer, processor, max_size=1000000000):

    def tokenize_sample(sample, max_length=3400):
        pixel_values = sample["pixel_values"]
        image_grid_thw = sample["image_grid_thw"]

        question_tokenized = sample["input_ids"]
        logging.debug(f"step length: {len(sample['steps'])}")

        steps_tokenized = [
            tokenizer.encode(s + "\n", add_special_tokens=False)
            for s in sample["steps"]
        ]
        sample["answer"] = str(sample["answer"])
        answer_tokenized = tokenizer.encode(
            "Therefore, the answer is " + sample["answer"], add_special_tokens=False
        ) + [tokenizer.eos_token_id]
        
        total_length = (
            len(question_tokenized)
            + sum(len(step) for step in steps_tokenized)
            + len(answer_tokenized)
        )

        if total_length > max_length:
            excess_length = total_length - max_length
            new_steps_tokenized = []
            current_length = 0
            for step in steps_tokenized:
                if current_length + len(step) <= (sum(len(s) for s in steps_tokenized) - excess_length):
                    new_steps_tokenized.append(step)
                    current_length += len(step)
                else:
                    break
            steps_tokenized = new_steps_tokenized

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
        # Tokenizer padding does not know about ignored LM labels or manually
        # shifted position ids, so these two fields are padded separately.

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

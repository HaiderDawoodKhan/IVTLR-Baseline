import argparse
import gc
import itertools
import logging
import os
import shutil

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import yaml
from datasets import disable_caching, load_dataset
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

from dataset import MyCollator
from utils import Config, set_seed

disable_caching()


lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    inference_mode=False,
)


class LazyScienceQABaseCollator:
    def __init__(
        self,
        tokenizer,
        processor,
        label_pad_token_id=-100,
        max_length=3400,
        image_size=280,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.label_pad_token_id = label_pad_token_id
        self.max_length = max_length
        self.image_size = image_size
        self.base_collator = MyCollator(
            tokenizer=tokenizer,
            latent_id=None,
            label_pad_token_id=label_pad_token_id,
        )

    def _build_one_feature(self, sample):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": sample["image"],
                        "resized_height": self.image_size,
                        "resized_width": self.image_size,
                    },
                    {"type": "text", "text": sample["question"]},
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        question_tokenized = inputs["input_ids"][0].tolist()
        steps_tokenized = [
            self.tokenizer.encode(step + "\n", add_special_tokens=False)
            for step in sample["steps"]
        ]
        answer_tokenized = (
            self.tokenizer.encode(
                "Therefore, the answer is " + str(sample["answer"]),
                add_special_tokens=False,
            )
            + [self.tokenizer.eos_token_id]
        )

        total_length = (
            len(question_tokenized)
            + sum(len(step) for step in steps_tokenized)
            + len(answer_tokenized)
        )
        if total_length > self.max_length:
            excess_length = total_length - self.max_length
            total_steps_len = sum(len(step) for step in steps_tokenized)
            keep_steps_len = max(0, total_steps_len - excess_length)
            new_steps_tokenized = []
            current_length = 0
            for step in steps_tokenized:
                if current_length + len(step) <= keep_steps_len:
                    new_steps_tokenized.append(step)
                    current_length += len(step)
                else:
                    break
            steps_tokenized = new_steps_tokenized

        target_tokens = list(itertools.chain.from_iterable(steps_tokenized)) + answer_tokenized
        tokens = question_tokenized + target_tokens
        labels = [self.label_pad_token_id] * len(question_tokenized) + target_tokens

        return {
            "input_ids": tokens,
            "labels": labels,
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"].squeeze(0),
            "idx": int(sample["idx"]),
        }

    def __call__(self, features):
        processed_features = [self._build_one_feature(sample) for sample in features]
        pixel_values_list = [f.pop("pixel_values") for f in processed_features]
        image_grid_thw_list = [f.pop("image_grid_thw") for f in processed_features]
        idx_list = [f.pop("idx") for f in processed_features]

        batch = self.base_collator(processed_features)
        batch["pixel_values"] = torch.cat(pixel_values_list, dim=0)
        batch["image_grid_thw"] = torch.stack(image_grid_thw_list, dim=0)
        batch["idx"] = torch.tensor(idx_list, dtype=torch.long)
        return batch


def baseline_run_name(name):
    return name if name.endswith("_base") else f"{name}_base"


def main():
    print("Initializing DeepSpeed baseline training!")
    parser = argparse.ArgumentParser(description="Qwen-VL ScienceQA baseline training")
    parser.add_argument("config_file")
    parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed")
    parser.add_argument("--deepspeed_config", default="ds_config.json", help="DeepSpeed config path")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by DeepSpeed")
    args = parser.parse_args()

    deepspeed.init_distributed()
    local_rank = args.local_rank if args.local_rank >= 0 else int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)

    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    configs = Config(config_dict)
    set_seed(configs.seed)
    run_name = baseline_run_name(configs.name)
    save_dir = os.path.join(configs.save_path, run_name)
    model_name = getattr(configs, "model_name", "Qwen/Qwen2-VL-2B-Instruct")

    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

    logging.getLogger().handlers.clear()
    logging.basicConfig(
        filename=os.path.join(save_dir, "qwen_scienceqa_base_train.log"),
        level=logging.DEBUG,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if configs.resume != 0:
        print(f"Skipping the first {configs.resume} epochs; checkpoint loading is unchanged from IVTLR scripts.")

    print(f"Loading baseline model {model_name}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    processor = AutoProcessor.from_pretrained(model_name, tokenizer=tokenizer)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Running DeepSpeed on rank = {rank}, world size = {world_size}")
    model = model.to(local_rank)
    if configs.bf16:
        model.to(torch.bfloat16)

    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=args.deepspeed_config,
        model_parameters=filter(lambda p: p.requires_grad, model.parameters()),
    )
    del model

    dataset = load_dataset("derek-thomas/ScienceQA")

    def has_image(example):
        return "image" in example and example["image"] is not None

    def process_example(example):
        # Baseline training keeps explicit ScienceQA rationales. Lecture and
        # solution are merged here so the model sees one CoT-style target.
        example["answer"] = str(example["answer"])
        lecture = example.get("lecture", "") or ""
        solution = example.get("solution", "") or ""

        if lecture and solution:
            rationale = (lecture.strip() + " " + solution.strip()).strip()
        elif lecture:
            rationale = lecture.strip()
        elif solution:
            rationale = solution.strip()
        else:
            rationale = example["answer"]
            print(f"Warning: Both lecture and solution are empty for question: {example['question']}")

        rationale = rationale.replace("\n", " ").strip()
        example["steps"] = rationale.split(". ")
        if example["steps"] and example["steps"][-1] == "":
            example["steps"].pop()

        if len(example["steps"]) > 3:
            total_steps = len(example["steps"])
            step_size = total_steps // 3
            remainder = total_steps % 3
            new_steps = []
            start = 0
            for i in range(3):
                end = start + step_size + (1 if i < remainder else 0)
                new_steps.append(". ".join(example["steps"][start:end]))
                start = end
            example["steps"] = new_steps

        question_with_braces = f"{{{example['question'].strip()}}}"
        choices = example.get("choices", [])
        if choices:
            choices_str = "[Options]:\n" + "\n".join(
                f"({chr(65 + i)}).{{{choice.strip()}}}" for i, choice in enumerate(choices)
            )
            example["question"] = f"[Question]:{question_with_braces}\n{choices_str}\nAnswer:\n"
        else:
            example["question"] = f"[Question]:{question_with_braces}\nAnswer:\n"

        for key in ("lecture", "solution", "choices"):
            if key in example:
                del example[key]
        return example

    train_dataset = dataset["train"].filter(has_image)
    if configs.debug:
        train_dataset = train_dataset.select(range(min(200, len(train_dataset))))

    train_dataset = train_dataset.map(
        process_example,
        num_proc=1,
        load_from_cache_file=False,
        desc="Formatting ScienceQA text only",
    )
    train_dataset = train_dataset.map(
        lambda example, idx: {"idx": idx},
        with_indices=True,
        num_proc=1,
        load_from_cache_file=False,
        desc="Adding indices",
    )

    best_acc = 0

    for epoch in range(configs.resume, configs.num_epochs):
        np.random.seed(epoch)
        collator = LazyScienceQABaseCollator(
            tokenizer=tokenizer,
            processor=processor,
            label_pad_token_id=-100,
            max_length=3400,
            image_size=280,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            num_workers=0,
            shuffle=False,
            pin_memory=False,
            batch_size=configs.batch_size_training,
            collate_fn=collator,
            sampler=DistributedSampler(train_dataset, shuffle=True),
        )

        model_engine.train()
        total_length = len(train_dataloader) // configs.gradient_accumulation_steps
        pbar = tqdm(
            colour="blue",
            desc=f"Training Epoch: {epoch + 1}",
            total=total_length,
            dynamic_ncols=True,
        )

        for step, batch in enumerate(train_dataloader):
            batch = {key: batch[key].to(local_rank) for key in batch.keys() if key != "idx"}
            outputs = model_engine(**batch)
            loss = outputs.loss
            print(f"loss: {loss}")
            model_engine.backward(loss)
            model_engine.step()
            pbar.set_description(
                f"Training Epoch: {epoch + 1}/{configs.num_epochs}, batch {step}/{len(train_dataloader)} "
                f"completed (loss: {round(float(loss.detach().float()), 4)})"
            )

        pbar.close()
        dist.barrier()

        if (not configs.debug) and (epoch + 1) % 4 == 0:
            epoch_save_dir = os.path.join(save_dir, f"epoch_{epoch + 1}_checkpoint")
            model_engine.save_checkpoint(
                save_dir=epoch_save_dir,
                tag=f"epoch_{epoch + 1}_zero3_bf32",
                client_state={"best_acc": best_acc, "current_epoch": epoch + 1},
            )

            if rank == 0:
                fp32_state_dict = get_fp32_state_dict_from_zero_checkpoint(
                    epoch_save_dir,
                    tag=f"epoch_{epoch + 1}_zero3_bf32",
                )
                fp32_output = os.path.join(save_dir, f"epoch_{epoch + 1}_full_model_fp32.pth")
                torch.save(fp32_state_dict, fp32_output)
                print(f"Epoch {epoch + 1} save to {fp32_output}")
                if os.path.exists(epoch_save_dir):
                    shutil.rmtree(epoch_save_dir)

            dist.barrier()
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

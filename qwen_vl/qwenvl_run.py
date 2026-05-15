import torch
import torch.distributed
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import timedelta
import deepspeed
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from torch.optim import AdamW
import shutil
import numpy as np
from torch.utils.data import Subset
from collections import OrderedDict
import re
import wandb

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_dataset, disable_caching
disable_caching()

import logging

from qwen_ivtlr import IVTLR
from dataset import (
    get_dataset,
    get_cot_latent_dataset,
    MyCollator,
)

from tqdm import tqdm
from copy import copy
import itertools
import os, sys
import yaml
import json
import gc
import argparse
import functools
from utils import Config, set_seed
import pdb
from peft import LoraConfig, get_peft_model

# LoRA
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    inference_mode=False
)
class LazyM3CoTCollator:
    def __init__(
        self,
        tokenizer,
        processor,
        configs,
        scheduled_stage,
        latent_id,
        start_id,
        end_id,
        label_pad_token_id=-100,
        max_length=3400,
        image_size=280,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.configs = configs
        self.scheduled_stage = scheduled_stage
        self.latent_id = latent_id
        self.start_id = start_id
        self.end_id = end_id
        self.label_pad_token_id = label_pad_token_id
        self.max_length = max_length
        self.image_size = image_size

        # Reuse the repo's original padding logic.
        self.base_collator = MyCollator(
            tokenizer=tokenizer,
            latent_id=latent_id,
            label_pad_token_id=label_pad_token_id,
        )

    def _build_one_feature(self, sample):
        # 1. Build Qwen-VL chat prompt with image.
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
                    {
                        "type": "text",
                        "text": sample["question"],
                    },
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        # 2. Process only this example's image right now.
        # This is the key fix: pixel_values are created lazily, not cached in HF Arrow.
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        question_tokenized = inputs["input_ids"][0].tolist()

        # 3. Tokenize CoT steps and answer.
        steps_tokenized = [
            self.tokenizer.encode(step + "\n", add_special_tokens=False)
            for step in sample["steps"]
        ]

        answer = str(sample["answer"])
        answer_tokenized = (
            self.tokenizer.encode(
                "Therefore, the answer is " + answer,
                add_special_tokens=False,
            )
            + [self.tokenizer.eos_token_id]
        )

        # 4. Truncate reasoning steps if sequence is too long.
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

        # 5. Apply IVT-LR scheduled latent training logic.
        scheduled_stage_to_train = self.scheduled_stage

        if scheduled_stage_to_train > self.configs.max_latent_stage:
            n_skip_steps = 10000

            if self.configs.pad_latent_to_max:
                n_latent_tokens = self.configs.max_latent_stage
            else:
                n_latent_tokens = min(
                    len(steps_tokenized),
                    self.configs.max_latent_stage,
                )
        else:
            n_skip_steps = scheduled_stage_to_train
            n_latent_tokens = scheduled_stage_to_train

        tokens = (
            question_tokenized
            + [self.latent_id] * n_latent_tokens
            + list(itertools.chain.from_iterable(steps_tokenized[n_skip_steps:]))
            + answer_tokenized
        )

        labels = (
            [-100] * (len(question_tokenized) + n_latent_tokens)
            + tokens[n_latent_tokens + len(question_tokenized):]
        )

        feature = {
            "input_ids": tokens,
            "labels": labels,
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"].squeeze(0),
            "idx": int(sample["idx"]),
        }

        return feature

    def __call__(self, features):
        processed_features = [self._build_one_feature(sample) for sample in features]

        # 1. Remove image tensors before tokenizer padding.
        pixel_values_list = [f.pop("pixel_values") for f in processed_features]
        image_grid_thw_list = [f.pop("image_grid_thw") for f in processed_features]
        idx_list = [f.pop("idx") for f in processed_features]

        # 2. Use original repo collator only for text fields.
        batch = self.base_collator(processed_features)

        # 3. Manually attach image fields after text padding.
        # Qwen2-VL expects pixel_values usually concatenated across examples.
        batch["pixel_values"] = torch.cat(pixel_values_list, dim=0)

        # image_grid_thw should be [batch_size, 3]
        batch["image_grid_thw"] = torch.stack(image_grid_thw_list, dim=0)

        batch["idx"] = torch.tensor(idx_list, dtype=torch.long)

        return batch
    
def main():
    print("Initializing DeepSpeed Training!")
    parser = argparse.ArgumentParser(description="ivtlr")
    parser.add_argument("config_file")
    parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed")
    parser.add_argument("--deepspeed_config", default="ds_config.json", help="DeepSpeed config path")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by DeepSpeed")
    args = parser.parse_args()

    # Initialize DeepSpeed
    deepspeed.init_distributed()
    local_rank = args.local_rank
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    print("line 57")
    # load the configuration file
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    configs = Config(config_dict)
    set_seed(configs.seed)
    save_dir = os.path.join(configs.save_path, configs.name)
    model_name = getattr(configs, "model_name", "Qwen/Qwen2-VL-2B-Instruct")

    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)

    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

    logging.getLogger().handlers.clear()
    logging.basicConfig(
        filename=os.path.join(save_dir, "qwen_m3cot_train.log"),
        level=logging.DEBUG,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    cur_ckpts = os.listdir(save_dir)


    # check if the job is preempted and resumed.
    # if len(cur_ckpts) > 0 and rank == 0:
    #     raise ValueError(
    #         f"Save directory {save_dir} is not empty! "
    #     )

    if configs.resume != 0:
        # by setting `resume`, we can skip a few epoches at the beginning.
        print(
            f"Loading from {configs.load_model_path} and skip the first {configs.resume} epochs"
        )
        
        
        
    print("start loading model")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="eager"
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=configs.lr)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    processor = AutoProcessor.from_pretrained(model_name, tokenizer=tokenizer)
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    print("latent_id: ", latent_id)
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    image_token_id = tokenizer.convert_tokens_to_ids(processor.image_token)
    visual_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    visual_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    model = get_peft_model(model, lora_config)

    loaded = False

    model.resize_token_embeddings(len(tokenizer))
    embeddings = model.get_input_embeddings()
    target_id = tokenizer.convert_tokens_to_ids("<<")
    # initialize the new token embeddings with a known token
    # it helps stablize the training
    for token_id in [latent_id, start_id, end_id]:
        target_embedding = embeddings.weight.data[token_id]
        embeddings.weight.data[token_id] = target_embedding

        lm_head = model.lm_head
        lm_head.weight.data[token_id] = lm_head.weight.data[target_id]
    
    model.print_trainable_parameters()

    model = IVTLR(model, latent_id, start_id, end_id, tokenizer.eos_token_id, image_token_id, visual_start_id, visual_end_id)

    print(f"Running Deepspeed on rank = {rank}, world size = {world_size}")
    model = model.to(rank)
    
    if configs.bf16:
        model.to(torch.bfloat16)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=args.deepspeed_config,
        # optimizer = optimizer,
        model_parameters=filter(lambda p: p.requires_grad, model.parameters())
    )

    del model

    dataset = load_dataset("LightChen2333/M3CoT")

    # def process_example(example):
    #     rationale = example["rationale"].replace("\n", " ").strip()
    #     example["steps"] = rationale.split(". ")
    #     if example["steps"][-1] == "":
    #         example["steps"].pop()

    #     if len(example["steps"]) > 3:
    #         total_steps = len(example["steps"])
    #         step_size = total_steps // 3
    #         remainder = total_steps % 3

    #         new_steps = []
    #         start = 0

    #         for i in range(3):
    #             end = start + step_size + (1 if i < remainder else 0)
    #             new_steps.append(". ".join(example["steps"][start:end]))
    #             start = end

    #         example["steps"] = new_steps


    #     question = example["question"]
    #     choices = example["choices"]
        

    #     choices_str = "[Options]:\n"+"\n".join([
    #         f"({chr(65 + i)}).{{{choice.strip()}}}"
    #         for i, choice in enumerate(choices)
    #     ])
    #     question = question
    #     question_with_braces = f"{{{question.strip()}}}"
    #     prefix_str = "Answer:"
        
    #     example["question"] = f"[Question]:{question_with_braces}\n{choices_str}\n{prefix_str}\n"
        
    #     del example["rationale"]
    #     del example["choices"]

    #     messages = [{
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "image": example["image"], "resized_height": 280, "resized_width": 280},
    #             {"type": "text", "text": example["question"]}
    #         ]
    #     }]

    #     example["question"] = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #     image_inputs, video_inputs = process_vision_info(messages)
    #     inputs = processor(
    #         text=[example["question"]],
    #         images=image_inputs,
    #         videos=video_inputs,
    #         padding=True,
    #         return_tensors="pt"
    #     )
    #     inputs = {k: v.tolist() for k, v in inputs.items()}
    #     example["input_ids"] = torch.tensor(inputs["input_ids"][0])
    #     example["image_grid_thw"] = torch.tensor(inputs["image_grid_thw"]).squeeze(0)
    #     example["pixel_values"] = torch.tensor(inputs["pixel_values"])

    #     return example

    def process_example(example):
        rationale = example["rationale"].replace("\n", " ").strip()
        example["steps"] = rationale.split(". ")

        if len(example["steps"]) > 0 and example["steps"][-1] == "":
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

        question = example["question"]
        choices = example["choices"]

        choices_str = "[Options]:\n" + "\n".join(
            [
                f"({chr(65 + i)}).{{{choice.strip()}}}"
                for i, choice in enumerate(choices)
            ]
        )

        question_with_braces = f"{{{question.strip()}}}"
        prefix_str = "Answer:"

        example["question"] = (
            f"[Question]:{question_with_braces}\n"
            f"{choices_str}\n"
            f"{prefix_str}\n"
        )

        # Remove large/unneeded text fields.
        del example["rationale"]
        del example["choices"]

        return example
    
    # print("start dataset")

    # def has_image(example):
    #     return (
    #         "image" in example and example["image"] is not None
    #     )

    # train_dataset = dataset["train"].filter(has_image)
    # # train_dataset = train_dataset.map(process_example, num_proc=2)
    # train_dataset = train_dataset.map(
    #     process_example,
    #     num_proc=2,
    #     keep_in_memory=False,
    #     load_from_cache_file=False,
    #     desc="Processing M3CoT"
    # )


    # base_dataset_train = get_dataset(
    #     train_dataset, tokenizer, processor, max_size=5000 if configs.debug else 100000000
    # )
    print("start dataset")

    def has_image(example):
        return "image" in example and example["image"] is not None

    train_dataset = dataset["train"].filter(has_image)

    if configs.debug:
        train_dataset = train_dataset.select(range(min(200, len(train_dataset))))

    # Lightweight map only. No image tensor processing here.
    train_dataset = train_dataset.map(
        process_example,
        num_proc=1,
        load_from_cache_file=False,
        desc="Formatting M3CoT text only",
    )

    # Add stable indices. This is lightweight.
    train_dataset = train_dataset.map(
        lambda example, idx: {"idx": idx},
        with_indices=True,
        num_proc=1,
        load_from_cache_file=False,
        desc="Adding indices",
    )

    base_dataset_train = train_dataset

    total_train_steps = 0

    # if not configs.debug and rank == 0:
    #     wandb_run = wandb.init(project=configs.project, name=configs.name)
    #     wandb_run.config.update(configs, allow_val_change=True)
    #     text_table = wandb.Table(columns=["step", "text"])

    # else:
    wandb_run = None


    best_acc = 0

    # collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)


    for epoch in range(configs.resume, configs.num_epochs):

        scheduled_stage = epoch // configs.epochs_per_stage

        np.random.seed(epoch) 

        # dataset_train = get_cot_latent_dataset(
        #     scheduled_stage,
        #     base_dataset_train,
        #     configs,
        #     start_id,
        #     latent_id,
        #     end_id,
        #     no_special_marker=True,
        #     shuffle=True,
        # )

        # train_dataloader = torch.utils.data.DataLoader(
        #     dataset_train,
        #     num_workers=1,
        #     shuffle=False,
        #     pin_memory=True,
        #     batch_size=configs.batch_size_training,
        #     collate_fn=collator,
        #     sampler=DistributedSampler(dataset_train, shuffle=True),
        # )
        dataset_train = base_dataset_train

        collator = LazyM3CoTCollator(
            tokenizer=tokenizer,
            processor=processor,
            configs=configs,
            scheduled_stage=scheduled_stage,
            latent_id=latent_id,
            start_id=start_id,
            end_id=end_id,
            label_pad_token_id=-100,
            max_length=3400,
            image_size=280,
        )

        train_dataloader = torch.utils.data.DataLoader(
            dataset_train,
            num_workers=0,
            shuffle=False,
            pin_memory=False,
            batch_size=configs.batch_size_training,
            collate_fn=collator,
            sampler=DistributedSampler(dataset_train, shuffle=True),
        )

        model_engine.train()
        total_length = len(train_dataloader) // configs.gradient_accumulation_steps
        pbar = tqdm(
            colour="blue",
            desc=f"Training Epoch: {epoch+1}",
            total=total_length,
            dynamic_ncols=True,
        )
        for step, batch in enumerate(train_dataloader):
            print("start")
            if step == 0 and wandb_run and rank == 0:
                print("logging training data")
                cur_bs = len(batch["input_ids"])
                text_str = ""
                for data_idx in range(cur_bs):
                    for token_idx in range(len(batch["input_ids"][data_idx])):
                        text_str += (
                            str(batch["input_ids"][data_idx][token_idx].item())
                            + " "
                            + str(batch["labels"][data_idx][token_idx].item())
                            + " "
                            + tokenizer.decode(
                                batch["input_ids"][data_idx][token_idx]
                            )
                            + "\n"
                        )
                    text_str += "====" * 10 + "\n"

                text_table.add_data(total_train_steps, text_str)

            total_train_steps += 1
            batch = {
                key: batch[key].to(rank) for key in batch.keys() if key != "idx"
            }

            outputs = model_engine(**batch)
            loss = outputs.loss
            print(f"loss: {loss}")
            model_engine.backward(loss)
            model_engine.step()
            
            if wandb_run and rank == 0:
                log_dict = {
                    "train/epoch": epoch + 1,
                    "train/step": epoch * len(train_dataloader) + step,
                    "train/loss": loss.detach().float()
                    # * configs.gradient_accumulation_steps,
                }
                wandb_run.log(log_dict)
            # print("line432")
            pbar.set_description(
                f"Training Epoch: {epoch+1}/{configs.num_epochs}, batch {step}/{len(train_dataloader)} "
                f"completed (loss: {round(float(loss.detach().float()), 4)}"
            )
            print("finish")
        pbar.close()
        dist.barrier()

        if (not configs.debug) and (epoch + 1) % 4 == 0:
            
            epoch_save_dir = os.path.join(save_dir, f"epoch_{epoch+1}_checkpoint")

            model_engine.save_checkpoint(
                save_dir=epoch_save_dir,
                tag=f"epoch_{epoch+1}_zero3_bf32",
                client_state={"best_acc": best_acc, "current_epoch": epoch+1}
            )

            if rank == 0:
                fp32_state_dict = get_fp32_state_dict_from_zero_checkpoint(epoch_save_dir, tag=f"epoch_{epoch+1}_zero3_bf32")
                fp32_output = os.path.join(save_dir, f"epoch_{epoch+1}_full_model_fp32.pth")

                torch.save(fp32_state_dict, fp32_output)
                
                print(f"Epoch {epoch+1} FP32 save to {fp32_output}")

                if os.path.exists(epoch_save_dir):
                    shutil.rmtree(epoch_save_dir)

            dist.barrier()
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
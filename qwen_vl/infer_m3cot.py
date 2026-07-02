from transformers import AutoTokenizer, AutoProcessor
from qwen_ivtlr import IVTLR  
from transformers import Qwen2VLForConditionalGeneration
import torch
import deepspeed
from peft import LoraConfig,get_peft_model
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from dataset import group_steps_to_max, split_rationale_into_sentences
import re
import logging
import json
import os
import time
from datetime import timedelta
import argparse
import yaml
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"
INFERENCE_LATENT_STEPS = 8
MAX_DYNAMIC_LATENT_STEPS = 8

def load_inference_model(
    checkpoint_path,
    model_name,
    disable_visual_insert=False,
    disable_reasoning=False,
):
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True,
        padding_side="right"
    )
    
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "<|start-latent|>",
            "<|end-latent|>",
            "<|latent|>"
        ]
    })
    
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    base_model.resize_token_embeddings(len(tokenizer))
    processor.tokenizer = tokenizer

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        inference_mode=False
    )
    base_model = get_peft_model(base_model, lora_config)
    
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    image_token_id = tokenizer.convert_tokens_to_ids(processor.image_token)
    visual_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    visual_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    
    model = IVTLR(
        base_model,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=tokenizer.eos_token_id,
        image_token_id=image_token_id,
        visual_start_id=visual_start_id, 
        visual_end_id=visual_end_id,
        disable_visual_insert=disable_visual_insert,
        disable_reasoning=disable_reasoning,
    )
    
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    print(state_dict.keys())
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=True)
    print(model)
    print("Successfully load")
    
    model = model.to(device)
    model.eval()
    return model, processor, tokenizer

parser = argparse.ArgumentParser(description="IVTLR inference on M3CoT")
parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
parser.add_argument("--config", default="args/qwen_m3cot.yaml", help="Path to config YAML")
parser.add_argument("--model_name", default=None, help="Override model name (e.g., Qwen/Qwen2-VL-2B-Instruct)")
parser.add_argument("--run_tag", default=None, help="Optional tag for per-run output subfolder (e.g., epoch_4)")
parser.add_argument(
    "--latent_steps",
    type=int,
    default=INFERENCE_LATENT_STEPS,
    help="Number of latent steps to append during inference",
)
parser.add_argument(
    "--dynamic_latent_steps",
    action="store_true",
    help="Use per-example latent steps derived from the CoT rationale (capped at 8)",
)
parser.add_argument(
    "--use-validation-set",
    "--use_validation_set",
    action="store_true",
    help="Use validation set for inference",
)
parser.add_argument(
    "--disable_visual_insert",
    action="store_true",
    help="Disable top-k visual token insertion during latent reasoning",
)
parser.add_argument(
    "--no-reasoning",
    action="store_true",
    help="Disable latent reasoning (no hidden-state replacement or visual insertion)",
)
parser.add_argument(
    "--disabled_model",
    action="store_true",
    help="If model was trained using disabled visual insert, make dir accordingly"
)
parser.add_argument(
    "--output_root",
    default="outuputs_dynamic_ivtlr",
    help="Root directory for inference outputs",
)
args = parser.parse_args()

os.makedirs(args.output_root, exist_ok=True)
base_output_dir = os.path.join(args.output_root, "inference", "m3cot")
if args.disabled_model:
    base_output_dir += "_no_vis"
output_dir = os.path.join(base_output_dir, args.run_tag) if args.run_tag else base_output_dir
os.makedirs(output_dir, exist_ok=True)
stdout_stderr_path = os.path.join(output_dir, "qwen_m3cot_infer_stdout_stderr.log")
_stdout_file = open(stdout_stderr_path, "a", encoding="utf-8")
_stdout_orig = sys.stdout
_stderr_orig = sys.stderr


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


sys.stdout = _Tee(_stdout_orig, _stdout_file)
sys.stderr = _Tee(_stderr_orig, _stdout_file)

logging.getLogger().handlers.clear()
logging.basicConfig(
    filename=os.path.join(output_dir, "qwen_m3cot_infer.log"),
    level=logging.DEBUG,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

model_name = args.model_name
if args.config and os.path.exists(args.config):
    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f) or {}
    model_name = config_dict.get("model_name", model_name)

if not model_name:
    raise ValueError("model_name must be provided via --model_name or config YAML")

model, processor, tokenizer = load_inference_model(
    args.checkpoint,
    model_name,
    disable_visual_insert=(args.disable_visual_insert or args.no_reasoning),
    disable_reasoning=args.no_reasoning,
)

def format_prompt(example):
    question = example["question"].strip()
    rationale = example["rationale"].replace("\n", " ").strip()
    answer = example["answer"].strip()
    choices = example["choices"]
    image = example["image"]

    choices_str = "\n".join([f"{chr(65+i)}.{{{choice.strip()}}}" for i, choice in enumerate(choices)])
    user_prompt = (
        f"[Question]:{{{question}}}\n"
        f"[Options]:\n{choices_str}\n"
        f"Answer:"
    )
    return user_prompt, rationale, answer, image

def process_func(example):
    prompt, rationale, answer, image = format_prompt(example)
    steps = split_rationale_into_sentences(rationale)
    steps = group_steps_to_max(steps, MAX_DYNAMIC_LATENT_STEPS)
    latent_steps = len(steps)

    return {
        "question_raw": prompt,
        "image_raw": image,
        "gt_answer": answer,
        "latent_steps": latent_steps,
        "id": example["id"],
        "choices": example["choices"],
        "domain": example["domain"],
        "topic": example["topic"]
    }

dataset = load_dataset("LightChen2333/M3CoT")
val_dataset = dataset["validation"] if args.use_validation_set else dataset["test"]
val_dataset = val_dataset.filter(lambda e: e["image"] is not None).map(process_func)

def evaluate_and_save(eval_dataset, model, processor):
    model.eval()
    correct = 0
    total = 0
    total_generated_tokens = 0 
    total_generate_time = 0.0  
    
    output_path = os.path.join(output_dir, "qwen_m3cot_predictions.jsonl")
    with open(output_path, "a", encoding="utf-8") as f_out:
        for ex in eval_dataset:
            input_text = ex["question_raw"]
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": ex["image_raw"], "resized_height": 280, "resized_width": 280},
                    {"type": "text", "text": input_text}
                ]
            }]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            latent_steps = args.latent_steps
            if args.dynamic_latent_steps:
                latent_steps = int(ex.get("latent_steps", 0))
            latent_steps = max(0, min(latent_steps, MAX_DYNAMIC_LATENT_STEPS))
            if not args.no_reasoning and latent_steps > 0:
                text = text + ("<|latent|>" * latent_steps)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(device)
            input_ids = inputs["input_ids"]
            prompt_length = input_ids.shape[1]
            
            generate_start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=torch.tensor(inputs["input_ids"]), 
                    attention_mask=torch.tensor(inputs["attention_mask"]),
                    pixel_values=torch.tensor(inputs["pixel_values"]),
                    image_grid_thw=torch.tensor(inputs["image_grid_thw"]),
                    max_new_tokens=512
                )
            generate_end_time = time.time()
            sample_generate_time = generate_end_time - generate_start_time
            total_generate_time += sample_generate_time
                        
            generated_tokens = outputs[0, prompt_length:]
            new_generated_text = processor.decode(generated_tokens, skip_special_tokens=True)
            output_text = processor.decode(outputs[0], skip_special_tokens=True)
            logging.debug(f"[OUTPUT] {output_text}")
            
            num_generated_tokens = len(generated_tokens)
            total_generated_tokens += num_generated_tokens

            cleaned_text = re.sub(
                r'(?<=answer:)\s*(\n+\s*)?assistant\b',
                '',
                output_text,
                flags=re.IGNORECASE
            )
            letter_matches = re.finditer(
                r'(?:the\s+answer\s+is|Answer:)\s*[\n\s]*([A-Z])',
                cleaned_text,
                flags=re.IGNORECASE | re.DOTALL
            )
            candidates = {match.group(1).upper() for match in letter_matches}

            digit_matches = re.finditer(
                r'(?:the\s+answer\s+is|Answer:)\s*[\n\s]*(\d)',
                cleaned_text,
                flags=re.IGNORECASE | re.DOTALL
            )
            for match in digit_matches:
                digit = int(match.group(1))
                if 0 <= digit <= 3:
                    candidates.add(chr(ord('A') + digit))

            gt_answer = ex["gt_answer"].strip().upper()
            if gt_answer.isdigit():
                digit = int(gt_answer)
                if 0 <= digit <= 3:
                    gt_answer = chr(ord('A') + digit)

            if gt_answer in candidates:
                correct += 1
                logging.debug(f"correct: True")
            total += 1
            logging.debug(f"[TOTAL] {total}")

            message_question = ex["question_raw"]
            message_question = message_question.replace("<image>", "", 1).replace("Answer:", "", 1).strip()
            message_question = message_question.split("Answer:")[0].strip()

            result = {
                "id": ex["id"],
                "choices": ex["choices"],
                "answer": ex["gt_answer"],
                "domain": ex["domain"],
                "topic": ex["topic"],
                "messages": [
                    message_question,
                    new_generated_text
                ]
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

        accuracy = correct / total if total > 0 else 0
        logging.info(f"[FINAL] Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2%}")
        print(f"[FINAL] Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2%}")
        print(f"Results saved to: {output_path}")
    
        f_out.flush()
            
        avg_generated_tokens = total_generated_tokens / total if total > 0 else 0
        avg_time_per_sample = total_generate_time / total if total > 0 else 0
    
        logging.info(f"[FINAL] Avg generated tokens per sample: {avg_generated_tokens:.1f}")
        logging.info(f"[FINAL] Total generate time: {total_generate_time:.2f}s ({timedelta(seconds=int(total_generate_time))})")
        logging.info(f"[FINAL] Avg generate time per sample: {avg_time_per_sample:.3f}s")
    
evaluate_and_save(val_dataset, model, processor)

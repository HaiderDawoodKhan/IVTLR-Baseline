from transformers import AutoTokenizer, AutoProcessor
from qwen_ivtlr import IVTLR  
from transformers import Qwen2VLForConditionalGeneration
import torch
import deepspeed
from peft import LoraConfig,get_peft_model
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
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
INFERENCE_LATENT_STEPS = 10

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

parser = argparse.ArgumentParser(description="IVTLR inference on ScienceQA")
parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
parser.add_argument("--config", default="args/qwen_sqa.yaml", help="Path to config YAML")
parser.add_argument("--model_name", default=None, help="Override model name (e.g., Qwen/Qwen2-VL-2B-Instruct)")
parser.add_argument("--run_tag", default=None, help="Optional tag for per-run output subfolder (e.g., epoch_4)")
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
base_output_dir = os.path.join(args.output_root, "inference", "sqa")
if args.disabled_model:
    base_output_dir += "_no_vis"
output_dir = os.path.join(base_output_dir, args.run_tag) if args.run_tag else base_output_dir
os.makedirs(output_dir, exist_ok=True)
stdout_stderr_path = os.path.join(output_dir, "qwen_scienceqa_infer_stdout_stderr.log")
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
    filename=os.path.join(output_dir, "qwen_scienceqa_infer.log"),
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
    answer = example["answer"] 
    choices = example.get("choices", [])
    image = example["image"]

    if choices:
        choices_str = "\n".join([f"({chr(65+i)}).{{{choice.strip()}}}" for i, choice in enumerate(choices)])
        user_prompt = (
            f"[Question]:{{{question}}}\n"
            f"[Options]:\n{choices_str}\n"
            f"Answer:"
        )
    else:
        user_prompt = f"[Question]:{{{question}}}\nAnswer:"
    
    return user_prompt, answer, image

def process_func(example, idx):
    prompt, answer, image = format_prompt(example)
    
    return {
        "idx": idx,  
        "question_raw": prompt,
        "image_raw": image,
        "gt_answer": answer,  
    }

dataset = load_dataset("derek-thomas/ScienceQA")
test_dataset = dataset["test"]

def has_image(example):
    return "image" in example and example["image"] is not None


test_dataset = test_dataset.map(lambda example, idx: {"original_idx": idx, **example}, with_indices=True)
test_dataset = test_dataset.filter(has_image)
test_dataset = test_dataset.map(lambda example: process_func(example, example["original_idx"]))

def evaluate_and_save(eval_dataset, model, processor):
    model.eval()
    correct = 0
    total = 0
    results = {} 
    total_generated_tokens = 0  
    total_generate_time = 0.0  
    
    output_json_path = os.path.join(output_dir, "qwen_scienceqa_predictions.json")
    
    for ex in eval_dataset:
        idx = str(ex["idx"]) 
        input_text = ex["question_raw"]
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": ex["image_raw"], "resized_height": 280, "resized_width": 280},
                {"type": "text", "text": input_text}
            ]
        }]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if not args.no_reasoning:
            text = text + ("<|latent|>" * INFERENCE_LATENT_STEPS)
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        prompt_length = inputs["input_ids"].shape[1]
        
        generate_start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"], 
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
                max_new_tokens=512
            )
        generate_end_time = time.time()
        sample_generate_time = generate_end_time - generate_start_time
        total_generate_time += sample_generate_time
        generated_tokens = outputs[0, prompt_length:]
        generated_text = processor.decode(generated_tokens, skip_special_tokens=True)
        num_generated_tokens = len(generated_tokens)
        total_generated_tokens += num_generated_tokens
        
        pred_answer = extract_answer(generated_text)
        

        results[idx] = pred_answer
        

        gt_answer = ex["gt_answer"]  
        if pred_answer == gt_answer:
            correct += 1
        
        total += 1
        
    
    output_data = {"results": results}
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    accuracy = correct / total if total > 0 else 0
    avg_generated_tokens = total_generated_tokens / total if total > 0 else 0
    avg_time_per_sample = total_generate_time / total if total > 0 else 0
    
    
    logging.info(f"[FINAL] Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2%}")
    logging.info(f"[FINAL] Avg generated tokens per sample: {avg_generated_tokens:.1f}")
    logging.info(f"[FINAL] Total generate time: {total_generate_time:.2f}s ({timedelta(seconds=int(total_generate_time))})")
    logging.info(f"[FINAL] Avg generate time per sample: {avg_time_per_sample:.3f}s")
    
    
    print(f"[FINAL] Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2%}")
    print(f"Results saved to: {output_json_path}")
    
    return accuracy

def extract_answer(text):
    digit_patterns = [
        r'Therefore,?\s*the\s+answer\s+is\s+(\d)',  
        r'the\s+answer\s+is\s+(\d)', 
        r'answer\s+is:?\s*(\d)',  
    ]
    
    for pattern in digit_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer_idx = int(match.group(1))
            logging.debug(f"Extracted answer (digit): {answer_idx}")
            return answer_idx
    

    letter_patterns = [
        r'Therefore,?\s*the\s+answer\s+is\s+([A-Z])', 
        r'the\s+answer\s+is\s+([A-Z])',
        r'answer\s+is:?\s*([A-Z])',  
    ]
    
    for pattern in letter_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()

            answer_idx = ord(letter) - ord('A')
            logging.debug(f"Extracted answer (letter): {letter} -> index {answer_idx}")
            return answer_idx
    

    logging.warning(f"No answer pattern found in text: {text[:200]}")
    return -1  

evaluate_and_save(test_dataset, model, processor)

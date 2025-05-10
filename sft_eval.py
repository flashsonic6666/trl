import os
import json
from PIL import Image
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch
from datasets import load_dataset
import random

# --------------------- Configuration ---------------------
CHECKPOINT = 25000
QWEN_MODEL = "2"
PARAM = "2B"
MODEL_NAME = f"Qwen/Qwen{QWEN_MODEL}-VL-{PARAM}-Instruct"
CHECKPOINT_PATH = f"Qwen{QWEN_MODEL}-VL-{PARAM}-SFT-Instruct-Full/checkpoint-{CHECKPOINT}"
OUTPUT_JSON = f"Qwen{QWEN_MODEL}-{PARAM}_sft_eval_results_{CHECKPOINT}_Full.json"
DATASET_FILE = "pubchem/eval_2k.csv" # Evaluation dataset
SAMPLE_SIZE = 300

# --------------------- Loading Model + Processor ---------------------
if "Qwen2-VL" in MODEL_NAME:
    processor = Qwen2VLProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(CHECKPOINT_PATH, torch_dtype=torch.bfloat16).to("cuda")
elif "Qwen2.5-VL" in MODEL_NAME:
    processor = Qwen2_5_VLProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(CHECKPOINT_PATH, torch_dtype=torch.bfloat16).to("cuda")
else:
    raise ValueError(f"Unknown model name: {MODEL_NAME}")
model.eval()

# --------------------- Load Dataset ---------------------
dataset = load_dataset("csv", data_files={"eval": DATASET_FILE})["eval"]

# --------------------- System Prompt ---------------------
system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

# --------------------- Format Sample ---------------------
def format_data(sample):
    return [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [
            {"type": "image", "image": sample["file_path"]},
            {"type": "text", "text": "Identify the chemical structure and return the SMILES inside <answer> </answer>."}
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": f"<answer>{sample['SMILES']}</answer>"}]}
    ]

# --------------------- Evaluate Some Samples ---------------------
random.seed(42)
indices = random.sample(range(len(dataset)), SAMPLE_SIZE)

subset = dataset.select(indices)

results = []
for i, sample in enumerate(subset):
    messages = format_data(sample)
    gt = sample["SMILES"]
    image_path = sample["file_path"]

    # Get model input
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    if image_inputs is None or len(image_inputs) == 0:
        print(f"[SKIP] No image for sample {i}")
        continue

    # Tokenize
    inputs = processor(
        text=[prompt],
        images=[image_inputs[0]],
        return_tensors="pt"
    ).to("cuda")

    # Generate
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        prediction = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

    results.append({
        "index": i,
        "image_path": image_path,
        "prompt": prompt,
        "ground_truth": gt,
        "prediction": prediction
    })

# --------------------- Save to file ---------------------
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

OUTPUT_JSON

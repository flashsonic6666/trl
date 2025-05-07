# experiments_sft.py — Supervised Fine‑Tuning Qwen2‑VL the “GRPO way”
# ----------------------------------------------------------------------------------
# Quick start (after `pip install -U trl transformers accelerate datasets peft pillow`):
#   accelerate launch experiments_sft.py
import wandb
import os
from datasets import load_dataset, disable_caching
from trl import SFTConfig, SFTTrainer
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
import torch
from qwen_vl_utils import process_vision_info
import dataclasses

# ----------------------------------------------------------------------------------
# 1) Configuration
# ----------------------------------------------------------------------------------
DATASET_FILE                = "synthetic/indigo_resize_train.csv"
EVAL_FILE                   = "synthetic/indigo_resize_eval.csv"
MODEL_NAME                  = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR                  = "Qwen2.5-VL-3B-SFT-Instruct [Organometallics]"
GROUND_TRUTH_COLUMN         = "SMILES"
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE  = 1
MAX_PROMPT_LENGTH           = 1024
MAX_COMPLETION_LENGTH       = 512

disable_caching()  # turn off datasets caching to avoid stale state

# ----------------------------------------------------------------------------------
# 2) Load original CSV + build a uniform `messages` column
# ----------------------------------------------------------------------------------
print("Loading CSV as a HuggingFace Dataset…")
data = load_dataset("csv", data_files={
    "train": DATASET_FILE,
    "eval": EVAL_FILE
})

def format_data(sample):
    image_path = sample["file_path"]
    smiles = sample["SMILES"]

    
    #try:
    #    image = Image.open(image_path).convert("RGB")
    #except Exception as e:
    #    print(f"[SKIP] Could not load image {image_path}: {e}")
    #    return []
    

    system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
        Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
        The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
        Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""
    
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Identify the chemical structure and return the SMILES inside <answer> </answer>."}
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": f"<answer>{smiles}</answer>"}]
        }
    ]

train_dataset = [format_data(sample) for sample in data["train"]]
eval_dataset = [format_data(sample) for sample in data["eval"]]

# ----------------------------------------------------------------------------------
# 3) Prepare the processor & collator
# ----------------------------------------------------------------------------------
if "Qwen2-VL" in MODEL_NAME:
    processor = Qwen2VLProcessor.from_pretrained(MODEL_NAME, use_fast=True)
elif "Qwen2.5-VL" in MODEL_NAME:
    processor = Qwen2_5_VLProcessor.from_pretrained(MODEL_NAME, use_fast=True)
else:
    raise ValueError(f"Unsupported model name: {MODEL_NAME}")

def collate_fn(examples):
    texts = []
    images = []

    for ex in examples:
        text = processor.apply_chat_template(ex, tokenize=False)
        vision_inputs, _ = process_vision_info(ex)
        if not vision_inputs or vision_inputs[0] is None:
            raise ValueError("Missing or invalid image.")
        texts.append(text)
        images.append(vision_inputs[0])  # use only first image

    batch = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

    # Ignore the image token index in the loss computation (model specific)
    if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
        image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    elif isinstance(processor, Qwen2_5_VLProcessor):  # Check if the processor is Qwen2_5_VLProcessor
        image_tokens = [151655]  # Specific image token IDs for Qwen2_5_VLProcessor
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels
    return batch

# ----------------------------------------------------------------------------------
# 4) SFT training arguments
# ----------------------------------------------------------------------------------
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    logging_steps=10,
    eval_steps=250,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_checkpointing=True,
    deepspeed="./deepspeed_config_zero2.json",
    bf16=True,
    save_total_limit=1,
    remove_unused_columns=False,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
    report_to="wandb",
)

wandb.init(
    project="MetalloScribe",
    name=OUTPUT_DIR,
    config=dataclasses.asdict(training_args),
)

# ----------------------------------------------------------------------------------
# 5) Model & Trainer
# ----------------------------------------------------------------------------------
if "Qwen2-VL" in MODEL_NAME:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
    )
elif "Qwen2.5-VL" in MODEL_NAME:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
    )
else:
    raise ValueError(f"Unsupported model name: {MODEL_NAME}")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    tokenizer = processor.tokenizer,
)

# ----------------------------------------------------------------------------------
# 6) Launch training
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

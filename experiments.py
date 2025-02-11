from datasets import load_dataset
from trl import GRPOConfig
from trl.trainer.vlmgpro_trainer import VLMGRPOTrainer
from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from torchvision import transforms

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set GPU to use

print("Loading dataset...")

# Load dataset
dataset = load_dataset("csv", data_files="synthetic/indigo_resize.csv")

# Load Qwen2-VL processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Function to load & preprocess images
def load_and_preprocess_image(example):
    """Loads an image from file_path and resizes it to 384x384 for Qwen2-VL."""
    image_path = example["file_path"]
    try:
        image = Image.open(image_path).convert("RGB")
        example["image"] = image  # Store as tensor
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        example["image"] = None  # Ensure this is not missing
    return example

# Ensure grid_thw is present in dataset
dataset = dataset.map(load_and_preprocess_image)['train']

# Filter out invalid samples
dataset = dataset.filter(lambda x: x["image"] is not None)

# Print first example to verify
print(dataset[0])

# Reward function (dummy example) (currently hardcoded in vlmgpro_trainer.py so this does nothing)
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

# GRPO config for training
training_args = GRPOConfig(
    output_dir="Qwen2-VL-2B-GRPO",
    logging_steps=10,
    per_device_train_batch_size=4,  # Adjust batch size based on GPU memory
    per_device_eval_batch_size=4,
    max_prompt_length=100,
    max_completion_length=100,
    temperature=0.9,
    use_vllm=False,  # No vLLM for now
    sync_ref_model=False,
    num_generations=2,  # Number of generations per prompt
)

# Load model
model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Instantiate Trainer with corrected inputs
trainer = VLMGRPOTrainer(
    model=model,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)

# Start Training
trainer.train()
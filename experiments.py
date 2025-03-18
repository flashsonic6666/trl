from datasets import load_dataset
from trl import GRPOConfig
from trl.trainer.grpo_vlm_trainer import VLMGRPOTrainer
from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from torchvision import transforms

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Set GPU to use

# Import utility functions
from reward_and_prompt_utils import get_reward_fn, get_prompt_template

# experiments.py

# Configuration variables
DATASET_FILE = "indigo_simple_render/simple_molecules.csv"  # Change dataset easily
REWARD_FN_NAME = "smiles_match"  # Options: "smiles_match", "has_aromatic_ring", "hydrogen_count"
PROMPT_TEMPLATE = "smiles"  # Options: "smiles", "aromatic", "hydrogen"
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"  # Model to use for training
OUTPUT_DIR = "Qwen2-VL-7B-GRPO-Good-Prompt-Instruct"  # Output directory for training
GROUND_TRUTH_COLUMN = "SMILES"  # Column name for ground truth
PER_DEVICE_TRAIN_BATCH_SIZE = 1  # Batch size for training
PER_DEVICE_EVAL_BATCH_SIZE = 1  # Batch size for evaluation
MAX_PROMPT_LENGTH = 1024  # Maximum prompt length
MAX_COMPLETION_LENGTH = 512  # Maximum completion length
TEMPERATURE = 0.9  # Temperature for sampling
USE_VLLM = False  # Use vLLM for training
SYNC_REF_MODEL = False  # Sync reference model
NUM_GENERATIONS = 2  # Number of generations per prompt

print("Loading dataset...")

# Load dataset
dataset = load_dataset("csv", data_files=DATASET_FILE)

# Load Qwen2-VL processor
processor = AutoProcessor.from_pretrained(MODEL_NAME)

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

# Shuffle dataset before training
dataset = dataset.shuffle(seed=42)  # Shuffle the dataset

# Print first example to verify
print(dataset[0])

# Choose the reward and prompt functions
reward_fn = get_reward_fn(REWARD_FN_NAME)
prompt_template_fn = get_prompt_template(PROMPT_TEMPLATE)

# GRPO config for training
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    logging_steps=10,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,  # Adjust batch size based on GPU memory
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    max_prompt_length=MAX_PROMPT_LENGTH,
    max_completion_length=MAX_COMPLETION_LENGTH,
    temperature=TEMPERATURE,
    use_vllm=USE_VLLM,  # No vLLM for now
    sync_ref_model=SYNC_REF_MODEL,
    num_generations=NUM_GENERATIONS,  # Number of generations per prompt
    deepspeed="./deepspeed_config_zero3.json",  # Path to your DeepSpeed config file (ending in zero2 or zero3)
    bf16=True,  # Matches --bf16
    gradient_checkpointing=True,  # Matches --gradient_checkpointing
    save_total_limit=1,  # Keep only 1 checkpoint to save space
)

# Load model
model = AutoModelForVision2Seq.from_pretrained(MODEL_NAME)

# Instantiate the trainer, now with custom reward and prompt functions.
trainer = VLMGRPOTrainer(
    model=model,
    #model_type="vision",  # Use 'causal' for a language model
    reward_funcs=reward_fn,
    args=training_args,
    train_dataset=dataset,
    prompt_template_fn=prompt_template_fn,
    ground_truth_column=GROUND_TRUTH_COLUMN,
)

# Start Training
trainer.train()
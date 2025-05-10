from datasets import load_dataset, Dataset
from PIL import Image
import pandas as pd
import os

DATASET_FILE = "indigo_simple_render/simple_molecules.csv"
OUTPUT_FILE = "indigo_simple_render/simple_molecules_sft_dataset.csv"
GROUND_TRUTH_COLUMN = "SMILES"

# Load dataset
raw_dataset = load_dataset("csv", data_files=DATASET_FILE)["train"]

# Function to load and attach images
def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[SKIP] Error loading {path}: {e}")
        return None

# Build new examples
new_examples = []
for example in raw_dataset:
    image_path = example.get("file_path")
    ground_truth = example.get(GROUND_TRUTH_COLUMN)
    image = load_image(image_path)

    if image is None or not isinstance(ground_truth, str):
        continue

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},  # Save path instead of image object
                {"type": "text", "text": "Identify the chemical structure and return the SMILES inside <answer> </answer>."}
            ]
        },
        {
            "role": "assistant",
            "content": f"<answer>{ground_truth}</answer>"
        }
    ]

    example["messages"] = messages
    new_examples.append(example)

# Convert to dataset & save as CSV
df = pd.DataFrame(new_examples)
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved SFT-ready dataset with {len(df)} examples to: {OUTPUT_FILE}")

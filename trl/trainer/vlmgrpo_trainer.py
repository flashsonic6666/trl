import os
from PIL import Image
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    GenerationConfig,
)
from transformers import AutoModelForVision2Seq  # assumed class for SmolVLM; adjust if needed
from trl import GRPOConfig, GRPOTrainer
import re

# ============================================================================
# 1. Define a custom reward function for hydrogen counting.
#    This function receives lists of prompts, completions, and ground-truth counts.
#    It converts the generated completion to an integer (if possible) and returns
#    a reward based on the negative absolute error.
# ============================================================================
def hydrogen_count_reward(prompts, completions, hydrogen_atom_count):
    rewards = []
    format_bonus = 0.2  # Small reward for correct <think> format
    format_penalty = -0.2  # Small penalty for incorrect format

    for prompt, completion, true_count in zip(prompts, completions, hydrogen_atom_count):
        try:
            # Attempt to extract an integer from the completion text.
            match = re.search(r"<think>(\d+)</think>", completion)
            if match:
                pred = int(match.group(1))
            else:
                pred = int(completion.strip())  # Fallback if no <think> tag is present
        except Exception:
            pred = 0  # If conversion fails, use zero (or apply other fallback logic)

        # Hydrogen count error-based reward
        reward = -1.0 + 2.0 * (1.0 - abs(pred - true_count) / max(true_count, 1))

        # Format correctness reward
        if re.search(r"<think>\d+</think>", completion):  # Checks for correctly formatted tags
            reward += format_bonus
        else:
            reward += format_penalty

        rewards.append(reward)

    return rewards


# ============================================================================
# 2. Create a subclass of GRPOTrainer that supports vision-language inputs.
#    We override the _prepare_inputs method so that it:
#      - Loads the image from the "file_path"
#      - Uses a constant text prompt (e.g. "Count hydrogen atoms in this molecule:")
#      - Uses the model’s multimodal processor to encode both image and text.
#      - Calls model.generate with the extra image tensor.
# ============================================================================
class VLMGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        """
        In addition to the usual GRPOTrainer arguments, the dataset is expected to have
        "file_path" and "hydrogen_atom_count" columns.
        """
        super().__init__(*args, **kwargs)
        # Instead of a text-only tokenizer, load the multimodal processor.
        # We assume the model’s identifier works for AutoProcessor.
        self.image_processor = AutoProcessor.from_pretrained(self.model.config._name_or_path)
        print(f"Model vocab size: {self.model.config.vocab_size}")
        print("All special tokens:", self.image_processor.tokenizer.special_tokens_map)
        print("Pad token ID:", self.image_processor.tokenizer.pad_token_id)
        print("Eos token ID:", self.image_processor.tokenizer.eos_token_id)
        print("Bos token ID:", self.image_processor.tokenizer.bos_token_id)

        # Adjust the generation configuration.
        self.generation_config = GenerationConfig(
            max_new_tokens=self.args.max_completion_length,
            do_sample=True,
            temperature=self.args.temperature,
            pad_token_id=self.image_processor.tokenizer.pad_token_id
            if hasattr(self.image_processor, "tokenizer")
            else self.image_processor.pad_token_id,
        )

        # In this example, we assume only one generation per prompt.
        self.num_generations = 1
        

    def _prepare_inputs(self, inputs):
        device = self.accelerator.device

        # ------------------------------
        # Process the images.
        # ------------------------------
        # Each input sample is expected to have an "file_path" key.
        images = [Image.open(x["file_path"]).convert("RGB") for x in inputs]

        # ------------------------------
        # Define a constant text prompt.
        # ------------------------------
        # Get the special image token (if defined in your tokenizer)
        image_token_id = self.model.config.image_token_id
        image_token = self.image_processor.tokenizer.decode([image_token_id])

        prompt_text = f"{image_token} Count the total number of hydrogen atoms in this molecule:"
        prompts = [prompt_text for _ in inputs]

        # ------------------------------
        # Use the multimodal processor to encode both text and images.
        # ------------------------------
        # The processor is expected to return a dictionary with:
        #   - "input_ids" and "attention_mask" for the text prompt
        #   - "pixel_values" for the image

        #Print the shape of the image
        #print("Number of images:", len(images))  # Debugging
        #print("Image shape before processor:", images[0].size)  # Debugging
        #print("Number of prompts:", len(prompts))  # Debugging
        
        processor_output = self.image_processor(
            text=prompts, images=images, return_tensors="pt", padding=True
        )
        input_ids = processor_output["input_ids"].to(device)
        attention_mask = processor_output["attention_mask"].to(device)
        pixel_values = processor_output["pixel_values"].to(device)
        image_grid_thw = processor_output["image_grid_thw"].to(device)

        #print("input ids shape after processor:", input_ids.shape)  # Debugging statement
        #print("pixel_values shape after processor:", pixel_values.shape) # Debugging statement
        #print("image_grid_thw shape:", image_grid_thw.shape)  # Should be (B, 3)

        # Ensure correct shape for pixel_values
        B = len(inputs)  # Batch size
        grid_h, grid_w = image_grid_thw[:, 1], image_grid_thw[:, 2]
        H, W = grid_h[0].item() * 14, grid_w[0].item() * 14  # Compute true image height and width, assuming 14x14 patches

        # The expected format for generation should be (B, C, H, W), so reshape it
        pixel_values = pixel_values.view(B, -1, H, W)  # Ensure correct shape
        #print("Reshaped pixel_values for generation:", pixel_values.shape)
        
        # print the image token
        #print("Image token:", image_token)  # Debugging
        #print("Image token id and image token:", image_token_id, image_token)  # Debugging

        # ------------------------------
        # Generate completions.
        # ------------------------------
        # We pass both text and image inputs to generate the output.
        with torch.no_grad():
            prompt_completion_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,  # use the stored grid
                generation_config=self.generation_config,
            )

        # ------------------------------
        # Separate the prompt and the generated completion.
        # ------------------------------
        prompt_length = input_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Create a mask that stops after the first EOS token.
        if hasattr(self.image_processor, "tokenizer"):
            eos_token_id = self.image_processor.tokenizer.eos_token_id
        else:
            eos_token_id = self.image_processor.eos_token_id
        is_eos = completion_ids == eos_token_id
        # For each sample, find the first EOS (or use full length)
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        has_eos = is_eos.any(dim=1)
        if has_eos.any():
            eos_idx[has_eos] = is_eos[has_eos].int().argmax(dim=1)
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # ------------------------------
        # Compute per-token log-probabilities using the reference model.
        # (The code here is very similar to the original GRPOTrainer.)
        # ------------------------------
        attention_mask_full = torch.cat([attention_mask, completion_mask], dim=1)
        full_input_ids = torch.cat([input_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, full_input_ids, attention_mask_full, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, full_input_ids, attention_mask_full, logits_to_keep
                    )

        # ------------------------------
        # Decode the generated completions.
        # ------------------------------
        completions_text = self.image_processor.tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        self._last_logged_completions = completions_text
        # ------------------------------
        # Compute rewards.
        # ------------------------------
        # For the reward function, we pass:
        #   - prompts (the same constant prompt for each sample)
        #   - completions (the generated output)
        #   - hydrogen_atom_count (extracted from the inputs)
        ground_truth = [x["hydrogen_atom_count"] for x in inputs]
        rewards = hydrogen_count_reward(prompts, completions_text, ground_truth)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)

        # (Since we assume one generation per prompt, advantages equal the rewards.
        # In a multi-generation setting, you would group and normalize these.)
        advantages = rewards_tensor

        return {
            "prompt_ids": input_ids,
            "prompt_mask": attention_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }


# ============================================================================
# 3. Example usage: loading a dataset and instantiating the trainer.
#
# Assume your dataset is in CSV format with two columns:
#   - "file_path": path to the molecule image.
#   - "hydrogen_atom_count": the true hydrogen atom count.
# ============================================================================
if __name__ == "__main__":
    # Load your dataset (adjust the paths as needed)
    dataset = load_dataset(
        "csv",
        data_files={
            "train": "path/to/train.csv",
            "validation": "path/to/validation.csv",
        },
    )

    # Initialize a GRPO configuration.
    config = GRPOConfig(
        output_dir="./outputs",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        max_prompt_length=50,
        max_completion_length=20,
        num_generations=2,  # one generation per prompt for simplicity
        temperature=1.0,
        use_vllm=False,  # not using vLLM in this example
        logging_steps=10,
        sync_ref_model=False,
    )

    # Load the SmolVLM model.
    # Replace "smol-vlm" with the actual model ID if different.
    model_name_or_path = "Qwen/Qwen2-VL-2B-Instruct"
    model = AutoModelForVision2Seq.from_pretrained(model_name_or_path)

    # Instantiate our custom VLM GRPO trainer.
    trainer = VLMGRPOTrainer(
        model=model,
        reward_funcs=[hydrogen_count_reward],  # our custom reward function
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    # Begin training.
    trainer.train()

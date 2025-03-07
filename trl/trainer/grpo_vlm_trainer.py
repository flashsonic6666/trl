import os
from PIL import Image
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    GenerationConfig,
    AutoModelForVision2Seq,
    AutoModelForCausalLM,
)
from trl import GRPOConfig, GRPOTrainer
import re
from rdkit import Chem
import torch.distributed as dist

# ============================================================================
# 1. Create a subclass of GRPOTrainer that supports vision-language inputs.
#    We override the _prepare_inputs method so that it:
#      - Loads the image from the "file_path"
#      - Uses a constant text prompt (e.g. "Count hydrogen atoms in this molecule:")
#      - Uses the model’s multimodal processor to encode both image and text.
#      - Calls model.generate with the extra image tensor.
# ============================================================================
class VLMGRPOTrainer(GRPOTrainer):
    def __init__(self, model_type: str = "vision", prompt_template_fn=None, ground_truth_column=None, *args, **kwargs):
        """
        In addition to the usual GRPOTrainer arguments, the dataset is expected to have
        "file_path" and "hydrogen_atom_count" columns.
        """
        assert model_type in ["vision"] # "model_type must be 'vision'
        self.model_type = model_type  # Store model type
        
        # Ensure model_type is passed down to GRPOTrainer
        kwargs["model_type"] = model_type

        # Determine the correct model class
        ModelClass = AutoModelForVision2Seq if model_type == "vision" else AutoModelForCausalLM

        # If model is already loaded, don't reload it.
        if isinstance(kwargs["model"], torch.nn.Module):
            self.model = kwargs["model"]
        else:
            self.model = ModelClass.from_pretrained(kwargs["model"])  


        super().__init__(*args, **kwargs)
        # Instead of a text-only tokenizer, load the multimodal processor.
        # We assume the model’s identifier works for AutoProcessor.
        # Use multimodal processor for vision models, otherwise tokenizer
        if model_type == "vision":
            self.image_processor = AutoProcessor.from_pretrained(self.model.config._name_or_path)
        else:
            self.processor = self.processing_class  # Use the tokenizer for causal models

        self.reward_funcs = self.reward_funcs[0] # We only have one reward function for now

        # Adjust the generation configuration.
        self.generation_config = GenerationConfig(
            max_new_tokens=self.args.max_completion_length,
            do_sample=True,
            temperature=self.args.temperature,
            pad_token_id=self.image_processor.tokenizer.pad_token_id
            if hasattr(self.image_processor, "tokenizer")
            else self.image_processor.pad_token_id,
            num_return_sequences=self.args.num_generations,
        )


        # Store the prompt template function (or use a default)
        self.prompt_template_fn = prompt_template_fn or (lambda image_token, completion_start: 
            f"<|im_start|>assistant\n{completion_start}")
        self.ground_truth_column = ground_truth_column

    def _prepare_inputs(self, inputs):
        device = self.accelerator.device

        # ------------------------------
        # Process the images.
        # ------------------------------
        # Each input sample is expected to have an "file_path" key.
        images = [Image.open(x["file_path"]).convert("RGB") for x in inputs]
        # Extract image filenames from the dataset inputs
        image_names = [os.path.basename(x["file_path"]) for x in inputs]  # Extract just the file names

        # Store the latest image names for logging
        self._last_logged_images = image_names

        # ------------------------------
        # Define a constant text prompt.
        # ------------------------------
        # Get the special image token (if defined in your tokenizer)
        image_token_id = self.model.config.image_token_id
        image_token = self.image_processor.tokenizer.decode([image_token_id])

        completion_start = "Let me solve this step by step. <think> "
        
        # Use the prompt template function
        prompt_text = self.prompt_template_fn(image_token, completion_start)
        prompts = [prompt_text for _ in inputs]

        # ------------------------------
        # Use the multimodal processor to encode both text and images.
        # ------------------------------
        # The processor is expected to return a dictionary with:
        #   - "input_ids" and "attention_mask" for the text prompt
        #   - "pixel_values" for the image
        processor_output = self.image_processor(
            text=prompts, images=images, return_tensors="pt", padding=True
        )
        input_ids = processor_output["input_ids"].to(device)
        attention_mask = processor_output["attention_mask"].to(device)
        pixel_values = processor_output["pixel_values"].to(device)
        image_grid_thw = processor_output["image_grid_thw"].to(device)

        # Ensure correct shape for pixel_values
        B = len(inputs)
        grid_h, grid_w = image_grid_thw[:, 1], image_grid_thw[:, 2]
        H, W = grid_h[0].item() * 14, grid_w[0].item() * 14
        pixel_values = pixel_values.view(B, -1, H, W)

        # ------------------------------
        # Generate completions.
        # ------------------------------
        # We pass both text and image inputs to generate the output.

        with torch.no_grad():
            prompt_completion_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                generation_config=self.generation_config,
            )

        # ------------------------------
        # Separate the prompt and the generated completion.
        # ------------------------------
        prompt_length    = input_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Create a mask that stops after the first EOS token.
        eos_token_id = (self.image_processor.tokenizer.eos_token_id 
                        if hasattr(self.image_processor, "tokenizer") 
                        else self.image_processor.eos_token_id)
        is_eos = completion_ids == eos_token_id

        # For each sample, find the first EOS (or use full length)
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        has_eos = is_eos.any(dim=1)
        if has_eos.any():
            eos_idx[has_eos] = is_eos[has_eos].int().argmax(dim=1)
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # --- FIX FOR BATCH SIZE: Repeat the attention_mask if needed ---
        # If batch sizes don't match, repeat the prompt tensors.
        if input_ids.shape[0] != completion_ids.shape[0]:
            num_return_sequences = completion_ids.shape[0] // input_ids.shape[0]
            print(f"Repeating input_ids {num_return_sequences} times to match completion_ids shape")
            input_ids = input_ids.repeat(num_return_sequences, 1)
            
        # Similarly, if attention_mask wasn't already repeated:
        if attention_mask.shape[0] != completion_ids.shape[0]:
            num_return_sequences = completion_ids.shape[0] // attention_mask.shape[0]
            print(f"Repeating attention_mask {num_return_sequences} times to match completion_ids shape")
            attention_mask = attention_mask.repeat(num_return_sequences, 1)


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
        completions_text = [
            f"{completion_start}{completion}" for completion in 
            self.image_processor.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        ]
        self._last_logged_completions = completions_text

        # ------------------------------
        # Compute rewards.
        # ------------------------------
        # Here, we expect the ground truth to be present in the dataset.
        # It might be different based on the prompt/reward you are using.
        # For instance, for SMILES, we use x["SMILES"]
        ground_truth = [x[self.ground_truth_column] for x in inputs]
        self._last_logged_ground_truth = ground_truth  # Save ground truth for logging

        # The reward function (passed in via the trainer constructor in experiments.py)
        rewards = self.reward_funcs(prompts, completions_text, ground_truth)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        self._metrics["reward"].append(rewards_tensor.mean().item())
        advantages = rewards_tensor

        return {
            "prompt_ids": input_ids,
            "prompt_mask": attention_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }
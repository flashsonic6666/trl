# reward_and_prompt_utils.py

import re
from rdkit import Chem

# Helper function to check that only allowed tags are present.
def only_has_allowed_tags(completion):
    allowed_tags = {"<think>", "</think>", "<answer>", "</answer>"}
    # Find all tags (ignoring extra whitespace)
    tags = re.findall(r"</?\s*\w+\s*>", completion)
    # Normalize tags by removing extra spaces.
    normalized_tags = {re.sub(r"\s+", "", tag) for tag in tags}
    return all(tag in allowed_tags for tag in normalized_tags)

# Modified hydrogen_count_reward function.
def hydrogen_count_reward(prompts, completions, hydrogen_atom_count):
    rewards = []
    for prompt, completion, true_count in zip(prompts, completions, hydrogen_atom_count):
        only_allowed = only_has_allowed_tags(completion)
        think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        answer_match = re.search(r"<answer>\s*(\d+)\s*</answer>", completion)
        bonus = 0.01 * len(think_match.group(1).split()) if think_match else 0.0
        if think_match and answer_match:
            answer_pred = int(answer_match.group(1))
            reward = 1.0 if answer_pred == true_count else 0
        else:
            reward = 0 if only_allowed else -1.0
        rewards.append(reward + bonus)
    return rewards

# Modified has_aromatic_ring_reward function.
def has_aromatic_ring_reward(prompts, completions, aromatic_ground_truth):
    rewards = []
    for prompt, completion, true_val in zip(prompts, completions, aromatic_ground_truth):
        only_allowed = only_has_allowed_tags(completion)
        think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        answer_match = re.search(r"<answer>\s*(\d+)\s*</answer>", completion)
        bonus = 0.01 * len(think_match.group(1).split()) if think_match else 0.0
        if think_match and answer_match:
            answer_pred = int(answer_match.group(1))
            if answer_pred in [0, 1]:
                reward = 1.0 if answer_pred == true_val else 0
            else:
                reward = -1.0
        else:
            reward = 0 if only_allowed else -1.0
        rewards.append(reward + bonus)
    return rewards

# import re
# from rdkit import Chem
# def smiles_match_reward(prompts, completions, ground_truth_smiles, length_reward=False, valid_reward=False):
#     rewards = []
#     for prompt, completion, true_smiles in zip(prompts, completions, ground_truth_smiles):
#         # Ensure completion matches exactly the expected format.
#         match = re.fullmatch(r"<think>(.*?)</think><answer>(.*?)</answer>", completion, re.DOTALL)
#         if not match:
#             rewards.append(-1.0)
#             continue  # Skip further processing if format is invalid
        
#         # Base reward for valid format.
#         base_reward = 0
        
#         # Initialize bonus values.
#         bonus_length = 0.0
#         bonus_valid = 0.0
        
#         # If enabled, calculate length bonus.
#         if length_reward:
#             bonus_length = 0.1 * (len(completion.split()) / 512)
        
#         # Extract answer content.
#         answer_content = match.group(2).strip()
        
#         try:
#             # Try to convert the answer SMILES.
#             mol_pred = Chem.MolFromSmiles(answer_content)
#             # If valid_reward is enabled and the SMILES is valid, add bonus.
#             if valid_reward and mol_pred is not None:
#                 bonus_valid = 0.1
#             # Convert ground-truth SMILES.
#             mol_true = Chem.MolFromSmiles(true_smiles.strip())
#             # If both SMILES are valid, compare their canonical forms.
#             if mol_pred is not None and mol_true is not None:
#                 canonical_pred = Chem.MolToSmiles(mol_pred, canonical=True)
#                 canonical_true = Chem.MolToSmiles(mol_true, canonical=True)
#                 if canonical_pred == canonical_true:
#                     base_reward = 1.0
#         except Exception:
#             # If any error occurs, keep base_reward as 0 (and no valid bonus is added).
#             pass
        
#         total_reward = base_reward + bonus_valid + bonus_length
#         rewards.append(total_reward)
#     return rewards

import re
from rdkit import Chem
def smiles_match_reward(prompts, completions, ground_truth_smiles, length_reward=False, valid_reward=False):
    rewards = []
    for prompt, completion, true_smiles in zip(prompts, completions, ground_truth_smiles):
        # Check that the completion is exactly in the desired format.
        match = re.fullmatch(r"Let me solve this step by step\.\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>", completion, re.DOTALL)

        if not match:
            rewards.append(-1.0)
            continue
        
        # Base reward for valid format.
        base_reward = 0
        
        # Calculate length bonus if enabled.
        bonus_length = 0.0
        if length_reward:
            bonus_length = 0.1 * (len(completion.split()) / 512)
        
        # Extract the SMILES from the answer tag.
        answer_content = match.group(2).strip()
        bonus_valid = 0.0
        
        try:
            # Parse the SMILES.
            mol_pred = Chem.MolFromSmiles(answer_content)
            mol_true = Chem.MolFromSmiles(true_smiles.strip())
            # If both SMILES are valid, compare their canonical forms.
            if mol_pred is not None and mol_true is not None:
                canonical_pred = Chem.MolToSmiles(mol_pred, canonical=True)
                canonical_true = Chem.MolToSmiles(mol_true, canonical=True)
                if canonical_pred == canonical_true:
                    # Correct answer: reward is exactly 1.
                    rewards.append(1.0)
                    continue
            # If not correct but valid_reward is enabled and the answer SMILES is valid, add bonus.
            if valid_reward and mol_pred is not None:
                bonus_valid = 0.1
        except Exception:
            # If an exception occurs, assume SMILES is invalid.
            pass
        
        total_reward = base_reward + bonus_valid + bonus_length
        rewards.append(total_reward)
    return rewards


# --- Prompt templates ---
def prompt_template_smiles(image_token, completion_start):
    return f"""<|im_start|>system
You are a helpful assistant. You first think about the reasoning process and then provide the user with the answer.
<|im_end|>

<|im_start|>user
Here is an image <|vision_start|> {image_token} <|vision_end|>. Identify the chemical structure shown and provide its SMILES representation.
Think step by step inside <think> </think> tags, and provide the final SMILES string inside <answer> </answer> tags.
For example: <think> The molecule consists of a benzene ring with an attached hydroxyl group. </think> <answer>Oc1ccccc1</answer>.
<|im_end|>

<|im_start|>assistant
{completion_start}"""

def prompt_template_smiles_no_instruct(image_token, completion_start, base_model_prompt=False):
    image = {image_token}
    question = f"Here is an image <|vision_start|> {image} <|vision_end|>. Identify the chemical structure shown and provide its SMILES representation. \
Think step by step inside <think> </think> tags, and provide the final SMILES string inside <answer> </answer> tags. \
For example: <think> The molecule consists of a benzene ring with an attached hydroxyl group. </think> <answer>Oc1ccccc1</answer>."
    question = question.replace("<image>", "")
    prompt = f'A conversation between User and Assistant. The user asks a question about the image, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\nUser: {question} \nAssistant: Let me solve this step by step.\n<think>'
    return "<image>" + prompt

def prompt_template_aromatic(image_token, completion_start):
    return f"""<|im_start|>system
You are a helpful assistant.
<|im_end|>

<|im_start|>user
Here is an image {image_token}. Determine whether the chemical structure shown contains an aromatic ring.
Provide your reasoning inside <think> </think> and your answer as "1" (aromatic) or "0" (non-aromatic) inside <answer> </answer>.
<|im_end|>

<|im_start|>assistant
{completion_start}"""

def prompt_template_hydrogen(image_token, completion_start):
    return f"""<|im_start|>system
You are a helpful assistant.
<|im_end|>

<|im_start|>user
Here is an image {image_token}. Count the total number of hydrogen atoms in the chemical structure shown.
Explain your reasoning inside <think> </think> and provide the final count inside <answer> </answer>.
<|im_end|>

<|im_start|>assistant
{completion_start}"""

from functools import partial
# A helper to select reward and prompt functions by name:
def get_reward_fn(name):
    if name == "hydrogen_count":
        return hydrogen_count_reward
    elif name == "has_aromatic_ring":
        return has_aromatic_ring_reward
    elif name == "smiles_match":
        return smiles_match_reward
    elif name == "smiles_match_with_length":
        return partial(smiles_match_reward, length_reward=True)
    elif name == "smiles_match_with_valid":
        return partial(smiles_match_reward, valid_reward=True)
    elif name == "smiles_match_with_length_and_valid":
        return partial(smiles_match_reward, length_reward=True, valid_reward=True)
    else:
        raise ValueError(f"Unknown reward function: {name}")

def get_prompt_template(name):
    if name == "hydrogen":
        return prompt_template_hydrogen
    elif name == "aromatic":
        return prompt_template_aromatic
    elif name == "smiles":
        return prompt_template_smiles
    elif name == "smiles_no_instruct":
        return prompt_template_smiles_no_instruct
    else:
        raise ValueError(f"Unknown prompt template: {name}")

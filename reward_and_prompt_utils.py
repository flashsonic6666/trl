# reward_and_prompt_utils.py

import re
from rdkit import Chem

# --- Reward functions ---
def hydrogen_count_reward(prompts, completions, hydrogen_atom_count):
    rewards = []
    for prompt, completion, true_count in zip(prompts, completions, hydrogen_atom_count):
        try:
            think_match = re.search(r"<think>.*?</think>", completion, re.DOTALL)
            answer_match = re.search(r"<answer>\s*(\d+)\s*</answer>", completion)
            if think_match and answer_match:
                answer_pred = int(answer_match.group(1))
                reward = 1.0 if answer_pred == true_count else -0.9
            else:
                reward = -1.0
        except Exception as e:
            reward = -1.0
        rewards.append(reward)
    return rewards

def has_aromatic_ring_reward(prompts, completions, aromatic_ground_truth):
    rewards = []
    for prompt, completion, true_val in zip(prompts, completions, aromatic_ground_truth):
        try:
            think_match = re.search(r"<think>.*?</think>", completion, re.DOTALL)
            answer_match = re.search(r"<answer>\s*(\d+)\s*</answer>", completion)
            if think_match and answer_match:
                answer_pred = int(answer_match.group(1))
                if answer_pred in [0, 1]:
                    reward = 1.0 if answer_pred == true_val else -0.9
                else:
                    reward = -1.0
            else:
                reward = -1.0
        except Exception as e:
            reward = -1.0
        rewards.append(reward)
    return rewards

def smiles_match_reward(prompts, completions, ground_truth_smiles):
    rewards = []
    for prompt, completion, true_smiles in zip(prompts, completions, ground_truth_smiles):
        try:
            smiles_match = re.search(r"<smiles>(.*?)</smiles>", completion, re.DOTALL)
            if smiles_match:
                predicted_smiles = smiles_match.group(1).strip()
                formatted_correctly = True
            else:
                formatted_correctly = False
        except Exception as e:
            formatted_correctly = False

        if not formatted_correctly:
            reward = -1.0
        else:
            mol_pred = Chem.MolFromSmiles(predicted_smiles)
            mol_true = Chem.MolFromSmiles(true_smiles)
            if mol_pred is None or mol_true is None:
                reward = -1.0
            else:
                canonical_pred = Chem.MolToSmiles(mol_pred, canonical=True)
                canonical_true = Chem.MolToSmiles(mol_true, canonical=True)
                reward = 1.0 if canonical_pred == canonical_true else -0.9
        rewards.append(reward)
    return rewards

# --- Prompt templates ---
def prompt_template_smiles(image_token, completion_start):
    return f"""<|im_start|>system
You are a helpful assistant. You first think about the reasoning process and then provide the user with the answer.
<|im_end|>

<|im_start|>user
Here is an image {image_token}. Identify the chemical structure shown and provide its SMILES representation.
Think step by step inside <think> </think> tags, and provide the final SMILES string inside <smiles> </smiles> tags.
For example: <think> The molecule consists of a benzene ring with an attached hydroxyl group. </think> <smiles>Oc1ccccc1</smiles>.
<|im_end|>

<|im_start|>assistant
{completion_start}"""

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

# A helper to select reward and prompt functions by name:
def get_reward_fn(name):
    if name == "hydrogen_count":
        return hydrogen_count_reward
    elif name == "has_aromatic_ring":
        return has_aromatic_ring_reward
    elif name == "smiles_match":
        return smiles_match_reward
    else:
        raise ValueError(f"Unknown reward function: {name}")

def get_prompt_template(name):
    if name == "hydrogen":
        return prompt_template_hydrogen
    elif name == "aromatic":
        return prompt_template_aromatic
    elif name == "smiles":
        return prompt_template_smiles
    else:
        raise ValueError(f"Unknown prompt template: {name}")

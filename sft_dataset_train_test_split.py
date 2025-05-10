import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split

# Input CSV path
csv_path = "data_organometallic/molecules_with_smiles.csv"
image_dir = "data_organometallic/images_organometallic"

# Load CSV
df = pd.read_csv(csv_path)

# Canonicalize SMILES
def canonicalize(smiles):
    return smiles

df["SMILES"] = df["SMILES"].apply(canonicalize)

# Drop any rows where canonicalization failed
df.dropna(subset=["SMILES"], inplace=True)

# Change "Image" to "file_path"
df["file_path"] = df["Image"].apply(lambda fn: f"{image_dir}/{fn}")
df.drop(columns=["Image"], inplace=True)

# Perform 80/20 split
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Save to CSV
train_df.to_csv("data_organometallic/train.csv", index=False)
eval_df.to_csv("data_organometallic/eval.csv", index=False)

print(f"Saved {len(train_df)} samples to data_organometallic/train.csv")
print(f"Saved {len(eval_df)} samples to data_organometallic/eval.csv")

import pandas as pd
from sklearn.model_selection import train_test_split

# Input CSV path
csv_path = "synthetic/indigo_resize.csv"

# Load the CSV
df = pd.read_csv(csv_path)

# Perform 80/20 train/eval split
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Save to new CSV files
train_df.to_csv("synthetic/indigo_resize_train.csv", index=False)
eval_df.to_csv("synthetic/indigo_resize_eval.csv", index=False)

print(f"Saved {len(train_df)} samples to simple_molecules_train.csv")
print(f"Saved {len(eval_df)} samples to simple_molecules_eval.csv")

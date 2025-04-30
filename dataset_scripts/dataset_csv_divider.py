import pandas as pd
from sklearn.model_selection import train_test_split
import os

filename = ""
df = pd.read_csv(filename,sep='\t')

# Ensure correct headers
assert 'input_values' in df.columns and 'labels' in df.columns, "CSV must contain 'input_values' and 'labels' columns"

df = df.sort_values(by=['labels', 'input_values']).reset_index(drop=True)

SEED = 42

percentages = [5, 10, 20, 30, 50, 80]

for pct in percentages:
    sample_size = int(len(df) * pct / 100)

    df_subset, _ = train_test_split(
        df,
        train_size=sample_size,
        stratify=df['labels'],
        random_state=SEED
    )

    base, ext = os.path.splitext(filename)
    new_filename = f"{base}_{pct}{ext}"
    df_subset.to_csv(new_filename, index=False,sep='\t')
    print(f"Saved {new_filename} with {len(df_subset)} rows")

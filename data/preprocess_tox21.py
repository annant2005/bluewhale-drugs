import pandas as pd
import os

df = pd.read_csv('tox21.csv')
# Keep only rows with non-null SR-p53 and smiles
filtered = df[['smiles', 'SR-p53']].dropna()
filtered = filtered.rename(columns={'SR-p53': 'label'})
filtered['label'] = filtered['label'].astype(int)  # Ensure integer labels
out_path = os.path.join('data', 'tox21_sr-p53.csv')
filtered.to_csv(out_path, index=False)
print(f'Preprocessed dataset saved as {out_path}. Total samples: {len(filtered)}') 
import pandas as pd
df = pd.read_parquet('data/processed/mag7_embeddings.parquet')
print(df.columns)
print(df.head(1))
print(f"Shape: {df.shape}")

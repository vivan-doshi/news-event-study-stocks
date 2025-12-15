import pandas as pd
import numpy as np

file_path = "data/master_analysis_data_advanced_clean.csv"
df = pd.read_csv(file_path)

# Filter for TSLA
mask = df['ticker_yf'].astype(str).str.contains("TSLA", case=False, na=False)
tsla = df[mask].sort_values('date')

col = "z_score_topic_0"

print(f"Checking {col} for TSLA...")
print(f"Total rows: {len(tsla)}")

# Check what pandas thinks the dtype is
print(f"Dtype: {tsla[col].dtype}")

# Filter out NaNs explicitly
valid = tsla[tsla[col].notna()]
print(f"Non-NA rows: {len(valid)}")

# Filter out 0s
valid_non_zero = valid[valid[col] != 0]
print(f"Non-NA & Non-Zero rows: {len(valid_non_zero)}")

if not valid_non_zero.empty:
    print("\nLast 10 values:")
    print(valid_non_zero[['date', col]].tail(10))
else:
    print("No valid non-zero numeric values found!")

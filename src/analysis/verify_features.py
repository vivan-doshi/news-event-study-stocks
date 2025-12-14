
import pandas as pd
import argparse

def main():
    path = "data/processed/mag7_augmented_features.parquet"
    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    
    # Check for new columns
    expected = ['sent_growth_zscore', 'sent_risk_magnitude']
    missing = [c for c in expected if c not in df.columns]
    
    if missing:
        print(f"ERROR: Missing columns: {missing}")
    else:
        print("SUCCESS: Target columns found.")
        
    # Sample display
    cols_to_show = ['symbol_query', 'final_date_for_news', 'sent_growth_zscore', 'sent_risk_magnitude']
    print("\n--- First 5 Rows ---")
    print(df[cols_to_show].head().to_string(index=False))
    
    print("\n--- Summary Stats ---")
    print(df[expected].describe().to_string())

if __name__ == "__main__":
    main()

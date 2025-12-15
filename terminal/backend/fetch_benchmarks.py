import yfinance as yf
import pandas as pd
import os

def fetch_sp500():
    print("Fetching S&P 500 (^GSPC) data from Yahoo Finance...")
    
    # Define Path
    output_path = "data/sp500_data.csv"
    
    # Download Data
    # Match the range of our analysis data (approx 2023 to late 2025)
    # We'll fetch a broad range to be safe.
    sp500 = yf.download("^GSPC", start="2023-01-01", end="2025-10-31", progress=False)
    
    if sp500.empty:
        print("Error: No data fetched for ^GSPC.")
        return

    # Reset index to get Date column
    sp500 = sp500.reset_index()
    
    # Standardize columns
    # yfinance returns MultiIndex columns in recent versions, explicitly handle
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.get_level_values(0)
    
    # Rename for consistency
    sp500 = sp500.rename(columns={"Date": "date", "Adj Close": "benchmark_price", "Close": "close"})
    
    # Ensure Date format
    sp500['date'] = pd.to_datetime(sp500['date']).dt.strftime('%Y-%m-%d')
    
    # Keep only necessary columns
    # We might need 'benchmark_price' (Adj Close)
    # Handle if 'Adj Close' is missing (sometimes yf issues) -> use 'Close'
    if 'benchmark_price' not in sp500.columns:
        if 'close' in sp500.columns:
            sp500['benchmark_price'] = sp500['close']
        else:
             print("Error: Could not find Close/Adj Close column.")
             print(sp500.columns)
             return

    final_df = sp500[['date', 'benchmark_price']]
    
    # Save
    final_df.to_csv(output_path, index=False)
    print(f"Success! S&P 500 data saved to {output_path}")
    print(final_df.head())

if __name__ == "__main__":
    fetch_sp500()

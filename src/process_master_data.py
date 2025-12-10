
import pandas as pd
import numpy as np
import os
import logging

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAG7_SYMBOLS = ['AAPL.US', 'MSFT.US', 'GOOGL.US', 'AMZN.US', 'META.US', 'NVDA.US', 'TSLA.US']
LAGS = [1, 2, 5, 10, 21]

def load_stock_data(file_path):
    """Loads and preprocesses stock data."""
    logger.info(f"Loading stock data from {file_path}...")
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)

    # Standardize columns
    df.columns = [c.lower() for c in df.columns]
    
    # Rename if needed (based on previous script logic)
    if 'symbol_query' in df.columns:
        df = df.rename(columns={'symbol_query': 'symbol', 'adj_close': 'adjusted_close'})
        
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['symbol', 'date'])
    
    # Filter Mag7
    df = df[df['symbol'].isin(MAG7_SYMBOLS)]
    
    # Calc Log Returns
    logger.info("Calculating returns and lags...")
    df['log_ret'] = df.groupby('symbol')['adjusted_close'].transform(lambda x: np.log(x / x.shift(1)))
    
    # Calc Lags
    for lag in LAGS:
        df[f'log_ret_lag{lag}'] = df.groupby('symbol')['log_ret'].shift(lag)
        
    df = df.dropna(subset=['log_ret']) # Drop the first row where return is NaN
    # Note: We keep rows with NaN lags for now (early dates), or drop?
    # Usually consistent datasets drop valid rows. Let's keep them and let the analysis decide or drop if critical.
    # The user wants "one file", so keeping as much data as possible is safer, analysis can dropna.
    
    return df

def load_factors(file_path):
    """Loads Fama-French factors."""
    logger.info(f"Loading factors from {file_path}...")
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)
        
    # Attempt to handle index/date column
    # If date is in columns, parse it.
    possible_date_cols = ['date', 'Date', 'index', 'Unnamed: 0']
    date_col = None
    for col in df.columns:
        if col in possible_date_cols:
            date_col = col
            break
            
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.rename(columns={date_col: 'date'})
    else:
        # Maybe index is date?
        try:
            df.index = pd.to_datetime(df.index)
            df = df.reset_index().rename(columns={'index': 'date'})
        except:
             logger.warning("Could not identify date column in factors. Assuming first column is date if looks like date.")
             first_col = df.columns[0]
             try:
                 df[first_col] = pd.to_datetime(df[first_col])
                 df = df.rename(columns={first_col: 'date'})
             except:
                 raise ValueError("Could not parse Date in factors file.")

    return df

def main():
    stock_path = 'data/processed/mag7_yf_2021_2025.parquet'
    factors_path = 'reports/fama_french/data/fama_french_factors.csv'
    output_dir = 'data'
    
    # Create output dir if not exists (should exist)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Stock
    stock_df = load_stock_data(stock_path)
    
    # 2. Load Factors
    factors_df = load_factors(factors_path)
    
    # 3. Merge
    logger.info("Merging stock data and factors...")
    # Merge on data. Stock has dates, Factors has dates.
    # Inner join to keep only matching dates
    master_df = pd.merge(stock_df, factors_df, on='date', how='inner')
    
    # 4. Save
    csv_path = os.path.join(output_dir, 'master_stock_data.csv')
    parquet_path = os.path.join(output_dir, 'master_stock_data.parquet')
    
    master_df.to_csv(csv_path, index=False)
    master_df.to_parquet(parquet_path, index=False)
    
    logger.info(f"Master dataset saved to:\n  {csv_path}\n  {parquet_path}")
    logger.info(f"Columns: {list(master_df.columns)}")

if __name__ == "__main__":
    main()

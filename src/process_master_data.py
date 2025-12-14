
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

def load_news_feature_data(file_path):
    """Loads the aggregated news + stock features data."""
    logger.info(f"Loading news & stock features from {file_path}...")
    df = pd.read_parquet(file_path)
    
    # Standardize Column Names
    # symbol_query -> symbol
    if 'symbol_query' in df.columns:
        df = df.rename(columns={'symbol_query': 'symbol'})
        
    # ret_log_1d -> log_ret
    if 'ret_log_1d' in df.columns:
        df = df.rename(columns={'ret_log_1d': 'log_ret'})
        
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['symbol', 'date'])
    
    # Ensure log_ret exists (calculate if missing but adj_close exists)
    if 'log_ret' not in df.columns and 'adj_close' in df.columns:
        logger.info("Calculating log returns...")
        df['log_ret'] = df.groupby('symbol')['adj_close'].transform(lambda x: np.log(x / x.shift(1)))
    
    # Calculate Lagged Returns
    logger.info("Calculating return lags...")
    for lag in LAGS:
        df[f'log_ret_lag{lag}'] = df.groupby('symbol')['log_ret'].shift(lag)
        
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
    # Input Paths
    news_features_path = 'data/processed/mag7_augmented_features.parquet'
    factors_path = 'reports/fama_french/data/fama_french_factors.csv'
    
    # Output Paths
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load News + Stock Features
    if not os.path.exists(news_features_path):
        logger.error(f"News features file not found: {news_features_path}")
        return
        
    master_df = load_news_feature_data(news_features_path)
    
    # 2. Load Factors
    factors_df = load_factors(factors_path)
    
    # 3. Merge
    logger.info("Merging news/stock data with Fama-French factors...")
    # Merge on date.
    final_df = pd.merge(master_df, factors_df, on='date', how='inner')
    
    # 4. Save
    csv_path = os.path.join(output_dir, 'master_analysis_data.csv')
    parquet_path = os.path.join(output_dir, 'master_analysis_data.parquet')
    
    final_df.to_csv(csv_path, index=False)
    final_df.to_parquet(parquet_path, index=False)
    
    logger.info(f"Master Analysis Dataset saved to:\n  {csv_path}\n  {parquet_path}")
    logger.info(f"Shape: {final_df.shape}")
    logger.info(f"Columns: {list(final_df.columns)}")

if __name__ == "__main__":
    main()

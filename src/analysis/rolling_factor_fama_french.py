
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import matplotlib.pyplot as plt
import argparse
import logging
import os
import sys

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants
MAG7_SYMBOLS = ['AAPL.US', 'MSFT.US', 'GOOGL.US', 'AMZN.US', 'META.US', 'NVDA.US', 'TSLA.US']
ETF_TICKERS = {
    'SPY': 'SPY',      # Market Proxy
    'IWM': 'IWM',      # Size (Small Cap)
    'IWD': 'IWD',      # Value
    'IWF': 'IWF',      # Growth
    'QUAL': 'QUAL',    # Profitability (Quality)
    'USMV': 'USMV',    # Investment (Min Volatility)
    'IRX': '^IRX'      # Risk-Free Rate (13 Week T-Bill)
}

def load_master_data(file_path, start_date, end_date):
    """Loads the master dataset with stock data, factors, and lags."""
    logger.info(f"Loading master data from {file_path}...")
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)
    
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter Date Range
    mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
    df = df[mask]
    
    # Ensure columns used in analysis are present
    required_cols = ['log_ret', 'RF', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Master file missing required columns. Found: {df.columns}")
        
    return df

def run_augmented_rolling_regression(df, window, output_dir):
    """Runs Rolling OLS with Fama-French 5 Factors + Lags."""
    
    logger.info(f"Running Augmented Rolling Regression (Window={window})...")
    
    # Create directories
    data_dir = os.path.join(output_dir, 'data')
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    all_results = []
    
    symbols = df['symbol'].unique()
    
    # Define Predictors
    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    lags = [c for c in df.columns if 'log_ret_lag' in c]
    predictors = factors + lags
    logger.info(f"Predictors: {predictors}")
    
    for sym in symbols:
        logger.info(f"Processing {sym}...")
        
        # Align Data
        sym_data = df[df['symbol'] == sym].set_index('date').copy()
        
        # Drop rows with missing values (crucial for lags)
        sym_data = sym_data.dropna(subset=['log_ret', 'RF'] + predictors)
        
        if len(sym_data) < window:
            logger.warning(f"Insufficient data for {sym}. Skipping.")
            continue
            
        # Prepare Variables
        # Excess Return = Stock Ret - RF
        y = sym_data['log_ret'] - sym_data['RF']
        
        # X variables
        X = sym_data[predictors]
        X = sm.add_constant(X)
        
        # Rolling OLS
        try:
            model = RollingOLS(y, X, window=window)
            results = model.fit()
            
            # Extract Results
            params = results.params.copy()
            
            # Rename columns for clarity if needed, but keeping original names is often safer for automated processing
            # We will just verify they are consistent
            
            # Annualize Alpha if present
            if 'const' in params.columns:
                params['Alpha_Annualized'] = params['const'] * 252
            
            params['R_Squared'] = results.rsquared
            params['symbol'] = sym
            
            # Drop NaN rows (start of window)
            params = params.dropna()
            
            # Save to list
            metrics_reset = params.reset_index().rename(columns={'date': 'date'})
            all_results.append(metrics_reset)
            
            # Generate Factor Exposure Grid Plot (Updated to handle more vars)
            plot_factor_exposure(params, sym, plots_dir, predictors)
            
        except Exception as e:
            logger.error(f"Error processing {sym}: {e}")

    # Save Consolidated Report
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        csv_path = os.path.join(data_dir, 'rolling_augmented_stats.csv')
        final_df.to_csv(csv_path, index=False)
        logger.info(f"Consolidated report saved to {csv_path}")
        
        parquet_path = os.path.join(data_dir, 'rolling_augmented_stats.parquet')
        final_df.to_parquet(parquet_path, index=False)
        logger.info(f"Consolidated report saved to {parquet_path}")

def plot_factor_exposure(metrics, symbol, output_dir, predictors):
    """Generates a grid subplot for Alpha and all Betas."""
    
    # Calculate grid size
    num_vars = len(predictors) + 1 # +1 for Alpha
    cols = 3
    rows = (num_vars + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    fig.suptitle(f"{symbol} - Rolling Exposure (Factors + Lags)", fontsize=16)
    
    axes = axes.flatten()
    
    # List of things to plot
    # Alpha
    if 'Alpha_Annualized' in metrics.columns:
        ax = axes[0]
        ax.plot(metrics.index, metrics['Alpha_Annualized'], label='Alpha (Ann.)', color='purple', linewidth=1.5)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax.set_title("Alpha (Annualized)")
        ax.grid(True, alpha=0.3)
    
    # Predictors
    for i, var in enumerate(predictors):
        ax_idx = i + 1
        if ax_idx >= len(axes): break
        
        ax = axes[ax_idx]
        ax.plot(metrics.index, metrics[var], label=f'Beta_{var}', color='tab:blue', linewidth=1.5)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax.set_title(f"Beta: {var}")
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for j in range(num_vars, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"{symbol}_augmented.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Augmented Fama-French Rolling Analysis")
    parser.add_argument("--start_date", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", default="2025-10-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--window", type=int, default=60, help="Rolling window size")
    parser.add_argument("--input_file", default="data/master_stock_data.csv", help="Path to master data")
    parser.add_argument("--output_dir", default="reports/fama_french_augmented", help="Output directory")
    
    args = parser.parse_args()
    
    # 1. Load Master Data
    master_df = load_master_data(args.input_file, args.start_date, args.end_date)
    
    # 2. Run Analysis
    run_augmented_rolling_regression(master_df, args.window, args.output_dir)
    
    logger.info("Augmented Analysis Complete.")

if __name__ == "__main__":
    main()

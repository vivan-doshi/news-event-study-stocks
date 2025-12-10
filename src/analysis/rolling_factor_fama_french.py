
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

def load_stock_data(file_path, start_date, end_date):
    """Loads target stock data and calculates returns."""
    logger.info(f"Loading stock data from {file_path}...")
    try:
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
            # Rename columns to match expected schema if needed
            if 'symbol_query' in df.columns:
                df = df.rename(columns={'symbol_query': 'symbol', 'adj_close': 'adjusted_close'})
        else:
            df = pd.read_csv(file_path)
        
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter Date Range
        mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
        df = df[mask]
        
        # Filter Mag7
        df = df[df['symbol'].isin(MAG7_SYMBOLS)]
        
        # Calculate Log Returns
        df = df.sort_values(by=['symbol', 'date'])
        df['log_ret'] = df.groupby('symbol')['adjusted_close'].transform(lambda x: np.log(x / x.shift(1)))
        df = df.dropna(subset=['log_ret'])
        
        return df
    except Exception as e:
        logger.error(f"Error loading stock data: {e}")
        raise

def fetch_etf_factors(start_date, end_date):
    """
    Fetches ETF data and constructs Fama-French factor proxies.
    Returns a DataFrame with daily factor returns.
    """
    logger.info("Fetching ETF data for factor construction...")
    
    # Extend fetch window slightly to account for returns calculation (t-1)
    fetch_start = pd.to_datetime(start_date) - pd.Timedelta(days=5)
    
    tickers = list(ETF_TICKERS.values())
    
    try:
        data = yf.download(tickers, start=fetch_start, end=end_date, progress=False, group_by='ticker')
        
        # Check if data might satisfy multi-index or single index depending on version
        # yfinance > 0.2 returns MultiIndex (Ticker, Price Type) output for multiple tickers
        
        factors = pd.DataFrame(index=data.index)
        
        # 1. Process Risk-Free Rate (^IRX)
        # ^IRX is yield in percent (e.g., 4.5 means 4.5% annualized)
        # Convert to daily return: (1 + yield/100)^(1/252) - 1
        irx_yield = data[ETF_TICKERS['IRX']]['Close']
        # Fill missing yields with previous day
        irx_yield = irx_yield.ffill()
        factors['RF'] = (1 + irx_yield / 100)**(1/252) - 1
        
        # 2. Process ETF Returns (Log Returns)
        for name, ticker in ETF_TICKERS.items():
            if name == 'IRX': continue
            
            price = data[ticker]['Close']
            # Forward fill prices to handle holidays/missing data
            price = price.ffill()
            factors[f'ret_{name}'] = np.log(price / price.shift(1))
            
        factors = factors.dropna()
        
        # 3. Construct Factors (The "Liquid 5" Methodology)
        
        # Mkt-RF: SPY - RF
        factors['Mkt-RF'] = factors['ret_SPY'] - factors['RF']
        
        # SMB: IWM (Small) - SPY (Large Proxy)
        factors['SMB'] = factors['ret_IWM'] - factors['ret_SPY']
        
        # HML: IWD (Value) - IWF (Growth)
        factors['HML'] = factors['ret_IWD'] - factors['ret_IWF']
        
        # RMW: QUAL (Quality) - SPY (Market)
        # (Proxying robust profitability vs market)
        factors['RMW'] = factors['ret_QUAL'] - factors['ret_SPY']
        
        # CMA: USMV (Conservative/Low Vol) - SPY (Market)
        # (Proxying conservative investment style)
        factors['CMA'] = factors['ret_USMV'] - factors['ret_SPY']
        
        # Keep only date and factors
        factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        factors = factors[factor_cols]
        
        # Filter back to exact start date
        factors = factors[factors.index >= pd.to_datetime(start_date)]
        
        return factors
        
    except Exception as e:
        logger.error(f"Error constructing factors: {e}")
        raise

def run_5factor_rolling_regression(stock_df, factors_df, window, output_dir):
    """Runs Rolling 5-Factor OLS for each stock."""
    
    logger.info(f"Running 5-Factor Rolling Regression (Window={window})...")
    
    # Create directories
    data_dir = os.path.join(output_dir, 'data')
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    all_results = []
    
    symbols = stock_df['symbol'].unique()
    
    for sym in symbols:
        logger.info(f"Processing {sym}...")
        
        # Align Data
        sym_data = stock_df[stock_df['symbol'] == sym].set_index('date')[['log_ret']]
        
        # Merge with factors
        reg_data = pd.merge(sym_data, factors_df, left_index=True, right_index=True, how='inner')
        
        if len(reg_data) < window:
            logger.warning(f"Insufficient data for {sym}. Skipping.")
            continue
            
        # Prepare Variables
        # Excess Return = Stock Ret - RF
        y = reg_data['log_ret'] - reg_data['RF']
        
        # Factors
        X = reg_data[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
        X = sm.add_constant(X)
        
        # Rolling OLS
        model = RollingOLS(y, X, window=window)
        results = model.fit()
        
        # Extract Results
        params = results.params.copy()
        
        # Compile MetricsDataFrame
        metrics = params.rename(columns={
            'const': 'Alpha',
            'Mkt-RF': 'Beta_Mkt',
            'SMB': 'Beta_SMB',
            'HML': 'Beta_HML',
            'RMW': 'Beta_RMW',
            'CMA': 'Beta_CMA'
        })
        
        # Annualize Alpha (Daily Alpha * 252)
        metrics['Alpha'] = metrics['Alpha'] * 252
        
        metrics['R_Squared'] = results.rsquared
        metrics['symbol'] = sym
        metrics = metrics.dropna()
        
        # Save to list for consolidated report
        metrics_reset = metrics.reset_index().rename(columns={'index': 'date'})
        all_results.append(metrics_reset)
        
        # Generate Factor Exposure Grid Plot
        plot_factor_exposure(metrics, sym, plots_dir)

    # Save Consolidated Report
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        csv_path = os.path.join(data_dir, 'rolling_5factor_stats.csv')
        final_df.to_csv(csv_path, index=False)
        logger.info(f"Consolidated report saved to {csv_path}")

def plot_factor_exposure(metrics, symbol, output_dir):
    """Generates a 6-grid subplot of Alpha and 5 Betas."""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f"{symbol} - Dynamic Fama-French 5-Factor Exposure (Rolling)", fontsize=16)
    
    # Plot layout
    plots = [
        ('Alpha', 'Alpha (Annualized)', 'purple'),
        ('Beta_Mkt', 'Market Beta', 'blue'),
        ('Beta_SMB', 'Size (SMB) Beta', 'green'),
        ('Beta_HML', 'Value (HML) Beta', 'red'),
        ('Beta_RMW', 'Profitability (RMW) Beta', 'orange'),
        ('Beta_CMA', 'Investment (CMA) Beta', 'brown')
    ]
    
    axes = axes.flatten()
    
    for i, (col, title, color) in enumerate(plots):
        ax = axes[i]
        ax.plot(metrics.index, metrics[col], label=col, color=color, linewidth=1.5)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        if col == 'Beta_Mkt':
             ax.axhline(1, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
             
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"{symbol}_5factor.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Dynamic Fama-French 5-Factor Rolling Analysis")
    parser.add_argument("--start_date", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", default="2025-10-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--window", type=int, default=60, help="Rolling window size")
    parser.add_argument("--input_file", default="data/stock_data.csv", help="Path to stock data")
    parser.add_argument("--output_dir", default="reports/fama_french", help="Output directory")
    
    args = parser.parse_args()
    
    # 1. Load Stock Data
    stock_df = load_stock_data(args.input_file, args.start_date, args.end_date)
    
    # 2. Fetch/Construct Factors
    factors_df = fetch_etf_factors(args.start_date, args.end_date)
    
    # 3. Run Analysis
    run_5factor_rolling_regression(stock_df, factors_df, args.window, args.output_dir)
    
    logger.info("Fama-French Analysis Complete.")

if __name__ == "__main__":
    main()

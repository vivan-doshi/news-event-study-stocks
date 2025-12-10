
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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

def load_data(file_path):
    """Loads stock data and prepares it for analysis."""
    logger.info(f"Loading data from {file_path}...")
    try:
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
            # Rename columns to match expected schema if needed
            if 'symbol_query' in df.columns:
                df = df.rename(columns={'symbol_query': 'symbol', 'adj_close': 'adjusted_close'})
        else:
            df = pd.read_csv(file_path)
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by symbol and date
        df = df.sort_values(by=['symbol', 'date'])
        
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def calculate_returns(df):
    """Calculates Log Returns for each symbol."""
    logger.info("Calculating Log Returns...")
    
    # Calculate log returns: ln(P_t / P_{t-1})
    # We use adjusted_close for accurate returns
    df['log_ret'] = df.groupby('symbol')['adjusted_close'].transform(lambda x: np.log(x / x.shift(1)))
    
    # Drop rows with NaN returns (first day of each symbol)
    df = df.dropna(subset=['log_ret'])
    
    return df

def create_market_proxy(df):
    """
    Constructs a Synthetic Market Index.
    Logic: Equal-Weighted Mean Return of all distinct symbols present on that day.
    """
    logger.info("Constructing Synthetic Market Proxy (Equal-Weighted)...")
    
    # Group by date and take the mean of log_ret across all symbols
    market_returns = df.groupby('date')['log_ret'].mean().rename('market_ret')
    
    return market_returns

def run_rolling_regression(df, market_returns, window, output_dir):
    """
    Runs Rolling OLS for each symbol against the market proxy.
    Saves metrics and generates plots.
    """
    # Create output directories
    data_dir = os.path.join(output_dir, 'data')
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    symbols = df['symbol'].unique()
    summary_list = []
    
    for sym in symbols:
        logger.info(f"Processing {sym} (Window={window})...")
        
        # Extract symbol data
        sym_data = df[df['symbol'] == sym].set_index('date')[['log_ret']].rename(columns={'log_ret': 'asset_ret'})
        
        # Merge with market returns
        regression_data = pd.merge(sym_data, market_returns, left_index=True, right_index=True, how='inner')
        
        if len(regression_data) < window:
            logger.warning(f"Not enough data for {sym} to run rolling regression. Skipping.")
            continue
            
        # Prepare X (Market) and y (Asset)
        X = sm.add_constant(regression_data['market_ret'])
        y = regression_data['asset_ret']
        
        # Run Rolling OLS
        # Model: R_asset = alpha + beta * R_market
        model = RollingOLS(y, X, window=window)
        results = model.fit()
        
        # Extract Parameters
        params = results.params.copy()
        rsquared = results.rsquared
        
        # Rename columns for clarity. params has 'const' (Alpha) and 'market_ret' (Beta)
        metrics = pd.DataFrame({
            'alpha': params['const'],
            'beta': params['market_ret'],
            'r_squared': rsquared
        })
        
        # Save Metrics
        metrics_file = os.path.join(data_dir, f"{sym}_rolling_metrics.csv")
        metrics.to_csv(metrics_file)
        
        # Generate Plots
        plot_rolling_metrics(metrics, sym, window, plots_dir)
        
        # --- Evaluation: Predictive Power ---
        # 1. Correct Shift: To predict return at t, we use Alpha/Beta from t-1 (rolling window ending yesterday)
        # metrics index is 'date'. params at 'date' are trained on window ending at 'date'.
        # So we shift metrics forward by 1 to align with 'future' returns.
        pred_params = metrics.shift(1).dropna()
        
        # Align with actual returns
        # regression_data has 'asset_ret', 'market_ret'
        eval_data = pd.merge(regression_data, pred_params, left_index=True, right_index=True, how='inner')
        
        if len(eval_data) > 0:
            # Predict: Alpha + Beta * Market
            eval_data['pred_ret'] = eval_data['alpha'] + eval_data['beta'] * eval_data['market_ret']
            
            # OOS R2 (Predictive R2)
            # Standard def: 1 - SSE/SST
            y_true = eval_data['asset_ret']
            y_pred = eval_data['pred_ret']
            oos_r2 = r2_score(y_true, y_pred)
            
            # Additional Metrics
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            
            # Hit Rate (Directional Accuracy)
            # 1 if signs match, 0 otherwise. Handle zeros if needed (sign(0)=0).
            # Usually simple sign match is sufficient.
            hit_rate = np.mean(np.sign(y_true) == np.sign(y_pred))
            
            # In-Sample R2 (Average of rolling R2s)
            in_sample_r2_avg = metrics['r_squared'].mean()
            
            summary_list.append({
                'symbol': sym,
                'in_fold_r2_avg': in_sample_r2_avg,
                'out_of_fold_r2': oos_r2,
                'rmse': rmse,
                'mae': mae,
                'hit_rate': hit_rate
            })
            
    # Save Summary
    if summary_list:
        summary_df = pd.DataFrame(summary_list)
        summary_file = os.path.join(output_dir, "rolling_predictive_performance.csv")
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Predictive Performance Summary saved to {summary_file}")
        print("\n=== Predictive Performance (R2) ===")
        print(summary_df)


def plot_rolling_metrics(metrics, symbol, window, output_dir):
    """Generates Alpha and Beta plots."""
    
    # Drop NaNs for plotting (start of window)
    clean_metrics = metrics.dropna()
    
    if clean_metrics.empty:
        return

    # 1. Rolling Beta
    plt.figure(figsize=(10, 6))
    plt.plot(clean_metrics.index, clean_metrics['beta'], label='Rolling Beta', color='blue')
    plt.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Market (Beta=1)')
    plt.title(f"{symbol} - {window}-Day Rolling Beta")
    plt.xlabel('Date')
    plt.ylabel('Beta')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{symbol}_rolling_beta.png"))
    plt.close()
    
    # 2. Rolling Alpha
    plt.figure(figsize=(10, 6))
    # Multiply alpha by 252? RollingOLS gives alpha per period (daily). 
    # Usually we plot daily alpha or annualized. Let's stick to daily alpha level for now or specific instructions?
    # Spec didn't say annualized. Keeping raw.
    plt.plot(clean_metrics.index, clean_metrics['alpha'], label='Rolling Alpha (Daily)', color='green')
    plt.axhline(0.0, color='red', linestyle='--', alpha=0.7, label='Zero Alpha')
    plt.title(f"{symbol} - {window}-Day Rolling Alpha")
    plt.xlabel('Date')
    plt.ylabel('Alpha', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{symbol}_rolling_alpha.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Rolling Market Regression Analysis (Alpha/Beta)")
    parser.add_argument("--input_file", default="data/stock_data.csv", help="Path to input stock data CSV")
    parser.add_argument("--window", type=int, default=60, help="Rolling window size in trading days")
    parser.add_argument("--start_date", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output_dir", default="reports/rolling_regression", help="Directory to save outputs")
    
    args = parser.parse_args()
    
    # 1. Load Data
    try:
        df = load_data(args.input_file)
    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input_file}")
        return

    # 2. Calculate Returns
    df_ret = calculate_returns(df)
    
    # Filter Date Range
    if args.start_date:
        logger.info(f"Filtering data start date: {args.start_date}")
        df_ret = df_ret[df_ret['date'] >= args.start_date]
        
    if args.end_date:
        logger.info(f"Filtering data end date: {args.end_date}")
        df_ret = df_ret[df_ret['date'] <= args.end_date]
    
    # 3. Create Market Proxy
    market_returns = create_market_proxy(df_ret)
    
    # 4. Run Analysis
    run_rolling_regression(df_ret, market_returns, args.window, args.output_dir)
    
    logger.info("Rolling Regression Analysis Complete.")

if __name__ == "__main__":
    main()

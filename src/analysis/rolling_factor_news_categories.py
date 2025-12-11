
import pandas as pd
import numpy as np
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

def load_master_data(file_path, start_date, end_date):
    """Loads the master dataset and prepares category features."""
    logger.info(f"Loading master data from {file_path}...")
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)
    
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter Date Range
    mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
    df = df[mask].copy()
    
    # Feature Engineering: Log of Category Counts
    # Identify category columns
    cat_cols = [c for c in df.columns if 'total_count_' in c]
    
    for col in cat_cols:
        # Create log version
        log_col_name = f"log_{col}"
        df[log_col_name] = np.log1p(df[col])
        
    return df, cat_cols

def run_category_augmented_rolling_regression(df, window, output_dir, cat_cols):
    """Runs Rolling OLS with F-F 5 Factors + Lags + Log Category Counts."""
    
    logger.info(f"Running Category Augmented Rolling Regression (Window={window})...")
    
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
    
    # Use the Log transformed category columns
    cat_predictors = [f"log_{c}" for c in cat_cols]
    
    predictors = factors + lags + cat_predictors
    logger.info(f"Num Predictors: {len(predictors)}")
    
    for sym in symbols:
        logger.info(f"Processing {sym}...")
        
        # Align Data
        sym_data = df[df['symbol'] == sym].set_index('date').copy()
        
        # Drop rows with missing values
        sym_data = sym_data.dropna(subset=['log_ret', 'RF'] + predictors)
        
        if len(sym_data) < window:
            logger.warning(f"Insufficient data for {sym}. Skipping.")
            continue
            
        # Prepare Variables
        y = sym_data['log_ret'] - sym_data['RF']
        X = sym_data[predictors]
        X = sm.add_constant(X)
        
        # Rolling OLS
        try:
            model = RollingOLS(y, X, window=window)
            results = model.fit()
            params = results.params.copy()
            
            # --- Metrics Calculation ---
            avg_is_r2 = results.rsquared.mean()
            
            # IS Fit
            y_hat_is = (X * params).sum(axis=1)
            mse_is = ((y - y_hat_is)**2).mean()
            rmse_is = np.sqrt(mse_is)
            
            # OOS Prediction (One-Step Ahead)
            params_shifted = params.shift(1)
            y_hat_oos = (X * params_shifted).sum(axis=1)
            
            valid_idx = y_hat_oos.dropna().index
            y_true_oos = y.loc[valid_idx]
            y_pred_oos = y_hat_oos.loc[valid_idx]
            
            mse_oos = ((y_true_oos - y_pred_oos)**2).mean()
            rmse_oos = np.sqrt(mse_oos)
            
            mse_baseline = ((y_true_oos - y_true_oos.mean())**2).mean()
            r2_oos = 1 - (mse_oos / mse_baseline)
            
            logger.info(f"{sym} Category Metrics -> IS R2: {avg_is_r2:.4f}, IS RMSE: {rmse_is:.4f} | OOS R2: {r2_oos:.4f}, OOS RMSE: {rmse_oos:.4f}")
            
            metrics_summary = {
                'symbol': sym,
                'avg_is_r2': avg_is_r2,
                'is_rmse': rmse_is,
                'oos_r2': r2_oos,
                'oos_rmse': rmse_oos
            }
            metrics_df = pd.DataFrame([metrics_summary])
            metrics_csv = os.path.join(data_dir, f'{sym}_metrics.csv')
            metrics_df.to_csv(metrics_csv, index=False)
            
            if 'const' in params.columns:
                params['Alpha_Annualized'] = params['const'] * 252
            
            params['R_Squared'] = results.rsquared
            params['symbol'] = sym
            params = params.dropna()
            
            metrics_reset = params.reset_index().rename(columns={'date': 'date'})
            all_results.append(metrics_reset)
            
            # Plot only significant vars or grouping to avoid 30 plots?
            # We will plot basic stats or a heatmap if possible, but let's stick to the grid for now, 
            # maybe just the Top 5 Categories by average absolute coefficient?
            plot_category_exposure(params, sym, plots_dir, cat_predictors, metrics_summary)
            
        except Exception as e:
            logger.error(f"Error processing {sym}: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        csv_path = os.path.join(data_dir, 'rolling_category_stats.csv')
        final_df.to_csv(csv_path, index=False)
        logger.info(f"Consolidated report saved to {csv_path}")

def plot_category_exposure(metrics, symbol, output_dir, cat_predictors, performance=None):
    """Generates a summary plot for Top impact categories."""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Identify Top 6 categories by max absolute beta
    max_betas = metrics[cat_predictors].abs().max().sort_values(ascending=False)
    top_cats = max_betas.head(6).index.tolist()
    
    cols = 3
    num_plots = len(top_cats)
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 6), sharex=True)
    
    title = f"{symbol} - Top 6 News Categories Exposure\n"
    if performance:
        title += f"Out-of-Sample R²: {performance['oos_r2']:.2%} | RMSE: {performance['oos_rmse']:.4f}"
        
    fig.suptitle(title, fontsize=20, weight='bold', y=0.98)
    axes = axes.flatten() if num_plots > 1 else [axes]
    
    for i, var in enumerate(top_cats):
        ax = axes[i]
        
        # Clean name
        clean_name = var.replace('log_total_count_', '').replace('News', '').strip()
        
        ax.plot(metrics.index, metrics[var], color='#e377c2', linewidth=2) # Pink for categories
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.set_title(f"Beta: {clean_name}", fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide empty
    for j in range(num_plots, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"{symbol}_category_top6.png"), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Category Augmented Fama-French Rolling Analysis")
    parser.add_argument("--start_date", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", default="2025-10-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--window", type=int, default=60, help="Rolling window size")
    parser.add_argument("--input_file", default="data/master_analysis_data.csv", help="Path to master data")
    parser.add_argument("--output_dir", default="reports/fama_french_news_categories", help="Output directory")
    
    args = parser.parse_args()
    
    df, cat_cols = load_master_data(args.input_file, args.start_date, args.end_date)
    run_category_augmented_rolling_regression(df, args.window, args.output_dir, cat_cols)
    
    logger.info("Category Augmented Analysis Complete.")

if __name__ == "__main__":
    main()

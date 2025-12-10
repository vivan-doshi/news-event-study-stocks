
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os
import argparse

# Global configurations mirroring notebook
FEATURES = [
    'day_sentiment',
    'sentiment_finbert_CEO & Influencer News',
    'sentiment_finbert_Corporate Earnings & Financials',
    'sentiment_finbert_Corporate Strategy & Growth',
    'sentiment_finbert_Economic Indicators & Fed Policy',
    'sentiment_finbert_General Business & News Reporting',
    'sentiment_finbert_Geopolitical & Trade News',
    'sentiment_finbert_Innovation & Future Tech',
    'sentiment_finbert_Investment & Hedge Fund Activity',
    'sentiment_finbert_Investment Analysis & Strategy',
    'sentiment_finbert_Layoffs & Corporate Restructuring',
    'sentiment_finbert_Legal & Regulatory Changes',
    'sentiment_finbert_Market Movements & Trading News',
    'sentiment_finbert_Mergers, Acquisitions & Deals',
    'sentiment_finbert_Product News & Updates',
    'sentiment_finbert_Semiconductor & Chip Industry News',
    # Enhanced Features
    'day_sentiment_lag1',
    'day_sentiment_lag2',
    'day_sentiment_lag3',
    'interaction_sentiment_volume'
]

TARGET = 'target_return_next_day' 
# Wait, checking inspect output: "target = 'target_day_end_raw_close_next_day'" in Cell 44.
# But Cell 2 output showed 'ret_log_1d'.
# Let's verify the target variable from the inspection output for OLS.
# Cell 39: "Uses global avg_sentiment_df, features, and target."
# I need to confirm what `target` was set to.
# In Cell 44 it says 'target = "target_day_end_raw_close_next_day"'.
# However, standard event studies use returns.
# Let's check `run_ols_for_symbol` usage in Cell 41.
# It doesn't explicitly set target in the function call, so it uses the global one.
# I will make target configurable but default to 'ret_log_1d' or widely used metric if not specified, 
# but the notebook snippet suggests 'target_day_end_raw_close_next_day' might be a specific constructed target.
# Actually, looking at Cell 41: "actual_next_day_close": y_test.values.
# This strongly implies the target is a PRICE, not a return.
# "target_day_end_raw_close_next_day" isn't in the initial columns of Cell 2.
# It must be created. I missed the target creation in feature engineering or notebook.
# Let's check feature engineering script I wrote. I didn't create a 'target_*' column.
# I need to add target creation to `prepare_symbol_data` or `feature_engineering.py`.
# Since `feature_engineering.py` is "common", and this target might be specific.
# But the inspection showed "Uses global ... target".
# Let's double check if I can find where`target` is defined or created.
# If it's next day close, I can create it on the fly.

def ensure_target_exists(df, target_col):
    if target_col in df.columns:
        return df
    
    # Debug
    # print(f"Target {target_col} not found. Available cols: {df.columns.tolist()}")

    # Try to construct it if it looks like a lag/lead
    if target_col == 'target_return_next_day':
        # Shift -1 on ret_log_1d (standard log returns)
        if 'ret_log_1d' in df.columns:
             df[target_col] = df['ret_log_1d'].shift(-1)
        elif 'ret_1d' in df.columns:
             df[target_col] = df['ret_1d'].shift(-1)
        else:
             print(f"Error: Could not find return source column for {target_col}")
        return df
    
    # Legacy support or fallback
    if target_col == 'target_day_end_raw_close_next_day':
        # Prioritize adj_close as verified in data
        if 'adj_close' in df.columns:
             df[target_col] = df['adj_close'].shift(-1)
        elif 'day_end_raw_close' in df.columns:
             df[target_col] = df['day_end_raw_close'].shift(-1)
        elif 'day_end_value' in df.columns:
             df[target_col] = df['day_end_value'].shift(-1)
        else:
             print(f"Error: Could not find source column for {target_col}")
        return df
    return df

def prepare_symbol_data(df, symbol, features, target, cutoff_date='2025-06-01'):
    """
    Mirror notebook logic for data split.
    """
    df_sym = df.loc[df['symbol_query'] == symbol, :].copy()
    df_sym = df_sym.sort_values('final_date_for_news') # Script output uses this name
    
    # Ensure target exists
    df_sym = ensure_target_exists(df_sym, target)

    # Drop rows where target or any feature is NaN
    # Check if features exist
    missing_feats = [f for f in features if f not in df_sym.columns]
    if missing_feats:
        print(f"Warning: Missing features in data: {missing_feats}")
        # Try to continue with available? Or fail.
        # Notebook features list is hardcoded.
        # Let's filter to available.
        features = [f for f in features if f in df_sym.columns]
    
    valid_cols = features + [target]
    df_sym = df_sym.dropna(subset=valid_cols)

    # Split
    # Date column in df is 'final_date_for_news' (string YYYY-MM-DD from my script)
    df_train = df_sym.loc[df_sym['final_date_for_news'] < cutoff_date, :].copy()
    df_test  = df_sym.loc[df_sym['final_date_for_news'] >= cutoff_date, :].copy()

    X_train = df_train[features]
    y_train = df_train[target]

    X_test  = df_test[features]
    y_test  = df_test[target]

    return X_train, y_train, X_test, y_test, df_sym, features

def run_ols_for_symbol(df, symbol, output_dir, target_col='target_day_end_raw_close_next_day'):
    
    # Create subdirs
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Running OLS for {symbol}...")
    
    X_train, y_train, X_test, y_test, df_sym, used_features = prepare_symbol_data(df, symbol, FEATURES, target_col)
    
    if X_train.empty:
        print(f"No training data for {symbol}. Skipping.")
        return None

    # Fit model
    X_train_const = sm.add_constant(X_train)
    # Check for NaNs/Infs again just in case statsmodels complains
    if not np.isfinite(X_train_const.values).all() or not np.isfinite(y_train.values).all():
         print(f"Data contains NaNs or Infs for {symbol}. Skipping.")
         return None
         
    model = sm.OLS(y_train, X_train_const)
    results = model.fit()

    # Predict
    X_test_const = sm.add_constant(X_test)
    if X_test_const.empty:
         print(f"No test data for {symbol}.")
         y_pred = []
         residuals = []
    else:
        # Handle case where X_test might be missing const if length 0, handled by empty check
        # Ensure const is present even if 1 row
        if 'const' not in X_test_const.columns:
            X_test_const['const'] = 1.0
            
        y_pred = results.predict(X_test_const)
        residuals = y_test - y_pred

    # Plots
    # 1. Residuals
    if len(y_pred) > 0:
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'{symbol} – Residuals vs. Predicted Values (OLS)')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, f'{symbol}_residuals.png'))
        plt.close()

        # 2. Actual vs Predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5, label='Observed points')
        
        # Regression line for plot
        if len(y_test) > 1:
            coeffs = np.polyfit(y_test, y_pred, 1)
            slope, intercept = coeffs[0], coeffs[1]
            x_line = np.linspace(y_test.min(), y_test.max(), 100)
            y_line = intercept + slope * x_line
            plt.plot(x_line, y_line, linestyle='--', color='red', label=f'Fit: y={intercept:.2f}+{slope:.2f}x')
        
        plt.xlabel(f'Actual {target_col}')
        plt.ylabel(f'Predicted {target_col}')
        plt.title(f'{symbol} – Actual vs Predicted')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, f'{symbol}_actual_vs_pred.png'))
        plt.close()

    # Calculate Out-of-Fold R2
    oos_r2 = np.nan
    if len(y_test) > 0 and len(y_pred) > 0:
        oos_r2 = r2_score(y_test, y_pred)

    return {
        'symbol': symbol,
        'n_obs': results.nobs,
        'r2_in_fold': results.rsquared,      # In-sample R2
        'r2_out_of_fold': oos_r2,           # Out-of-sample R2
        'adj_r2': results.rsquared_adj,
        'sigma_resid': np.sqrt(results.mse_resid) if results.mse_resid > 0 else 0,
        'params': results.params.to_dict(),
        'pvalues': results.pvalues.to_dict()
    }

def event_study_main(data_path, output_dir, symbols=None, target=None):
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    if symbols is None:
        # Default to Mag7 from notebook
        symbols = ['AAPL.US', 'MSFT.US', 'GOOGL.US', 'AMZN.US', 'META.US', 'NVDA.US', 'TSLA.US']
    else:
        symbols = symbols.split(',')
        
    if target is None:
        target = TARGET # Use global default

    results_list = []
    
    for sym in symbols:
        # Handle symbol aliases if necessary (e.g. GOOG vs GOOGL)
        # Notebook used GOOGL.US in list but filename was google_mag7. 
        # Check data for available symbols
        if sym not in df['symbol_query'].unique():
            print(f"Symbol {sym} not found in data. Available: {df['symbol_query'].unique()}")
            continue
            
        res = run_ols_for_symbol(df, sym, output_dir, target)
        if res:
            results_list.append(res)
            
    # Save summary
    if results_list:
        summary_rows = []
        for r in results_list:
            row = {k: v for k, v in r.items() if k not in ['params', 'pvalues']}
            summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        save_path = os.path.join(output_dir, 'tables', 'ols_summary.csv')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        summary_df.to_csv(save_path, index=False)
        print(f"Summary saved to {save_path}")
        print(summary_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--symbols", help="Comma-separated list of symbols")
    parser.add_argument("--target", help="Target column name")
    
    args = parser.parse_args()
    
    event_study_main(args.data_path, args.output_dir, args.symbols, args.target)

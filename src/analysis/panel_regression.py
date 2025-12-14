
import pandas as pd
import numpy as np
import statsmodels.api as sm
import argparse
import logging
import os
import sys
from tqdm import tqdm

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def run_rolling_panel_regression(df, window, output_dir):
    """
    Runs Rolling Panel Regression (Pooled OLS with Fixed Effects).
    """
    
    # Define Models
    base_vars = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 
                 'log_ret_lag1', 'log_ret_lag2', 'log_ret_lag5', 'log_ret_lag10', 'log_ret_lag21']
    
    # Identify dummy columns
    dummy_cols = [c for c in df.columns if c.startswith('dummy_')]
    
    # 3 Model Configurations
    models = {
        'Panel_Baseline': base_vars + dummy_cols,
        'Panel_News': base_vars + dummy_cols + ['log_total_news', 'log_total_news_lag1'],
        'Panel_Sentiment': base_vars + dummy_cols + ['day_sentiment', 'day_sentiment_lag1'],
        'Panel_Thematic_Shocks': base_vars + dummy_cols + ['sent_growth_zscore_lag1', 'sent_risk_zscore_lag1', 'sent_macro_zscore_lag1', 'sent_earnings_zscore_lag1'],
        'Panel_Risk_Shock': base_vars + dummy_cols + ['sent_risk_zscore_lag1'],
        'Panel_Signal_Conviction': base_vars + dummy_cols + ['day_sentiment_magnitude_lag1'] + ['log_total_news', 'log_total_news_lag1'] # Include vol control? Maybe just interaction? 
        # Interaction term usually requires main effects too. 
        # So: Sentiment + Vol + Sentiment*Vol. 
        # Let's add main effects: 'day_sentiment_lag1', 'log_total_news_lag1' + 'day_sentiment_magnitude_lag1'
    }
    # Update Signal Conviction to be statistically sound (Main Effects + Interaction)
    models['Panel_Signal_Conviction'] = base_vars + dummy_cols + ['day_sentiment_lag1', 'log_total_news_lag1', 'day_sentiment_magnitude_lag1']
    
    # Get unique dates sorted
    dates = sorted(df['date'].unique())
    num_dates = len(dates)
    
    if num_dates <= window:
        logger.error("Not enough dates for the specified window.")
        return

    # Prepare results container
    all_predictions = [] 
    all_params = [] # Store coefficients
    
    logger.info(f"Starting Rolling Panel Regression (Window={window} days)...")
    
    start_idx = window 
    
    for i in tqdm(range(start_idx, num_dates)):
        target_date = dates[i]
        
        # Define window range (exclusive of target_date)
        window_start_date = dates[i - window]
        train_mask = (df['date'] >= window_start_date) & (df['date'] < target_date)
        
        train_df = df[train_mask]
        test_df = df[df['date'] == target_date]
        
        if train_df.empty or test_df.empty:
            if train_df.empty and test_df.empty:
                 pass # skip silent for common case
            else:
                 # logger.debug(f"Skipping {target_date}: Train {len(train_df)}, Test {len(test_df)}")
                 pass
            continue
            
        for model_name, predictors in models.items():
            # Drop clean
            train_sub = train_df.dropna(subset=['log_ret', 'RF'] + predictors)
            
            if len(train_sub) < 50: 
                # logger.debug(f"Skipping {target_date} {model_name}: Insufficient train data {len(train_sub)}")
                continue
                
            y_train = (train_sub['log_ret'] - train_sub['RF']).astype(float)
            X_train = sm.add_constant(train_sub[predictors].astype(float))
            
            try:
                # Fit Model
                model = sm.OLS(y_train, X_train).fit()
                
                # STORE COEFFICIENTS
                params = model.params.to_frame().T
                params['model'] = model_name
                params['date'] = target_date # The date these params are valid for predicting
                all_params.append(params)
                
                # Predict
                test_sub = test_df.dropna(subset=['log_ret', 'RF'] + predictors)
                if test_sub.empty:
                    # logger.debug(f"Skipping {target_date} {model_name}: Empty test set")
                    continue
                
                # Ensure X_test aligns with params
                X_test = sm.add_constant(test_sub[predictors].astype(float), has_constant='add')
                
                y_pred = model.predict(X_test)
                
                # Capture Results
                res_df = pd.DataFrame({
                    'date': test_sub['date'],
                    'symbol': test_sub['symbol'],
                    'model': model_name,
                    'y_true': test_sub['log_ret'] - test_sub['RF'],
                    'y_pred': y_pred
                })
                
                all_predictions.append(res_df)
                
            except Exception as e:
                logger.error(f"Error in {model_name} at {target_date}: {e}")
                pass

    # Post-Process Results
    save_results(all_predictions, all_params, output_dir)

def save_results(predictions_list, params_list, output_dir):
    """Calculates Metrics and Saves Panel Results + Coefficients."""
    os.makedirs(output_dir, exist_ok=True)
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save Predictions
    if predictions_list:
        full_preds = pd.concat(predictions_list, ignore_index=True)
        full_preds.to_csv(os.path.join(data_dir, 'panel_predictions.csv'), index=False)
        
        # Calculate Metrics
        metrics = []
        symbols = full_preds['symbol'].unique()
        models = full_preds['model'].unique()
        
        for sym in symbols:
            for mod in models:
                sub = full_preds[(full_preds['symbol'] == sym) & (full_preds['model'] == mod)]
                if sub.empty:
                    continue
                mse_oos = ((sub['y_true'] - sub['y_pred'])**2).mean()
                mse_base = ((sub['y_true'] - sub['y_true'].mean())**2).mean()
                r2_oos = 1 - (mse_oos / mse_base)
                rmse_oos = np.sqrt(mse_oos)
                metrics.append({'symbol': sym, 'model': mod, 'oos_r2': r2_oos, 'oos_rmse': rmse_oos})
                
        pd.DataFrame(metrics).to_csv(os.path.join(data_dir, 'panel_metrics.csv'), index=False)
        logger.info(f"Metrics saved.")
    else:
        logger.warning("No predictions generated.")

    # Save Coefficients
    if params_list:
        full_params = pd.concat(params_list, ignore_index=True)
        # Reorder to put date/model first
        cols = ['date', 'model'] + [c for c in full_params.columns if c not in ['date', 'model']]
        full_params = full_params[cols]
        full_params.to_csv(os.path.join(data_dir, 'rolling_coefficients.csv'), index=False)
        logger.info(f"Rolling coefficients saved to {os.path.join(data_dir, 'rolling_coefficients.csv')}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--output_dir", default="reports/panel_regression")
    parser.add_argument("--input_file", default="data/master_analysis_data.csv")
    args = parser.parse_args()
    
    # Load raw
    raw = pd.read_csv(args.input_file)
    raw['date'] = pd.to_datetime(raw['date'])
    raw = raw[raw['date'] >= '2023-01-01'] 
    
    # Create dummies
    # Note: 'symbol' is needed for tracking. get_dummies removes it.
    df_dummies = pd.get_dummies(raw, columns=['symbol'], prefix='dummy', drop_first=True)
    df_dummies['symbol'] = raw['symbol'] # Re-attach symbol
    
    # Feature Engineering (Apply Logs)
    if 'total_news' in df_dummies.columns:
        df_dummies['log_total_news'] = np.log1p(df_dummies['total_news'])
        # Ensure lag exists if not present
        if 'log_total_news_lag1' not in df_dummies.columns:
             df_dummies['log_total_news_lag1'] = df_dummies.groupby('symbol')['log_total_news'].shift(1)
             
    # Ensure sentiment lags exist
    if 'day_sentiment_lag1' not in df_dummies.columns and 'day_sentiment' in df_dummies.columns:
        df_dummies['day_sentiment_lag1'] = df_dummies.groupby('symbol')['day_sentiment'].shift(1)
        
    # --- FEATURE ENGINEERING FOR NEW COLUMNS ---
    
    # 1. Z-Scores (Shocks) Lags
    shock_cols = [c for c in df_dummies.columns if c.endswith('_zscore')]
    for col in shock_cols:
        lag_col = f"{col}_lag1"
        if lag_col not in df_dummies.columns:
            df_dummies[lag_col] = df_dummies.groupby('symbol')[col].shift(1)
            
    # 2. Magnitude (Signal * Vol) Lags
    mag_cols = [c for c in df_dummies.columns if c.endswith('_magnitude')]
    for col in mag_cols:
        lag_col = f"{col}_lag1"
        if lag_col not in df_dummies.columns:
             df_dummies[lag_col] = df_dummies.groupby('symbol')[col].shift(1)
             
    # --- MODEL DEFINITIONS ---
    base_vars = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 
                 'log_ret_lag1', 'log_ret_lag2', 'log_ret_lag5', 'log_ret_lag10', 'log_ret_lag21']
    dummy_cols = [c for c in df_dummies.columns if c.startswith('dummy_')]
    
    # Define Feature Sets
    # Thematic Shocks: growth, risk, macro, earnings
    thematic_shock_vars = ['sent_growth_zscore_lag1', 'sent_risk_zscore_lag1', 'sent_macro_zscore_lag1', 'sent_earnings_zscore_lag1']
    # Risk Shock Only
    risk_shock_vars = ['sent_risk_zscore_lag1']
    # Signal Conviction (Magnitude)
    conviction_vars = ['day_sentiment_magnitude_lag1'] # Global Sentiment * Vol
    
    # Update Models Dictionary in run_rolling function by passing new df
    # But run_rolling defines models internally. We need to pass these extra vars or update run_rolling.
    # EASIER: We will update run_rolling to accept 'extra_models' or just hardcode them there.
    # Let's update run_rolling signature to match or just move logic there.
    # Actually, let's just pass the FULL df_dummies to run_rolling and update the valid columns list there.
    
    run_rolling_panel_regression(df_dummies, args.window, args.output_dir)

# We need to update run_rolling_panel_regression definition too.
# Let's do it in one go. But since I can only replace a block, I will replace main() here 
# and then replace run_rolling separately. Actually, run_rolling has hardcoded models.
# I need to update run_rolling primarily.

    run_rolling_panel_regression(df_dummies, args.window, args.output_dir)

if __name__ == "__main__":
    main()

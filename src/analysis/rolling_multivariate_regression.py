
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import argparse
import logging
import os
import json

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- DEEP DIVE MODEL SPECIFICATIONS (20+ Models) ---
# Features
NEWS_CNT = ['news_volume']
NEWS_SENT = ['avg_sentiment']
NEWS_INTERACT = ['interaction_term']
NEWS_SHOCK = ['z_score_sentiment']
TOPIC_DEV = ['sent_topic_0', 'sent_topic_1', 'sent_topic_2', 'sent_topic_3', 'sent_topic_4']
TOPIC_INTERACT = ['interaction_topic_0', 'interaction_topic_1', 'interaction_topic_2', 'interaction_topic_3', 'interaction_topic_4']
TOPIC_SHOCK = ['z_score_topic_0', 'z_score_topic_1', 'z_score_topic_2', 'z_score_topic_3', 'z_score_topic_4']

# Factors
CAPM = ['Mkt-RF']
FF3 = ['Mkt-RF', 'SMB', 'HML']
FF5 = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

MODELS = {
    # --- Baselines ---
    '01_CAPM': CAPM,
    '02_FF3': FF3,
    '03_FF5': FF5,
    
    # --- CAPM Variants ---
    '04_CAPM_NewsCount': CAPM + NEWS_CNT,
    '05_CAPM_Sentiment': CAPM + NEWS_SENT,
    '06_CAPM_Interaction': CAPM + NEWS_INTERACT,
    '07_CAPM_Shock': CAPM + NEWS_SHOCK,
    '08_CAPM_Topics': CAPM + TOPIC_DEV,
    '09_CAPM_TopicInteract': CAPM + TOPIC_INTERACT,
    
    # --- FF3 Variants ---
    '10_FF3_NewsCount': FF3 + NEWS_CNT,
    '11_FF3_Sentiment': FF3 + NEWS_SENT,
    '12_FF3_Interaction': FF3 + NEWS_INTERACT,
    '13_FF3_Shock': FF3 + NEWS_SHOCK,
    '14_FF3_Topics': FF3 + TOPIC_DEV,
    '15_FF3_TopicInteract': FF3 + TOPIC_INTERACT,
    
    # --- FF5 Variants ---
    '16_FF5_NewsCount': FF5 + NEWS_CNT,
    '17_FF5_Sentiment': FF5 + NEWS_SENT,
    '18_FF5_Interaction': FF5 + NEWS_INTERACT,
    '19_FF5_Shock': FF5 + NEWS_SHOCK,
    '20_FF5_Topics': FF5 + TOPIC_DEV,
    '21_FF5_TopicInteract': FF5 + TOPIC_INTERACT,
    
    # --- Topic Shocks ---
    '22_FF3_TopicShock': FF3 + TOPIC_SHOCK,
    '23_FF5_TopicShock': FF5 + TOPIC_SHOCK,
    
    # --- Hybrid ---
    '24_FF3_TopicShock_VolInteract': FF3 + TOPIC_SHOCK + NEWS_INTERACT
}

# Add Lasso manually in the loop logic, defined by ALL features
ALL_FEATURES = list(set(FF5 + NEWS_CNT + NEWS_SENT + NEWS_INTERACT + NEWS_SHOCK + TOPIC_DEV + TOPIC_INTERACT + TOPIC_SHOCK))

def load_data(file_path):
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def calculate_metrics(y_true, y_pred, model_name):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # R2
    r2 = r2_score(y_true, y_pred)
    
    # Hit Rate (Directional Accuracy)
    # Using sign of return. If return is 0, arguably sign is 0.
    hit_rate = np.mean(np.sign(y_true) == np.sign(y_pred))
    
    # --- Trading Strategy Metrics ---
    # Position: derived from PREDICTION of T+1 Return
    position = np.sign(y_pred)
    
    # Strategy Return: Position(at close T) * Return(T+1)
    # Note: Transaction costs ignored.
    strategy_ret = position * y_true
    
    # Cumulative Return (for DD)
    cum_ret = np.cumprod(1 + strategy_ret)
    
    # 1. Sharpe Ratio
    mean_ret = np.mean(strategy_ret)
    std_ret = np.std(strategy_ret)
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
    
    # 2. Sortino Ratio
    downside_returns = strategy_ret[strategy_ret < 0]
    downside_std = np.std(downside_returns)
    sortino = (mean_ret / downside_std) * np.sqrt(252) if downside_std > 0 else 0
    
    # 3. Max Drawdown
    running_max = np.maximum.accumulate(cum_ret)
    drawdown = (cum_ret / running_max) - 1
    max_drawdown = np.min(drawdown)
    
    # 4. Profit Factor
    gross_profit = np.sum(strategy_ret[strategy_ret > 0])
    gross_loss = np.abs(np.sum(strategy_ret[strategy_ret < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    return {
        'model': model_name,
        'oos_r2': r2,
        'hit_rate': hit_rate,
        'sharpe': sharpe,
        'sortino': sortino,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown
    }

def run_rolling_analysis(df, window=120, output_dir='reports/regression_t1'):
    os.makedirs(output_dir, exist_ok=True)
    summary_results = []
    
    symbols = df['symbol'].unique()
    
    # Pre-calculate excess returns
    if 'excess_ret' not in df.columns:
        df['excess_ret'] = df['daily_return'] - df['RF']
    
    # --- FIX 1: SHIFT TARGET FOR PREDICTIVE ACCURACY (T+1) ---
    # We want X_t to predict Y_{t+1}.
    # So we create a column 'target' which is 'excess_ret' shifted by -1 within each symbol group.
    df = df.sort_values(['symbol', 'date'])
    df['target_t1'] = df.groupby('symbol')['excess_ret'].shift(-1)
    
    # Verify shift (debug print)
    # temp = df[df['symbol'] == 'AAPL'][['date', 'excess_ret', 'target_t1']].head()
    # print(temp)
    
    for sym in symbols:
        logger.info(f"Processing {sym} (Target: T+1 Return)...")
        sym_df = df[df['symbol'] == sym].copy().set_index('date').sort_index()
        
        # Define superset of all needed columns + target
        all_cols_needed = list(set(ALL_FEATURES)) + ['target_t1']
        
        # Drop NaN (Last day will be dropped because it has no T+1 target)
        sym_df = sym_df.dropna(subset=all_cols_needed)
        
        if len(sym_df) < window + 20:
            logger.warning(f"Not enough data for {sym}. Skipping.")
            continue

        # Common Test Index (OOS period)
        # We start predicting from index = window
        test_index = sym_df.index[window:]
        
        # --- 1. RUN STANDARD MODELS (OLS) ---
        for model_name, features in MODELS.items():
            if not all(f in sym_df.columns for f in features):
                continue
                
            # RollingOls from statsmodels
            # X comes from T, y comes from T+1
            y = sym_df['target_t1']
            X = sm.add_constant(sym_df[features])
            
            try:
                # RollingOLS fits on (y, X) windows.
                # ideally, at time t, it uses window [t-W : t] to fit, and provides params for t.
                # However, RollingOLS.fit() returns params aligned with the *end* of the window.
                # So params at index t are based on data up to t.
                # We use these params to predict y_{t+1} (which is aligned at index t in our shifted df? No.)
                
                # Careful: We have shifted "target_t1". 
                # Row T contains: X_t and Y_{t+1}.
                # If we fit OLS on this row, we are learning the relationship X_t -> Y_{t+1}.
                # This is correct.
                # So we can just use RollingOLS on the shifted dataframe directly.
                # Params at index T will capture relation X_{t-w}...X_t  ->  Y_{t-w+1}...Y_{t+1}.
                # Then we use Params_T * X_{T_{next}} to predict??? NO.
                
                # Wait. In a true walk-forward:
                # To predict Y_{T+1}, we must use a model trained on data strictly BEFORE T+1.
                # We can use data up to T (pairs of X_{t-k}, Y_{t-k+1}).
                # So yes, fitting on the shifted dataframe is correct. The coefficients `params[T]` 
                # minimize error for X_t -> Y_{t+1}.
                
                # BUT, to predict the *next* step (Test set), we take coefficients trained up to T, 
                # and apply them to X_{T+1} to predict Y_{T+2}.
                # Effectively, we use `params.shift(1)` * `X`.
                
                rolling_model = RollingOLS(y, X, window=window)
                results = rolling_model.fit()
                
                # Shift params by 1 to use "yesterday's model" on "today's features"
                # to predict "tomorrow's return" (which is in today's target column)
                pred_params = results.params.shift(1).dropna()
                
                # Align
                common_idx = pred_params.index.intersection(X.index)
                pred_params = pred_params.loc[common_idx]
                X_curr = X.loc[common_idx]
                y_curr = y.loc[common_idx]
                
                y_pred = (X_curr * pred_params).sum(axis=1) # Dot product
                
                metrics = calculate_metrics(y_curr, y_pred, model_name)
                metrics['symbol'] = sym
                summary_results.append(metrics)
                
            except Exception as e:
                logger.error(f"Error {model_name} {sym}: {e}")
                continue

        # --- 2. RUN LASSO (Regularized, Rolling) ---
        # RollingOLS doesn't support Lasso. We loop manually.
        # This is slower but necessary.
        model_name = "25_Lasso_All"
        features = ALL_FEATURES
        
        # Prepare Data
        X_lasso = sym_df[features]
        y_lasso = sym_df['target_t1']
        
        preds_lasso = []
        actuals_lasso = []
        dates_lasso = []
        
        # Simple rolling loop
        # Start at 'window'
        # Train on [i-window : i]
        # Predict on [i]
        
        scaler = StandardScaler()
        
        # Optimization: Don't retrain every single day if slow. 
        # But for 250 days * 7 stocks it's fine.
        
        indices = range(window, len(sym_df))
        
        # If too slow, we can step.
        for i in indices:
            # Training Window
            X_train = X_lasso.iloc[i-window:i]
            y_train = y_lasso.iloc[i-window:i]
            
            # Test Point (Current Day T, predicting T+1)
            X_test = X_lasso.iloc[[i]] # 2D array
            y_true_test = y_lasso.iloc[i]
            
            # Scale
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Fit Lasso
            # Alpha 0.001 is a reasonable guess for financial returns (noisy)
            lasso = Lasso(alpha=0.0005, random_state=42) 
            lasso.fit(X_train_scaled, y_train)
            
            pred = lasso.predict(X_test_scaled)[0]
            
            preds_lasso.append(pred)
            actuals_lasso.append(y_true_test)
            dates_lasso.append(sym_df.index[i])
            
        # Calc Metrics
        if len(preds_lasso) > 0:
            metrics = calculate_metrics(actuals_lasso, preds_lasso, model_name)
            metrics['symbol'] = sym
            summary_results.append(metrics)
            
    # --- AGGREGATE ---
    summary_df = pd.DataFrame(summary_results)
    if not summary_df.empty:
        summary_path = os.path.join(output_dir, 'deep_dive_model_performance.csv')
        summary_df.to_csv(summary_path, index=False)
        
        # Average across tickers
        avg_metrics = summary_df.groupby('model')[['oos_r2', 'hit_rate', 'sharpe', 'sortino', 'profit_factor', 'max_drawdown']].mean()
        
        print("\n=== CORRECTED (T+1) Model Comparison (Average) ===")
        print(avg_metrics.sort_values('sharpe', ascending=False))
        
        avg_path = os.path.join(output_dir, 'deep_dive_benchmark_avg.csv')
        avg_metrics.to_csv(avg_path)
    else:
        logger.warning("No results generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="data/master_analysis_data.csv")
    parser.add_argument("--window", type=int, default=120)
    args = parser.parse_args()
    
    df = load_data(args.input_path)
    run_rolling_analysis(df, args.window)

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from sklearn.metrics import r2_score, mean_squared_error
import argparse
import logging
import os
import json

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    '20_FF5_Topics': FF5 + TOPIC_DEV,
    '21_FF5_TopicInteract': FF5 + TOPIC_INTERACT,
    
    # --- Topic Shocks ---
    '22_FF3_TopicShock': FF3 + TOPIC_SHOCK,
    '23_FF5_TopicShock': FF5 + TOPIC_SHOCK,
    
    # --- Hybrid ---
    '24_FF3_TopicShock_VolInteract': FF3 + TOPIC_SHOCK + NEWS_INTERACT
}

def load_data(file_path):
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def calculate_metrics(y_true, y_pred, model_name):
    # R2
    r2 = r2_score(y_true, y_pred)
    
    # Hit Rate
    hit_rate = np.mean(np.sign(y_true) == np.sign(y_pred))
    
    # --- Trading Strategy Metrics ---
    position = np.sign(y_pred)
    strategy_ret = position * y_true
    
    # Cumulative Return (for DD)
    cum_ret = (1 + strategy_ret).cumprod()
    
    # 1. Sharpe Ratio
    mean_ret = strategy_ret.mean()
    std_ret = strategy_ret.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
    
    # 2. Sortino Ratio
    downside_returns = strategy_ret[strategy_ret < 0]
    downside_std = downside_returns.std()
    sortino = (mean_ret / downside_std) * np.sqrt(252) if downside_std > 0 else 0
    
    # 3. Max Drawdown
    running_max = cum_ret.cummax()
    drawdown = (cum_ret / running_max) - 1
    max_drawdown = drawdown.min()
    
    # 4. Recovery Factor
    total_return = cum_ret.iloc[-1] - 1
    recovery_factor = abs(total_return / max_drawdown) if max_drawdown < 0 else 0
    
    # 5. Risk/Reward
    winning_days = strategy_ret[strategy_ret > 0]
    losing_days = strategy_ret[strategy_ret < 0]
    avg_win = winning_days.mean() if len(winning_days) > 0 else 0
    avg_loss = abs(losing_days.mean()) if len(losing_days) > 0 else 1e-9
    risk_reward = avg_win / avg_loss
    
    # 6. Profit Factor (Gross Profit / Gross Loss)
    gross_profit = strategy_ret[strategy_ret > 0].sum()
    gross_loss = abs(strategy_ret[strategy_ret < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    return {
        'model': model_name,
        'oos_r2': r2,
        'hit_rate': hit_rate,
        'sharpe': sharpe,
        'sortino': sortino,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'recovery_factor': recovery_factor,
        'risk_reward': risk_reward
    }

def run_rolling_analysis(df, window=120, output_dir='reports/regression'):
    os.makedirs(output_dir, exist_ok=True)
    summary_results = []
    
    symbols = df['symbol'].unique()
    
    # Pre-calculate excess returns if not present
    if 'excess_ret' not in df.columns:
        df['excess_ret'] = df['daily_return'] - df['RF']

    for sym in symbols:
        logger.info(f"Processing {sym}...")
        sym_df = df[df['symbol'] == sym].copy().set_index('date').sort_index()
        
        # Define superset of all needed columns
        all_features = set()
        for m in MODELS.values():
            all_features.update(m)
        all_cols = list(all_features) + ['excess_ret']
        
        # Drop NaN
        sym_df = sym_df.dropna(subset=all_cols)
        
        if len(sym_df) < window + 10:
            logger.warning(f"Not enough data for {sym}. Skipping.")
            continue

        for model_name, features in MODELS.items():
            if not all(f in sym_df.columns for f in features):
                # Check specifics
                missing = [f for f in features if f not in sym_df.columns]
                logger.warning(f"Missing features {missing} for {model_name} in {sym}. Skipping.")
                continue
                
            y = sym_df['excess_ret']
            X = sm.add_constant(sym_df[features])
            
            # Check for constant columns (e.g. topic count 0)
            # RollingOLS might fail or produce bad results if X is singleton.
            # But statsmodels RollingOLS is usually robust-ish.
            
            try:
                rolling_model = RollingOLS(y, X, window=window)
                results = rolling_model.fit()
                
                # Prediction
                pred_params = results.params.shift(1).dropna()
                
                common_idx = pred_params.index.intersection(X.index)
                if len(common_idx) == 0:
                    continue
                    
                pred_params = pred_params.loc[common_idx]
                X_curr = X.loc[common_idx]
                y_curr = y.loc[common_idx]
                
                y_pred = (X_curr * pred_params).sum(axis=1)
                
                metrics = calculate_metrics(y_curr, y_pred, model_name)
                metrics['symbol'] = sym
                summary_results.append(metrics)
                
            except Exception as e:
                logger.error(f"Error running {model_name} for {sym}: {e}")
                continue
            
    # Aggregate
    summary_df = pd.DataFrame(summary_results)
    if not summary_df.empty:
        summary_path = os.path.join(output_dir, 'deep_dive_model_performance.csv')
        summary_df.to_csv(summary_path, index=False)
        
        # Average
        avg_metrics = summary_df.groupby('model')[['oos_r2', 'hit_rate', 'sharpe', 'sortino', 'profit_factor', 'max_drawdown', 'recovery_factor', 'risk_reward']].mean()
        print("\n=== Deep Dive Model Comparison (Average) ===")
        print(avg_metrics.sort_values('sharpe', ascending=False)) # Rank by Sharpe
        
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

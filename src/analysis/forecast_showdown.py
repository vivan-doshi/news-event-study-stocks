import pandas as pd
import numpy as np
import os
import logging
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from sklearn.metrics import r2_score

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FORECAST SHOWDOWN MODELS ---
FF3 = ['Mkt-RF', 'SMB', 'HML']
FF5 = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

# Features
SHOCK = ['z_score_sentiment']
VOL_INTERACT = ['interact_Vol_Sent']
TOPIC_SHOCK = ['z_score_topic_0', 'z_score_topic_1', 'z_score_topic_2', 'z_score_topic_3', 'z_score_topic_4']

MODELS = {
    '00_Benchmark_FF5_T1': FF5,
    '01_FF3_Shock_T1': FF3 + SHOCK,
    '02_FF3_VolInteract_T1': FF3 + VOL_INTERACT,
    '03_FF3_Combined_T1': FF3 + SHOCK + VOL_INTERACT,
    '04_FF3_TopicShock_T1': FF3 + TOPIC_SHOCK,
    '05_FF3_TopicShock_VolInteract_T1': FF3 + TOPIC_SHOCK + VOL_INTERACT
}

def load_data(file_path):
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def calculate_metrics(y_true, y_pred, model_name):
    # R2
    r2 = r2_score(y_true, y_pred)
    
    # Hit Rate (Directional Accuracy)
    hit_rate = np.mean(np.sign(y_true) == np.sign(y_pred))
    
    # Trading Strategy
    position = np.sign(y_pred)
    strategy_ret = position * y_true
    
    # Cumulative & Stats
    cum_ret = (1 + strategy_ret).cumprod()
    mean_ret = strategy_ret.mean()
    std_ret = strategy_ret.std() if len(strategy_ret) > 0 else 0
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
    
    downside_returns = strategy_ret[strategy_ret < 0]
    downside_std = downside_returns.std()
    sortino = (mean_ret / downside_std) * np.sqrt(252) if downside_std > 0 else 0
    
    running_max = cum_ret.cummax()
    drawdown = (cum_ret / running_max) - 1
    max_drawdown = drawdown.min()
    
    gross_profit = strategy_ret[strategy_ret > 0].sum()
    gross_loss = abs(strategy_ret[strategy_ret < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    total_return = cum_ret.iloc[-1] - 1
    recovery_factor = abs(total_return / max_drawdown) if max_drawdown < 0 else 0
    
    return {
        'model': model_name,
        'oos_r2': r2,
        'hit_rate': hit_rate,
        'sharpe': sharpe,
        'sortino': sortino,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'recovery_factor': recovery_factor
    }

def run_forecast_showdown(df, window=120, output_dir='reports/forecast_showdown'):
    os.makedirs(output_dir, exist_ok=True)
    summary_results = []
    
    # Ensure target
    if 'excess_ret' not in df.columns:
        df['excess_ret'] = df['daily_return'] - df['RF']
    
    symbols = df['symbol'].unique()
    
    for sym in symbols:
        logger.info(f"Processing {sym}...")
        sym_df = df[df['symbol'] == sym].sort_values('date').set_index('date')
        
        # Check required columns
        all_feats = set()
        for m in MODELS.values():
            all_feats.update(m)
        required = list(all_feats) + ['excess_ret']
        
        sym_df = sym_df.dropna(subset=required)
        
        if len(sym_df) < window + 10:
            continue
            
        for model_name, features in MODELS.items():
            y = sym_df['excess_ret']
            X = sm.add_constant(sym_df[features])
            
            # --- THE KEY CHANGE: LAG PREDICTORS FOR T+1 FORECAST ---
            # We want to predict y_t using X_{t-1}
            # Shift X down by 1 so that the row at time t contains X_{t-1} and y_t
            X_lagged = X.shift(1)
            
            # Drop the first row (NaN shift)
            # Combine X_lagged and y to align
            combined = pd.concat([y, X_lagged], axis=1).dropna()
            
            y_aligned = combined['excess_ret']
            X_aligned = combined.drop(columns=['excess_ret'])
            
            try:
                # Rolling Window on the ALIGNED data
                # This means we use (X_{t-w-1}...X_{t-2}) to predict y_{t-w}...y_{t-1} to learn beta
                # Then apply beta to X_{t-1} to predict y_t
                
                rolling_model = RollingOLS(y_aligned, X_aligned, window=window)
                results = rolling_model.fit()
                
                # Get the beta meant for the NEXT observation
                # Results.params are the betas estimated using window ending at t.
                # We use these betas to predict y_{t+1} using X_t (which is in row t+1 of X_aligned? No)
                
                # With rollingOLS, params[t] is fit on y[t-w+1:t+1] and X[t-w+1:t+1].
                # We want to predict y[t+1] using X_aligned[t+1] (which is X_original[t]) and params[t].
                
                # Shift params by 1 to align with the future prediction target
                pred_params = results.params.shift(1).dropna()
                common_idx = pred_params.index.intersection(X_aligned.index)
                
                if len(common_idx) == 0: continue
                
                pred_params = pred_params.loc[common_idx]
                X_curr = X_aligned.loc[common_idx]
                y_curr = y_aligned.loc[common_idx]
                
                y_pred = (X_curr * pred_params).sum(axis=1)
                
                metrics = calculate_metrics(y_curr, y_pred, model_name)
                metrics['symbol'] = sym
                summary_results.append(metrics)
                
            except Exception as e:
                logger.error(f"Error {model_name} {sym}: {e}")
                
    # Aggregate
    res_df = pd.DataFrame(summary_results)
    if not res_df.empty:
        res_df.to_csv(os.path.join(output_dir, 'forecast_results.csv'), index=False)
        
        avg = res_df.groupby('model').mean(numeric_only=True)
        print("\n=== FORECASTING (T+1) RESULTS (Average) ===")
        print(avg.sort_values('sharpe', ascending=False))
        print("\nNote: OOS R2 is typically negative for T+1 forecasts in efficient markets.")
        avg.to_csv(os.path.join(output_dir, 'forecast_avg.csv'))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/master_analysis_data_advanced.csv")
    args = parser.parse_args()
    
    df = load_data(args.data_path)
    run_forecast_showdown(df)

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

# --- FINAL SHOWDOWN MODELS ---
FF3 = ['Mkt-RF', 'SMB', 'HML']
FF5 = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

# Features
SHOCK = ['z_score_sentiment']
VOL_INTERACT = ['interact_Vol_Sent']

MODELS = {
    '00_Benchmark_FF5': FF5,
    '01_FF3_Shock': FF3 + SHOCK,
    '02_FF3_VolInteract': FF3 + VOL_INTERACT,
    '03_FF3_Combined': FF3 + SHOCK + VOL_INTERACT
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

def run_showdown(df, window=120, output_dir='reports/final_showdown'):
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
            
            try:
                rolling_model = RollingOLS(y, X, window=window)
                results = rolling_model.fit()
                
                pred_params = results.params.shift(1).dropna()
                common_idx = pred_params.index.intersection(X.index)
                
                if len(common_idx) == 0: continue
                
                pred_params = pred_params.loc[common_idx]
                X_curr = X.loc[common_idx]
                y_curr = y.loc[common_idx]
                
                y_pred = (X_curr * pred_params).sum(axis=1)
                
                metrics = calculate_metrics(y_curr, y_pred, model_name)
                metrics['symbol'] = sym
                summary_results.append(metrics)
                
            except Exception as e:
                logger.error(f"Error {model_name} {sym}: {e}")
                
    # Aggregate
    res_df = pd.DataFrame(summary_results)
    if not res_df.empty:
        res_df.to_csv(os.path.join(output_dir, 'showdown_results.csv'), index=False)
        
        avg = res_df.groupby('model').mean(numeric_only=True)
        print("\n=== FINAL SHOWDOWN RESULTS (Average) ===")
        print(avg.sort_values('sharpe', ascending=False))
        avg.to_csv(os.path.join(output_dir, 'showdown_avg.csv'))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/master_analysis_data_advanced.csv")
    args = parser.parse_args()
    
    df = load_data(args.data_path)
    run_showdown(df)

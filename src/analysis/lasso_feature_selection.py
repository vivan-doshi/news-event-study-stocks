import pandas as pd
import numpy as np
import os
import argparse
import logging
import statsmodels.api as sm
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def calculate_metrics(y_true, y_pred, model_name):
    # Same metric calculation logic for consistency
    r2 = r2_score(y_true, y_pred)
    hit_rate = np.mean(np.sign(y_true) == np.sign(y_pred))
    
    position = np.sign(y_pred)
    strategy_ret = position * y_true
    
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
    
    return {
        'model': model_name,
        'oos_r2': r2,
        'hit_rate': hit_rate,
        'sharpe': sharpe,
        'sortino': sortino,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown
    }

def run_rolling_lasso(df, window=120, output_dir='reports/lasso'):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Define Feature Universe (All potential alpha sources)
    # Exclude basic IDs and Targets AND Direct proxies of target
    target_col = 'excess_ret'
    if target_col not in df.columns:
         df[target_col] = df['daily_return'] - df['RF']
         
    exclude_cols = ['symbol', 'date', 'daily_return', 'RF', target_col, 
                    'return_lag_1d', 'return_lag_5d', 'return_lag_10d', 'return_lag_21d',
                    'symbol_query', 'company_name', 'topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 
                    'final_date_for_news', 'published_at', 'adj_close', 'adjusted_close', 'close', 'high', 'low', 'open', 'volume',
                     'open_int', 'log_volume', 'abs_ret',
                     # CRITICAL: Exclude leakages (any current day return)
                     'ret_1d', 'log_ret', 'excess_ret', 'daily_ret', 'return']
    
    # Also exclude raw price columns if any
    
    potential_features = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [np.float64, np.float32, int, np.int64]]
    
    # Double check: Remove any column starting with 'ret' or 'log_ret' unless it has 'lag'
    potential_features = [f for f in potential_features if not (f.startswith('ret') or f.startswith('log_ret')) or 'lag' in f]
    
    # Check if we have NaN
    df = df.dropna(subset=potential_features + [target_col])
    
    logger.info(f"Running Lasso on {len(potential_features)} features: {potential_features[:5]}... (and {len(potential_features)-5} more)")
    
    symbols = df['symbol'].unique()
    summary_results = []
    feature_importance_list = []
    
    for sym in symbols:
        logger.info(f"Processing {sym}...")
        sym_df = df[df['symbol'] == sym].sort_values('date').set_index('date')
        
        if len(sym_df) < window + 20:
            logger.warning(f"Not enough data for {sym}")
            continue
            
        y = sym_df[target_col]
        X = sym_df[potential_features]
        
        preds = []
        actuals = []
        dates = []
        coefs_dict = {f: 0 for f in potential_features}
        count = 0
        
        # Rolling Loop
        # Needs to be efficient. LassoCV is slow if run every day.
        # Strategy: Re-train every 20 days? Or every day but with fixed Alpha?
        # Let's try LassoCV every 60 days to find alpha, then uses that alpha?
        # Or faster: Lasso with fixed small alpha.
        # User asked for Lasso. Let's use sk-learn.
        
        # Iterating daily is slow. 
        # For 'rolling' in sk-learn, we must loop manually.
        
        t_start = window
        t_end = len(sym_df)
        
        # To speed up: Re-fit model every 20 days (approx 1 month)
        refit_freq = 20
        current_model = None
        scaler = StandardScaler()
        
        for t in range(t_start, t_end):
            row_date = X.index[t]
            
            if t % refit_freq == 0 or current_model is None:
                X_train = X.iloc[t-window:t]
                y_train = y.iloc[t-window:t]
                
                # Standardize (Lasso requirement)
                X_train_scaled = scaler.fit_transform(X_train)
                
                # Fit
                # LassoCV finds best alpha
                # Optimization: n_jobs=-1, reduce max_iter slightly if causing hang, or keep it.
                current_model = LassoCV(cv=3, random_state=42, max_iter=1000, n_jobs=-1)
                current_model.fit(X_train_scaled, y_train)
                
                # Store coeffs importance
                for feat, coef in zip(potential_features, current_model.coef_):
                    if abs(coef) > 0:
                        coefs_dict[feat] += 1
                count += 1
            
            # Predict for current day
            # SCALE current input using the scaler fitted on training window
            X_curr = X.iloc[t:t+1]
            X_curr_scaled = scaler.transform(X_curr)
            
            pred = current_model.predict(X_curr_scaled)[0]
            
            preds.append(pred)
            actuals.append(y.iloc[t])
            dates.append(row_date)
            
        # Analysis
        y_pred_series = pd.Series(preds, index=dates)
        y_true_series = pd.Series(actuals, index=dates)
        
        metrics = calculate_metrics(y_true_series, y_pred_series, f'Lasso_Rolling_{sym}')
        metrics['symbol'] = sym
        summary_results.append(metrics)
        
        # Feature Importance for this stock
        # normalize counts
        if count > 0:
            norm_coefs = {k: v/count for k, v in coefs_dict.items()}
            norm_coefs['symbol'] = sym
            feature_importance_list.append(norm_coefs)
            
    # Save Results
    res_df = pd.DataFrame(summary_results)
    if not res_df.empty:
        res_df.to_csv(os.path.join(output_dir, 'lasso_performance.csv'), index=False)
        print("\n=== Lasso Performance ===")
        print(res_df.mean(numeric_only=True))
        
        # Feature Importance Aggregation
        feat_df = pd.DataFrame(feature_importance_list)
        feat_df.to_csv(os.path.join(output_dir, 'lasso_feature_selection_freq.csv'), index=False)
        
        # Top Global Features
        global_feats = feat_df.drop('symbol', axis=1).mean().sort_values(ascending=False)
        print("\n=== Top Selected Features (Selection Frequency) ===")
        print(global_feats.head(10))
        global_feats.head(20).to_csv(os.path.join(output_dir, 'top_lasso_features.csv'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/master_analysis_data_advanced.csv")
    parser.add_argument("--window", type=int, default=120)
    args = parser.parse_args()
    
    df = load_data(args.data_path)
    run_rolling_lasso(df, window=args.window)

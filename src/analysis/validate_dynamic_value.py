
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

# Configure Plotting Style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.family'] = 'sans-serif'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamic_pred_path", default="reports/dynamic_analysis/dynamic_predictions.csv")
    parser.add_argument("--baseline_pred_path", default="reports/panel_regression/data/panel_predictions.csv")
    parser.add_argument("--output_dir", default="reports/dynamic_analysis/validation")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Predictions
    if not os.path.exists(args.dynamic_pred_path):
        print(f"Dynamic predictions not found at {args.dynamic_pred_path}")
        return
        
    dyn_df = pd.read_csv(args.dynamic_pred_path)
    dyn_df['date'] = pd.to_datetime(dyn_df['date'])
    dyn_df = dyn_df.rename(columns={'y_pred': 'y_pred_dynamic'})
    
    # Load Baseline
    base_df = pd.read_csv(args.baseline_pred_path)
    base_df = base_df[base_df['model'] == 'Panel_Baseline'].copy()
    base_df['date'] = pd.to_datetime(base_df['date'])
    base_df = base_df.rename(columns={'y_pred': 'y_pred_baseline'})
    
    # Merge
    merged = pd.merge(
        dyn_df[['date', 'symbol', 'y_true', 'y_pred_dynamic']], 
        base_df[['date', 'symbol', 'y_pred_baseline']], 
        on=['date', 'symbol'], 
        how='inner'
    )
    
    if merged.empty:
        print("No overlapping dates between Dynamic and Baseline.")
        return
        
    # Metrics
    mse_dyn = ((merged['y_true'] - merged['y_pred_dynamic'])**2).mean()
    r2_dyn = 1 - (mse_dyn / ((merged['y_true'] - merged['y_true'].mean())**2).mean())
    
    mse_base = ((merged['y_true'] - merged['y_pred_baseline'])**2).mean()
    r2_base = 1 - (mse_base / ((merged['y_true'] - merged['y_true'].mean())**2).mean())
    
    print(f"Comparison (Overlapping N={len(merged)}):")
    print(f"Dynamic Pipeline OOS R2: {r2_dyn:.4f}")
    print(f"Baseline Panel OOS R2:   {r2_base:.4f}")
    
    # Cumulative Performance Plot
    # Strategy: Long Top 1, Short Bottom 1 (Daily)
    def backtest(sub, pred_col):
        # Long/Short Portfolio
        longs = sub.nlargest(1, pred_col)
        shorts = sub.nsmallest(1, pred_col)
        
        ret = longs['y_true'].mean() - shorts['y_true'].mean()
        return ret
        
    dyn_dates = merged['date'].unique()
    dyn_returns = []
    base_returns = []
    dates = []
    
    for d in sorted(dyn_dates):
        sub = merged[merged['date'] == d]
        if len(sub) < 2: continue
        
        dyn_ret = backtest(sub, 'y_pred_dynamic')
        base_ret = backtest(sub, 'y_pred_baseline')
        
        dyn_returns.append(dyn_ret)
        base_returns.append(base_ret)
        dates.append(d)
        
    perf_df = pd.DataFrame({
        'date': dates,
        'Dynamic Strategy': dyn_returns,
        'Baseline Strategy': base_returns
    }).set_index('date')
    
    cum_perf = (1 + perf_df).cumprod()
    
    plt.figure()
    cum_perf.plot(linewidth=2)
    plt.title('Dynamic Topic vs Baseline: L/S Strategy Performance')
    plt.ylabel('Cumulative Return')
    plt.savefig(os.path.join(args.output_dir, "strategy_comparison.png"))
    plt.close()
    
    print(f"Saved validation plot to {args.output_dir}/strategy_comparison.png")

if __name__ == "__main__":
    main()

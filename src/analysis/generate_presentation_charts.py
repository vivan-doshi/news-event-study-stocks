
# src/analysis/generate_presentation_charts.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(df_strategy):
    """
    Calculate Sharpe, Sortino, Max Drawdown, Profit Factor.
    Assume 'strategy_return' is daily excess return.
    """
    returns = df_strategy['strategy_return']
    
    # Annualized Sharpe (Assuming 252 trading days, risk-free is netted out in excess return)
    mean_ret = returns.mean() * 252
    vol = returns.std() * np.sqrt(252)
    sharpe = mean_ret / vol if vol != 0 else 0
    
    # Sortino (Downside risk only)
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    sortino = mean_ret / downside_vol if downside_vol != 0 else 0
    
    # Max Drawdown
    cumulative = (1 + returns).cumprod()
    peaks = cumulative.cummax()
    drawdowns = (cumulative - peaks) / peaks
    max_dd = drawdowns.min()
    
    # Profit Factor (Gross Win / |Gross Loss|)
    winning_days = returns[returns > 0].sum()
    losing_days = abs(returns[returns < 0].sum())
    profit_factor = winning_days / losing_days if losing_days != 0 else 0
    
    return {
        "Annualized Return": mean_ret,
        "Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
        "Profit Factor": profit_factor
    }

def main():
    INPUT_PATH = "reports/mag7_benchmark_analysis/preds_window_100.csv"
    OUTPUT_DIR = "reports/visuals"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading data from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. ACTUAL VS PREDICTED SCATTER
    plt.figure(figsize=(10, 6))
    # Sample if too large for scatter aesthetics
    plot_df = df.sample(5000) if len(df) > 5000 else df
    
    sns.regplot(x='y_true', y='y_pred', data=plot_df, 
                scatter_kws={'alpha':0.3, 'color': 'blue'}, 
                line_kws={'color': 'red'})
    
    plt.title('Actual vs Predicted Returns (Window=100)')
    plt.xlabel('Actual Excess Return')
    plt.ylabel('Predicted Excess Return')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/actual_vs_pred.png")
    print(f"Saved {OUTPUT_DIR}/actual_vs_pred.png")
    plt.close()
    
    # 2. STRATEGY SIMULATION (Long/Short vs Buy & Hold)
    # Strategy: Long if y_pred > 0, Short if y_pred < 0
    # Note: Using next day return (y_true is t+1 return usually in these models, we check logic)
    # In 'preds.csv', y_true matches y_pred for the same row. 
    # Usually y_pred is prediction for 'y_true'.
    
    df['position'] = np.sign(df['y_pred']) # 1 or -1
    
    # Strategy Return = Position * Actual Return
    # Transaction costs ignored for simplified presentation view
    df['strategy_return'] = df['position'] * df['y_true']
    
    # Buy & Hold (Benchmark) - essentially average of Mag7 equal weight here?
    # Or just y_true (which is individual stock return). 
    # To get Portfolio view, we aggregate by date using mean.
    
    portfolio = df.groupby('date')[['y_true', 'strategy_return']].mean().reset_index()
    
    # Cumulative Calculation
    portfolio['Cumulative Benchmark'] = (1 + portfolio['y_true']).cumprod()
    portfolio['Cumulative Strategy'] = (1 + portfolio['strategy_return']).cumprod()
    
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio['date'], portfolio['Cumulative Strategy'], label='Topic-Augmented Strategy (L/S)', color='green', linewidth=2)
    plt.plot(portfolio['date'], portfolio['Cumulative Benchmark'], label='Mag7 Equal Weight (Buy & Hold)', color='gray', linestyle='--')
    
    plt.title('Cumulative Performance: Topic Strategy vs Benchmark (Window 100)')
    plt.xlabel('Date')
    plt.ylabel('Growth of $1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cumulative_returns.png")
    print(f"Saved {OUTPUT_DIR}/cumulative_returns.png")
    plt.close()
    
    # 3. METRICS
    metrics = calculate_metrics(portfolio)
    print("\n=== STRATEGY METRICS (Window 100) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()

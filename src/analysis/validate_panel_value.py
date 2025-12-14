
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def main():
    # Paths
    input_path = "reports/panel_regression/data/panel_predictions.csv"
    output_dir = "reports/panel_regression/commercial_val"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    # 1. Data Loading
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for Thematic Shocks Model (most promising)
    sent_df = df[df['model'] == 'Panel_Thematic_Shocks'].copy()
    if sent_df.empty:
        print("Warning: Panel_Thematic_Shocks not found. Falling back to Panel_Sentiment.")
        sent_df = df[df['model'] == 'Panel_Sentiment'].copy()
    
    # ==========================================
    # Diagnostic 1: The "Pulse" Check (Rolling IC)
    # ==========================================
    print("Generating Pulse Check (Rolling IC)...")
    
    # Calculate daily IC (Spearman correlation between pred and true)
    daily_ic = sent_df.groupby('date').apply(lambda x: x['y_pred'].corr(x['y_true'], method='spearman'))
    
    # Calculate 30-Day Rolling Mean
    rolling_ic = daily_ic.rolling(window=30).mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_ic.index, rolling_ic, label='30-Day Rolling IC', color='#2ca02c', linewidth=2)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title('The "Pulse" Check: Information Coefficient (IC) Stability', fontsize=16, weight='bold')
    plt.ylabel('Spearman Correlation', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pulse_check_ic.png'), dpi=150)
    plt.close()
    
    # ==========================================
    # Diagnostic 2: The "Antigravity" Backtest
    # ==========================================
    print("Generating Antigravity Backtest...")
    
    strategy_returns = []
    dates = sorted(sent_df['date'].unique())
    
    for d in dates:
        day_data = sent_df[sent_df['date'] == d]
        
        # We need at least 4 stocks to pick Top 2 and Bottom 2
        if len(day_data) < 4:
            strategy_returns.append({'date': d, 'strategy': 0, 'benchmark': day_data['y_true'].mean()})
            continue
            
        # Top 2 Long, Bottom 2 Short
        sorted_day = day_data.sort_values('y_pred', ascending=False)
        longs = sorted_day.head(2)
        shorts = sorted_day.tail(2)
        
        # Equal weighted returns
        long_ret = longs['y_true'].mean()
        short_ret = shorts['y_true'].mean()
        
        # Strategy: Long - Short
        # Note: If we assume fully funded long/short, return is (L - S) / Capital?
        # Usually simplified as r_long - r_short for a dollar-neutral portfolio.
        # The equation provided was: 0.5(R_l1 + R_l2) - 0.5(R_s1 + R_s2)
        # Which is exactly Mean(Longs) - Mean(Shorts)
        strat_ret = long_ret - short_ret
        
        # Benchmark: Equal weight of all available stocks
        bench_ret = day_data['y_true'].mean()
        
        strategy_returns.append({'date': d, 'strategy': strat_ret, 'benchmark': bench_ret})
        
    perf_df = pd.DataFrame(strategy_returns).set_index('date')
    
    # Cumulative Returns
    cum_perf = (1 + perf_df).cumprod() - 1
    
    plt.figure(figsize=(12, 6))
    plt.plot(cum_perf.index, cum_perf['strategy'], label='L/S Strategy (Top 2 vs Bottom 2)', color='#1f77b4', linewidth=3)
    plt.plot(cum_perf.index, cum_perf['benchmark'], label='Market Benchmark (Eq Weight)', color='gray', linestyle='--', linewidth=2)
    
    plt.title('Long/Short Strategy (Thematic Shocks) Backtest', fontsize=16, weight='bold')
    plt.ylabel('Cumulative Excess Return', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add annotation for total return
    final_strat = cum_perf['strategy'].iloc[-1]
    final_bench = cum_perf['benchmark'].iloc[-1]
    plt.annotate(f"Total Strategy: {final_strat:.1%}", xy=(cum_perf.index[-1], final_strat), xytext=(10, 0), textcoords='offset points', color='#1f77b4', weight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ls_sentiment_backtest.png'), dpi=150)
    plt.close()
    
    # ==========================================
    # Diagnostic 3: Model Comparison Bar Chart
    # ==========================================
    print("Generating Model Comparison...")
    
    mse_scores = []
    mse_scores = []
    # Add new models to comparison
    model_list = ['Panel_Baseline', 'Panel_News', 'Panel_Sentiment', 'Panel_Thematic_Shocks', 'Panel_Signal_Conviction', 'Panel_Risk_Shock']
    
    for model_name in model_list:
        sub = df[df['model'] == model_name]
        if sub.empty:
            continue
        mse = ((sub['y_true'] - sub['y_pred'])**2).mean()
        rmse = np.sqrt(mse)
        mse_scores.append({'Model': model_name, 'RMSE': rmse})
        
    mse_df = pd.DataFrame(mse_scores)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=mse_df, x='Model', y='RMSE', hue='Model', palette='viridis', legend=False)
    plt.title('Model Comparison: Root Mean Squared Error (RMSE)', fontsize=16, weight='bold')
    plt.ylabel('RMSE (Lower is Better)', fontsize=12)
    
    # Add formatted labels
    for index, row in mse_df.iterrows():
        plt.text(index, row.RMSE, f'{row.RMSE:.5f}', color='black', ha="center", va="bottom")
        
    plt.ylim(mse_df['RMSE'].min() * 0.95, mse_df['RMSE'].max() * 1.05) # Zoom in slightly to show diffs
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_rmse.png'), dpi=150)
    plt.close()
    
    print(f"Validation complete. Plots saved to {output_dir}")

if __name__ == "__main__":
    main()

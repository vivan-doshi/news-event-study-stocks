import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12

OUTPUT_DIR = 'reports/visuals'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    contemporaneous = pd.read_csv('reports/regression/deep_dive_benchmark_avg.csv')
    forecast = pd.read_csv('reports/forecast_showdown/forecast_avg.csv')
    return contemporaneous, forecast

def plot_sharpe_comparison(df):
    # Filter for top models + Baseline
    models_of_interest = ['02_FF3', '03_FF5', '13_FF3_Shock', '02_FF3_VolInteract', '24_FF3_TopicShock_VolInteract', '22_FF3_TopicShock']
    # Startswith filter because names might vary slightly in CSV vs code map
    # Actually, in the CSV the names are exact.
    
    subset = df[df['model'].isin(models_of_interest)].copy()
    subset = subset.sort_values('sharpe', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=subset, x='sharpe', y='model', palette='viridis')
    plt.title('Sharpe Ratio Comparison (Contemporaneous)', fontsize=16)
    plt.xlabel('Sharpe Ratio')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/sharpe_comparison.png')
    plt.close()

def plot_sortino_comparison(df):
    # Comparing Topic Shocks vs Benchmark
    models = ['03_FF5', '02_FF3_VolInteract', '22_FF3_TopicShock']
    subset = df[df['model'].isin(models)].copy()
    subset = subset.sort_values('sortino', ascending=False)
    
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=subset, x='sortino', y='model', palette='magma')
    plt.title('Sortino Ratio: The "Topic Shock" Advantage', fontsize=16)
    plt.xlabel('Sortino Ratio')
    
    # Add labels
    for i, v in enumerate(subset['sortino']):
        ax.text(v + 0.5, i, f'{v:.1f}', va='center')
        
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/sortino_advantage.png')
    plt.close()

def plot_t1_decay(df_t1):
    # Forecast models
    subset = df_t1.sort_values('sharpe', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=subset, x='sharpe', y='model', palette='rocket')
    plt.title('T+1 Forecasting: Topic Shocks Decay Overnight', fontsize=16)
    plt.xlabel('Sharpe Ratio (T+1)')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/t1_decay.png')
    plt.close()

def plot_risk_reward_scatter(df):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='oos_r2', y='profit_factor', hue='model', s=100, legend=False)
    
    # Annotate key models
    key_models = ['22_FF3_TopicShock', '23_FF5_TopicShock', '02_FF3_VolInteract', '03_FF5']
    for i, row in df.iterrows():
        if row['model'] in key_models:
            plt.text(row['oos_r2']+0.01, row['profit_factor'], row['model'], fontsize=9, weight='bold')
            
    plt.title('Risk vs Reward: The "Sniper" Nature of Topic Shocks', fontsize=16)
    plt.xlabel('OOS R-Squared (Predictive Power)')
    plt.ylabel('Profit Factor (Gross Win / Gross Loss)')
    plt.axvline(0, color='red', linestyle='--') # Zero R2 line
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/risk_reward_scatter.png')
    plt.close()

if __name__ == "__main__":
    contemp, forecast = load_data()
    plot_sharpe_comparison(contemp)
    plot_sortino_comparison(contemp)
    plot_t1_decay(forecast)
    plot_risk_reward_scatter(contemp)
    print(f"Visuals generated in {OUTPUT_DIR}")

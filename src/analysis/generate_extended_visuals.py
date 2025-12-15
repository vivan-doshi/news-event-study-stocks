
# src/analysis/generate_extended_visuals.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set Aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300

OUTPUT_DIR = "reports/visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. LOAD DATA
NEWS_PATH = "data/processed/mag7_news_with_sentiment_and_topics_labeledV2.parquet"
LASSO_PATH = "reports/lasso/top_lasso_features.csv"
MODEL_PREDS_PATH = "reports/mag7_benchmark_analysis/preds_window_100.csv"

def plot_topic_evolution(df):
    """Stacked Area Chart of News Volume by Topic (Monthly)"""
    print("Generating Topic Volume Evolution...")
    
    # Preprocessing
    df['date'] = pd.to_datetime(df['published_at'])
    df['month'] = df['date'].dt.to_period('M')
    
    # Aggregate
    topic_map = {
        0: 'AI & EV Rally',
        1: 'Trump/Macro',
        2: 'Analyst Ratings',
        3: 'Corporate Actions',
        4: 'Earnings'
    }
    df['topic_label'] = df['topic_id_kmeans'].map(topic_map)
    
    vol_data = df.groupby(['month', 'topic_label']).size().unstack(fill_value=0)
    
    # Plot
    plt.figure(figsize=(12, 6))
    vol_data.plot(kind='area', stacked=True, alpha=0.8, figsize=(12, 6)) # Plotting directly from DF handles axis nicely
    
    plt.title('Evolution of Narrative Dominance: Monthly News Volume by Topic', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/topic_volume_evolution.png")
    plt.close()

def plot_sentiment_violin(df):
    """Violin Plot of Sentiment Distribution per Topic"""
    print("Generating Topic Sentiment Distribution...")
    
    topic_map = {
        0: 'AI & EV Rally',
        1: 'Trump/Macro',
        2: 'Analyst Ratings',
        3: 'Corporate Actions',
        4: 'Earnings'
    }
    df['topic_label'] = df['topic_id_kmeans'].map(topic_map)
    
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='topic_label', y='sentiment_finbert', data=df, inner='quartile', palette='muted')
    
    plt.title('Sentiment Volatility by Narrative Cluster', fontsize=14)
    plt.xlabel('Topic')
    plt.ylabel('FinBERT Sentiment Score')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/topic_sentiment_dist.png")
    plt.close()

def plot_lasso_importance():
    """Horizontal Bar Chart of Lasso Coefficients"""
    print("Generating Lasso Feature Importance...")
    
    if not os.path.exists(LASSO_PATH):
        print(f"Skipping Lasso Plot: {LASSO_PATH} not found.")
        return

    df = pd.read_csv(LASSO_PATH, header=None, names=['feature', 'frequency'])
    # Clean names for display
    df['feature'] = df['feature'].str.replace('interact_', 'Interact: ').str.replace('_', ' ')
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='frequency', y='feature', data=df.head(15), palette='viridis')
    
    plt.title('Lasso Feature Selection Frequency (Top 15)', fontsize=14)
    plt.xlabel('Selection Frequency (0.0 - 1.0)')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/lasso_importance.png")
    plt.close()

def plot_sortino_comparison():
    """Bar Chart of Sortino Ratios"""
    print("Generating Sortino Comparison...")
    
    # Hardcoded from final results for clean presentation (or load from CSV if preferred)
    data = {
        'Model': ['FF5 Benchmark', 'FF3 + Global Shock', 'Lasso Optimized'],
        'Sortino Ratio': [25.03, 24.57, 49.83]
    }
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x='Model', y='Sortino Ratio', data=df, palette=['grey', 'teal', 'gold'])
    
    # Add labels
    for i, v in enumerate(df['Sortino Ratio']):
        ax.text(i, v + 0.5, str(v), color='black', ha='center')
        
    plt.title('Risk-Adjusted Performance: The "Alpha" Gap', fontsize=14)
    plt.ylabel('Sortino Ratio (Reward / Downside Vol)')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_comparison_sortino.png")
    plt.close()

def plot_drawdown_curve():
    """Drawdown Curve Comparison"""
    print("Generating Drawdown Comparison...")
    
    if not os.path.exists(MODEL_PREDS_PATH):
        print(f"Skipping Drawdown Plot: {MODEL_PREDS_PATH} not found.")
        return

    df = pd.read_csv(MODEL_PREDS_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    # Strategy Logic
    df['position'] = np.sign(df['y_pred'])
    df['strat_ret'] = df['position'] * df['y_true']
    
    # Aggregate to Portfolio level
    port = df.groupby('date')[['y_true', 'strat_ret']].mean().reset_index()
    
    # Cumulative & Drawdown Function
    def get_drawdown(ret_series):
        cum = (1 + ret_series).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return dd * 100 # Percentage
        
    port['DD_Bench'] = get_drawdown(port['y_true'])
    port['DD_Strat'] = get_drawdown(port['strat_ret'])
    
    plt.figure(figsize=(12, 6))
    plt.plot(port['date'], port['DD_Strat'], label='Topic Shock Strategy', color='green', linewidth=1.5)
    plt.plot(port['date'], port['DD_Bench'], label='Benchmark (Mag7)', color='red', alpha=0.5, linewidth=1)
    
    plt.fill_between(port['date'], port['DD_Bench'], 0, color='red', alpha=0.1)
    
    plt.title('Drawdown Analysis: Capital Preservation During 2022/23 Volatility', fontsize=14)
    plt.ylabel('Drawdown %')
    plt.xlabel('Date')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/drawdown_comparison.png")
    plt.close()


def main():
    # Load News Data for 1 & 2
    if os.path.exists(NEWS_PATH):
        print(f"Loading News Data: {NEWS_PATH}...")
        df_news = pd.read_parquet(NEWS_PATH)
        plot_topic_evolution(df_news)
        plot_sentiment_violin(df_news)
    else:
        print(f"Skipping News Plots: {NEWS_PATH} not found.")
        
    # Plot 3
    plot_lasso_importance()
    
    # Plot 4
    plot_sortino_comparison()
    
    # Plot 5
    plot_drawdown_curve()
    
    print(f"Done. Visuals saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

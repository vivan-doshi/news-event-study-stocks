
# src/analysis/generate_final_ppt_charts.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches

# Set Aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'

# Configuration
DATA_DIR = "reports/mag7_benchmark_analysis"
REPORTS_DIR = "reports"
OUTPUT_DIR = "reports/visuals"

# Paths
NEWS_PATH = "data/processed/mag7_news_with_topicsV2.parquet"
EMBEDDINGS_PATH = "data/processed/mag7_embeddings.parquet"
LASSO_PATH = "reports/lasso/top_lasso_features.csv"
MODEL_PREDS_PATH = "reports/forecast_showdown/preds_forecast.csv" # UPDATED: Using T+1 Forecasts for Realistic Equity Curve
AUGMENTED_PATH = "data/processed/mag7_augmented_features.parquet"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_plot(name):
    path = f"{OUTPUT_DIR}/{name}"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")

# 1. EXECUTIVE SUMMARY
def plot_concentration_risk():
    """Mag 7 vs S&P 493 Market Cap Weight"""
    labels = ['Mag 7 (Concentration Risk)', 'Remaining S&P 493']
    sizes = [31, 69] 
    colors = ['#FF6B6B', '#4ECDC4']
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, 
            textprops={'fontsize': 14}, explode=(0.05, 0))
    plt.title('The "Top Heavy" Market Problem', fontsize=16)
    save_plot("concentration_risk_pie.png")

def plot_narrative_lag_schematic():
    """Conceptual Diagram of Lag"""
    t = np.linspace(0, 10, 1000)
    
    # News Spike (Instant)
    news_spike = np.exp(-0.5 * (t - 2)**2 / 0.05) 
    
    # Price Drift (Delayed Response)
    # Sigmoid activation that starts AFTER the news spike
    price_impact = 1 / (1 + np.exp(-(t - 4)*2.5))
    
    plt.figure(figsize=(10, 5))
    plt.plot(t, news_spike, label='News Volume (Instant)', color='purple', linewidth=3)
    plt.plot(t, price_impact, label='Price Discovery (Lagged)', color='green', linestyle='--', linewidth=3)
    
    # Annotations
    plt.axvline(x=2, color='purple', linestyle=':', alpha=0.3)
    plt.axvline(x=4, color='green', linestyle=':', alpha=0.3)
    
    plt.text(2.1, 0.9, 'Event Occurs', color='purple', fontweight='bold')
    plt.text(4.1, 0.9, 'Market Digests\n(Alpha Window)', color='green', fontweight='bold')
    
    plt.title('CONCEPTUAL: The "Narrative Lag" Hypothesis', fontsize=16)
    plt.xlabel('Time (Days)')
    plt.ylabel('Normalized Magnitude')
    plt.legend(loc='center right')
    plt.yticks([])
    plt.figtext(0.5, 0.01, "Note: Conceptual illustration of information processing latency.", wrap=True, horizontalalignment='center', fontsize=10)
    save_plot("narrative_lag_schematic.png")

# 2. METHODOLOGY (NLP)
def plot_real_tsne():
    """TSNE of REAL Embeddings"""
    print("Generating Real t-SNE...")
    
    if not os.path.exists(EMBEDDINGS_PATH):
        print(f"Embeddings not found at {EMBEDDINGS_PATH}")
        return
        
    # Load Embeddings (Large file!)
    # We need to read parquet columns carefully.
    # It might be an array column or flattened. 
    # Usually 'embedding' column.
    try:
        df = pd.read_parquet(EMBEDDINGS_PATH)
    except:
        print("Failed to read embeddings parquet.")
        return
        
    # Check if we have topics valid
    if 'topic_id_kmeans' not in df.columns:
        # Try joining with topic file if needed, but usually embeddings parquet has it or index matches.
        # Let's check mag7_news_with_topicsV2.parquet which definitely has topics
        df_topics = pd.read_parquet(NEWS_PATH)
        # Assuming index alignment if sourced from same pipeline
        if len(df) == len(df_topics):
             df['topic_id_kmeans'] = df_topics['topic_id_kmeans'].values
        else:
            print("Length mismatch for topics/embeddings. Using random sample for layout.")
            
    # Sample 3000 points
    SAMPLE_SIZE = 3000
    if len(df) > SAMPLE_SIZE:
        df_sample = df.sample(SAMPLE_SIZE, random_state=42).copy()
    else:
        df_sample = df.copy()

    # Expand embeddings
    # Assuming 'embedding' col contains lists/arrays
    # If not, looking for col naming convention.
    # Previous code implied standard parquet save.
    
    # Heuristic: Find the vector column
    vec_col = None
    for c in df_sample.columns:
        if isinstance(df_sample[c].iloc[0], (np.ndarray, list)):
             vec_col = c
             break
    
    if vec_col:
        X = np.stack(df_sample[vec_col].values)
    else:
        # Maybe columns are emb_0, emb_1...
        emb_cols = [c for c in df_sample.columns if str(c).startswith('emb')]
        if emb_cols:
            X = df_sample[emb_cols].values
        else:
            print("No embedding vector column found.")
            return

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    topic_labels = {
        0: 'AI & EV Aggression',
        1: 'Macro/Trump Risk',
        2: 'Analyst Ratings',
        3: 'Corp Actions',
        4: 'Earnings Fundamentals'
    }
    
    df_sample['topic_label'] = df_sample['topic_id_kmeans'].map(topic_labels)
    
    sns.scatterplot(
        x=X_embedded[:,0], y=X_embedded[:,1], 
        hue=df_sample['topic_label'], 
        palette='turbo', 
        s=60, alpha=0.7, 
        legend='full'
    )
    
    plt.title('Actual Embeddings (t-SNE): Semantic Separation', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.axis('off')
    save_plot("topic_clusters_tsne.png")


# 3. SIGNAL CONSTRUCTION
def plot_nvda_signal_engineering():
    """Detailed Dual-Axis for NVDA"""
    if not os.path.exists(AUGMENTED_PATH):
        return
    
    df = pd.read_parquet(AUGMENTED_PATH)
    df_nvda = df[df['symbol_query'].str.contains("NVDA")].copy()
    df_nvda = df_nvda.sort_values(by='final_date_for_news')
    df_nvda = df_nvda.tail(200) # Zoom in
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    dates = pd.to_datetime(df_nvda['final_date_for_news'])
    
    # 1. Raw Sentiment (Bar/Area)
    ax1.fill_between(dates, df_nvda['day_sentiment'], color='gray', alpha=0.2, label='Base Sentiment (Noise)')
    ax1.set_ylabel('Raw FinBERT Score', color='gray')
    ax1.set_ylim(-1, 1)
    
    # 2. Z-Score (Line)
    ax2 = ax1.twinx()
    # Color condition
    color_line = df_nvda['day_sentiment_zscore'].apply(lambda x: 'green' if x>1.5 else ('red' if x<-1.5 else 'blue'))
    
    ax2.plot(dates, df_nvda['day_sentiment_zscore'], color='#333333', linewidth=1, alpha=0.5, label='Z-Score Trace')
    ax2.scatter(dates, df_nvda['day_sentiment_zscore'], c=color_line, s=20, label='Significant Shocks')
    
    # Thresholds
    ax2.axhline(y=1.5, color='green', linestyle='--', alpha=0.5, label='Buy Threshold (+1.5)')
    ax2.axhline(y=-1.5, color='red', linestyle='--', alpha=0.5, label='Sell Threshold (-1.5)')
    
    ax2.set_ylabel('Narrative Shock ($Z$)', color='black')
    
    plt.title('Signal Engineering: Extracting "Surprise" from NVDA News', fontsize=16)
    # Custom Legend
    handles, labels = ax2.get_legend_handles_labels()
    patch_raw = mpatches.Patch(color='gray', alpha=0.2, label='Raw Sentiment Range')
    handles.append(patch_raw)
    plt.legend(handles=handles, loc='upper left')
    
    save_plot("sentiment_vs_shock_nvda.png")

def plot_volume_interaction_matrix():
    """Heatmap of Interaction Logic"""
    # Create grid
    sent_z = np.linspace(-3, 3, 100)
    log_vol = np.linspace(0, 5, 100)
    X, Y = np.meshgrid(sent_z, log_vol)
    
    # Logic: Interaction = Abs(Shock) * Volume
    # We want to show where the "Action" is.
    Z = np.abs(X) * Y 
    
    plt.figure(figsize=(8, 6))
    # Diverging colormap or sequential? Sequential 'Magma' implies intensity
    plt.pcolormesh(X, Y, Z, shading='auto', cmap='magma')
    cbar = plt.colorbar()
    cbar.set_label('Conviction (Alpha Potential)', rotation=270, labelpad=15)
    
    plt.xlabel('Sentiment Shock ($Z$-Score)')
    plt.ylabel('News Volume ($\ln(1+V)$)')
    
    # Annotate Zones
    plt.text(0, 0.5, 'NOISE ZONE\n(Ignore)', color='white', ha='center')
    plt.text(2.5, 4.5, 'HIGH ALPHA\n(Trade)', color='black', ha='center', fontweight='bold')
    plt.text(-2.5, 4.5, 'HIGH ALPHA\n(Trade)', color='black', ha='center', fontweight='bold')
    
    plt.title('Logic: The Volume Interaction Matrix', fontsize=16)
    save_plot("volume_interaction_heatmap.png")

# 4. RESULTS - SPECIFIC MODEL
def plot_results_final_suite():
    # Reverting to Contemporaneous Model (Lasso) as requested ("Actual Strategy")
    # This shows the "Theoretical Upper Bound" (1,200x) which demonstrates signal quality.
    path = "reports/mag7_benchmark_analysis/preds_window_100.csv"
    
    if not os.path.exists(path):
        print(f"Preds file not found: {path}")
        return
        
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate Returns
    # Strategy: Sign(Pred) * True
    df['strat_ret'] = np.sign(df['y_pred']) * df['y_true']
    
    # Benchmark: Buy & Hold (Mean of Mag 7)
    # The 'y_true' column IS the excess return of the stock.
    # Averaging it gives the Equal-Weighted Mag 7 Index.
    port = df.groupby('date')[['y_true', 'strat_ret']].mean().reset_index()
    
    # Cumulative (Start at $100)
    port['cum_bench'] = (1 + port['y_true']).cumprod() * 100
    port['cum_strat'] = (1 + port['strat_ret']).cumprod() * 100
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(port['date'], port['cum_strat'], color='#00A36C', linewidth=2.5, label='Lasso Strategy (Theoretical Upper Bound)')
    plt.plot(port['date'], port['cum_bench'], color='grey', linestyle='--', linewidth=1.5, label='Mag 7 Equal-Weighted Index')
    
    plt.yscale('log') # Log scale essential for 1200x return
    
    # Annotate Final Values
    final_strat = port['cum_strat'].iloc[-1]
    final_bench = port['cum_bench'].iloc[-1]
    
    plt.title(f'Cumulative Growth ($100 Invested): Strategy (${final_strat:,.0f}) vs Benchmark (${final_bench:,.0f})', fontsize=16)
    plt.ylabel('Portfolio Value ($) [Log Scale]')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, which='both')
    
    save_plot("cumulative_return_equity.png")
    
    # 2. Rolling Sortino (Better representation)
    # 6 Month Rolling window
    W = 126
    
    def calc_sortino_series(series):
        # Downside Dev
        neg_ret = series.copy()
        neg_ret[neg_ret > 0] = 0
        downside_std = neg_ret.rolling(W).std() * np.sqrt(252)
        mean_ret = series.rolling(W).mean() * 252
        return mean_ret / downside_std.replace(0, np.nan)

    port['roll_sort_bench'] = calc_sortino_series(port['y_true'])
    port['roll_sort_strat'] = calc_sortino_series(port['strat_ret'])
    
    plt.figure(figsize=(12, 6))
    # Area chart for Strategy to show dominance
    plt.fill_between(port['date'], port['roll_sort_strat'], color='#00A36C', alpha=0.3, label='Strategy Sortino')
    plt.plot(port['date'], port['roll_sort_strat'], color='#00A36C', linewidth=1)
    
    plt.plot(port['date'], port['roll_sort_bench'], color='#B0B0B0', label='Benchmark Sortino', linestyle='--')
    
    plt.axhline(0, color='black', linewidth=0.5)
    plt.title('Rolling Risk-Adjusted Return (6-Month Sortino)', fontsize=16)
    plt.legend(loc='upper left')
    save_plot("rolling_sortino.png")
    
    # 3. Underwater Plot (Explained)
    # DD is Peak to Trough
    def calc_dd(series):
        peak = series.cummax()
        return (series - peak) / peak
        
    port['dd_bench'] = calc_dd(port['cum_bench'])
    port['dd_strat'] = calc_dd(port['cum_strat'])
    
    plt.figure(figsize=(12, 5))
    plt.plot(port['date'], port['dd_strat'], color='#00A36C', linewidth=1.5, label='Strategy Drawdown')
    plt.fill_between(port['date'], port['dd_bench'], 0, color='red', alpha=0.2, label='Benchmark Drawdown')
    
    plt.title('Capital Preservation: Underwater Plot', fontsize=16)
    plt.ylabel('Drawdown %')
    plt.legend()
    save_plot("drawdown_underwater.png")

def main():
    print("Generating Final Presentation Charts...")
    plot_concentration_risk()
    plot_narrative_lag_schematic()
    # plot_kmeans_elbow() -> REPLACED BY DENDROGRAM (Existing artifact used)
    plot_real_tsne()
    plot_nvda_signal_engineering()
    plot_volume_interaction_matrix()
    plot_results_final_suite()
    print("Done.")

if __name__ == "__main__":
    main()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def main():
    # Paths
    params_path = "reports/panel_regression/data/rolling_coefficients.csv"
    output_dir = "reports/panel_regression/inference_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(params_path):
        print(f"Error: {params_path} not found.")
        return
        
    df = pd.read_csv(params_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter Models
    # We want to see how 'Total Sentiment' beta evolves in the Sentiment Model
    sent_model = df[df['model'] == 'Panel_Sentiment'].copy()
    news_model = df[df['model'] == 'Panel_News'].copy()
    base_model = df[df['model'] == 'Panel_Baseline'].copy()
    
    # 1. Plot Sentiment Beta Over Time
    if not sent_model.empty:
        plt.figure(figsize=(12, 6))
        # Plot Lagged Sentiment Beta
        plt.plot(sent_model['date'], sent_model['day_sentiment_lag1'], label='Lagged Sentiment Beta', color='#9467bd', linewidth=2)
        # Plot Contemporaneous Sentiment Beta
        plt.plot(sent_model['date'], sent_model['day_sentiment'], label='Current Day Sentiment Beta', color='#d62728', alpha=0.6, linewidth=1.5)
        
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.title('Performance Inference: Rolling Sentiment Sensitivity (252-Day Window)', fontsize=14, weight='bold')
        plt.ylabel('Beta (Unit Impact of Sentiment)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rolling_sentiment_beta.png'), dpi=150)
        plt.close()
        
    # 2. Plot News Volume Beta Over Time
    if not news_model.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(news_model['date'], news_model['log_total_news_lag1'], label='Lagged News Volume Beta', color='#ff7f0e', linewidth=2)
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.title('Performance Inference: Rolling News Volume Sensitivity (252-Day Window)', fontsize=14, weight='bold')
        plt.ylabel('Beta (Unit Impact of log Volume)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rolling_news_beta.png'), dpi=150)
        plt.close()
        
    # 3. Plot Market Beta Stability (Panel vs Static?)
    # Just show Panel Market Beta from Baseline
    if not base_model.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(base_model['date'], base_model['Mkt-RF'], label='Market Beta (Panel)', color='black', linewidth=2)
        plt.axhline(1, color='red', linestyle='--', alpha=0.5, label='Beta = 1.0')
        plt.title('Performance Inference: Market Beta Stability', fontsize=14, weight='bold')
        plt.ylabel('Market Beta', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rolling_market_beta.png'), dpi=150)
        plt.close()
        
    print(f"Rolling Beta plots saved to {output_dir}")

if __name__ == "__main__":
    main()

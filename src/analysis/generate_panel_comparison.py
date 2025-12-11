
import pandas as pd
import os
import logging
import sys

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    # Paths
    panel_metrics_path = "reports/panel_regression/data/panel_metrics.csv"
    individual_baseline_dir = "reports/window_252d/baseline/data"
    
    if not os.path.exists(panel_metrics_path):
        logger.error("Panel metrics not found.")
        return

    # Load Panel Metrics
    # Columns: symbol, model, oos_r2, oos_rmse
    panel_df = pd.read_csv(panel_metrics_path)
    
    # Load Individual Baseline Metrics
    # We want to compare Panel Baseline vs Individual Baseline
    # And Panel News vs Panel Baseline
    
    ind_metrics = []
    files = [f for f in os.listdir(individual_baseline_dir) if f.endswith('_metrics.csv')]
    for f in files:
        sym = f.replace('_metrics.csv', '')
        df = pd.read_csv(os.path.join(individual_baseline_dir, f))
        r2 = df.iloc[0]['oos_r2']
        rmse = df.iloc[0]['oos_rmse']
        ind_metrics.append({'symbol': sym, 'ind_baseline_r2': r2, 'ind_baseline_rmse': rmse})
        
    ind_df = pd.DataFrame(ind_metrics)
    
    # Merge
    # We need to pivot panel_df to have columns for each model
    panel_pivot = panel_df.pivot(index='symbol', columns='model', values='oos_r2').reset_index()
    # Panel models: Panel_Baseline, Panel_News, Panel_Sentiment
    
    merged = pd.merge(ind_df, panel_pivot, on='symbol', how='inner')
    
    # Generate Report
    lines = []
    lines.append("# Panel Regression Analysis Results")
    lines.append("## Overview")
    lines.append("Comparison of **Pooled Panel Models** (learning from all stocks simultaneously with Fixed Effects) vs. **Individual Stock Models**.")
    lines.append("Rolling Window: 252 Days.")
    lines.append("")
    
    lines.append("## Key Comparison: Individual vs. Pooled")
    
    header = "| Symbol | Ind. Baseline (FF5) | Panel Baseline | Panel News | Panel Sentiment | Best Approach |"
    lines.append(header)
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for idx, row in merged.iterrows():
        sym = row['symbol']
        ind_base = row['ind_baseline_r2']
        pan_base = row.get('Panel_Baseline', None)
        pan_news = row.get('Panel_News', None)
        pan_sent = row.get('Panel_Sentiment', None)
        
        # Determine Winner
        scores = {
            'Individual': ind_base,
            'Panel Base': pan_base,
            'Panel News': pan_news,
            'Panel Sent': pan_sent
        }
        # Filter None
        valid_scores = {k: v for k, v in scores.items() if pd.notnull(v)}
        winner = max(valid_scores, key=valid_scores.get)
        
        def fmt(v): return f"{v:.4f}" if pd.notnull(v) else "N/A"
        
        line = f"| **{sym}** | {fmt(ind_base)} | {fmt(pan_base)} | {fmt(pan_news)} | {fmt(pan_sent)} | **{winner}** |"
        lines.append(line)
        
    lines.append("")
    lines.append("## Aggregate Performance (Mean OOS R²)")
    mean_ind = merged['ind_baseline_r2'].mean()
    mean_pan_base = merged['Panel_Baseline'].mean()
    mean_pan_news = merged['Panel_News'].mean()
    mean_pan_sent = merged['Panel_Sentiment'].mean()
    
    lines.append(f"- **Individual Baseline:** {mean_ind:.4f}")
    lines.append(f"- **Panel Baseline:** {mean_pan_base:.4f}")
    lines.append(f"- **Panel News:** {mean_pan_news:.4f}")
    lines.append(f"- **Panel Sentiment:** {mean_pan_sent:.4f}")
    
    output_path = "reports/panel_regression/panel_comparison_summary.md"
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
        
    logger.info(f"Panel comparison generated at {output_path}")
    print(open(output_path).read())

if __name__ == "__main__":
    main()

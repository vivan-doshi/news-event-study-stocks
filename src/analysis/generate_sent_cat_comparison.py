
import pandas as pd
import os
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    # Paths
    baseline_dir = "reports/fama_french_augmented/data"
    sent_cat_dir = "reports/fama_french_sentiment_categories/data"
    output_report_path = "reports/fama_french_sentiment_categories/sent_cat_analysis_summary.md"
    
    # Load List of processed symbols
    metrics_files = [f for f in os.listdir(sent_cat_dir) if f.endswith('_metrics.csv')]
    symbols = sorted([f.replace('_metrics.csv', '') for f in metrics_files])
    
    report_lines = []
    report_lines.append("# Sentiment Category Augmented Analysis Results")
    report_lines.append("## ⚠️ CRITICAL WARNING: Overfitting Detected (Again)")
    report_lines.append("Similar to the News Count analysis, adding 15 individual sentiment scores (even without lags) increased the model dimensionality too much for the 60-day rolling window.")
    report_lines.append("")
    report_lines.append("While some stocks (AAPL, MSFT) survived with reasonable (though lower) performance, others (AMZN, NVDA, TSLA) suffered catastrophic failure due to multicollinearity/noise.")
    report_lines.append("")
    
    summary_table = []
    summary_table.append("| Symbol | Baseline OOS R² | Sent. Cat. OOS R² | Change | Baseline RMSE | Sent. Cat. RMSE | Improvement? |")
    summary_table.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for sym in symbols:
        baseline_file = os.path.join(baseline_dir, f"{sym}_metrics.csv")
        sent_cat_file = os.path.join(sent_cat_dir, f"{sym}_metrics.csv")
        
        if not os.path.exists(baseline_file) or not os.path.exists(sent_cat_file):
            continue
            
        base = pd.read_csv(baseline_file).iloc[0]
        sent = pd.read_csv(sent_cat_file).iloc[0]
        
        base_r2 = base['oos_r2']
        sent_r2 = sent['oos_r2']
        diff_r2 = sent_r2 - base_r2
        
        base_rmse = base['oos_rmse']
        sent_rmse = sent['oos_rmse']
        
        improved = "✅ YES" if sent_r2 > base_r2 else "❌ NO"
        
        # Handle extremely large negative numbers for display
        if sent_r2 < -10:
            sent_r2_str = "Collapsed (< -10)"
        else:
            sent_r2_str = f"{sent_r2:.4f}"
            
        summary_table.append(f"| {sym} | {base_r2:.4f} | {sent_r2_str} | {diff_r2:.2f} | {base_rmse:.5f} | {sent_rmse:.5f} | {improved} |")
        
    report_lines.extend(summary_table)
    report_lines.append("")
    
    report_lines.append("## Detailed Exposure (Top Sentiment Categories)")
    
    for sym in symbols:
        report_lines.append(f"### {sym}")
        report_lines.append(f"![{sym} Sent. Categories]({os.path.abspath(f'reports/fama_french_sentiment_categories/plots/{sym}_sent_cat_top6.png')})")
        report_lines.append("")

    with open(output_report_path, 'w') as f:
        f.write('\n'.join(report_lines))
        
    logger.info(f"Comparison report generated at {output_report_path}")

if __name__ == "__main__":
    main()

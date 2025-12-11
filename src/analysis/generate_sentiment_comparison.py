
import pandas as pd
import os
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    # Paths
    baseline_dir = "reports/fama_french_augmented/data"
    sentiment_dir = "reports/fama_french_sentiment/data"
    output_report_path = "reports/fama_french_sentiment/sentiment_analysis_summary.md"
    
    # Load List of processed symbols
    metrics_files = [f for f in os.listdir(sentiment_dir) if f.endswith('_metrics.csv')]
    symbols = sorted([f.replace('_metrics.csv', '') for f in metrics_files])
    
    report_lines = []
    report_lines.append("# Sentiment Augmented Analysis Results")
    report_lines.append("Comparing Baseline (Fama-French + Lags) vs. Sentiment Augmented (FF + Lags + Daily Sentiment)")
    report_lines.append("")
    
    summary_table = []
    summary_table.append("| Symbol | Baseline OOS R² | Sentiment OOS R² | Change | Baseline RMSE | Sentiment RMSE | Improvement? |")
    summary_table.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for sym in symbols:
        baseline_file = os.path.join(baseline_dir, f"{sym}_metrics.csv")
        sentiment_file = os.path.join(sentiment_dir, f"{sym}_metrics.csv")
        
        if not os.path.exists(baseline_file) or not os.path.exists(sentiment_file):
            continue
            
        base = pd.read_csv(baseline_file).iloc[0]
        sent = pd.read_csv(sentiment_file).iloc[0]
        
        base_r2 = base['oos_r2']
        sent_r2 = sent['oos_r2']
        diff_r2 = sent_r2 - base_r2
        
        base_rmse = base['oos_rmse']
        sent_rmse = sent['oos_rmse']
        
        improved = "✅ YES" if sent_r2 > base_r2 else "❌ NO"
        
        summary_table.append(f"| {sym} | {base_r2:.4f} | {sent_r2:.4f} | {diff_r2:+.4f} | {base_rmse:.5f} | {sent_rmse:.5f} | {improved} |")
        
        # Detailed Individual Sections
    
    report_lines.extend(summary_table)
    report_lines.append("")
    
    report_lines.append("## Detailed Exposure")
    
    for sym in symbols:
        report_lines.append(f"### {sym}")
        report_lines.append(f"![{sym} With Sentiment]({os.path.abspath(f'reports/fama_french_sentiment/plots/{sym}_sentiment_augmented.png')})")
        report_lines.append("")

    with open(output_report_path, 'w') as f:
        f.write('\n'.join(report_lines))
        
    logger.info(f"Comparison report generated at {output_report_path}")

if __name__ == "__main__":
    main()

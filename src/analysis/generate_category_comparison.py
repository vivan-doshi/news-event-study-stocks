
import pandas as pd
import os
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    # Paths
    baseline_dir = "reports/fama_french_augmented/data"
    news_dir = "reports/fama_french_news_categories/data"
    output_report_path = "reports/fama_french_news_categories/category_analysis_summary.md"
    
    # Load List of processed symbols (assuming same for both)
    # We can inspect the metrics files in news_dir
    metrics_files = [f for f in os.listdir(news_dir) if f.endswith('_metrics.csv')]
    symbols = sorted([f.replace('_metrics.csv', '') for f in metrics_files])
    
    report_lines = []
    report_lines.append("# News Category Augmented Analysis Results")
    report_lines.append("## ⚠️ CRITICAL WARNING: Overfitting Detected")
    report_lines.append("Adding 15 individually detailed news categories to the model drastically increased the number of parameters. With a 60-day window, this caused **catastrophic overfitting**.")
    report_lines.append("")
    report_lines.append("The model learned to fit the noise in the training window almost perfectly (High In-Sample R²), but completely failed to predict the next day (Extremely Negative OOS R²).")
    report_lines.append("")
    
    summary_table = []
    summary_table.append("| Symbol | Baseline OOS R² | News OOS R² | Change | Baseline RMSE | News RMSE | Improvement? |")
    summary_table.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for sym in symbols:
        baseline_file = os.path.join(baseline_dir, f"{sym}_metrics.csv")
        news_file = os.path.join(news_dir, f"{sym}_metrics.csv")
        
        if not os.path.exists(baseline_file) or not os.path.exists(news_file):
            continue
            
        base = pd.read_csv(baseline_file).iloc[0]
        news = pd.read_csv(news_file).iloc[0]
        
        base_r2 = base['oos_r2']
        news_r2 = news['oos_r2']
        diff_r2 = news_r2 - base_r2
        
        base_rmse = base['oos_rmse']
        news_rmse = news['oos_rmse']
        
        improved = "✅ YES" if news_r2 > base_r2 else "❌ NO"
        
        # Handle extremely large negative numbers for display
        if news_r2 < -10:
            news_r2_str = "Collapsed (< -10)"
        else:
            news_r2_str = f"{news_r2:.4f}"
            
        summary_table.append(f"| {sym} | {base_r2:.4f} | {news_r2_str} | {diff_r2:.2f} | {base_rmse:.5f} | {news_rmse:.5f} | {improved} |")
        
        # Detailed Individual Sections
    
    report_lines.extend(summary_table)
    report_lines.append("")
    
    report_lines.append("## Detailed Exposure (Top Categories)")
    
    for sym in symbols:
        report_lines.append(f"### {sym}")
        report_lines.append(f"![{sym} Categories]({os.path.abspath(f'reports/fama_french_news_categories/plots/{sym}_category_top6.png')})")
        report_lines.append("")

    with open(output_report_path, 'w') as f:
        f.write('\n'.join(report_lines))
        
    logger.info(f"Comparison report generated at {output_report_path}")

if __name__ == "__main__":
    main()

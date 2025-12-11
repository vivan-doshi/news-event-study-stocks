
import pandas as pd
import os
import logging
import sys

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    # Root Dir for 252d analysis
    root_dir = "reports/window_252d"
    output_report_path = "reports/window_252d/master_comparison_summary.md"
    
    # Models to compare
    models = {
        'Baseline (FF5+Lags)': 'baseline',
        'News Volume': 'news_vol',
        'Total Sentiment': 'sentiment',
        'News Categories (15)': 'news_cat',
        'Sent Categories (15)': 'sent_cat'
    }
    
    # Symbols (get from baseline)
    baseline_path = os.path.join(root_dir, 'baseline', 'data')
    files = [f for f in os.listdir(baseline_path) if f.endswith('_metrics.csv')]
    symbols = sorted([f.replace('_metrics.csv', '') for f in files])
    
    # Data container
    results = {} # sym -> {model: oos_r2}
    
    for sym in symbols:
        results[sym] = {}
        for model_name, folder_name in models.items():
            path = os.path.join(root_dir, folder_name, 'data', f"{sym}_metrics.csv")
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    if not df.empty:
                        results[sym][model_name] = df.iloc[0]['oos_r2']
                    else:
                         results[sym][model_name] = None
                except:
                    results[sym][model_name] = None
            else:
                results[sym][model_name] = None
                
    # Generate Report
    lines = []
    lines.append("# 252-Day Rolling Window Analysis Comparison")
    lines.append("## Overview")
    lines.append("Comparing Out-of-Sample R² across 5 models using a **252-day (1 year)** rolling window.")
    lines.append("A longer window provides more stability and reduces overfitting for complex models.")
    lines.append("")
    
    # Table Header
    header = "| Symbol | Baseline | News Vol | Sentiment | News Cat | Sent Cat | Best Model |"
    lines.append(header)
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for sym in symbols:
        row_data = results.get(sym, {})
        
        # Get values
        base = row_data.get('Baseline (FF5+Lags)')
        news = row_data.get('News Volume')
        sent = row_data.get('Total Sentiment')
        ncat = row_data.get('News Categories (15)')
        scat = row_data.get('Sent Categories (15)')
        
        # Determine Winner
        # Filter None
        valid_scores = {k: v for k, v in row_data.items() if v is not None}
        if not valid_scores:
            continue
            
        winner_name = max(valid_scores, key=valid_scores.get)
        winner_val = valid_scores[winner_name]
        
        # Format columns (Highlight winner with bold if significant?)
        def fmt(val):
            if val is None: return "N/A"
            if val < -1: return "Fail (< -1)"
            return f"{val:.4f}"
            
        line = f"| **{sym}** | {fmt(base)} | {fmt(news)} | {fmt(sent)} | {fmt(ncat)} | {fmt(scat)} | **{winner_name}** |"
        lines.append(line)
        
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- **Baseline (FF5 + Lags):** The standard financial model.")
    lines.append("- **News/Sentiment Models:** Did specific news features improve over the baseline?")
    lines.append("- **Overfitting Check:** With 252 days, do the Category models still crash (negative R²) or do they perform well?")
    lines.append("")
    
    with open(output_report_path, 'w') as f:
        f.write('\n'.join(lines))
        
    logger.info(f"Master comparison generated at {output_report_path}")
    print(open(output_report_path).read()) # Print to stdout for immediate view

if __name__ == "__main__":
    main()

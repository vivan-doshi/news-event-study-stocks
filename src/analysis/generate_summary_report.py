
import pandas as pd
import statsmodels.api as sm
import os
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    # Paths
    master_data_path = 'data/master_analysis_data.csv'
    rolling_stats_path = 'reports/fama_french_augmented/data/rolling_augmented_stats.csv'
    output_report_path = 'reports/fama_french_augmented/analysis_summary.md'
    
    # Load Data
    df = pd.read_csv(master_data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Define Predictors for Static Regression
    # Included Lags as per previous request
    predictors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 
                  'log_ret_lag1', 'log_ret_lag2', 'log_ret_lag5', 'log_ret_lag10', 'log_ret_lag21']
    
    # Prepare Report Content
    report_lines = []
    report_lines.append("# Fama-French Augmented Analysis Results")
    report_lines.append(f"**Date Range:** {df['date'].min().date()} to {df['date'].max().date()}")
    report_lines.append("")
    
    symbols = df['symbol'].unique()
    
    for sym in symbols:
        report_lines.append(f"## {sym}")
        
        # 1. Static OLS (Standard Output)
        sym_data = df[df['symbol'] == sym].copy()
        sym_data = sym_data.dropna(subset=['log_ret', 'RF'] + predictors)
        
        if sym_data.empty:
            report_lines.append("Insufficient data for static regression.")
            continue
            
        y = sym_data['log_ret'] - sym_data['RF']
        X = sym_data[predictors]
        X = sm.add_constant(X)
        
        model = sm.OLS(y, X)
        results = model.fit()
        
        report_lines.append("### Static Regression Output")
        report_lines.append("```text")
        report_lines.append(str(results.summary()))
        report_lines.append("```")
        report_lines.append("")
        
        # 2. Rolling Model Performance Metrics
        metrics_file = f"reports/fama_french_augmented/data/{sym}_metrics.csv"
        if os.path.exists(metrics_file):
            metrics = pd.read_csv(metrics_file).iloc[0]
            report_lines.append("### Rolling Model Performance (Avg Window / OOS)")
            report_lines.append("| Metric | Value | Description |")
            report_lines.append("| :--- | :--- | :--- |")
            report_lines.append(f"| **In-Sample Avg R²** | {metrics['avg_is_r2']:.4f} | Average R² across all rolling windows |")
            report_lines.append(f"| **In-Sample RMSE** | {metrics['is_rmse']:.5f} | RMS Error of in-sample fits |")
            report_lines.append(f"| **Out-of-Sample R²** | {metrics['oos_r2']:.4f} | 1-Step Ahead Prediction R² |")
            report_lines.append(f"| **Out-of-Sample RMSE** | {metrics['oos_rmse']:.5f} | RMS Error of 1-step ahead predictions |")
            report_lines.append("")
        
        # 3. Rolling Factor Exposure Plots
        report_lines.append("### Rolling Factor Exposure Plots")
        report_lines.append(f"![{sym} Rolling Factors]({os.path.abspath(f'reports/fama_french_augmented/plots/{sym}_augmented.png')})")
        report_lines.append("")
        
    # Write Report
    with open(output_report_path, 'w') as f:
        f.write('\n'.join(report_lines))
        
    logger.info(f"Report generated at {output_report_path}")

if __name__ == "__main__":
    main()


import pandas as pd
import numpy as np
import statsmodels.api as sm
import argparse
import os

def main():
    input_path = "data/master_analysis_data.csv"
    output_dir = "reports/panel_regression/inference"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= '2023-01-01'].copy()
    
    # 2. Feature Engineering (Same as Rolling)
    df['log_total_news'] = np.log1p(df['total_news'])
    df['log_total_news_lag1'] = df.groupby('symbol')['log_total_news'].shift(1)
    df['day_sentiment_lag1'] = df.groupby('symbol')['day_sentiment'].shift(1)
    
    # Create Dummies (Fixed Effects)
    df_dummies = pd.get_dummies(df, columns=['symbol'], prefix='dummy', drop_first=True)
    
    # 3. Define Models to Test
    base_vars = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 
                 'log_ret_lag1', 'log_ret_lag2', 'log_ret_lag5', 'log_ret_lag10', 'log_ret_lag21']
    
    # Identify dummy columns
    dummy_cols = [c for c in df_dummies.columns if c.startswith('dummy_')]
    
    models_config = {
        'Panel_Baseline': base_vars + dummy_cols,
        'Panel_News': base_vars + dummy_cols + ['log_total_news', 'log_total_news_lag1'],
        'Panel_Sentiment': base_vars + dummy_cols + ['day_sentiment', 'day_sentiment_lag1']
    }
    
    results_str = []
    
    for name, predictors in models_config.items():
        # Prepare Data
        sub = df_dummies.dropna(subset=['log_ret', 'RF'] + predictors)
        y = (sub['log_ret'] - sub['RF']).astype(float)
        X = sm.add_constant(sub[predictors].astype(float))
        
        # Fit Model (Robust Covariance type 'HC1' for heteroskedasticity)
        model = sm.OLS(y, X).fit(cov_type='HC1')
        
        results_str.append(f"==================================================")
        results_str.append(f"Model: {name}")
        results_str.append(f"Observations: {len(y)}")
        results_str.append(f"Adj. R-squared: {model.rsquared_adj:.4f}")
        results_str.append(f"==================================================")
        results_str.append(str(model.summary()))
        results_str.append("\n")
        
    # Save to file
    with open(os.path.join(output_dir, 'static_inference_results.txt'), 'w') as f:
        f.write('\n'.join(results_str))
        
    print(f"Inference output saved to {os.path.join(output_dir, 'static_inference_results.txt')}")
    print('\n'.join(results_str))

if __name__ == "__main__":
    main()

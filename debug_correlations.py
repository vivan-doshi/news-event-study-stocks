
import pandas as pd
import numpy as np

def check_correlations():
    df = pd.read_csv('data/master_analysis_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date'])
    
    # Calculate T+1 Return
    df['ret_t1'] = df.groupby('symbol')['daily_return'].shift(-1)
    
    # Check Correlations
    # 1. Contemporaneous (Should be High)
    corr_contemp = df[['daily_return', 'Mkt-RF', 'news_volume', 'avg_sentiment']].corr()['daily_return']
    
    # 2. Predictive (Should be Low)
    # We correlate X_t (Mkt-RF, etc.) with Ret_{t+1}
    # We just explicitly look at the correlation between 'ret_t1' and factors
    corr_pred = df[['ret_t1', 'daily_return', 'Mkt-RF', 'news_volume', 'avg_sentiment']].corr()['ret_t1']
    
    print("=== CONTEMPORANEOUS CORRELATION (X_t vs Ret_t) ===")
    print(corr_contemp)
    print("\n=== PREDICTIVE CORRELATION (X_t vs Ret_{t+1}) ===")
    print(corr_pred)
    
    # Check Autocorrelation of Returns
    df['ret_lag1'] = df.groupby('symbol')['daily_return'].shift(1)
    autocorr = df['daily_return'].corr(df['ret_lag1'])
    print(f"\nReturn Autocorrelation (Lag 1): {autocorr:.4f}")

if __name__ == "__main__":
    check_correlations()

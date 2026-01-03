
import pandas as pd
import numpy as np

def test_baselines():
    df = pd.read_csv('data/master_analysis_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date'])
    
    # Calculate excess ret T+1 (Target)
    df['excess_ret'] = df['daily_return'] - df['RF']
    df['target_t1'] = df.groupby('symbol')['excess_ret'].shift(-1)
    
    df = df.dropna(subset=['target_t1', 'Mkt-RF', 'excess_ret'])
    
    # 1. Always Buy Strategy
    # Pred = 1.0 always
    strat_buy = 1.0 * df['target_t1']
    sharpe_buy = (strat_buy.mean() / strat_buy.std()) * np.sqrt(252)
    hit_buy = np.mean(df['target_t1'] > 0)
    
    # 2. Momentum Strategy (Buy if Today was Green)
    # Pred = Sign(excess_ret)
    strat_mom = np.sign(df['excess_ret']) * df['target_t1']
    sharpe_mom = (strat_mom.mean() / strat_mom.std()) * np.sqrt(252)
    hit_mom = np.mean(np.sign(df['excess_ret']) == np.sign(df['target_t1']))
    
    print(f"=== BASELINE CHECKS (N={len(df)}) ===")
    print(f"Total Positive Days (Hit Rate for Always Buy): {hit_buy:.2%}")
    print(f"Sharpe for Always Buy: {sharpe_buy:.4f}")
    
    print("\n=== MOMENTUM CHECK ===")
    print(f"Hit Rate for 'Buy if Green': {hit_mom:.2%}")
    print(f"Sharpe for 'Buy if Green': {sharpe_mom:.4f}")
    
    # Check NVDA specifically
    nvda = df[df['symbol'].str.contains('NVDA')]
    strat_buy_nvda = nvda['target_t1']
    sharpe_nvda = (strat_buy_nvda.mean() / strat_buy_nvda.std()) * np.sqrt(252)
    print(f"\n=== NVDA ONLY ===")
    print(f"Sharpe for NVDA Buy & Hold: {sharpe_nvda:.4f}")
    print(f"Max T+1 Return: {nvda['target_t1'].max():.2%}")

if __name__ == "__main__":
    test_baselines()

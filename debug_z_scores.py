import pandas as pd
import os

file_path = "data/master_analysis_data_advanced_clean.csv"

if not os.path.exists(file_path):
    print("File not found!")
    exit()

df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])

tickers = ["AAPL", "MSFT", "TSLA", "NVDA"]

# Mapping from previous logic
# 0: AI and EV rally
# 3: Trump/Macro

print(f"Total Rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

for ticker in tickers:
    print(f"\n--- {ticker} ---")
    if 'ticker_yf' in df.columns:
        t_col = 'ticker_yf'
    else:
        t_col = 'symbol'
        
    mask = df[t_col].astype(str).str.contains(ticker, case=False, na=False)
    sub = df[mask].sort_values('date')
    
    if sub.empty:
        print("No data found.")
        continue
        
    print(f"Rows for {ticker}: {len(sub)}")
    print(f"Date Range: {sub['date'].min()} to {sub['date'].max()}")
    
    for t_id in [0, 1, 2, 3, 4]:
        col = f"z_score_topic_{t_id}"
        if col not in sub.columns:
            print(f"Column {col} missing!")
            continue
            
        non_zeros = sub[sub[col] != 0]
        cnt = len(non_zeros)
        print(f"Topic {t_id} ({col}): {cnt} non-zeros ({cnt/len(sub)*100:.1f}%)")
        if cnt > 0:
            last_date = non_zeros.iloc[-1]['date']
            last_val = non_zeros.iloc[-1][col]
            print(f"  Last Non-Zero: {last_date.date()} -> {last_val:.4f}")
        else:
            print("  Last Non-Zero: NEVER")


import pandas as pd
import numpy as np

def main():
    try:
        df = pd.read_csv('reports/regression_t1/deep_dive_model_performance.csv')
    except FileNotFoundError:
        # Fallback if I messed up the path in previous thoughts
        df = pd.read_csv('reports/regression/deep_dive_model_performance.csv')
    
    # Group by model to get stats
    stats = df.groupby('model')['sharpe'].agg(['min', 'median', 'max', 'mean']).reset_index()
    
    # Sort by Median (descending)
    stats = stats.sort_values('median', ascending=False)
    
    print("| Rank | Model | Median Sharpe | Sharpe Range (Min - Max) |")
    print("| :--- | :--- | :--- | :--- |")
    
    rank = 1
    for _, row in stats.iterrows():
        model = row['model']
        med = row['median']
        min_s = row['min']
        max_s = row['max']
        
        # Formatting range string
        range_str = f"{min_s:.2f} - {max_s:.2f}"
        
        print(f"| {rank} | {model} | **{med:.2f}** | {range_str} |")
        rank += 1

if __name__ == "__main__":
    main()


import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime, time

def load_data(news_path, stock_path, map_path):
    print(f"Loading news data from {news_path}...")
    try:
        df = pd.read_parquet(news_path)
    except Exception as e:
        print(f"Error loading news data: {e}")
        raise

    print(f"Loading stock data from {stock_path}...")
    try:
        yf_df = pd.read_parquet(stock_path)
    except Exception as e:
        print(f"Error loading stock data: {e}")
        raise

    print(f"Loading topic map from {map_path}...")
    schema = {
        'topic_id': 'Int32',
        'label': 'string',
    }
    try:
        map_df = pd.read_csv(map_path, encoding="utf-8", dtype=schema)
    except Exception as e:
        print(f"Error loading topic map: {e}")
        raise

    return df, yf_df, map_df

def adjust_dates(df):
    print("Adjusting dates (4PM cutoff, weekends, holidays)...")
    
    # 1. Merge with map (moved here to ensure topic_label_auto/label is available if needed, though topic_id_kmeans is key)
    # The notebook did this merge early on. 
    # Logic from notebook:
    # df1 = pd.merge(df, map_df, how='left', left_on='topic_id_kmeans', right_on='topic_id')
    
    # Holiday list (hardcoded from notebook)
    nasdaq_holidays = [
        '2023-01-02', '2023-01-16', '2023-02-20', '2023-04-07', '2023-05-29', 
        '2023-06-19', '2023-07-04', '2023-09-04', '2023-11-23', '2023-12-25',
        '2024-01-01', '2024-01-15', '2024-02-19', '2024-03-29', '2024-05-27', 
        '2024-06-19', '2024-07-04', '2024-09-02', '2024-11-28', '2024-12-25',
        '2025-01-01', '2025-01-20', '2025-02-17', '2025-04-18', '2025-05-26', 
        '2025-06-19', '2025-07-04', '2025-09-01', '2025-11-27', '2025-12-25'
    ]
    nasdaq_holidays = pd.to_datetime(nasdaq_holidays).date
    holidays_np = np.array(nasdaq_holidays, dtype='datetime64[D]')

    cutoff_hour = 16

    # Step 1: move timestamps after cutoff to next day
    # Be careful with copies
    next_days = pd.to_datetime(df['published_at']).copy()
    
    # Vectorized check for cutoff
    # We need to access the hour. ensure it's datetime
    add_day_mask = (next_days.dt.hour >= cutoff_hour)
    next_days += pd.to_timedelta(add_day_mask.astype(int), unit='D')

    # Step 2: normalize time to midnight
    next_days = next_days.dt.normalize()

    # Step 3: vectorized loop to skip weekends and holidays
    # Ensure next_days is datetime64[ns] to work with dt accessor
    
    mask = (next_days.dt.weekday >= 5) | np.isin(next_days.dt.date, holidays_np)
    
    # Safety counter to prevent infinite loops if something is wrong (though unlikely with dates)
    loop_count = 0
    while mask.any() and loop_count < 100:
        next_days.loc[mask] += pd.Timedelta(days=1)
        mask = (next_days.dt.weekday >= 5) | np.isin(next_days.dt.date, holidays_np)
        loop_count += 1
    
    if loop_count >= 100:
        print("Warning: Date adjustment loop hit limit. Check holiday logic.")

    df['final_date_for_news'] = next_days.dt.strftime('%Y-%m-%d')
    return df

def aggregate_data(df, map_df):
    print("Aggregating data (Pivot tables)...")
    
    # Merge map first to get labels
    # Note: notebook used topic_id_kmeans to join with map_df's topic_id
    merged_df = pd.merge(df, map_df, how='left', left_on='topic_id_kmeans', right_on='topic_id')
    
    # Create pivots
    
    # 1. Sentiment Finbert (Mean)
    pivoted_sentiment_finbert = merged_df.pivot_table(
        index=['symbol_query', 'final_date_for_news'],
        columns='label',
        values='sentiment_finbert',
        aggfunc='mean'
    ).fillna(0)
    pivoted_sentiment_finbert.columns = ['sentiment_finbert_' + col for col in pivoted_sentiment_finbert.columns]

    # 2. Total Count (Count) - using article_id or similar unique identifier?
    # Notebook typically just counts rows. if 'topic_id_kmeans' is present, we count it.
    pivoted_total_count = merged_df.pivot_table(
        index=['symbol_query', 'final_date_for_news'],
        columns='label',
        values='topic_id_kmeans', # Count any non-null column
        aggfunc='count'
    ).fillna(0)
    pivoted_total_count.columns = ['total_count_' + col for col in pivoted_total_count.columns]
    
    # 3. Sentiment Negative (Mean)
    pivoted_sentiment_neg = merged_df.pivot_table(
        index=['symbol_query', 'final_date_for_news'],
        columns='label',
        values='sent_neg',
        aggfunc='mean'
    ).fillna(0)
    pivoted_sentiment_neg.columns = ['sentiment_neg_' + col for col in pivoted_sentiment_neg.columns]

    # 4. Sentiment Neutral (Mean)
    pivoted_sentiment_neu = merged_df.pivot_table(
        index=['symbol_query', 'final_date_for_news'],
        columns='label',
        values='sent_neu',
        aggfunc='mean'
    ).fillna(0)
    pivoted_sentiment_neu.columns = ['sentiment_neutral_' + col for col in pivoted_sentiment_neu.columns]

    # 5. Sentiment Positive (Mean)
    pivoted_sentiment_pos = merged_df.pivot_table(
        index=['symbol_query', 'final_date_for_news'],
        columns='label',
        values='sent_pos',
        aggfunc='mean'
    ).fillna(0)
    pivoted_sentiment_pos.columns = ['sentiment_pos_' + col for col in pivoted_sentiment_pos.columns]

    # Merge all pivots
    dfs = [pivoted_sentiment_finbert, pivoted_total_count, pivoted_sentiment_neg, pivoted_sentiment_neu, pivoted_sentiment_pos]
    final_pivot = pd.concat(dfs, axis=1)
    
    return final_pivot.reset_index()

def feature_engineering_main(news_path, stock_path, map_path, output_path):
    print("Starting Feature Engineering...")
    
    # 1. Load
    df, yf_df, map_df = load_data(news_path, stock_path, map_path)
    
    # 2. Adjust Dates
    df = adjust_dates(df)
    
    # 3. Aggregate
    aggregated_df = aggregate_data(df, map_df)
    
    # 4. Merge with Stock Data
    print("Merging with stock data...")
    # Ensure date types match for merge
    # aggregated_df['final_date_for_news'] is string YYYY-MM-DD
    # yf_df['date'] might be datetime or string. check notebook.
    # Notebook: final_df = pd.merge(pivoted_df, yf_df, how='left', left_on=['symbol_query', 'final_date_for_news'], right_on=['symbol_query', 'date'])
    
    # Let's ensure yf_df date is string for safe merge, or convert both to datetime.
    # Safe bet: convert to datetime then back to string or just use datetime.
    # In notebook: df1['final_date_for_news'] = df1['final_date_for_news'].dt.strftime('%Y-%m-%d')
    # So it uses string.
    
    # Inspect yf_df in a real run if possible, but safe assumption is to standardise.
    if 'date' in yf_df.columns:
         if not pd.api.types.is_string_dtype(yf_df['date']):
             yf_df['date_str'] = pd.to_datetime(yf_df['date']).dt.strftime('%Y-%m-%d')
         else:
             yf_df['date_str'] = yf_df['date']
    else:
        # If date is index
        yf_df = yf_df.reset_index()
        yf_df['date_str'] = pd.to_datetime(yf_df['date']).dt.strftime('%Y-%m-%d')

    final_df = pd.merge(aggregated_df, yf_df, how='left', 
                        left_on=['symbol_query', 'final_date_for_news'], 
                        right_on=['symbol_query', 'date_str'])

    # Drop the extra key if needed
    if 'date_str' in final_df.columns:
        final_df.drop(columns=['date_str'], inplace=True)

    # 5. Add Derived Features (from common notebook logic)
    print("Adding derived features (total_news, day_sentiment)...")
    
    # total_news: Sum of all total_count_* columns
    count_cols = [c for c in final_df.columns if c.startswith('total_count_')]
    final_df['total_news'] = final_df[count_cols].sum(axis=1)
    
    # day_sentiment: Mean of non-zero sentiment_finbert_* columns
    # Logic: df[cols].replace(0, np.nan).mean(axis=1)
    finbert_cols = [c for c in final_df.columns if c.startswith('sentiment_finbert_')]
    final_df['day_sentiment'] = final_df[finbert_cols].replace(0, np.nan).mean(axis=1)
    
    # 6. Save
    print(f"Saving aggregated data to {output_path}...")
    final_df.to_parquet(output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Engineering for Mag7 Event Study")
    parser.add_argument("--news_path", required=True, help="Path to news parquet file")
    parser.add_argument("--stock_path", required=True, help="Path to stock data parquet file")
    parser.add_argument("--map_path", required=True, help="Path to topic map CSV")
    parser.add_argument("--output_path", required=True, help="Path to save output parquet")
    
    args = parser.parse_args()
    
    feature_engineering_main(
        args.news_path,
        args.stock_path,
        args.map_path,
        args.output_path
    )

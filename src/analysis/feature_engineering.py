
import pandas as pd
import numpy as np
import os
import argparse
import logging
import sys

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# THEMATIC MAPPING
THEME_MAPPING = {
    'Growth': [1, 6, 23, 29, 34, 40, 48],
    'Risk': [8, 9, 30, 33, 44, 45],
    'Macro': [10, 15, 18, 36],
    'Earnings': [13, 14, 32, 37]
}

def load_data(news_path, stock_path, map_path):
    logger.info(f"Loading news data from {news_path}...")
    df = pd.read_parquet(news_path)
    
    logger.info(f"Loading stock data from {stock_path}...")
    yf_df = pd.read_parquet(stock_path)
    
    # Map not strictly needed if we hardcode topics, but kept for consistency
    # logger.info(f"Loading topic map from {map_path}...")
    # map_df = pd.read_csv(map_path) 
    
    # Invert mapping for easier lookup: topic_id -> theme
    topic_to_theme = {}
    for theme, topics in THEME_MAPPING.items():
        for t in topics:
            topic_to_theme[t] = theme
    
    return df, yf_df, topic_to_theme

def adjust_dates(df):
    logger.info("Adjusting dates (4PM cutoff, weekends, holidays)...")
    # Using simple date logic for speed, assuming data might be pre-adjusted or simple aggregation is sufficient
    # Re-using the robust logic from previous version
    
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
    next_days = pd.to_datetime(df['published_at']).copy()
    
    # 4PM Cutoff
    add_day_mask = (next_days.dt.hour >= cutoff_hour)
    next_days += pd.to_timedelta(add_day_mask.astype(int), unit='D')
    next_days = next_days.dt.normalize()

    # Skip Weekends/Holidays
    mask = (next_days.dt.weekday >= 5) | np.isin(next_days.dt.date, holidays_np)
    loop_count = 0
    while mask.any() and loop_count < 100:
        next_days.loc[mask] += pd.Timedelta(days=1)
        mask = (next_days.dt.weekday >= 5) | np.isin(next_days.dt.date, holidays_np)
        loop_count += 1

    df['final_date_for_news'] = next_days.dt.strftime('%Y-%m-%d')
    return df

def aggregate_thematic(df, topic_to_theme):
    logger.info("Aggregating Thematic Sentiment...")
    
    # 1. Map Topics to Themes
    df['theme'] = df['topic_id_kmeans'].map(topic_to_theme)
    # Fill unmapped topics with 'Other'? Or strictly ignore?
    # User only specified 4 themes. Let's keep others as 'Other' or just purely use those 4.
    # We will prioritize the 4 themes.
    
    # 2. Global Sentiment (All topics)
    global_pivot = df.pivot_table(
        index=['symbol_query', 'final_date_for_news'],
        values=['sentiment_finbert', 'topic_id_kmeans'],
        aggfunc={'sentiment_finbert': 'mean', 'topic_id_kmeans': 'count'}
    ).rename(columns={'sentiment_finbert': 'day_sentiment', 'topic_id_kmeans': 'total_news'})
    
    # 3. Thematic Sentiment
    # Filter only mapped themes
    df_themes = df[df['theme'].notna()].copy()
    
    theme_pivot = df_themes.pivot_table(
        index=['symbol_query', 'final_date_for_news'],
        columns='theme',
        values='sentiment_finbert',
        aggfunc='mean'
    )
    theme_pivot.columns = [f'sent_{col.lower()}' for col in theme_pivot.columns]
    
    # Merge
    final_pivot = pd.concat([global_pivot, theme_pivot], axis=1).fillna(0) # Fill NaNs (no news for theme) with 0?
    # Sentiment 0 is Neutral. If there is NO news, sentiment is 0 (Neutral).
    # That is a reasonable assumption for feature vectors.
    
    return final_pivot.reset_index()

def calculate_sentiment_shocks(df):
    """
    Calculates Rolling Z-Score (Shock) using STRICT lag to prevent lookahead.
    Formula: (Today - Rolling_Mean_Lag1) / Rolling_Std_Lag1
    """
    logger.info("Calculating Sentiment Shocks (Z-Scores)...")
    
    # Identify sentiment columns (Global + Thematic)
    sent_cols = ['day_sentiment'] + [c for c in df.columns if c.startswith('sent_')]
    
    # Sort for rolling
    df = df.sort_values(by=['symbol_query', 'final_date_for_news'])
    
    for col in sent_cols:
        # 1. Shift by 1 to get "Past"
        lagged_series = df.groupby('symbol_query')[col].shift(1)
        
        # 2. Rolling stats on the Lagged Series
        # Window=20
        roll_mean = lagged_series.transform(lambda x: x.rolling(window=20, min_periods=5).mean())
        roll_std = lagged_series.transform(lambda x: x.rolling(window=20, min_periods=5).std())
        
        # 3. Z-Score = (Current - Prior_Mean) / Prior_Std
        # Handle division by zero
        z_score = (df[col] - roll_mean) / roll_std
        z_score = z_score.replace([np.inf, -np.inf], 0).fillna(0)
        
        df[f'{col}_zscore'] = z_score
        
    return df

def calculate_signal_magnitude(df):
    """
    Signal = Sentiment * log(1 + Volume)
    """
    logger.info("Calculating Signal Magnitude...")
    
    sent_cols = ['day_sentiment'] + [c for c in df.columns if c.startswith('sent_') and not c.endswith('_zscore')]
    
    # Use Global Volume for magnitude weighting? 
    # Or should we calculate volume per theme?
    # User said "log(1 + Volume)". Usually implies Total Volume unless specified.
    # However, weighting "Growth Sentiment" by "Total Volume" (which might be mostly Risk news) is noisy.
    # But collecting per-theme volume requires another pivot.
    # "Weight sentiment by volume conviction."
    # Let's check user prompt: "Implement an interaction feature that weights sentiment by volume conviction."
    # Let's stick to Total Volume for simplicity unless we refactor to get theme counts.
    # Actually, getting theme counts is easy. Let's do it for better quality.
    
    # Wait, I need to fetch theme counts in aggregation step.
    # I'll update aggregate_thematic really quick.
    
    vol = df['total_news'] # Global Volume
    log_vol = np.log1p(vol)
    
    for col in sent_cols:
        # Interaction
        df[f'{col}_magnitude'] = df[col] * log_vol
        
    return df

def feature_engineering_main(news_path, stock_path, map_path, output_path):
    
    # 1. Load
    df, yf_df, topic_to_theme = load_data(news_path, stock_path, map_path)
    
    # 2. Adjust Dates
    df = adjust_dates(df)
    
    # 3. Aggregate
    # Create Theme Count Pivot here just in case? 
    # Let's stick to Global Volume for Magnitude as per simplified instructions unless specified.
    # "Formula: Signal = Sentiment * log(1 + Volume)" - likely implies total volume.
    aggregated_df = aggregate_thematic(df, topic_to_theme)
    
    # 4. Merge with Stock Data
    logger.info("Merging with stock data...")
    if 'date' in yf_df.columns:
         if not pd.api.types.is_string_dtype(yf_df['date']):
             yf_df['date_str'] = pd.to_datetime(yf_df['date']).dt.strftime('%Y-%m-%d')
         else:
             yf_df['date_str'] = yf_df['date']
    else:
        yf_df = yf_df.reset_index()
        yf_df['date_str'] = pd.to_datetime(yf_df['date']).dt.strftime('%Y-%m-%d')

    final_df = pd.merge(aggregated_df, yf_df, how='left', 
                        left_on=['symbol_query', 'final_date_for_news'], 
                        right_on=['symbol_query', 'date_str'])
    
    if 'date_str' in final_df.columns:
        final_df.drop(columns=['date_str'], inplace=True)
        
    # 5. Shocks & Magnitude
    final_df = calculate_sentiment_shocks(final_df)
    final_df = calculate_signal_magnitude(final_df)
    
    # 6. Save
    logger.info(f"Saving augmented features to {output_path}...")
    final_df.to_parquet(output_path)
    logger.info(f"Done. Columns: {list(final_df.columns)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--news_path", required=True)
    parser.add_argument("--stock_path", required=True)
    parser.add_argument("--map_path", required=True)
    parser.add_argument("--output_path", required=True)
    
    args = parser.parse_args()
    
    feature_engineering_main(
        args.news_path,
        args.stock_path,
        args.map_path,
        args.output_path
    )

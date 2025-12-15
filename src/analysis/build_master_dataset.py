import pandas as pd
import numpy as np
import os
import argparse
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(news_path, stock_path, ff_path):
    logger.info("Loading datasets...")
    
    # News & Stock
    news = pd.read_parquet(news_path)
    stock = pd.read_parquet(stock_path)
    
    # Load FF5
    # FF files often have header info. We'll try to sniff it or assume standard.
    # Usually first few lines are descriptions.
    try:
        # Try reading with skiprows=3 which is common for FF CSVs from library
        ff = pd.read_csv(ff_path, skiprows=3)
        # Rename first column to Date if it's unnamed
        if 'Unnamed: 0' in ff.columns:
            ff.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
        
        # Check if we have the right columns
        required_ff = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        if not all(col in ff.columns for col in required_ff):
            # Try reading without skiprows
            ff = pd.read_csv(ff_path)
            if 'Unnamed: 0' in ff.columns:
                ff.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    except Exception as e:
        logger.error(f"Error reading FF file: {e}")
        raise

    # Clean FF dates
    ff['date'] = pd.to_datetime(ff['date'], format='%Y%m%d', errors='coerce')
    if pd.api.types.is_datetime64_any_dtype(ff['date']):
        ff['date'] = ff['date'].dt.tz_localize(None)
    ff = ff.dropna(subset=['date'])
    
    return news, stock, ff

def calculate_features(news_df, stock_df, ff_df):
    logger.info("Calculating features...")
    
    # 1. Aggregate News Sentiment Daily per Stock
    # Assuming news_df has 'symbol_query', 'final_date_for_news', 'sentiment_finbert'
    # Check column names
    if 'final_date_for_news' not in news_df.columns:
         # Fallback or create it (logic from feature_engineering.py should be applied if raw)
         # Assuming input news_df IS ALREADY processed by standard pipeline or we need to apply date logic?
         # The task said "Build master_analysis_input.csv". 
         # We should probably assume news_df is the raw embeddings/processed file.
         # Let's rely on 'published_at' if available and doing simple day agg for now/
         if 'published_at' in news_df.columns:
             news_df['date'] = pd.to_datetime(news_df['published_at']).dt.normalize()
         else:
             raise ValueError("News data missing date column")
    else:
        news_df['date'] = pd.to_datetime(news_df['final_date_for_news'])

    # Aggregation - Global
    # Ensure date is TZ-naive
    if pd.api.types.is_datetime64_any_dtype(news_df['date']):
        news_df['date'] = news_df['date'].dt.tz_localize(None)

    daily_news = news_df.groupby(['symbol_query', 'date']).agg(
        avg_sentiment=('sentiment_finbert', 'mean'),
        news_volume=('sentiment_finbert', 'count')
    ).reset_index()
    
    # Aggregation - Per Topic
    # We want mean sentiment AND count for each topic per day per stock
    if 'topic_id_kmeans' in news_df.columns:
        logger.info("Aggregating Per-Topic Sentiment and Counts...")
        
        # 1. Pivot Mean Sentiment
        topic_sent_pivot = news_df.pivot_table(
            index=['symbol_query', 'date'],
            columns='topic_id_kmeans',
            values='sentiment_finbert',
            aggfunc='mean'
        ).reset_index()
        
        # 2. Pivot Count
        topic_count_pivot = news_df.pivot_table(
            index=['symbol_query', 'date'],
            columns='topic_id_kmeans',
            values='sentiment_finbert', # Any col works for count
            aggfunc='count'
        ).reset_index()
        
        # Rename columns
        sent_cols = []
        count_cols = []
        
        # Helper to rename keys to keep order
        topic_keys = [c for c in topic_sent_pivot.columns if c not in ['symbol_query', 'date']]
        
        for k in topic_keys:
            topic_sent_pivot.rename(columns={k: f'sent_topic_{k}'}, inplace=True)
            sent_cols.append(f'sent_topic_{k}')
            
            topic_count_pivot.rename(columns={k: f'count_topic_{k}'}, inplace=True)
            count_cols.append(f'count_topic_{k}')

        # Merge back to daily_news
        daily_news = pd.merge(daily_news, topic_sent_pivot, on=['symbol_query', 'date'], how='left')
        daily_news = pd.merge(daily_news, topic_count_pivot, on=['symbol_query', 'date'], how='left')
        
        # Fill NaNs
        # No news = 0 sentiment impact, 0 count
        daily_news[sent_cols] = daily_news[sent_cols].fillna(0)
        daily_news[count_cols] = daily_news[count_cols].fillna(0)
        
        # --- Advanced Features: Per-Topic Shocks & Interactions ---
        logger.info("Calculating Per-Topic Shocks & Interactions...")
        daily_news = daily_news.sort_values(['symbol_query', 'date'])
        
        for k in topic_keys:
            s_col = f'sent_topic_{k}'
            c_col = f'count_topic_{k}'
            
            # Shock (Z-Score 252d)
            roll_mean = daily_news.groupby('symbol_query')[s_col].transform(lambda x: x.rolling(window=252, min_periods=60).mean())
            roll_std = daily_news.groupby('symbol_query')[s_col].transform(lambda x: x.rolling(window=252, min_periods=60).std())
            
            z_col = f'z_score_topic_{k}'
            daily_news[z_col] = (daily_news[s_col] - roll_mean) / roll_std
            daily_news[z_col] = daily_news[z_col].fillna(0)
            
            # Interaction: Sentiment * ln(1 + Topic_Volume)
            # Use specific topic volume!
            i_col = f'interaction_topic_{k}'
            log_vol = np.log1p(daily_news[c_col])
            daily_news[i_col] = daily_news[s_col] * log_vol

    else:
        logger.warning("topic_id_kmeans column missing. Skipping topic features.")
    
    # 2. Z-Score (Global) - 252d
    logger.info("Calculating Global Z-Scores (252d)...")
    daily_news = daily_news.sort_values(['symbol_query', 'date'])
    
    daily_news['sentiment_mean_252'] = daily_news.groupby('symbol_query')['avg_sentiment'].transform(
        lambda x: x.rolling(window=252, min_periods=60).mean()
    )
    daily_news['sentiment_std_252'] = daily_news.groupby('symbol_query')['avg_sentiment'].transform(
        lambda x: x.rolling(window=252, min_periods=60).std()
    )
    
    daily_news['z_score_sentiment'] = (daily_news['avg_sentiment'] - daily_news['sentiment_mean_252']) / daily_news['sentiment_std_252']
    daily_news['z_score_sentiment'] = daily_news['z_score_sentiment'].fillna(0)
    
    # 3. Interaction Term
    # ln(1 + volume)
    daily_news['log_volume'] = np.log1p(daily_news['news_volume'])
    daily_news['interaction_term'] = daily_news['avg_sentiment'] * daily_news['log_volume']
    
    # 4. Merge Stock Data (Calculate Lags)
    logger.info("Merging Stock Data and Calculating Lags...")
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    if pd.api.types.is_datetime64_any_dtype(stock_df['date']):
        stock_df['date'] = stock_df['date'].dt.tz_localize(None)
    stock_df = stock_df.sort_values(['symbol', 'date'])
    
    # Check if FF columns exist in stock_df and drop them to avoid collision/confusion with official FF source
    ff_cols_in_stock = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
    stock_df = stock_df.drop(columns=[c for c in ff_cols_in_stock if c in stock_df.columns], errors='ignore')

    # Calculate Returns
    # Use 'adjusted_close'
    if 'adjusted_close' in stock_df.columns:
        price_col = 'adjusted_close'
    elif 'adj_close' in stock_df.columns:
        price_col = 'adj_close'
    else:
        raise KeyError("No adjusted close column found in stock data")

    stock_df['daily_return'] = stock_df.groupby('symbol')[price_col].pct_change()
    
    for lag in [1, 5, 10, 21]:
        # Log returns often preferred, but let's stick to simple for lags unless specified 
        #(Actually req said "Lagged Returns"). 
        # But if we want Log Returns for regression:
        # stock_df['log_ret'] = np.log(stock_df[price_col] / stock_df[price_col].shift(1))
        # Let's keep 'daily_return' (simple) as base feature, but can compute log too.
        stock_df[f'return_lag_{lag}d'] = stock_df.groupby('symbol')['daily_return'].shift(lag)
        
    # 5. Merge Everything
    # Stock + News
    master = pd.merge(stock_df, daily_news, left_on=['symbol', 'date'], right_on=['symbol_query', 'date'], how='left')
    
    # Fill missing news features with 0 (No news = Neutral/Zero signal)
    news_features = ['avg_sentiment', 'news_volume', 'z_score_sentiment', 'interaction_term']
    master[news_features] = master[news_features].fillna(0)
    
    # FF5
    logger.info("Merging Fama-French Factors...")
    master = pd.merge(master, ff_df, on='date', how='left')
    
    # --- PHASE 5: Advanced Feature Engineering (Lags & Interactions) ---
    logger.info("Generating Advanced Features (Lags & Interactions)...")
    master = master.sort_values(['symbol', 'date'])
    
    # 1. Lags
    for lag in [1, 3]:
        master[f'avg_sentiment_lag{lag}'] = master.groupby('symbol')['avg_sentiment'].shift(lag).fillna(0)
        master[f'interaction_term_lag{lag}'] = master.groupby('symbol')['interaction_term'].shift(lag).fillna(0)
        master[f'z_score_sentiment_lag{lag}'] = master.groupby('symbol')['z_score_sentiment'].shift(lag).fillna(0)

    # 2. Factor-News Interactions (Must happen after FF merge)
    # Market * Sentiment (Does news matter more in specific market regimes?)
    if 'Mkt-RF' in master.columns:
        master['interact_Mkt_Sent'] = master['Mkt-RF'] * master['avg_sentiment']
        master['interact_Mkt_Shock'] = master['Mkt-RF'] * master['z_score_sentiment']
        
    # Volatility * Sentiment (Using squared Magnitude of return as proxy for volatility if VIX not present)
    # |Return| * Sentiment
    master['abs_ret'] = master['daily_return'].abs()
    master['interact_Vol_Sent'] = master['abs_ret'] * master['avg_sentiment']

    
    # Drop rows where Stock data is missing (weekends/holidays already handled in stock data usually)
    master = master.dropna(subset=['adjusted_close'])
    
    return master

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--news_path", required=True)
    parser.add_argument("--stock_path", required=True)
    parser.add_argument("--ff_path", required=True)
    parser.add_argument("--output_path", required=True)
    
    args = parser.parse_args()
    
    news, stock, ff = load_data(args.news_path, args.stock_path, args.ff_path)
    master_df = calculate_features(news, stock, ff)
    
    logger.info(f"Saving Master Dataset to {args.output_path}...")
    master_df.to_csv(args.output_path, index=False)
    logger.info("Done.")

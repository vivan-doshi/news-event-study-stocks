"""
Pull historical data for 2021-2023 and combine with existing data
"""

import pandas as pd
from data_collector import EODHDDataCollector
from dotenv import load_dotenv
import os
from datetime import datetime

# Load environment variables
load_dotenv('config.env')
API_KEY = os.getenv('EODHD_API_KEY')

def main():
    # Initialize data collector
    collector = EODHDDataCollector(API_KEY)

    # Define symbols and date range for 2021-2023
    symbols = ['AAPL.US', 'TSLA.US']
    start_date = '2021-01-01'
    end_date = '2023-12-31'

    print(f"Collecting historical data from {start_date} to {end_date}")
    print(f"Symbols: {symbols}")

    # Fetch stock data for 2021-2023
    print("\n=== Fetching Stock Data (2021-2023) ===")
    stock_data_2021_2023 = collector.fetch_multiple_symbols_data(symbols, start_date, end_date, 'stock')

    if not stock_data_2021_2023.empty:
        stock_data_2021_2023.to_csv('data/stock_data_2021_2023.csv', index=False)
        print(f"Stock data saved: {len(stock_data_2021_2023)} records")
        print(f"Date range: {stock_data_2021_2023['date'].min()} to {stock_data_2021_2023['date'].max()}")
        print(f"Symbols: {stock_data_2021_2023['symbol'].unique()}")
    else:
        print("No stock data collected")

    # Fetch news data for 2021-2023
    print("\n=== Fetching News Data (2021-2023) ===")
    news_data_2021_2023 = collector.fetch_multiple_symbols_data(symbols, start_date, end_date, 'news')

    if not news_data_2021_2023.empty:
        news_data_2021_2023.to_csv('data/news_data_2021_2023.csv', index=False)
        print(f"News data saved: {len(news_data_2021_2023)} records")
        print(f"Date range: {news_data_2021_2023['date'].min()} to {news_data_2021_2023['date'].max()}")
        print(f"Symbols: {news_data_2021_2023['symbol'].unique()}")
    else:
        print("No news data collected")

    # Load existing data
    print("\n=== Loading Existing Data ===")
    existing_stock = pd.read_csv('data/stock_data.csv')
    existing_news = pd.read_csv('data/news_data.csv')

    print(f"Existing stock data: {len(existing_stock)} records")
    print(f"Existing news data: {len(existing_news)} records")

    # Combine stock data
    print("\n=== Combining Stock Data ===")
    if not stock_data_2021_2023.empty:
        combined_stock = pd.concat([stock_data_2021_2023, existing_stock], ignore_index=True)
        # Remove duplicates based on date and symbol
        combined_stock['date'] = pd.to_datetime(combined_stock['date'])
        combined_stock = combined_stock.drop_duplicates(subset=['date', 'symbol'], keep='first')
        combined_stock = combined_stock.sort_values(['symbol', 'date']).reset_index(drop=True)

        combined_stock.to_csv('data/stock_data_combined.csv', index=False)
        print(f"Combined stock data saved: {len(combined_stock)} records")
        print(f"Date range: {combined_stock['date'].min()} to {combined_stock['date'].max()}")

        # Show data distribution
        print("\nRecords per symbol:")
        print(combined_stock.groupby('symbol').size())
        print("\nDate range per symbol:")
        for symbol in combined_stock['symbol'].unique():
            symbol_data = combined_stock[combined_stock['symbol'] == symbol]
            print(f"{symbol}: {symbol_data['date'].min()} to {symbol_data['date'].max()}")

    # Combine news data
    print("\n=== Combining News Data ===")
    if not news_data_2021_2023.empty:
        combined_news = pd.concat([news_data_2021_2023, existing_news], ignore_index=True)
        # Remove duplicates (news items might have unique IDs, so we'll keep all for now)
        combined_news['date'] = pd.to_datetime(combined_news['date'])
        combined_news = combined_news.sort_values(['symbol', 'date']).reset_index(drop=True)

        combined_news.to_csv('data/news_data_combined.csv', index=False)
        print(f"Combined news data saved: {len(combined_news)} records")
        print(f"Date range: {combined_news['date'].min()} to {combined_news['date'].max()}")

        # Show data distribution
        print("\nRecords per symbol:")
        print(combined_news.groupby('symbol').size())

    print("\n=== Complete! ===")
    print("Combined files created:")
    print("- data/stock_data_combined.csv")
    print("- data/news_data_combined.csv")

if __name__ == "__main__":
    main()

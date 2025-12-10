"""
Create comprehensive combined dataset with all available data
"""

import pandas as pd
import os

def main():
    print("=== Creating Combined Dataset ===\n")

    # Load existing data
    print("Loading data files...")
    stock_data = pd.read_csv('data/stock_data.csv')
    news_data = pd.read_csv('data/news_data.csv')

    # Convert dates
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    news_data['date'] = pd.to_datetime(news_data['date'])

    # Sort and clean stock data
    stock_data = stock_data.sort_values(['symbol', 'date']).reset_index(drop=True)
    stock_data = stock_data.drop_duplicates(subset=['date', 'symbol'], keep='first')

    # Sort news data
    news_data = news_data.sort_values(['symbol', 'date']).reset_index(drop=True)

    # Save combined files
    stock_output = 'data/stock_data_complete.csv'
    news_output = 'data/news_data_complete.csv'

    stock_data.to_csv(stock_output, index=False)
    news_data.to_csv(news_output, index=False)

    # Print summary statistics
    print(f"\n{'='*60}")
    print("STOCK DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Total records: {len(stock_data):,}")
    print(f"Date range: {stock_data['date'].min().date()} to {stock_data['date'].max().date()}")
    print(f"Number of symbols: {stock_data['symbol'].nunique()}")
    print(f"\nRecords per symbol:")
    print(stock_data.groupby('symbol').size())

    print(f"\n{'='*60}")
    print("Records by year:")
    print(f"{'='*60}")
    stock_data['year'] = stock_data['date'].dt.year
    year_counts = stock_data.groupby('year').size().sort_index()
    for year, count in year_counts.items():
        print(f"{year}: {count:,} records")

    # Detailed breakdown for 2021-2023
    print(f"\n{'='*60}")
    print("2021-2023 DETAILED BREAKDOWN")
    print(f"{'='*60}")
    data_2021_2023 = stock_data[(stock_data['date'] >= '2021-01-01') & (stock_data['date'] <= '2023-12-31')]
    print(f"Total records (2021-2023): {len(data_2021_2023):,}")
    print(f"Date range: {data_2021_2023['date'].min().date()} to {data_2021_2023['date'].max().date()}")
    print("\nRecords per symbol (2021-2023):")
    print(data_2021_2023.groupby('symbol').size())

    print(f"\n{'='*60}")
    print("NEWS DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Total records: {len(news_data):,}")
    print(f"Date range: {news_data['date'].min()} to {news_data['date'].max()}")
    print(f"Number of symbols: {news_data['symbol'].nunique()}")
    print(f"\nRecords per symbol:")
    print(news_data.groupby('symbol').size())

    print(f"\n{'='*60}")
    print("FILES CREATED")
    print(f"{'='*60}")
    print(f"✓ {stock_output}")
    print(f"  Size: {os.path.getsize(stock_output) / 1024 / 1024:.2f} MB")
    print(f"✓ {news_output}")
    print(f"  Size: {os.path.getsize(news_output) / 1024 / 1024:.2f} MB")

    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

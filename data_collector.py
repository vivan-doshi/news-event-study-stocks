"""
Data Collection Script for News Event Study Stocks
Pulls stock data and news data for Apple and Tesla using EODHD API
"""

import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import time
from datetime import datetime, timedelta
import json

# Load environment variables
load_dotenv('config.env')
API_KEY = os.getenv('EODHD_API_KEY')

class EODHDDataCollector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://eodhd.com/api"
        
    def fetch_stock_data(self, symbol, start_date, end_date):
        """
        Fetch historical stock data for a given symbol and date range
        """
        url = f"{self.base_url}/eod/{symbol}"
        params = {
            'from': start_date,
            'to': end_date,
            'api_token': self.api_key,
            'fmt': 'json'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df['symbol'] = symbol
                return df
            else:
                print(f"No data returned for {symbol}")
                return pd.DataFrame()
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching stock data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_news_data(self, symbol, start_date, end_date, limit=1000):
        """
        Fetch news data for a given symbol and date range
        """
        url = f"{self.base_url}/news"
        params = {
            's': symbol,
            'from': start_date,
            'to': end_date,
            'api_token': self.api_key,
            'limit': limit
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df['symbol'] = symbol
                return df
            else:
                print(f"No news data returned for {symbol}")
                return pd.DataFrame()
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching news data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_multiple_symbols_data(self, symbols, start_date, end_date, data_type='stock'):
        """
        Fetch data for multiple symbols
        """
        all_data = []
        
        for symbol in symbols:
            print(f"Fetching {data_type} data for {symbol}...")
            
            if data_type == 'stock':
                data = self.fetch_stock_data(symbol, start_date, end_date)
            elif data_type == 'news':
                data = self.fetch_news_data(symbol, start_date, end_date)
            
            if not data.empty:
                all_data.append(data)
            
            # Add delay to respect API rate limits
            time.sleep(1)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()

def main():
    # Initialize data collector
    collector = EODHDDataCollector(API_KEY)
    
    # Define symbols and date range
    symbols = ['AAPL.US', 'TSLA.US']
    
    # Fama-French data typically starts from July 1963, but we'll use a more recent period
    # since Apple IPO was in 1980 and Tesla IPO was in 2010
    start_date = '2010-06-29'  # Tesla IPO date
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Collecting data from {start_date} to {end_date}")
    print(f"Symbols: {symbols}")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Fetch stock data
    print("\n=== Fetching Stock Data ===")
    stock_data = collector.fetch_multiple_symbols_data(symbols, start_date, end_date, 'stock')
    
    if not stock_data.empty:
        stock_data.to_csv('data/stock_data.csv', index=False)
        print(f"Stock data saved: {len(stock_data)} records")
        print(f"Date range: {stock_data['date'].min()} to {stock_data['date'].max()}")
        print(f"Symbols: {stock_data['symbol'].unique()}")
    else:
        print("No stock data collected")
    
    # Fetch news data
    print("\n=== Fetching News Data ===")
    news_data = collector.fetch_multiple_symbols_data(symbols, start_date, end_date, 'news')
    
    if not news_data.empty:
        news_data.to_csv('data/news_data.csv', index=False)
        print(f"News data saved: {len(news_data)} records")
        print(f"Date range: {news_data['date'].min()} to {news_data['date'].max()}")
        print(f"Symbols: {news_data['symbol'].unique()}")
    else:
        print("No news data collected")
    
    # Display sample data
    if not stock_data.empty:
        print("\n=== Sample Stock Data ===")
        print(stock_data.head())
        print(f"\nStock data columns: {list(stock_data.columns)}")
    
    if not news_data.empty:
        print("\n=== Sample News Data ===")
        print(news_data.head())
        print(f"\nNews data columns: {list(news_data.columns)}")

if __name__ == "__main__":
    main()

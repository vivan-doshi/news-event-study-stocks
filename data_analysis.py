"""
Data Analysis Script for News Event Study Stocks
Analyzes the collected stock and news data for Apple and Tesla
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def analyze_data():
    """Analyze the collected stock and news data"""
    
    # Load the data
    stock_data = pd.read_csv('data/stock_data.csv')
    news_data = pd.read_csv('data/news_data.csv')
    
    # Convert date columns
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    news_data['date'] = pd.to_datetime(news_data['date'])
    
    print("=== DATA COLLECTION SUMMARY ===")
    print(f"Stock Data Records: {len(stock_data):,}")
    print(f"News Data Records: {len(news_data):,}")
    print()
    
    # Stock Data Analysis
    print("=== STOCK DATA ANALYSIS ===")
    print(f"Date Range: {stock_data['date'].min().strftime('%Y-%m-%d')} to {stock_data['date'].max().strftime('%Y-%m-%d')}")
    print(f"Symbols: {', '.join(stock_data['symbol'].unique())}")
    print()
    
    # Stock data by symbol
    for symbol in stock_data['symbol'].unique():
        symbol_data = stock_data[stock_data['symbol'] == symbol]
        print(f"{symbol}:")
        print(f"  Records: {len(symbol_data):,}")
        print(f"  Date Range: {symbol_data['date'].min().strftime('%Y-%m-%d')} to {symbol_data['date'].max().strftime('%Y-%m-%d')}")
        print(f"  Price Range: ${symbol_data['close'].min():.2f} - ${symbol_data['close'].max():.2f}")
        print(f"  Average Volume: {symbol_data['volume'].mean():,.0f}")
        print()
    
    # News Data Analysis
    print("=== NEWS DATA ANALYSIS ===")
    print(f"Date Range: {news_data['date'].min().strftime('%Y-%m-%d')} to {news_data['date'].max().strftime('%Y-%m-%d')}")
    print(f"Symbols: {', '.join(news_data['symbol'].unique())}")
    print()
    
    # News data by symbol
    for symbol in news_data['symbol'].unique():
        symbol_news = news_data[news_data['symbol'] == symbol]
        print(f"{symbol}:")
        print(f"  News Articles: {len(symbol_news):,}")
        print(f"  Date Range: {symbol_news['date'].min().strftime('%Y-%m-%d')} to {symbol_news['date'].max().strftime('%Y-%m-%d')}")
        
        # Sentiment analysis if available
        if 'sentiment' in symbol_news.columns:
            sentiment_counts = symbol_news['sentiment'].value_counts()
            print(f"  Sentiment Distribution:")
            for sentiment, count in sentiment_counts.items():
                print(f"    {sentiment}: {count}")
        print()
    
    # Create visualizations
    create_visualizations(stock_data, news_data)
    
    # Data quality checks
    print("=== DATA QUALITY CHECKS ===")
    print(f"Stock data missing values: {stock_data.isnull().sum().sum()}")
    print(f"News data missing values: {news_data.isnull().sum().sum()}")
    print()
    
    # Check for duplicate dates in stock data
    duplicate_dates = stock_data.groupby(['symbol', 'date']).size()
    duplicates = duplicate_dates[duplicate_dates > 1]
    if len(duplicates) > 0:
        print(f"Duplicate dates found: {len(duplicates)}")
    else:
        print("No duplicate dates in stock data")
    
    return stock_data, news_data

def create_visualizations(stock_data, news_data):
    """Create visualizations of the data"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Apple and Tesla Data Analysis', fontsize=16, fontweight='bold')
    
    # 1. Stock Price Trends
    ax1 = axes[0, 0]
    for symbol in stock_data['symbol'].unique():
        symbol_data = stock_data[stock_data['symbol'] == symbol]
        ax1.plot(symbol_data['date'], symbol_data['close'], label=symbol, linewidth=2)
    ax1.set_title('Stock Price Trends')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Closing Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Trading Volume
    ax2 = axes[0, 1]
    for symbol in stock_data['symbol'].unique():
        symbol_data = stock_data[stock_data['symbol'] == symbol]
        ax2.plot(symbol_data['date'], symbol_data['volume'], label=symbol, alpha=0.7)
    ax2.set_title('Trading Volume')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volume')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. News Count by Date
    ax3 = axes[1, 0]
    news_daily = news_data.groupby(['date', 'symbol']).size().unstack(fill_value=0)
    news_daily.plot(kind='bar', stacked=True, ax=ax3)
    ax3.set_title('Daily News Count')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Number of Articles')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Price Volatility (30-day rolling standard deviation)
    ax4 = axes[1, 1]
    for symbol in stock_data['symbol'].unique():
        symbol_data = stock_data[stock_data['symbol'] == symbol].copy()
        symbol_data['returns'] = symbol_data['close'].pct_change()
        symbol_data['volatility'] = symbol_data['returns'].rolling(window=30).std()
        ax4.plot(symbol_data['date'], symbol_data['volatility'], label=f'{symbol} Volatility')
    ax4.set_title('30-Day Rolling Volatility')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Volatility')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved to 'data/data_analysis.png'")

def create_summary_report(stock_data, news_data):
    """Create a summary report of the data"""
    
    report = f"""
# News Event Study Data Collection Report

## Overview
This report summarizes the data collected for Apple (AAPL) and Tesla (TSLA) stocks for news event study analysis.

## Data Collection Details
- **Collection Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Data Source**: EODHD API
- **Time Period**: {stock_data['date'].min().strftime('%Y-%m-%d')} to {stock_data['date'].max().strftime('%Y-%m-%d')}

## Stock Data Summary
- **Total Records**: {len(stock_data):,}
- **Symbols**: {', '.join(stock_data['symbol'].unique())}
- **Data Fields**: {', '.join(stock_data.columns)}

### By Symbol:
"""
    
    for symbol in stock_data['symbol'].unique():
        symbol_data = stock_data[stock_data['symbol'] == symbol]
        report += f"""
#### {symbol}
- Records: {len(symbol_data):,}
- Date Range: {symbol_data['date'].min().strftime('%Y-%m-%d')} to {symbol_data['date'].max().strftime('%Y-%m-%d')}
- Price Range: ${symbol_data['close'].min():.2f} - ${symbol_data['close'].max():.2f}
- Average Volume: {symbol_data['volume'].mean():,.0f}
"""
    
    report += f"""
## News Data Summary
- **Total Records**: {len(news_data):,}
- **Symbols**: {', '.join(news_data['symbol'].unique())}
- **Data Fields**: {', '.join(news_data.columns)}

### By Symbol:
"""
    
    for symbol in news_data['symbol'].unique():
        symbol_news = news_data[news_data['symbol'] == symbol]
        report += f"""
#### {symbol}
- News Articles: {len(symbol_news):,}
- Date Range: {symbol_news['date'].min().strftime('%Y-%m-%d')} to {symbol_news['date'].max().strftime('%Y-%m-%d')}
"""
    
    report += f"""
## Data Quality
- Stock data missing values: {stock_data.isnull().sum().sum()}
- News data missing values: {news_data.isnull().sum().sum()}

## Files Generated
- `data/stock_data.csv`: Historical stock data
- `data/news_data.csv`: News articles data
- `data/data_analysis.png`: Data visualization charts

## Next Steps
1. Align stock and news data by date
2. Perform event study analysis
3. Calculate abnormal returns around news events
4. Statistical significance testing
"""
    
    # Save the report
    with open('data/data_collection_report.md', 'w') as f:
        f.write(report)
    
    print("Summary report saved to 'data/data_collection_report.md'")

if __name__ == "__main__":
    # Run the analysis
    stock_data, news_data = analyze_data()
    
    # Create summary report
    create_summary_report(stock_data, news_data)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Check the 'data' folder for:")
    print("- stock_data.csv: Historical stock prices")
    print("- news_data.csv: News articles")
    print("- data_analysis.png: Visualization charts")
    print("- data_collection_report.md: Detailed summary report")

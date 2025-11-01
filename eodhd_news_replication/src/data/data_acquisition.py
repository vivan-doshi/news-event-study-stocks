"""
EODHD Data Acquisition Module
Handles fetching news and price data from EODHD API
"""

import os
import json
import hashlib
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from dotenv import load_dotenv
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EODHDDataAcquisition:
    """Class for acquiring data from EODHD API"""

    def __init__(self, config_path: str = "conf/experiment.yaml"):
        """Initialize data acquisition with configuration"""
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv('EODHD_API_KEY')
        if not self.api_key:
            raise ValueError("EODHD_API_KEY not found in environment variables")

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.base_url = "https://eodhd.com/api"
        self.rate_limit_delay = self.config['data_acquisition']['rate_limit_delay']

        # Create necessary directories
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        Path("data/clean").mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)

    def generate_article_id(self, article: Dict) -> str:
        """Generate unique article ID using hash"""
        unique_string = f"{article.get('source', '')}" \
                       f"{article.get('url', '')}" \
                       f"{article.get('title', '')}" \
                       f"{article.get('published_at', '')}"
        return hashlib.md5(unique_string.encode()).hexdigest()

    def fetch_news(self, symbol: str, start_date: str, end_date: str) -> List[Dict]:
        """Fetch news for a specific symbol and date range"""
        all_news = []
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        while current_date <= end_dt:
            # Format dates for API
            from_date = current_date.strftime("%Y-%m-%d")
            to_date = min(current_date + timedelta(days=30), end_dt).strftime("%Y-%m-%d")

            # Construct API URL
            url = f"{self.base_url}/news"
            params = {
                's': symbol,
                'api_token': self.api_key,
                'from': from_date,
                'to': to_date,
                'limit': 1000,
                'fmt': 'json'
            }

            try:
                response = requests.get(url, params=params)
                response.raise_for_status()

                data = response.json()

                # Handle different response formats
                if isinstance(data, list):
                    news_items = data
                elif isinstance(data, dict) and 'news' in data:
                    news_items = data['news']
                else:
                    news_items = []

                for article in news_items:
                    # Add metadata
                    article['symbol_query'] = symbol
                    article['fetch_date'] = datetime.now().isoformat()
                    all_news.append(article)

                logger.info(f"Fetched {len(news_items)} articles for {symbol} from {from_date} to {to_date}")

                # Rate limiting
                time.sleep(self.rate_limit_delay)

            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching news for {symbol} from {from_date} to {to_date}: {e}")

            # Move to next batch
            current_date = current_date + timedelta(days=31)

        return all_news

    def fetch_all_news(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Fetch news for all symbols and save to parquet"""
        if symbols is None:
            # Load symbols from config file
            with open("conf/symbols_us.txt", 'r') as f:
                symbols = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        # Get date range from config
        start_date = self.config['data_acquisition']['time_range']['start_date']
        end_date = self.config['data_acquisition']['time_range']['end_date']

        all_articles = []
        article_ids = set()  # For deduplication

        for i, symbol in enumerate(symbols):
            logger.info(f"Processing symbol {i+1}/{len(symbols)}: {symbol}")

            news_items = self.fetch_news(symbol, start_date, end_date)

            for article in news_items:
                # Generate unique article ID
                article_id = self.generate_article_id(article)

                # Deduplicate
                if article_id not in article_ids:
                    article_ids.add(article_id)

                    # Structure the article data
                    structured_article = {
                        'article_id': article_id,
                        'published_at': article.get('date', ''),
                        'title': article.get('title', ''),
                        'content': article.get('content', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', ''),
                        'symbols': article.get('symbols', []),
                        'tags': article.get('tags', []),
                        'sentiment': article.get('sentiment', {}),
                        'symbol_query': article.get('symbol_query', ''),
                        'fetch_date': article.get('fetch_date', '')
                    }

                    all_articles.append(structured_article)

        # Convert to DataFrame
        df = pd.DataFrame(all_articles)

        # Save raw data
        timestamp = datetime.now().strftime("%Y%m%d")
        raw_file_path = f"data/raw/news_raw_{timestamp}.parquet"
        df.to_parquet(raw_file_path, compression='snappy')
        logger.info(f"Saved {len(df)} unique articles to {raw_file_path}")

        # Log statistics
        self._log_statistics(df)

        return df

    def fetch_price_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical price data for a symbol"""
        url = f"{self.base_url}/eod/{symbol}"
        params = {
            'api_token': self.api_key,
            'from': start_date,
            'to': end_date,
            'fmt': 'json'
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            df = pd.DataFrame(data)

            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df['symbol'] = symbol

            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching price data for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_all_prices(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Fetch price data for all symbols"""
        if symbols is None:
            # Default to index and sector ETFs
            symbols = self.config['symbols']['index_tickers'] + self.config['symbols']['sector_etfs']

        start_date = self.config['data_acquisition']['time_range']['start_date']
        end_date = self.config['data_acquisition']['time_range']['end_date']

        all_prices = []

        for symbol in symbols:
            logger.info(f"Fetching price data for {symbol}")
            df = self.fetch_price_data(symbol, start_date, end_date)

            if not df.empty:
                all_prices.append(df)

            time.sleep(self.rate_limit_delay)

        if all_prices:
            # Combine all price data
            combined_df = pd.concat(all_prices)

            # Save to parquet
            price_file_path = "data/raw/prices_raw.parquet"
            combined_df.to_parquet(price_file_path, compression='snappy')
            logger.info(f"Saved price data for {len(symbols)} symbols to {price_file_path}")

            return combined_df

        return pd.DataFrame()

    def calculate_returns(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns from price data"""
        returns_list = []

        for symbol in prices_df['symbol'].unique():
            symbol_data = prices_df[prices_df['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_index()

            # Calculate log returns
            symbol_data['log_return'] = np.log(symbol_data['adjusted_close'] /
                                               symbol_data['adjusted_close'].shift(1))

            # Calculate simple returns
            symbol_data['simple_return'] = symbol_data['adjusted_close'].pct_change()

            returns_list.append(symbol_data)

        returns_df = pd.concat(returns_list)

        # Save returns data
        returns_df.to_parquet("data/derived/returns_daily.parquet", compression='snappy')

        # Create monthly returns
        monthly_returns = returns_df.groupby([pd.Grouper(freq='M'), 'symbol']).agg({
            'log_return': 'sum',
            'simple_return': lambda x: (1 + x).prod() - 1
        })

        monthly_returns.to_parquet("data/derived/returns_monthly.parquet", compression='snappy')

        logger.info("Calculated and saved daily and monthly returns")

        return returns_df

    def _log_statistics(self, df: pd.DataFrame):
        """Log statistics about the fetched news data"""
        stats = {
            'total_articles': len(df),
            'unique_sources': df['source'].nunique(),
            'date_range': f"{df['published_at'].min()} to {df['published_at'].max()}",
            'articles_with_content': df['content'].notna().sum(),
            'articles_with_sentiment': df['sentiment'].apply(lambda x: bool(x)).sum()
        }

        # Save statistics to log file
        with open("logs/data_acquisition_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Data acquisition statistics: {stats}")

        # Create daily article counts
        if not df.empty and 'published_at' in df.columns:
            df['published_date'] = pd.to_datetime(df['published_at']).dt.date
            daily_counts = df.groupby('published_date').size()

            # Log days with zero coverage
            date_range = pd.date_range(
                start=df['published_date'].min(),
                end=df['published_date'].max(),
                freq='D'
            )
            missing_dates = set(date_range.date) - set(daily_counts.index)

            if missing_dates:
                logger.warning(f"Days with zero coverage: {len(missing_dates)}")
                with open("logs/missing_coverage_dates.txt", 'w') as f:
                    for date in sorted(missing_dates):
                        f.write(f"{date}\n")


def main():
    """Main function to run data acquisition"""
    # Initialize data acquisition
    acquisition = EODHDDataAcquisition()

    # Fetch news data
    logger.info("Starting news data acquisition...")
    news_df = acquisition.fetch_all_news()

    # Fetch price data
    logger.info("Starting price data acquisition...")
    prices_df = acquisition.fetch_all_prices()

    # Calculate returns
    if not prices_df.empty:
        acquisition.calculate_returns(prices_df)

    logger.info("Data acquisition complete!")


if __name__ == "__main__":
    main()
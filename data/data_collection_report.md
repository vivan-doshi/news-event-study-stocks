
# News Event Study Data Collection Report

## Overview
This report summarizes the data collected for Apple (AAPL) and Tesla (TSLA) stocks for news event study analysis.

## Data Collection Details
- **Collection Date**: 2025-10-27 18:33:47
- **Data Source**: EODHD API
- **Time Period**: 2010-06-29 to 2025-10-27

## Stock Data Summary
- **Total Records**: 7,714
- **Symbols**: AAPL.US, TSLA.US
- **Data Fields**: date, open, high, low, close, adjusted_close, volume, symbol

### By Symbol:

#### AAPL.US
- Records: 3,857
- Date Range: 2010-06-29 to 2025-10-27
- Price Range: $90.28 - $702.10
- Average Volume: 206,346,911

#### TSLA.US
- Records: 3,857
- Date Range: 2010-06-29 to 2025-10-27
- Price Range: $15.80 - $2238.75
- Average Volume: 97,012,463

## News Data Summary
- **Total Records**: 2,000
- **Symbols**: AAPL.US, TSLA.US
- **Data Fields**: date, title, content, link, symbols, tags, sentiment, symbol

### By Symbol:

#### AAPL.US
- News Articles: 1,000
- Date Range: 2025-10-09 to 2025-10-27

#### TSLA.US
- News Articles: 1,000
- Date Range: 2025-10-12 to 2025-10-27

## Data Quality
- Stock data missing values: 0
- News data missing values: 0

## Files Generated
- `data/stock_data.csv`: Historical stock data
- `data/news_data.csv`: News articles data
- `data/data_analysis.png`: Data visualization charts

## Next Steps
1. Align stock and news data by date
2. Perform event study analysis
3. Calculate abnormal returns around news events
4. Statistical significance testing

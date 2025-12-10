# Mag7 News Event Study Analysis

This project performs an event study analysis to investigate the impact of news sentiment on the stock performance of the "Magnificent 7" companies (Apple, Amazon, Google, Meta, Microsoft, Nvidia, Tesla).

It refactors the original Jupyter notebook workflow into a robust, modular Python pipeline for reproducibility and scalability.

## 📂 Project Structure

```
.
├── config/
│   └── topic_to_label_map_v2.csv        # Topic mapping configuration
├── data/
│   ├── processed/
│   │   ├── mag7_news_with_sentiment...  # Input news data
│   │   ├── mag7_yf_2021_2025.parquet    # Input stock data
│   │   └── mag7_aggregated_features...  # Generated features (Intermediate)
│   └── outputs/
│       └── results/
│           ├── plots/                   # Generated regression plots
│           └── tables/                  # Summary statistics (CSV)
├── src/
│   ├── analysis/
│   │   ├── feature_engineering.py       # Data aggregation logic
│   │   └── event_study.py               # OLS regression & plotting
│   └── run_analysis.py                  # Main CLI entry point
└── archive/                             # Archive of legacy notebooks
```

## 🛠️ Setup & Installation

Ensure you have Python installed (3.8+ recommended).

### Dependencies
The project relies on standard data science libraries:
*   `pandas`
*   `numpy`
*   `statsmodels`
*   `matplotlib`
*   `pyarrow` (for Parquet support)

Install them via pip:
```bash
pip install pandas numpy statsmodels matplotlib pyarrow
```

## 🚀 Usage

The entire pipeline is orchestrated by the `src/run_analysis.py` driver script.

### 1. Run the Full Pipeline (Recommended)
This runs both feature engineering and the event study for all configured symbols.
```bash
python src/run_analysis.py --stage all
```

### 2. Run Feature Engineering Only
aggregates raw news and stock data into a single dataset.
```bash
python src/run_analysis.py --stage feature_engineering
```
**Output**: `data/processed/mag7_aggregated_features.parquet`

### 3. Run Event Study Only
Runs OLS regressions on the aggregated data. You can specify a single symbol or a list.
```bash
# Run for all symbols
python src/run_analysis.py --stage event_study

# Run for specific symbols (e.g., Google and Apple)
python src/run_analysis.py --stage event_study --symbol GOOGL.US,AAPL.US
```
**Output**: Plots in `data/outputs/results/plots/` and summary tables in `data/outputs/results/tables/`.

## 📊 Methodology

### Feature Engineering
1.  **Data Loading**: Loads Parquet files for news and stock history.
2.  **Date Adjustment**: Aligns news timestamps to trading days:
    *   News published after 4:00 PM EST is moved to the next day.
    *   Weekends and NASDAQ holidays are skipped to find the next valid trading day.
3.  **Aggregation**:
    *   Calculates daily sentiment metrics (FinBERT) per topic.
    *   Counts news volume per topic.
4.  **Merging**: joins aggregated news features with daily stock returns (`adj_close`).

### Event Study (OLS Regression)
*   **Model**: Regresses next-day stock price (derived from `adj_close`) against daily sentiment scores across various topics.
*   **Validation**: Splits data into training (pre-June 2025) and testing sets.
    *   *Note: The cutoff date is currently set to '2025-06-01' matching the original study parameters.*
*   **Vizualization**: Generates "Actual vs. Predicted" and "Residuals" plots to assess model fit.

## 📝 Configuration
*   **Topic Map**: Mappings between topic IDs and human-readable labels are stored in `config/topic_to_label_map_v2.csv`.
*   **Target Variable**: The analysis targets the **Next Day's Adjusted Close** price by default.

## 📧 Contact
For questions regarding the analysis implementation, please refer to the source code definitions in `src/analysis/`.

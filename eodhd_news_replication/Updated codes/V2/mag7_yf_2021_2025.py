"""
mag7_yf_2021_2025.py

Download daily price data for the Magnificent 7 from yfinance
for 2021-01-01 to 2025-10-31, compute returns, and save as a
tidy Parquet that can be merged with your news panel.
"""

import numpy as np
import pandas as pd
import yfinance as yf

# ==========================
# CONFIG
# ==========================

# EODHD-style -> yfinance ticker
MAG7_MAP = {
    "AAPL.US": "AAPL",
    "MSFT.US": "MSFT",
    "AMZN.US": "AMZN",
    "GOOGL.US": "GOOGL",
    "META.US": "META",
    "TSLA.US": "TSLA",
    "NVDA.US": "NVDA",
}

START_DATE = "2021-01-01"
END_DATE   = "2025-10-31"
OUTPUT_PARQUET_PATH = "mag7_yf_2021_2025.parquet"


def main():
    tickers = list(MAG7_MAP.values())
    print(f"Downloading {tickers} from {START_DATE} to {END_DATE} ...")

    px = yf.download(
        tickers=tickers,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        progress=True,
    )

    if px.empty:
        print("No data returned from yfinance (DataFrame is empty).")
        return

    # ==========================
    # TIDY FORMAT (ROBUST)
    # ==========================
    # px has:
    #   - index: Date
    #   - columns: MultiIndex (field, ticker) OR sometimes other structures.
    # We only need Adj Close, then stack to long format.
    adj = px["Adj Close"]

    # Make sure we have a DataFrame, even if it's a single ticker
    if not isinstance(adj, pd.DataFrame):
        adj = adj.to_frame()

    # After stack + reset_index, we DON'T assume any column names,
    # we rename by position: [date, ticker_yf, adj_close].
    adj_long = adj.stack().reset_index()
    adj_long.columns = ["date", "ticker_yf", "adj_close"]

    # Map back to your symbol_query style (AAPL.US, etc.)
    rev_map = {v: k for k, v in MAG7_MAP.items()}
    adj_long["symbol_query"] = adj_long["ticker_yf"].map(rev_map)

    # Drop rows we can't map (defensive)
    adj_long = adj_long.dropna(subset=["symbol_query"])

    # Clean date + sort
    adj_long["date"] = pd.to_datetime(adj_long["date"]).dt.date
    adj_long = adj_long.sort_values(["symbol_query", "date"])

    # ==========================
    # RETURNS
    # ==========================
    adj_long["ret_1d"] = (
        adj_long
        .groupby("symbol_query")["adj_close"]
        .pct_change()
    )

    adj_long["ret_log_1d"] = np.log(
        adj_long["adj_close"]
        / adj_long.groupby("symbol_query")["adj_close"].shift(1)
    )

    # ==========================
    # SAVE
    # ==========================
    adj_long.to_parquet(OUTPUT_PARQUET_PATH, index=False)

    print(f"Saved: {OUTPUT_PARQUET_PATH}")
    print(adj_long.head())


if __name__ == "__main__":
    main()

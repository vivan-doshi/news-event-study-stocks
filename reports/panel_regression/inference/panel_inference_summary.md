
# Deep Dive Inference: Panel Regression Analysis

This report provides a statistical interpretation of the Panel Regression results, moving beyond simple "prediction accuracy" (R²) to understanding the **relationships** (Inference) discovered in the data (2023-2025).

## 1. The "Power of Pooling" (Why Panel wins)
The Panel Model achieved an **OOS R² of 0.44** (vs 0.34 for individual models).
*   **Reason:** By combining 7 stocks, the model had ~4,400 observations to learn from, instead of ~600 per stock.
*   **What it learned:** It solidified the "Universal Physics" of these Tech Giants:
    *   **Market Beta (~0.97):** They move almost 1:1 with the market.
    *   **HML (-0.75):** They are heavily "Growth" oriented (negative value loading).
    *   **CMA (-0.42):** They invest aggressively (negative conservative loading).
    *   **T-Statistics:** These factors had t-stats of 19.0, -10.5, and -3.5 respectively (extremely robust).

## 2. Unlocking the "News" Signal (Statistical Significance)
The predictive model (Rolling R²) suggested news added "no value". However, the **Static Inference** reveals a more nuanced story.

### A. News Volume (Buzz)
*   **Finding:** Lagged News Volume (`log_total_news_lag1`) is **Statistically Significant** (P-Value: **0.022**).
*   **Coefficient:** **-0.0012**
*   **Interpretation:** When news volume was high *yesterday*, the stock tends to slightly *underperform* (lower alpha) *today*.
*   **Why didn't R² improve?** The effect size is tiny compared to the Market Beta. It provides a real signal, but it's drowned out by the noise of the market direction.

### B. Sentiment (Mood)
*   **Finding 1 (Predictive):** Past Sentiment (`day_sentiment_lag1`) is **NOT Significant** (P-Value: 0.237).
    *   **Meaning:** Yesterday's mood does *not* reliably predict today's return. This explains why your "Backtest" and "OOS R²" didn't show big gains. The market absorbs the sentiment immediately.
*   **Finding 2 (Contemporaneous):** Current Sentiment (`day_sentiment`) is **Highly Significant** (P-Value: **0.005**).
    *   **Coefficient:** **-0.0035** (Negative!)
    *   **Meaning:** On days with *high* net sentiment, the stocks actually generated *lower* Excess Returns (Alpha) than expected by their Beta.
    *   **Theory:** "Buy the rumor, sell the news" or mean reversion? When the news is overwhelmingly positive, the price might be overextended.

## 3. Commercial Conclusion
*   **Risk Model:** The Panel approach is superior for estimating risk (Beta/Factor loadings). Use pooling for all future risk models.
*   **Alpha Signal:**
    *   **Sentiment:** No predictive edge (market is efficient).
    *   **Attention (Volume):** Slight "Reversal" signal (High buzz -> Low return), but weak.
*   **Recommendation:** Do not use daily news sentiment as a primary directional trigger for these liquid large-cap stocks. The market is too fast.

# 252-Day Rolling Window Analysis Comparison
## Overview
Comparing Out-of-Sample R² across 5 models using a **252-day (1 year)** rolling window.
A longer window provides more stability and reduces overfitting for complex models.

| Symbol | Baseline | News Vol | Sentiment | News Cat | Sent Cat | Best Model |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AAPL.US** | 0.3692 | 0.3664 | 0.3629 | 0.3496 | 0.3360 | **Baseline (FF5+Lags)** |
| **AMZN.US** | 0.3476 | 0.3433 | 0.3428 | 0.3247 | 0.3351 | **Baseline (FF5+Lags)** |
| **GOOGL.US** | 0.2451 | 0.2392 | 0.2439 | 0.2300 | 0.2384 | **Baseline (FF5+Lags)** |
| **META.US** | 0.2514 | 0.2433 | 0.2486 | 0.2146 | 0.2113 | **Baseline (FF5+Lags)** |
| **MSFT.US** | 0.3718 | 0.3658 | 0.3682 | 0.3472 | 0.3599 | **Baseline (FF5+Lags)** |
| **NVDA.US** | 0.4628 | 0.4596 | 0.4637 | 0.4477 | 0.4514 | **Total Sentiment** |
| **TSLA.US** | 0.3070 | 0.2997 | 0.3048 | 0.2696 | 0.2710 | **Baseline (FF5+Lags)** |

## Interpretation
- **Baseline (FF5 + Lags):** The standard financial model.
- **News/Sentiment Models:** Did specific news features improve over the baseline?
- **Overfitting Check:** With 252 days, do the Category models still crash (negative R²) or do they perform well?

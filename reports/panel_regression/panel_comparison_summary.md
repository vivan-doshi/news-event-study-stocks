# Panel Regression Analysis Results
## Overview
Comparison of **Pooled Panel Models** (learning from all stocks simultaneously with Fixed Effects) vs. **Individual Stock Models**.
Rolling Window: 252 Days.

## Key Comparison: Individual vs. Pooled
| Symbol | Ind. Baseline (FF5) | Panel Baseline | Panel News | Panel Sentiment | Best Approach |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NVDA.US** | 0.4628 | 0.5883 | 0.5860 | 0.5919 | **Panel Sent** |
| **MSFT.US** | 0.3718 | 0.3862 | 0.3805 | 0.3789 | **Panel Base** |
| **AMZN.US** | 0.3476 | 0.5943 | 0.5938 | 0.5927 | **Panel Base** |
| **GOOGL.US** | 0.2451 | 0.3468 | 0.3403 | 0.3489 | **Panel Sent** |
| **AAPL.US** | 0.3692 | 0.3100 | 0.3150 | 0.3047 | **Individual** |
| **TSLA.US** | 0.3070 | 0.3457 | 0.3463 | 0.3488 | **Panel Sent** |
| **META.US** | 0.2514 | 0.4937 | 0.4986 | 0.4906 | **Panel News** |

## Aggregate Performance (Mean OOS R²)
- **Individual Baseline:** 0.3364
- **Panel Baseline:** 0.4379
- **Panel News:** 0.4372
- **Panel Sentiment:** 0.4366
# Panel Regression Analysis Results
## Overview
Comparison of **Pooled Panel Models** (learning from all stocks simultaneously with Fixed Effects) vs. **Individual Stock Models**.
Rolling Window: 252 Days.

## Key Comparison: Individual vs. Pooled
| Symbol | Ind. Baseline | Panel Base | Panel Thematic | Panel Conviction | Panel Risk | Best Approach |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NVDA.US** | 0.4628 | 0.5883 | 0.4833 | 0.5840 | 0.5881 | **Panel Base** |
| **MSFT.US** | 0.3718 | 0.3862 | 0.3723 | 0.3825 | 0.3833 | **Panel Base** |
| **AMZN.US** | 0.3476 | 0.5943 | 0.5932 | 0.5982 | 0.5906 | **Panel Conv** |
| **GOOGL.US** | 0.2451 | 0.3468 | 0.3430 | 0.3495 | 0.3425 | **Panel Conv** |
| **AAPL.US** | 0.3692 | 0.3100 | 0.3000 | 0.3107 | 0.3069 | **Individual** |
| **TSLA.US** | 0.3070 | 0.3457 | 0.3471 | 0.3469 | 0.3455 | **Panel Them** |
| **META.US** | 0.2514 | 0.4937 | 0.4995 | 0.4959 | 0.4981 | **Panel Them** |

## Aggregate Performance (Mean OOS R²)
- **Individual Baseline:** 0.3364
- **Panel Baseline:** 0.4379
- **Panel News:** 0.4372
- **Panel Sentiment:** 0.4361
- **Panel Thematic Shocks:** 0.4198
- **Panel Signal Conviction:** 0.4382
- **Panel Risk Shock:** 0.4364
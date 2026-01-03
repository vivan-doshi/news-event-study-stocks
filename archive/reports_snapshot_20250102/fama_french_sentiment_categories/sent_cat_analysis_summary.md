# Sentiment Category Augmented Analysis Results
## ⚠️ CRITICAL WARNING: Overfitting Detected (Again)
Similar to the News Count analysis, adding 15 individual sentiment scores (even without lags) increased the model dimensionality too much for the 60-day rolling window.

While some stocks (AAPL, MSFT) survived with reasonable (though lower) performance, others (AMZN, NVDA, TSLA) suffered catastrophic failure due to multicollinearity/noise.

| Symbol | Baseline OOS R² | Sent. Cat. OOS R² | Change | Baseline RMSE | Sent. Cat. RMSE | Improvement? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| AAPL.US | 0.3958 | 0.1311 | -0.26 | 0.01279 | 0.01534 | ❌ NO |
| AMZN.US | 0.4486 | Collapsed (< -10) | -18545.74 | 0.01470 | 2.69546 | ❌ NO |
| GOOGL.US | 0.2846 | -0.1584 | -0.44 | 0.01598 | 0.02034 | ❌ NO |
| META.US | 0.2764 | -0.3661 | -0.64 | 0.02047 | 0.02813 | ❌ NO |
| MSFT.US | 0.4822 | 0.1115 | -0.37 | 0.01065 | 0.01395 | ❌ NO |
| NVDA.US | 0.5378 | Collapsed (< -10) | -387718410921890945507470082048.00 | 0.02239 | 20505567736811.07422 | ❌ NO |
| TSLA.US | 0.2704 | -8.0757 | -8.35 | 0.03248 | 0.11456 | ❌ NO |

## Detailed Exposure (Top Sentiment Categories)
### AAPL.US
![AAPL.US Sent. Categories](/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/reports/fama_french_sentiment_categories/plots/AAPL.US_sent_cat_top6.png)

### AMZN.US
![AMZN.US Sent. Categories](/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/reports/fama_french_sentiment_categories/plots/AMZN.US_sent_cat_top6.png)

### GOOGL.US
![GOOGL.US Sent. Categories](/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/reports/fama_french_sentiment_categories/plots/GOOGL.US_sent_cat_top6.png)

### META.US
![META.US Sent. Categories](/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/reports/fama_french_sentiment_categories/plots/META.US_sent_cat_top6.png)

### MSFT.US
![MSFT.US Sent. Categories](/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/reports/fama_french_sentiment_categories/plots/MSFT.US_sent_cat_top6.png)

### NVDA.US
![NVDA.US Sent. Categories](/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/reports/fama_french_sentiment_categories/plots/NVDA.US_sent_cat_top6.png)

### TSLA.US
![TSLA.US Sent. Categories](/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/reports/fama_french_sentiment_categories/plots/TSLA.US_sent_cat_top6.png)

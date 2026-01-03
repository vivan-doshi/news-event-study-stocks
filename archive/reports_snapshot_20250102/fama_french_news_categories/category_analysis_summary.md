# News Category Augmented Analysis Results
## ⚠️ CRITICAL WARNING: Overfitting Detected
Adding 15 individually detailed news categories to the model drastically increased the number of parameters. With a 60-day window, this caused **catastrophic overfitting**.

The model learned to fit the noise in the training window almost perfectly (High In-Sample R²), but completely failed to predict the next day (Extremely Negative OOS R²).

| Symbol | Baseline OOS R² | News OOS R² | Change | Baseline RMSE | News RMSE | Improvement? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| AAPL.US | 0.3958 | 0.0268 | -0.37 | 0.01279 | 0.01623 | ❌ NO |
| AMZN.US | 0.4486 | 0.2056 | -0.24 | 0.01470 | 0.01764 | ❌ NO |
| GOOGL.US | 0.2846 | -0.0296 | -0.31 | 0.01598 | 0.01918 | ❌ NO |
| META.US | 0.2764 | 0.1346 | -0.14 | 0.02047 | 0.02239 | ❌ NO |
| MSFT.US | 0.4822 | Collapsed (< -10) | -12220347883973808928581484544.00 | 0.01065 | 1635877875351.58789 | ❌ NO |
| NVDA.US | 0.5378 | Collapsed (< -10) | -581530122592624911895958126592.00 | 0.02239 | 25113063444763.36328 | ❌ NO |
| TSLA.US | 0.2704 | -0.5400 | -0.81 | 0.03248 | 0.04719 | ❌ NO |

## Detailed Exposure (Top Categories)
### AAPL.US
![AAPL.US Categories](/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/reports/fama_french_news_categories/plots/AAPL.US_category_top6.png)

### AMZN.US
![AMZN.US Categories](/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/reports/fama_french_news_categories/plots/AMZN.US_category_top6.png)

### GOOGL.US
![GOOGL.US Categories](/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/reports/fama_french_news_categories/plots/GOOGL.US_category_top6.png)

### META.US
![META.US Categories](/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/reports/fama_french_news_categories/plots/META.US_category_top6.png)

### MSFT.US
![MSFT.US Categories](/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/reports/fama_french_news_categories/plots/MSFT.US_category_top6.png)

### NVDA.US
![NVDA.US Categories](/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/reports/fama_french_news_categories/plots/NVDA.US_category_top6.png)

### TSLA.US
![TSLA.US Categories](/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/reports/fama_french_news_categories/plots/TSLA.US_category_top6.png)

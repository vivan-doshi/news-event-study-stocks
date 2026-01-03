# Fama-French Augmented Analysis Results
**Date Range:** 2023-01-03 to 2025-10-30

## AAPL.US
### Static Regression Output
```text
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.539
Model:                            OLS   Adj. R-squared:                  0.532
Method:                 Least Squares   F-statistic:                     73.85
Date:                Wed, 10 Dec 2025   Prob (F-statistic):           2.44e-99
Time:                        14:45:26   Log-Likelihood:                 1974.7
No. Observations:                 642   AIC:                            -3927.
Df Residuals:                     631   BIC:                            -3878.
Df Model:                          10                                         
Covariance Type:            nonrobust                                         
=================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
const            -0.0003      0.000     -0.735      0.463      -0.001       0.001
Mkt-RF            1.1083      0.067     16.545      0.000       0.977       1.240
SMB              -0.1569      0.072     -2.168      0.031      -0.299      -0.015
HML              -0.2766      0.107     -2.594      0.010      -0.486      -0.067
RMW              -0.4028      0.219     -1.837      0.067      -0.833       0.028
CMA               0.1247      0.162      0.772      0.440      -0.193       0.442
log_ret_lag1      0.0833      0.028      3.028      0.003       0.029       0.137
log_ret_lag2      0.0392      0.027      1.440      0.150      -0.014       0.093
log_ret_lag5     -0.0313      0.027     -1.141      0.254      -0.085       0.023
log_ret_lag10    -0.0294      0.027     -1.079      0.281      -0.083       0.024
log_ret_lag21    -0.0203      0.027     -0.749      0.454      -0.074       0.033
==============================================================================
Omnibus:                       70.470   Durbin-Watson:                   1.901
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              510.435
Skew:                           0.042   Prob(JB):                    1.45e-111
Kurtosis:                       7.367   Cond. No.                         535.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

### Rolling Model Performance (Avg Window / OOS)
| Metric | Value | Description |
| :--- | :--- | :--- |
| **In-Sample Avg R²** | 0.5866 | Average R² across all rolling windows |
| **In-Sample RMSE** | 0.01006 | RMS Error of in-sample fits |
| **Out-of-Sample R²** | 0.3958 | 1-Step Ahead Prediction R² |
| **Out-of-Sample RMSE** | 0.01279 | RMS Error of 1-step ahead predictions |

### Rolling Factor Exposure Plots
![AAPL.US Rolling Factors](/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/reports/fama_french_augmented/plots/AAPL.US_augmented.png)

## AMZN.US
### Static Regression Output
```text
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.593
Model:                            OLS   Adj. R-squared:                  0.587
Method:                 Least Squares   F-statistic:                     93.81
Date:                Wed, 10 Dec 2025   Prob (F-statistic):          1.16e-118
Time:                        14:45:26   Log-Likelihood:                 1934.2
No. Observations:                 655   AIC:                            -3846.
Df Residuals:                     644   BIC:                            -3797.
Df Model:                          10                                         
Covariance Type:            nonrobust                                         
=================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
const             0.0003      0.001      0.550      0.582      -0.001       0.001
Mkt-RF            0.9190      0.075     12.192      0.000       0.771       1.067
SMB               0.0919      0.080      1.142      0.254      -0.066       0.250
HML              -0.7339      0.118     -6.223      0.000      -0.965      -0.502
RMW              -0.5430      0.245     -2.216      0.027      -1.024      -0.062
CMA              -0.2610      0.180     -1.450      0.147      -0.614       0.092
log_ret_lag1     -0.0478      0.025     -1.882      0.060      -0.098       0.002
log_ret_lag2     -0.1088      0.025     -4.293      0.000      -0.159      -0.059
log_ret_lag5     -0.0199      0.025     -0.790      0.430      -0.069       0.030
log_ret_lag10     0.0087      0.025      0.346      0.729      -0.041       0.058
log_ret_lag21    -0.0319      0.025     -1.273      0.204      -0.081       0.017
==============================================================================
Omnibus:                       95.190   Durbin-Watson:                   1.983
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              862.799
Skew:                           0.280   Prob(JB):                    4.42e-188
Kurtosis:                       8.595   Cond. No.                         536.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

### Rolling Model Performance (Avg Window / OOS)
| Metric | Value | Description |
| :--- | :--- | :--- |
| **In-Sample Avg R²** | 0.6636 | Average R² across all rolling windows |
| **In-Sample RMSE** | 0.01242 | RMS Error of in-sample fits |
| **Out-of-Sample R²** | 0.4486 | 1-Step Ahead Prediction R² |
| **Out-of-Sample RMSE** | 0.01470 | RMS Error of 1-step ahead predictions |

### Rolling Factor Exposure Plots
![AMZN.US Rolling Factors](/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/reports/fama_french_augmented/plots/AMZN.US_augmented.png)

## GOOGL.US
### Static Regression Output
```text
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.455
Model:                            OLS   Adj. R-squared:                  0.447
Method:                 Least Squares   F-statistic:                     53.37
Date:                Wed, 10 Dec 2025   Prob (F-statistic):           1.20e-77
Time:                        14:45:26   Log-Likelihood:                 1852.1
No. Observations:                 649   AIC:                            -3682.
Df Residuals:                     638   BIC:                            -3633.
Df Model:                          10                                         
Covariance Type:            nonrobust                                         
=================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
const             0.0003      0.001      0.484      0.629      -0.001       0.001
Mkt-RF            0.6873      0.083      8.248      0.000       0.524       0.851
SMB               0.0090      0.090      0.101      0.920      -0.167       0.185
HML              -0.6151      0.134     -4.605      0.000      -0.877      -0.353
RMW               0.0552      0.272      0.203      0.839      -0.479       0.589
CMA              -0.4005      0.202     -1.987      0.047      -0.796      -0.005
log_ret_lag1      0.0259      0.030      0.875      0.382      -0.032       0.084
log_ret_lag2      0.0347      0.030      1.173      0.241      -0.023       0.093
log_ret_lag5     -0.0195      0.030     -0.656      0.512      -0.078       0.039
log_ret_lag10    -0.0315      0.029     -1.080      0.281      -0.089       0.026
log_ret_lag21     0.0355      0.030      1.203      0.230      -0.022       0.094
==============================================================================
Omnibus:                      121.715   Durbin-Watson:                   1.987
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1215.227
Skew:                          -0.498   Prob(JB):                    1.31e-264
Kurtosis:                       9.629   Cond. No.                         536.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

### Rolling Model Performance (Avg Window / OOS)
| Metric | Value | Description |
| :--- | :--- | :--- |
| **In-Sample Avg R²** | 0.5601 | Average R² across all rolling windows |
| **In-Sample RMSE** | 0.01359 | RMS Error of in-sample fits |
| **Out-of-Sample R²** | 0.2846 | 1-Step Ahead Prediction R² |
| **Out-of-Sample RMSE** | 0.01598 | RMS Error of 1-step ahead predictions |

### Rolling Factor Exposure Plots
![GOOGL.US Rolling Factors](/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/reports/fama_french_augmented/plots/GOOGL.US_augmented.png)

## META.US
### Static Regression Output
```text
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.518
Model:                            OLS   Adj. R-squared:                  0.510
Method:                 Least Squares   F-statistic:                     66.88
Date:                Wed, 10 Dec 2025   Prob (F-statistic):           6.58e-92
Time:                        14:45:26   Log-Likelihood:                 1694.4
No. Observations:                 634   AIC:                            -3367.
Df Residuals:                     623   BIC:                            -3318.
Df Model:                          10                                         
Covariance Type:            nonrobust                                         
=================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
const             0.0006      0.001      0.823      0.411      -0.001       0.002
Mkt-RF            0.9718      0.102      9.532      0.000       0.772       1.172
SMB              -0.2297      0.108     -2.128      0.034      -0.442      -0.018
HML              -0.2419      0.161     -1.501      0.134      -0.558       0.075
RMW               2.0634      0.331      6.231      0.000       1.413       2.714
CMA              -1.2854      0.248     -5.187      0.000      -1.772      -0.799
log_ret_lag1      0.0094      0.029      0.328      0.743      -0.047       0.066
log_ret_lag2     -0.0356      0.028     -1.248      0.213      -0.092       0.020
log_ret_lag5     -0.0006      0.028     -0.021      0.983      -0.056       0.055
log_ret_lag10     0.0010      0.028      0.034      0.973      -0.055       0.056
log_ret_lag21     0.0424      0.029      1.486      0.138      -0.014       0.098
==============================================================================
Omnibus:                      376.082   Durbin-Watson:                   1.828
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            16272.353
Skew:                           1.961   Prob(JB):                         0.00
Kurtosis:                      27.507   Cond. No.                         544.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

### Rolling Model Performance (Avg Window / OOS)
| Metric | Value | Description |
| :--- | :--- | :--- |
| **In-Sample Avg R²** | 0.5981 | Average R² across all rolling windows |
| **In-Sample RMSE** | 0.01746 | RMS Error of in-sample fits |
| **Out-of-Sample R²** | 0.2764 | 1-Step Ahead Prediction R² |
| **Out-of-Sample RMSE** | 0.02047 | RMS Error of 1-step ahead predictions |

### Rolling Factor Exposure Plots
![META.US Rolling Factors](/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/reports/fama_french_augmented/plots/META.US_augmented.png)

## MSFT.US
### Static Regression Output
```text
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.648
Model:                            OLS   Adj. R-squared:                  0.642
Method:                 Least Squares   F-statistic:                     113.9
Date:                Wed, 10 Dec 2025   Prob (F-statistic):          3.40e-133
Time:                        14:45:26   Log-Likelihood:                 2089.2
No. Observations:                 630   AIC:                            -4156.
Df Residuals:                     619   BIC:                            -4108.
Df Model:                          10                                         
Covariance Type:            nonrobust                                         
=================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
const          6.678e-05      0.000      0.185      0.853      -0.001       0.001
Mkt-RF            0.8395      0.053     15.897      0.000       0.736       0.943
SMB              -0.0627      0.057     -1.105      0.270      -0.174       0.049
HML              -0.9003      0.083    -10.823      0.000      -1.064      -0.737
RMW              -0.2724      0.174     -1.562      0.119      -0.615       0.070
CMA               0.4680      0.127      3.672      0.000       0.218       0.718
log_ret_lag1      0.0168      0.024      0.694      0.488      -0.031       0.064
log_ret_lag2     -0.0564      0.024     -2.354      0.019      -0.103      -0.009
log_ret_lag5      0.0401      0.024      1.677      0.094      -0.007       0.087
log_ret_lag10    -0.0078      0.024     -0.329      0.742      -0.055       0.039
log_ret_lag21     0.0185      0.024      0.781      0.435      -0.028       0.065
==============================================================================
Omnibus:                      176.031   Durbin-Watson:                   2.075
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             3831.061
Skew:                           0.678   Prob(JB):                         0.00
Kurtosis:                      15.004   Cond. No.                         537.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

### Rolling Model Performance (Avg Window / OOS)
| Metric | Value | Description |
| :--- | :--- | :--- |
| **In-Sample Avg R²** | 0.7181 | Average R² across all rolling windows |
| **In-Sample RMSE** | 0.00924 | RMS Error of in-sample fits |
| **Out-of-Sample R²** | 0.4822 | 1-Step Ahead Prediction R² |
| **Out-of-Sample RMSE** | 0.01065 | RMS Error of 1-step ahead predictions |

### Rolling Factor Exposure Plots
![MSFT.US Rolling Factors](/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/reports/fama_french_augmented/plots/MSFT.US_augmented.png)

## NVDA.US
### Static Regression Output
```text
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.674
Model:                            OLS   Adj. R-squared:                  0.669
Method:                 Least Squares   F-statistic:                     119.8
Date:                Wed, 10 Dec 2025   Prob (F-statistic):          6.63e-134
Time:                        14:45:26   Log-Likelihood:                 1507.5
No. Observations:                 590   AIC:                            -2993.
Df Residuals:                     579   BIC:                            -2945.
Df Model:                          10                                         
Covariance Type:            nonrobust                                         
=================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
const             0.0016      0.001      2.026      0.043    5.01e-05       0.003
Mkt-RF            0.8776      0.118      7.461      0.000       0.647       1.109
SMB              -0.1875      0.126     -1.485      0.138      -0.435       0.061
HML              -1.0833      0.184     -5.896      0.000      -1.444      -0.722
RMW               2.7703      0.380      7.296      0.000       2.025       3.516
CMA              -1.9361      0.283     -6.830      0.000      -2.493      -1.379
log_ret_lag1     -0.0495      0.024     -2.076      0.038      -0.096      -0.003
log_ret_lag2      0.0276      0.024      1.151      0.250      -0.020       0.075
log_ret_lag5     -0.0127      0.024     -0.531      0.595      -0.060       0.034
log_ret_lag10    -0.0157      0.024     -0.663      0.507      -0.062       0.031
log_ret_lag21     0.0183      0.024      0.775      0.439      -0.028       0.065
==============================================================================
Omnibus:                      126.803   Durbin-Watson:                   1.981
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1397.314
Skew:                           0.592   Prob(JB):                    3.78e-304
Kurtosis:                      10.446   Cond. No.                         531.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

### Rolling Model Performance (Avg Window / OOS)
| Metric | Value | Description |
| :--- | :--- | :--- |
| **In-Sample Avg R²** | 0.7447 | Average R² across all rolling windows |
| **In-Sample RMSE** | 0.01759 | RMS Error of in-sample fits |
| **Out-of-Sample R²** | 0.5378 | 1-Step Ahead Prediction R² |
| **Out-of-Sample RMSE** | 0.02239 | RMS Error of 1-step ahead predictions |

### Rolling Factor Exposure Plots
![NVDA.US Rolling Factors](/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/reports/fama_french_augmented/plots/NVDA.US_augmented.png)

## TSLA.US
### Static Regression Output
```text
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.443
Model:                            OLS   Adj. R-squared:                  0.434
Method:                 Least Squares   F-statistic:                     49.61
Date:                Wed, 10 Dec 2025   Prob (F-statistic):           7.96e-73
Time:                        14:45:26   Log-Likelihood:                 1362.7
No. Observations:                 636   AIC:                            -2703.
Df Residuals:                     625   BIC:                            -2654.
Df Model:                          10                                         
Covariance Type:            nonrobust                                         
=================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
const            -0.0007      0.001     -0.581      0.561      -0.003       0.002
Mkt-RF            1.4141      0.173      8.159      0.000       1.074       1.754
SMB               0.6775      0.183      3.711      0.000       0.319       1.036
HML              -1.3145      0.271     -4.842      0.000      -1.848      -0.781
RMW              -4.3731      0.566     -7.723      0.000      -5.485      -3.261
CMA               0.1668      0.420      0.397      0.691      -0.657       0.991
log_ret_lag1      0.0608      0.030      2.014      0.044       0.002       0.120
log_ret_lag2      0.0151      0.030      0.500      0.617      -0.044       0.074
log_ret_lag5     -0.0035      0.030     -0.117      0.907      -0.063       0.056
log_ret_lag10    -0.0387      0.030     -1.305      0.192      -0.097       0.020
log_ret_lag21    -0.0120      0.030     -0.405      0.686      -0.070       0.046
==============================================================================
Omnibus:                       70.619   Durbin-Watson:                   2.104
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              448.555
Skew:                           0.208   Prob(JB):                     3.96e-98
Kurtosis:                       7.093   Cond. No.                         544.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

### Rolling Model Performance (Avg Window / OOS)
| Metric | Value | Description |
| :--- | :--- | :--- |
| **In-Sample Avg R²** | 0.5018 | Average R² across all rolling windows |
| **In-Sample RMSE** | 0.02666 | RMS Error of in-sample fits |
| **Out-of-Sample R²** | 0.2704 | 1-Step Ahead Prediction R² |
| **Out-of-Sample RMSE** | 0.03248 | RMS Error of 1-step ahead predictions |

### Rolling Factor Exposure Plots
![TSLA.US Rolling Factors](/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/reports/fama_french_augmented/plots/TSLA.US_augmented.png)

# Fama-French Augmented Analysis Results
**Date Range:** 2023-01-03 to 2025-10-30

## AAPL.US
### Static Regression Output
```text
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.525
Model:                            OLS   Adj. R-squared:                  0.518
Method:                 Least Squares   F-statistic:                     77.23
Date:                Wed, 10 Dec 2025   Prob (F-statistic):          5.34e-106
Time:                        14:19:12   Log-Likelihood:                 2174.6
No. Observations:                 710   AIC:                            -4327.
Df Residuals:                     699   BIC:                            -4277.
Df Model:                          10                                         
Covariance Type:            nonrobust                                         
=================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
const            -0.0001      0.000     -0.307      0.759      -0.001       0.001
Mkt-RF            1.1093      0.065     17.118      0.000       0.982       1.237
SMB              -0.1784      0.070     -2.542      0.011      -0.316      -0.041
HML              -0.2354      0.102     -2.297      0.022      -0.437      -0.034
RMW              -0.3606      0.214     -1.689      0.092      -0.780       0.059
CMA               0.0684      0.155      0.441      0.659      -0.236       0.373
log_ret_lag1      0.0960      0.026      3.626      0.000       0.044       0.148
log_ret_lag2      0.0311      0.026      1.186      0.236      -0.020       0.083
log_ret_lag5     -0.0396      0.026     -1.507      0.132      -0.091       0.012
log_ret_lag10    -0.0362      0.026     -1.387      0.166      -0.088       0.015
log_ret_lag21    -0.0234      0.026     -0.901      0.368      -0.075       0.028
==============================================================================
Omnibus:                       72.589   Durbin-Watson:                   1.906
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              461.010
Skew:                           0.129   Prob(JB):                    7.81e-101
Kurtosis:                       6.939   Cond. No.                         542.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

### Rolling Factor Exposure Plots
![AAPL.US Rolling Factors](/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/reports/fama_french_augmented/plots/AAPL.US_augmented.png)

## AMZN.US
### Static Regression Output
```text
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.582
Model:                            OLS   Adj. R-squared:                  0.576
Method:                 Least Squares   F-statistic:                     97.34
Date:                Wed, 10 Dec 2025   Prob (F-statistic):          2.87e-125
Time:                        14:19:12   Log-Likelihood:                 2081.0
No. Observations:                 710   AIC:                            -4140.
Df Residuals:                     699   BIC:                            -4090.
Df Model:                          10                                         
Covariance Type:            nonrobust                                         
=================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
const             0.0002      0.000      0.391      0.696      -0.001       0.001
Mkt-RF            0.9398      0.074     12.713      0.000       0.795       1.085
SMB               0.0975      0.080      1.215      0.225      -0.060       0.255
HML              -0.7005      0.117     -5.969      0.000      -0.931      -0.470
RMW              -0.5708      0.244     -2.339      0.020      -1.050      -0.092
CMA              -0.3083      0.177     -1.738      0.083      -0.656       0.040
log_ret_lag1     -0.0345      0.025     -1.396      0.163      -0.083       0.014
log_ret_lag2     -0.0910      0.025     -3.696      0.000      -0.139      -0.043
log_ret_lag5     -0.0112      0.024     -0.459      0.646      -0.059       0.037
log_ret_lag10    -0.0090      0.024     -0.367      0.714      -0.057       0.039
log_ret_lag21    -0.0321      0.024     -1.313      0.190      -0.080       0.016
==============================================================================
Omnibus:                       93.585   Durbin-Watson:                   1.990
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              834.613
Skew:                           0.183   Prob(JB):                    5.84e-182
Kurtosis:                       8.299   Cond. No.                         545.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

### Rolling Factor Exposure Plots
![AMZN.US Rolling Factors](/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/reports/fama_french_augmented/plots/AMZN.US_augmented.png)

## GOOGL.US
### Static Regression Output
```text
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.445
Model:                            OLS   Adj. R-squared:                  0.437
Method:                 Least Squares   F-statistic:                     56.09
Date:                Wed, 10 Dec 2025   Prob (F-statistic):           9.63e-83
Time:                        14:19:12   Log-Likelihood:                 2015.3
No. Observations:                 710   AIC:                            -4009.
Df Residuals:                     699   BIC:                            -3958.
Df Model:                          10                                         
Covariance Type:            nonrobust                                         
=================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
const             0.0003      0.001      0.607      0.544      -0.001       0.001
Mkt-RF            0.7178      0.081      8.861      0.000       0.559       0.877
SMB               0.0100      0.088      0.114      0.910      -0.163       0.182
HML              -0.6306      0.129     -4.907      0.000      -0.883      -0.378
RMW               0.1645      0.267      0.617      0.538      -0.359       0.688
CMA              -0.3623      0.194     -1.865      0.063      -0.744       0.019
log_ret_lag1      0.0315      0.028      1.110      0.267      -0.024       0.087
log_ret_lag2      0.0182      0.029      0.637      0.524      -0.038       0.074
log_ret_lag5     -0.0177      0.029     -0.617      0.537      -0.074       0.039
log_ret_lag10    -0.0248      0.028     -0.874      0.382      -0.081       0.031
log_ret_lag21     0.0400      0.028      1.404      0.161      -0.016       0.096
==============================================================================
Omnibus:                      111.673   Durbin-Watson:                   1.976
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1279.039
Skew:                          -0.248   Prob(JB):                    1.82e-278
Kurtosis:                       9.557   Cond. No.                         542.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

### Rolling Factor Exposure Plots
![GOOGL.US Rolling Factors](/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/reports/fama_french_augmented/plots/GOOGL.US_augmented.png)

## META.US
### Static Regression Output
```text
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.503
Model:                            OLS   Adj. R-squared:                  0.496
Method:                 Least Squares   F-statistic:                     70.78
Date:                Wed, 10 Dec 2025   Prob (F-statistic):           2.84e-99
Time:                        14:19:12   Log-Likelihood:                 1894.1
No. Observations:                 710   AIC:                            -3766.
Df Residuals:                     699   BIC:                            -3716.
Df Model:                          10                                         
Covariance Type:            nonrobust                                         
=================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
const             0.0009      0.001      1.396      0.163      -0.000       0.002
Mkt-RF            0.9586      0.096      9.961      0.000       0.770       1.148
SMB              -0.2452      0.105     -2.345      0.019      -0.450      -0.040
HML              -0.2706      0.152     -1.777      0.076      -0.570       0.028
RMW               1.8595      0.316      5.880      0.000       1.239       2.480
CMA              -1.2270      0.230     -5.327      0.000      -1.679      -0.775
log_ret_lag1     -0.0101      0.027     -0.368      0.713      -0.064       0.044
log_ret_lag2     -0.0370      0.027     -1.358      0.175      -0.090       0.016
log_ret_lag5     -0.0110      0.027     -0.404      0.686      -0.064       0.042
log_ret_lag10     0.0239      0.027      0.884      0.377      -0.029       0.077
log_ret_lag21     0.0214      0.027      0.796      0.426      -0.031       0.074
==============================================================================
Omnibus:                      441.218   Durbin-Watson:                   1.966
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            18163.013
Skew:                           2.153   Prob(JB):                         0.00
Kurtosis:                      27.401   Cond. No.                         542.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

### Rolling Factor Exposure Plots
![META.US Rolling Factors](/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/reports/fama_french_augmented/plots/META.US_augmented.png)

## MSFT.US
### Static Regression Output
```text
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.630
Model:                            OLS   Adj. R-squared:                  0.625
Method:                 Least Squares   F-statistic:                     119.1
Date:                Wed, 10 Dec 2025   Prob (F-statistic):          1.05e-143
Time:                        14:19:12   Log-Likelihood:                 2343.6
No. Observations:                 710   AIC:                            -4665.
Df Residuals:                     699   BIC:                            -4615.
Df Model:                          10                                         
Covariance Type:            nonrobust                                         
=================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
const         -4.938e-05      0.000     -0.144      0.886      -0.001       0.001
Mkt-RF            0.8470      0.051     16.594      0.000       0.747       0.947
SMB              -0.0641      0.055     -1.159      0.247      -0.173       0.044
HML              -0.9025      0.081    -11.180      0.000      -1.061      -0.744
RMW              -0.2444      0.169     -1.450      0.147      -0.575       0.086
CMA               0.4723      0.122      3.869      0.000       0.233       0.712
log_ret_lag1      0.0168      0.023      0.718      0.473      -0.029       0.063
log_ret_lag2     -0.0574      0.023     -2.486      0.013      -0.103      -0.012
log_ret_lag5      0.0271      0.023      1.171      0.242      -0.018       0.072
log_ret_lag10    -0.0113      0.023     -0.494      0.622      -0.056       0.034
log_ret_lag21     0.0167      0.023      0.729      0.466      -0.028       0.062
==============================================================================
Omnibus:                      171.907   Durbin-Watson:                   2.074
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             3610.767
Skew:                           0.515   Prob(JB):                         0.00
Kurtosis:                      14.000   Cond. No.                         544.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

### Rolling Factor Exposure Plots
![MSFT.US Rolling Factors](/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/reports/fama_french_augmented/plots/MSFT.US_augmented.png)

## NVDA.US
### Static Regression Output
```text
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.671
Model:                            OLS   Adj. R-squared:                  0.666
Method:                 Least Squares   F-statistic:                     142.6
Date:                Wed, 10 Dec 2025   Prob (F-statistic):          2.26e-161
Time:                        14:19:12   Log-Likelihood:                 1839.4
No. Observations:                 710   AIC:                            -3657.
Df Residuals:                     699   BIC:                            -3606.
Df Model:                          10                                         
Covariance Type:            nonrobust                                         
=================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
const             0.0015      0.001      2.120      0.034       0.000       0.003
Mkt-RF            0.8840      0.104      8.462      0.000       0.679       1.089
SMB              -0.1794      0.113     -1.591      0.112      -0.401       0.042
HML              -1.1112      0.165     -6.748      0.000      -1.434      -0.788
RMW               2.5881      0.341      7.597      0.000       1.919       3.257
CMA              -1.8500      0.250     -7.404      0.000      -2.341      -1.359
log_ret_lag1     -0.0374      0.022     -1.708      0.088      -0.080       0.006
log_ret_lag2      0.0187      0.022      0.853      0.394      -0.024       0.062
log_ret_lag5     -0.0082      0.022     -0.377      0.707      -0.051       0.035
log_ret_lag10    -0.0005      0.022     -0.023      0.982      -0.043       0.042
log_ret_lag21     0.0086      0.022      0.397      0.692      -0.034       0.051
==============================================================================
Omnibus:                      155.835   Durbin-Watson:                   2.032
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1707.137
Skew:                           0.638   Prob(JB):                         0.00
Kurtosis:                      10.488   Cond. No.                         542.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

### Rolling Factor Exposure Plots
![NVDA.US Rolling Factors](/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/reports/fama_french_augmented/plots/NVDA.US_augmented.png)

## TSLA.US
### Static Regression Output
```text
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.423
Model:                            OLS   Adj. R-squared:                  0.415
Method:                 Least Squares   F-statistic:                     51.32
Date:                Wed, 10 Dec 2025   Prob (F-statistic):           5.65e-77
Time:                        14:19:12   Log-Likelihood:                 1505.0
No. Observations:                 710   AIC:                            -2988.
Df Residuals:                     699   BIC:                            -2938.
Df Model:                          10                                         
Covariance Type:            nonrobust                                         
=================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
const            -0.0003      0.001     -0.242      0.809      -0.002       0.002
Mkt-RF            1.4452      0.168      8.615      0.000       1.116       1.775
SMB               0.7434      0.180      4.120      0.000       0.389       1.098
HML              -1.3421      0.264     -5.076      0.000      -1.861      -0.823
RMW              -3.9147      0.547     -7.159      0.000      -4.988      -2.841
CMA               0.0964      0.400      0.241      0.810      -0.690       0.883
log_ret_lag1      0.0429      0.029      1.475      0.141      -0.014       0.100
log_ret_lag2     -0.0028      0.029     -0.095      0.924      -0.060       0.054
log_ret_lag5      0.0230      0.029      0.801      0.423      -0.033       0.079
log_ret_lag10    -0.0125      0.028     -0.438      0.661      -0.068       0.043
log_ret_lag21    -0.0115      0.028     -0.403      0.687      -0.067       0.044
==============================================================================
Omnibus:                       70.708   Durbin-Watson:                   2.075
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              423.013
Skew:                           0.152   Prob(JB):                     1.39e-92
Kurtosis:                       6.769   Cond. No.                         543.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

### Rolling Factor Exposure Plots
![TSLA.US Rolling Factors](/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/reports/fama_french_augmented/plots/TSLA.US_augmented.png)

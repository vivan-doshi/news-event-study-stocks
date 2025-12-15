# Performance Analysis & Model Validation

## 1. Metrics Definition & Trading Performance
We evaluate model performance using both statistical metrics (regression fit) and trading simulation metrics (financial viability).

### **Statistical Metrics**
#### **In-Fold R² (In-Sample Fit)**
*   **Definition:** The average Coefficient of Determination ($R^2$) calculated *within* each rolling training window (e.g., the 252 days used to estimate coefficients).
*   **Interpretation:** Measures how well the model explains the *past* variance of the stock returns.

#### **Out-of-Fold (OOS) R² (Predictive Power)**
*   **Definition:** The $R^2$ calculated using 1-step ahead predictions on data *not* seen by the model during training.
*   **Why it is the Gold Standard:** Strictly penalizes overfitting and "look-ahead bias." A positive OOS R² confirms the model captures genuine, persistent signal rather than random noise.

### **Trading Strategy Metrics**
To assess real-world viability, we simulate a strategy that goes Long/Short based on the sign of the model's predicted return.

#### **Sharpe Ratio**
*   **Definition:** The average excess return per unit of total risk (volatility).
    $$ \text{Sharpe} = \frac{\bar{R}_{strategy}}{\sigma_{strategy}} \times \sqrt{252} $$
*   **Interpretation:** Higher is better. A Sharpe > 1.0 is generally considered "investable" for potential alpha strategies. It rewards consistent returns and penalizes volatility.

#### **Sortino Ratio**
*   **Definition:** Similar to Sharpe, but only penalizes *downside* volatility (negative returns).
    $$ \text{Sortino} = \frac{\bar{R}_{strategy}}{\sigma_{downside}} \times \sqrt{252} $$
*   **Interpretation:** A more precise metric for asymmetric strategies. If a model has high volatility but mostly on the upside (big wins), the Sharpe Ratio penalizes it, but the Sortino Ratio will correctly identify it as favorable.

#### **Profit Factor**
*   **Definition:** The ratio of Total Gross Profits to Total Gross Losses.
    $$ \text{Profit Factor} = \frac{\sum \text{Winning Days}}{\sum |\text{Losing Days}|} $$
*   **Interpretation:** A measure of "bang for your buck."
    *   **> 1.0:** Profitable system.
    *   **> 1.5:** Robust system.
    *   **< 1.0:** Losing system.

#### **Maximum Drawdown (Max DD)**
*   **Definition:** The largest peak-to-trough decline in the strategy's cumulative equity curve over the entire period.
*   **Interpretation:** Measures worst-case risk. If a strategy has 20% annual returns but a 50% Max Drawdown, it is likely too risky for most institutional mandates. We aim for low Drawdowns relative to returns.

#### **Risk/Reward (Risk Factor)**
*   **Definition:** The average profit on winning days divided by the average loss on losing days.
    $$ \text{Risk/Reward} = \frac{\text{Avg Win}}{\text{Avg Loss}} $$
*   **Interpretation:** Helps determine if the strategy relies on a high win rate (hit rate) or high payoff per trade. A strategy with a low hit rate (e.g., 40%) can still be profitable if its Risk/Reward is very high (e.g., 3:1), meaning it cuts losses quickly and lets winners run.

## 2. Comparative Analysis: The "Sortino" Insight

While $R^2$ measures fit, our primary focus for validty is the **Sortino Ratio**, which penalizes *downside* volatility. This is critical because for invalid models, errors often manifest as "crashes" (large negative residuals), which the Sortino Ratio captures better than Sharpe.

### **Key Findings**
*   **Baseline (FF5) Performance:**
    *   **Sortino:** ~25.0
    *   **Interpretation:** The Fama-French 5-factor model provides a solid floor for risk-adjusted returns, effectively explaining systemic variance.
*   **Sentiment Augmented (FF5 + Sentiment):**
    *   **Sortino:** ~25.4
    *   **Insight:** Adding "Average Sentiment" provides only a marginal improvement (+1.6%). This suggests that generic "positive/negative" sentiment is too coarse to capture specific risk events.
*   **Topic Models (The Breakdown)**
    *   **Sortino:** **~49.7 (FF5 + Topics)**
    *   **Impact:** Incorporating *Topic Deviations* (shifts in specific narrative clusters) **doubles the Sortino Ratio** compared to the Baseline.
    *   **Why?** Topic features likely flag specific downside risks (e.g., "Product Recalls", "Regulatory Hits") that generic sentiment misses. By identifying these specific negative shocks, the model avoids large drawdown days, drastically improving the Sortino metric.

### **Conclusion**
The **Topic-Augmented** models are far superior in risk-adjusted terms. While they may not always have the highest raw $R^2$, their ability to preserve capital during downside shocks (High Sortino) makes them the preferred specification for a robust trading strategy.

## 3. Model Equation Specifications

Below are the exact regression specifications for the key models tested. All regressions are run on **Daily Excess Returns** ($R_{t} - R_{f,t}$).

### **1. Baseline (FF5)**
The standard Fama-French 5-Factor Model.
$$ R_{t} - R_{f,t} = \alpha + \beta_{mkt}(Mkt-RF) + \beta_{SMB}SMB + \beta_{HML}HML + \beta_{RMW}RMW + \beta_{CMA}CMA + \epsilon_t $$

### **2. Sentiment Augmented (FF5_Sentiment)**
Adds a single aggregate sentiment score.
$$ R_{t} - R_{f,t} = \text{Baseline Factors} + \beta_{sent} S_{t-1} + \epsilon_t $$
*   $S_{t-1}$: Average Sentiment Score of all news on day $t-1$.

### **3. Topic Augmented (FF5_Topics)**
Adds specific sentiment scores for distinct topics (clusters).
$$ R_{t} - R_{f,t} = \text{Baseline Factors} + \sum_{k=0}^{4} \beta_{topic,k} T_{k, t-1} + \epsilon_t $$
*   $T_{k, t-1}$: Sentiment score for Topic Cluster $k$ (0-4) on day $t-1$.

### **4. Topic Shock (FF5_TopicShock)**
Uses Z-Scores to measure *abnormal* shocks in topic sentiment.
$$ R_{t} - R_{f,t} = \text{Baseline Factors} + \sum_{k=0}^{4} \beta_{shock,k} Z(T_{k, t-1}) + \epsilon_t $$
*   $Z(T_{k, t-1})$: Rolling Z-Score of the topic sentiment, capturing "shocks" relative to the recent norm.

## 4. Chart Generation & Trends

### **1. Actual vs. Predicted Returns**
*   **Visualization:** A scatter plot with Actual Returns ($x$-axis) vs. Predicted Returns ($y$-axis).
*   **Key Trend to Highlight:** The **Residuals (Errors)**.
    *   **Fat Tails (Kurtosis):** The plot will likely show outliers where the model under-predicts extreme moves (both crashes and rallies). This validates the high Kurtosis values seen in the summary stats (e.g., Kurtosis > 7).
    *   **Implication:** The model works well in "normal" regimes but struggles to capture "Black Swan" or extreme news shocks.

### **2. Rolling Beta**
*   **Visualization:** Line chart of the Time-Varying Beta ($\beta_{mkt}$) over the 252-day window.
*   **Key Trend:**
    *   **Regime Shifts:** Look for sudden shifts in Beta around earnings releases or major macro events. A rising Beta indicates the stock is becoming riskier and more correlated with the market; a falling Beta suggests it is decoupling (moving on its own idiosyncratic news).
    *   **Validation:** For NVDA, we expect to see Beta volatility, whereas AAPL's Rolling Beta should be relatively smoother around 1.0-1.2.

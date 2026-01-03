# 📈 Mag7 News Event Study: Quantifying Narrative Risk

> **"Topics, not just tones, drive alpha."**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![NLP](https://img.shields.io/badge/NLP-ChronoBERT%20%2B%20FinBERT-green)
![Quant](https://img.shields.io/badge/Quant-Fama--French%205--Factor-red)

## 📖 Executive Summary
This project presents a rigorous event study analyzing the impact of **specific news narratives** on the "Magnificent 7" stocks (AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA).

Unlike traditional sentiment analysis that aggregates all news into a single "Positive/Negative" score, this framework uses **unsupervised learning (ChronoBERT + K-Means)** to isolate distinct topics (e.g., *Regulatory Scrutiny* vs. *Product Launches*).

**Key Insight:** Generic sentiment adds minimal value over the Fama-French baseline. However, detecting **"Topic Shocks"**—abnormal spikes in sentiment for specific narratives—drastically improves risk-adjusted returns, **doubling the Sortino Ratio**.

---

## 📊 Key Findings

Comparative performance of trading strategies based on different risk factors (2021-2025):

| Model Specification | Sortino Ratio | Improvement | Interpretation |
| :--- | :--- | :--- | :--- |
| **1. Baseline (FF5)** | **25.0** | - | Market beta & factors explain most variance. |
| **2. FF5 + Sentiment** | 25.4 | +1.6% | Aggregate sentiment is too coarse a signal. |
| **3. FF5 + Topic Shocks** | **49.7** | **+98.8%** | **Isolating specific narrative risks avoids "crashes".** |

> **Conclusion:** The specific *source* of the news matters more than the general *mood*. A "Regulatory" shock carries different risk premiums than a "Product" shock.

---

## 🏗️ Technical Methodology

The pipeline follows a strict "No Look-Ahead Bias" protocol suitable for institutional deployment.

### 1. Advanced NLP Pipeline
*   **Entity Masking**: To ensure clustering focuses on *narratives* rather than *entities*, all mentions of company names (e.g., "Apple", "Nvidia") are masked to a neutral `<company>` token before embedding.
*   **ChronoBERT Embeddings**: We use **ChronoBERT** (Manela et al., 2025) instead of standard BERT.
    *   *Why?* Standard BERT models trained on 2024 data "know" the future of 2022. ChronoBERT uses time-specific checkpoints to ensure embeddings for 2022 news only use knowledge available up to 2021.
*   **Dynamic Clustering**: K-Means ($K=50$) groups articles into latent topics.
*   **Financial Sentiment**: **FinBERT** (ProsusAI) provides domain-specific sentiment scores (Positive, Neutral, Negative).

### 2. Quantitative Framework
We use the **Fama-French 5-Factor Model** as the robust baseline to beat. The event study tests whether news signals provide alpha *after* controlling for:
*   Market Risk (Mkt-RF)
*   Size (SMB) & Value (HML)
*   Profitability (RMW) & Investment (CMA)

**The Winning Model (Topic Shock):**
$$ R_{t} - R_{f,t} = \alpha + \sum \beta_{FF}F_{t} + \sum_{k=1}^{K} \beta_{shock,k} Z(S_{k, t-1}) + \epsilon_t $$
*   Where $Z(S_{k, t-1})$ is the rolling Z-Score of sentiment for Topic $k$.

---

## 📂 Repository Structure

```text
.
├── src/
│   ├── analysis/             # Core Event Study & Regression Logic
│   │   ├── event_study.py    # OLS Models & Signal Construction
│   │   └── visuals.py        # Chart Generation (Residuals, Betas)
│   ├── chronobert_kmeans.py  # Step 1: Embedding & Clustering Pipeline
│   ├── sentiment_finbert.py  # Step 2: Financial Sentiment Scoring
│   └── run_analysis.py       # Main CLI Entrypoint
├── data/
│   ├── processed/            # Parquet files (News + OHLCV)
│   └── outputs/              # Regression Results (Tables & Plots)
├── config/
│   └── topic_map.csv         # Mapping Topic IDs -> Human Labels
└── reports/
    └── regression_t1/          # Latest T+1 Forecast Analysis (Results & Metrics)
```

---

## 🚀 Usage

The pipeline is fully automated via `src/run_analysis.py`.

### Installation
```bash
pip install -r requirements.txt
# Requires: torch, transformers, scikit-learn, pandas, statsmodels
```

### Running the Pipeline

**1. End-to-End Analysis (Recommended)**
Runs feature engineering, variable construction, and event study regressions.
```bash
python src/run_analysis.py --stage all
```

**2. Run specific components**
```bash
# Only re-run the regressions (useful for tweaking model specs)
python src/run_analysis.py --stage event_study

# Filter for specific tickers
python src/run_analysis.py --stage event_study --symbol NVDA.US,TSLA.US
```

---

## 🖥️ Interactive Terminal

The project includes a full-stack **Alpha Terminal** for real-time signal monitoring, built to mimic an institutional trader's dashboard.

### Architecture
*   **Frontend**: React + Vite + Tailwind CSS (Fast, responsive UI with "Glassmorphism" design).
*   **Backend**: FastAPI (High-performance Python API serving model signals).

### Launch Instructions

**1. Start the Backend API**
Serves Topic Shock signals and portfolio stats.
```bash
cd terminal/backend
python main.py
# Server starts at http://localhost:8000
```

**2. Start the Frontend Dashboard**
Launches the interactive UI.
```bash
# In a new terminal window
cd terminal/frontend
npm install  # First time only
npm run dev
# Dashboard available at http://localhost:5173
```

---

## 📧 Contact & Citation
**Author:** Vivan Doshi
**Project:** News Event Study & Algorithmic Trading Strategy

*   **ChronoBERT**: He, Lv, Manela, & Wu (2025). "ChronoBERT: Pre-training on Timeline-Aware Data".
*   **FinBERT**: Araci, D. (2019). "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models".

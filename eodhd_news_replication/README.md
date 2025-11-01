# EODHD News Event Study - Business News and Business Cycles Replication

A comprehensive implementation of the "Business News and Business Cycles" paper methodology using EODHD news data to analyze the relationship between financial news topics and market returns.

## 📊 Project Overview

This project replicates and extends the methodology from academic research on news-driven market dynamics, implementing:

- **News Data Collection**: Automated fetching of financial news from EODHD API
- **Text Processing**: Advanced NLP pipeline for cleaning and tokenization
- **Topic Modeling**: LDA (Latent Dirichlet Allocation) with hyperparameter optimization
- **Time Series Analysis**: Topic attention aggregation and trend analysis
- **Predictive Modeling**: LASSO regression and VAR models for return forecasting
- **Trading Strategy**: Backtested news-based trading signals

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- EODHD API key (stored in `.env` file)
- 8GB+ RAM recommended for full pipeline
- 10GB+ disk space for data storage

### Installation

1. Clone the repository:
```bash
cd eodhd_news_replication
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create .env file with your API key
echo "EODHD_API_KEY=your_api_key_here" > .env
```

4. Download NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Running the Pipeline

#### Full Pipeline Execution
```bash
python main_pipeline.py
```

#### Run Specific Steps
```bash
# Data acquisition only
python main_pipeline.py --steps data

# Text preprocessing and DTM construction
python main_pipeline.py --steps preprocess dtm

# Topic modeling with custom K
python main_pipeline.py --steps lda --n-topics 100

# Skip data acquisition if data exists
python main_pipeline.py --skip-data-acquisition
```

## 📁 Project Structure

```
eodhd_news_replication/
├── conf/                      # Configuration files
│   ├── experiment.yaml        # Main configuration
│   ├── symbols_us.txt         # US stock symbols
│   ├── stopwords.txt          # Custom stopwords
│   └── vocabulary.csv         # Generated vocabulary
│
├── data/                      # Data storage
│   ├── raw/                   # Raw API data
│   │   ├── news_raw_*.parquet
│   │   └── prices_raw.parquet
│   ├── clean/                 # Preprocessed data
│   │   └── news_clean.parquet
│   └── derived/               # Processed features
│       ├── dtm.npz            # Document-term matrix
│       ├── returns_*.parquet  # Calculated returns
│       └── topics_*.parquet   # Topic time series
│
├── models/                    # Trained models
│   ├── lda_k=*/              # LDA model artifacts
│   │   ├── model.pkl         # Trained model
│   │   ├── theta.parquet     # Document-topics
│   │   ├── phi.parquet       # Topic-terms
│   │   └── top_terms.csv     # Top words per topic
│   └── lda_cv_results.csv    # Cross-validation results
│
├── results/                   # Analysis outputs
│   ├── predictions_oos.csv   # Out-of-sample predictions
│   ├── trading_performance.json
│   ├── var_summary.json
│   ├── equity_curve.png
│   └── irf_plots/            # Impulse response plots
│
├── logs/                      # Pipeline logs
│   ├── pipeline_*.log
│   └── *_stats.json
│
└── src/                       # Source code
    ├── data/                  # Data modules
    │   ├── data_acquisition.py
    │   ├── text_preprocessing.py
    │   └── document_term_matrix.py
    ├── models/                # Model modules
    │   ├── lda_topic_modeling.py
    │   └── predictive_models.py
    └── analysis/              # Analysis modules
        └── time_aggregation.py
```

## 🔧 Configuration

The pipeline is configured via `conf/experiment.yaml`. Key parameters:

### Data Acquisition
- **Time Range**: 2019-01-01 to 2024-10-31 (default)
- **Markets**: US equities
- **Symbols**: SPY + sector ETFs + top 100 stocks

### Text Processing
- **Min Text Length**: 100 characters
- **Min Tokens**: 30 per document
- **Duplicate Threshold**: 0.95 cosine similarity

### Topic Modeling
- **K Values**: [50, 80, 120, 150, 180]
- **Dirichlet Priors**: α=1.0, β=1.0
- **Iterations**: 300
- **Cross-validation**: 10-fold

### Predictive Models
- **Forecast Horizon**: 1 month
- **Training Window**: 120 months (expanding)
- **Test Split**: 30%
- **Transaction Costs**: 15 bps

## 📊 Pipeline Steps

### 1. Data Acquisition (`data_acquisition.py`)
- Fetches news articles via EODHD News API
- Retrieves price data for market indices and stocks
- Calculates daily and monthly returns
- Deduplicates articles by content hash

### 2. Text Preprocessing (`text_preprocessing.py`)
- Cleans HTML, URLs, and boilerplate text
- Tokenizes and lemmatizes text
- Removes stopwords (custom + NLTK)
- Extracts unigrams and bigrams
- Filters by document frequency thresholds

### 3. Document-Term Matrix (`document_term_matrix.py`)
- Builds sparse matrix representation
- Calculates TF-IDF weighting
- Generates term and document indices
- Computes corpus statistics

### 4. Topic Modeling (`lda_topic_modeling.py`)
- Performs hyperparameter search over K
- Trains batch LDA for analysis
- Trains online LDA for forecasting (no look-ahead)
- Calculates perplexity and coherence metrics
- Extracts top terms per topic

### 5. Time Aggregation (`time_aggregation.py`)
- Aggregates document topics to time series
- Supports daily and monthly frequencies
- Implements mean and length-weighted averaging
- Identifies topic spikes during events
- Merges with market returns

### 6. Predictive Models (`predictive_models.py`)
- LASSO feature selection for topic predictors
- Rolling window out-of-sample backtesting
- Vector Autoregression (VAR) modeling
- Impulse response analysis
- Trading strategy simulation

## 📈 Key Outputs

### Topic Analysis
- **Topic Coherence**: Semantic quality of discovered topics
- **Top Terms**: Most representative words per topic
- **Topic Attention Series**: Time-varying topic prevalence
- **Event Spikes**: Abnormal topic attention during major events

### Predictive Performance
- **Out-of-Sample R²**: Forecast accuracy vs benchmark
- **Sharpe Ratio**: Risk-adjusted returns of strategy
- **Hit Rate**: Directional accuracy of predictions
- **Maximum Drawdown**: Worst peak-to-trough loss

### VAR Results
- **Granger Causality**: Statistical relationships
- **Impulse Responses**: Dynamic effects of shocks
- **Variance Decomposition**: Contribution to volatility

## 🧪 Robustness Checks

The pipeline includes several robustness tests:

1. **Varying K**: Test different numbers of topics
2. **Aggregation Methods**: Compare mean vs weighted averaging
3. **Vocabulary Thresholds**: Sensitivity to term filtering
4. **Text Sources**: Headlines vs full content
5. **Model Alternatives**: LASSO vs Elastic Net vs Ridge

## 📝 Usage Examples

### Custom Symbol List
```python
from src.data.data_acquisition import EODHDDataAcquisition

acquisition = EODHDDataAcquisition()
custom_symbols = ['AAPL.US', 'GOOGL.US', 'MSFT.US']
news_df = acquisition.fetch_all_news(symbols=custom_symbols)
```

### Specific Topic Count
```python
from src.models.lda_topic_modeling import run_lda_pipeline

# Train with exactly 100 topics
results = run_lda_pipeline(n_topics=100)
```

### Custom Date Range
Edit `conf/experiment.yaml`:
```yaml
data_acquisition:
  time_range:
    start_date: "2020-01-01"
    end_date: "2023-12-31"
```

## ⚠️ Important Notes

### Data Storage
- Keep API keys secure and never commit to version control
- Raw data can be large (several GB for multi-year periods)
- Use parquet format for efficient storage

### Computational Requirements
- LDA training can take 30-60 minutes for large corpora
- Cross-validation multiplies training time by fold count
- Consider using subset for initial testing

### No Look-Ahead Protocol
- Online LDA ensures no future information leakage
- Predictions use only historically available data
- Critical for valid backtesting results

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please respect:
- EODHD terms of service for data usage
- Academic citation requirements if using methodology
- Proprietary data restrictions

## 📚 References

Based on methodology from:
- "Business News and Business Cycles" (academic paper)
- EODHD API documentation
- Scikit-learn LDA implementation

## 🆘 Troubleshooting

### Common Issues

1. **API Rate Limits**
   - Adjust `rate_limit_delay` in configuration
   - Use smaller symbol lists for testing

2. **Memory Errors**
   - Reduce vocabulary size (`max_vocab_size`)
   - Process data in smaller batches
   - Use online learning methods

3. **Missing Data**
   - Check API key validity
   - Verify symbol formats (e.g., "AAPL.US")
   - Ensure date ranges have available data

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Check logs in `logs/` directory for debugging
- Review configuration in `conf/experiment.yaml`

---

**Note**: This implementation is designed for research and educational purposes. Always validate results and consider market risks before any trading decisions.
# Replicating “Business News and Business Cycles” with EODHD News  
*A step-by-step, code-free playbook for an AI to execute*

---

## 0) Objective and Scope

- Build topic time series from financial news  
- Link topic attention to market returns and (optionally) macro indicators  
- Mimic the paper’s logic while using your EODHD News feed instead of WSJ archives

**Deliverables**

- Cleaned article dataset with standardized fields  
- Topic model artifacts (topics, top words, per-document topic weights)  
- Monthly topic-attention time series  
- Linked panel with market returns and optional macro series  
- Model outputs: predictive topics, VAR results, backtest metrics  
- A reproducible folder with logs, configs, and metadata

---

## 1) Project Structure and Conventions

- Create top-level folders:  
  - `/data/raw`, `/data/clean`, `/data/derived`, `/models`, `/results`, `/logs`, `/conf`  
- Use consistent file naming:  
  - `news_raw_<yyyymmdd>.parquet`, `news_clean.parquet`, `dtm.npz`, `lda_model.k=050.pkl`, `monthly_attention.parquet`  
- Track metadata in `/conf/experiment.yaml`:  
  - API keys redacted  
  - Time windows, symbols, markets  
  - Text filters, LDA params, seeds  
  - Train/validation/test splits  
- Set fixed random seeds for all stochastic steps

---

## 2) Data Acquisition via EODHD

**2.1 Sources and Time Range**

- News: EODHD News endpoint  
- Prices: EODHD EOD/Intraday endpoints  
- Macro (optional): FRED (or other academic sources)

**2.2 Symbol Universe**

- Start with US market, broad coverage  
- Include:  
  - Major index ticker (e.g., SPY.US)  
  - Top 100–500 US equities by market cap  
  - Add sector ETFs if desired  
- Save the symbol list to `/conf/symbols_us.txt`

**2.3 Pull Strategy**

- Pull by symbol and by date range  
- Time horizon: minimum 3–5 years; more is better  
- Rate-limit compliant batching  
- Store raw JSON responses in `/data/raw/news_*.jsonl`  
- Convert to a canonical table with these fields:  
  - `article_id` (hash of source+url+title+published_at)  
  - `published_at` (UTC)  
  - `source_domain`, `source_name`  
  - `title`, `content` (body or summary; keep both)  
  - `tickers` (list)  
  - `url`  
  - `language`  
  - `sentiment` (if provided)  
- De-duplicate on `article_id`; keep first occurrence

**2.4 Completeness Checks**

- Log counts per day, per source, per ticker  
- Flag days with zero coverage for major symbols  
- Record start/end dates actually available

---

## 3) Text Normalization Pipeline

**3.1 Language and Eligibility**

- Keep English only  
- Drop articles with missing `title` and `content`  
- Drop articles with total text length < 100 characters

**3.2 Boilerplate and Noise Removal**

- Strip HTML, ads, disclaimers, tickers spam  
- Remove URL fragments and emails  
- Remove all-caps stock lists at the end of articles

**3.3 Tokenization and Lemmatization**

- Lowercase  
- Remove punctuation, numbers, and non-alphabetic tokens  
- Remove stopwords (standard English list + finance stopwords like “market”, “stock”, “company” if needed; save list in `/conf/stopwords.txt`)  
- Lemmatize tokens (verb/noun normalization)  
- Keep tokens of length ≥ 3

**3.4 N-grams and Vocabulary Construction**

- Build uni-grams and bi-grams  
- Minimum document frequency thresholds:  
  - Unigrams: ≥ 0.1% of documents  
  - Bigrams: ≥ 0.05% of documents  
- Maximum vocabulary size cap: 10–30k terms  
- Save the final vocabulary and thresholds to `/conf/vocabulary.yaml`

**3.5 Document Quality Filters**

- Drop documents with fewer than 30 valid tokens after cleaning  
- Drop near-duplicates: cosine similarity of tf-idf vectors > 0.95 within ±3 days  
- Log and count removals

---

## 4) Document–Term Matrix (DTM)

- Rows: documents (articles)  
- Columns: vocabulary terms  
- Values: integer term counts (not tf-idf)  
- Persist in sparse format (`.npz`), with index mapping files:  
  - `/data/derived/dtm.npz`  
  - `/data/derived/doc_index.parquet` (maps row → article_id)  
  - `/data/derived/term_index.parquet` (maps col → term)

---

## 5) Topic Modeling (LDA)

**5.1 Model Choice**

- Use LDA on count data (to mirror paper)  
- Batch LDA for exploration and topic quality  
- Online LDA for forecasting experiments to avoid look-ahead

**5.2 Hyperparameters**

- Number of topics, K: search grid {50, 80, 120, 150, 180}  
- Dirichlet priors: α = 1.0, β = 1.0 (paper-style)  
- Iterations: at least 300 for batch LDA; monitor convergence  
- Random seed fixed

**5.3 Model Selection**

- Evaluate per-word log-likelihood on held-out folds (10-fold CV)  
- Use approximate marginal likelihood / Bayes factor criteria if available  
- Choose the K with best out-of-sample fit and interpretable topics

**5.4 Outputs to Persist**

- Topic–term matrix Φ (size K × V)  
- Top-N terms per topic (e.g., N = 15)  
- Document–topic weights Θ (size D × K)  
- Topic coherence scores  
- Save as:  
  - `/models/lda_k=<K>/phi.parquet`  
  - `/models/lda_k=<K>/theta.parquet`  
  - `/models/lda_k=<K>/top_terms.csv`  
  - `/models/lda_k=<K>/metrics.json`

---

## 6) Time Aggregation: Topic Attention

**6.1 Time Indexing**

- Convert `published_at` to exchange-timezone date  
- Create both **daily** and **monthly** indices

**6.2 Aggregation Rule**

- For each period t and topic k, compute attention:  
  - Mean of per-document topic weight θ over all articles in period t  
  - Also compute alternative: token-weighted average using document length as weights  
- Persist two versions: simple mean and length-weighted

**6.3 Smoothing and Seasonality**

- Keep raw attentions for modeling  
- Optionally store a 3-month moving average for visualization only

**6.4 Final Series**

- Save daily: `/data/derived/topics_daily.parquet` (columns: topic_1..topic_K)  
- Save monthly: `/data/derived/topics_monthly.parquet`

---

## 7) Price and Macro Panels

**7.1 Market Returns**

- Pull EODHD adjusted close for:  
  - SPY.US (market proxy)  
  - Optionally sector ETFs (e.g., XLF.US, XLK.US)  
- Compute log returns at daily and monthly frequencies  
- Align to the same index as topic attention  
- Save to `/data/derived/returns_{daily,monthly}.parquet`

**7.2 Optional Macro Series**

- If included, fetch monthly: IP growth, unemployment rate, CPI inflation, term spread, VIX, EPU index (if accessible)  
- Align on month-end dates  
- Save to `/data/derived/macro_monthly.parquet`

**7.3 Merge Panel**

- Inner join on date with topics and returns (and macro if used)  
- Store: `/data/derived/panel_{daily,monthly}.parquet`

---

## 8) No Look-Ahead Protocol

**8.1 Online Topic Estimation**

- Use Online LDA (oLDA) for forecasting use-cases  
- Train incrementally in chronological order  
- At each time t, update the model with articles up to t  
- Infer θ for articles at t with parameters estimated **only** from ≤ t  
- Re-aggregate topic attention by period  
- Save the online attention series separately:  
  - `/data/derived/topics_monthly_online.parquet`

**8.2 Audit**

- Log the last training date used to produce each period’s topics  
- Verify no article with `published_at` > period end is included

---

## 9) Variable Construction and Screening

**9.1 Standardization**

- Standardize topic series to zero mean, unit variance within the estimation window  
- Standardize returns similarly for comparability

**9.2 Correlation and Stability Checks**

- Compute pairwise correlations among topics; flag > 0.9 absolute correlation  
- Drop unstable topics with near-zero variance  
- Keep a core set for modeling

**9.3 Topic Labeling**

- Auto-label each topic with its top 5 terms  
- Manually review for interpretability  
- Save a simple dictionary `topic_id → label`

---

## 10) Predictive Regressions (Monthly)

**10.1 Setup**

- Target: next-month SPY return (t+1)  
- Predictors: current month topic attentions (t)  
- Rolling window: 120 months expanding or 60-month fixed window  
- Out-of-sample evaluation on last 30–40% of months

**10.2 Feature Selection**

- LASSO with cross-validation on training window  
- Cap selected topics to the top 5–10 for interpretability  
- Record the selected topic IDs and coefficients at each re-fit date

**10.3 Metrics**

- OOS R² vs historical mean benchmark  
- Diebold–Mariano test vs benchmark forecast  
- Hit rate on sign of return  
- Annualized Sharpe of timing strategy (see 10.4)

**10.4 Simple Timing Strategy**

- Forecast sign using selected model  
- Long SPY if forecast > 0, else T-bill proxy  
- Transaction cost assumption: 10–25 bps per switch  
- Compute cumulative return and drawdown

**10.5 Outputs**

- `/results/predictive_topics_oos.csv` (time, selected topics, coefs)  
- `/results/forecast_metrics.json` (R², DM-pvalue, hit-rate)  
- `/results/timing_equity_curve.parquet`

---

## 11) Text-Augmented VAR (Monthly)

**11.1 Variables**

- Endogenous:  
  - SPY returns (or excess returns)  
  - 3–6 selected topic attentions (use stable, interpretable ones)  
  - Optional: macro series (IP growth, unemployment, term spread)  
- Lags: start with 3

**11.2 Specification**

- Estimate standard VAR first for baseline  
- Estimate sparse VAR using group selection (group = all lags of a variable)  
- Criteria: AIC/BIC for lag length and variable inclusion

**11.3 Impulse Responses**

- Identify via Cholesky ordering:  
  - Order macro first, topics next, returns last (or as theory suggests)  
- Horizon: 6–12 months  
- Bootstrap percentile intervals (≥ 1,000 reps)

**11.4 Outputs**

- `/results/var_summary.txt`  
- `/results/irf_plots/` (one per shock variable)  
- `/results/fevd.csv` (optional)

---

## 12) Robustness and Sensitivity

- Vary K: {50, 80, 120, 150, 180}  
- Vary weighting: equal vs length-weighted topic attention  
- Vary vocabulary thresholds  
- Use headline-only vs headline+summary text  
- Replace LASSO with Elastic Net  
- Use daily frequency with weekly aggregation as a check  
- Replace returns target with volatility or drawdown probability

Record each run in `/conf/experiment_runs.csv` with parameters and metrics.

---

## 13) Quality and Integrity Checks

- Topic coherence: ensure top words make sense  
- Time-series sanity: spikes during known events (e.g., COVID crash)  
- Stability: selected topics are not random every month  
- Data leakage: confirm online topics use only past data  
- Missingness: report % missing per series, imputation policy (prefer no imputation)

---

## 14) Documentation and Artifacts

- README.md summarizing pipeline and main findings  
- Model cards for LDA and VAR in `/models/cards/` describing:  
  - Data used, date ranges  
  - Hyperparameters  
  - Known limitations and biases  
- Versioned outputs with timestamps

---

## 15) Optional Extensions

- Firm-level panel: aggregate topics by ticker, run panel regressions on next-month excess returns  
- Sentiment-topics: interact EODHD sentiment with topic attention  
- Narrative retrieval: for a given month and topic spike, list top contributing articles (title + URL)  
- Event study: align articles to firm earnings dates and analyze abnormal returns

---

## 16) Acceptance Criteria

- At least one K produces coherent, interpretable topics  
- Monthly topic attention series show intuitive dynamics and known event spikes  
- Predictive regressions deliver positive OOS R² vs mean  
- Timing strategy Sharpe exceeds 0.5 net of costs in the test window (target, not guarantee)  
- VAR IRFs display economically meaningful responses with bootstrapped intervals

---

## 17) Reproducibility Keys

- All steps driven by config files in `/conf`  
- All randomness controlled by fixed seeds stored in config  
- All intermediate data persisted; no transient, hidden steps  
- Logs with timing and record counts for each stage in `/logs`

---

## 18) Privacy and Licensing

- Keep API keys out of repo; use environment variables  
- Respect EODHD terms of use for data storage and sharing  
- Do not redistribute proprietary content; store only metadata and derived features when necessary

---

### Final Note

Follow the steps in order.  
Do not skip logging, de-duplication, or the no-look-ahead protocol.  
Persist every intermediate artifact so you can audit and rerun any stage quickly.



import pandas as pd
import numpy as np
import os
import argparse
import logging
import sys
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans


from sklearn.feature_extraction.text import TfidfVectorizer
import statsmodels.api as sm

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(embeddings_path, stock_path, factors_path):
    logger.info("Loading Data...")
    
    # 1. Embeddings & Text
    # Expects columns: article_id, symbol_query, date, embedding (list), sentiment_finbert, text_for_nlp
    emb_df = pd.read_parquet(embeddings_path)
    emb_df['date'] = pd.to_datetime(emb_df['date'])
    
    # Convert embedding column from list to numpy matrix per row is slow
    # Best to keep as DF for slicing, then stack when needed
    
    # 2. Stock Data
    stock_df = pd.read_parquet(stock_path)
    # Ensure date
    if 'date' in stock_df.columns:
        stock_df['date'] = pd.to_datetime(stock_df['date'])
    if 'symbol_query' in stock_df.columns:
        stock_df = stock_df.rename(columns={'symbol_query': 'symbol'})
    if 'ret_log_1d' in stock_df.columns:
        stock_df = stock_df.rename(columns={'ret_log_1d': 'log_ret'})
        
    # 3. Factors
    factors_df = pd.read_csv(factors_path)
    if 'Date' in factors_df.columns:
        factors_df = factors_df.rename(columns={'Date': 'date'})
        
    factors_df['date'] = pd.to_datetime(factors_df['date'])
    
    return emb_df, stock_df, factors_df

def get_top_terms(texts, n_top=3):
    """Extract top TF-IDF terms for a cluster."""
    if not texts:
        return "empty"
    try:
        tfidf = TfidfVectorizer(stop_words='english', max_features=100)
        tfidf_matrix = tfidf.fit_transform(texts)
        # Sum tfidf for each term
        sum_scores = tfidf_matrix.sum(axis=0)
        # Get map
        terms = tfidf.get_feature_names_out()
        
        # Sort
        scores = [(terms[i], sum_scores[0, i]) for i in range(len(terms))]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return "_".join([x[0] for x in scores[:n_top]])
    except:
        return "unknown"

def run_dynamic_pipeline(emb_df, stock_df, factors_df, window=252, n_clusters=6, output_dir="reports/dynamic_analysis"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare Master Loop
    dates = sorted(emb_df['date'].unique())
    dates = [d for d in dates if d in stock_df['date'].values] # Sync
    dates = sorted(list(set(dates)))
    
    results = []
    topic_history = []
    
    start_idx = window
    logger.info(f"Starting Dynamic Pipeline. Total Dates: {len(dates)}. Window: {window}")
    
    for i in tqdm(range(start_idx, len(dates))):
        target_date = dates[i]
        
        # 1. Define Window (Exclusive of target)
        window_start = dates[i - window]
        train_mask = (emb_df['date'] >= window_start) & (emb_df['date'] < target_date)
        
        train_emb_df = emb_df[train_mask]
        
        if len(train_emb_df) < 50:
            continue
            
        # 2. RUN K-MEANS
        # Stack embeddings
        X_train = np.vstack(train_emb_df['embedding'].values)
        
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=1024,
            n_init='auto'
        )
        train_labels = kmeans.fit_predict(X_train)
        
        # 3. NAME TOPICS (AI-Simulated via TF-IDF)
        train_emb_df = train_emb_df.copy()
        train_emb_df['cluster'] = train_labels
        
        cluster_names = {}
        for c in range(n_clusters):
            texts = train_emb_df[train_emb_df['cluster'] == c]['text_for_nlp'].tolist()
            name = get_top_terms(texts)
            cluster_names[c] = name
            
        # Log Topic Snapshot for this window (store for report)
        topic_history.append({
            'date': target_date,
            'topics': cluster_names
        })
        
        # 4. AGGREGATE SENTIMENT PER TOPIC (In-Sample Construction)
        # Pivot: Date | Symbol | sent_topic_0 ... sent_topic_5
        # We need daily sentiment per topic per symbol
        
        # Add Cluster Name to DF
        train_emb_df['topic_name'] = train_emb_df['cluster'].map(cluster_names)
        
        # Aggregate
        # We want: Index=[Date, Symbol], Cols=[sent_topic_0, ... sent_topic_5]
        # Actually easier to use cluster ID for regression column names to avoid changing schemas
        
        pivot = train_emb_df.pivot_table(
            index=['date', 'symbol_query'],
            columns='cluster',
            values='sentiment_finbert',
            aggfunc='mean'
        ).fillna(0)
        
        pivot.columns = [f"sent_cluster_{c}" for c in pivot.columns]
        
        # 5. MERGE WITH TARGETS
        # We need stock returns for these dates
        stock_window = stock_df[(stock_df['date'] >= window_start) & (stock_df['date'] < target_date)]
        
        # Merge Factors
        # ... (Skipping factors for simplicity or adding them? User said "along with other metrics")
        # Let's add factors.
        
        fs_window = factors_df[(factors_df['date'] >= window_start) & (factors_df['date'] < target_date)]
        
        # Merge All
        reg_df = pd.merge(stock_window, pivot, left_on=['date', 'symbol'], right_on=['date', 'symbol_query'], how='left').fillna(0)
        reg_df = pd.merge(reg_df, fs_window, on='date', how='left')
        
        # Prepare Regression
        features = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'] + [f"sent_cluster_{c}" for c in range(n_clusters)]
        # Add lags? User said "like we do". Usually means lags. 
        # But dynamic clustering changes every day! Topic 0 today != Topic 0 tomorrow.
        # CRITICAL ISSUE: Rolling K-Means means "Cluster 0" assumes different meaning every window shift.
        # You cannot easily run OLS on "Cluster 0" if Cluster 0 is "AI" today and "Oil" tomorrow.
        # UNLESS we are running the regression *only* on the current window to predict *tomorrow*.
        # Yes, that is the Rolling Regression paradigm. 
        # We train a model VALID FOR THIS WINDOW.
        
        target_col = 'log_ret' # or Excess return
        
        # Filter valid
        mask = reg_df[[target_col, 'RF'] + features].notna().all(axis=1)
        valid_train = reg_df[mask]
        
        if len(valid_train) < 50:
            continue
            
        y = valid_train[target_col] - valid_train['RF']
        X = sm.add_constant(valid_train[features])
        
        try:
            model = sm.OLS(y, X).fit()
        except:
            continue
            
        # 6. PREDICT TEST DAY (T)
        # We need embeddings for Target Date
        test_emb_df = emb_df[emb_df['date'] == target_date]
        if test_emb_df.empty:
            continue
            
        # Project Test Embeddings to Clusters
        X_test_emb = np.vstack(test_emb_df['embedding'].values)
        test_labels = kmeans.predict(X_test_emb)
        
        test_emb_df = test_emb_df.copy()
        test_emb_df['cluster'] = test_labels
        
        # Aggregate Test Sentiment
        test_pivot = test_emb_df.pivot_table(
            index=['date', 'symbol_query'],
            columns='cluster',
            values='sentiment_finbert',
            aggfunc='mean'
        ).fillna(0)
        test_pivot.columns = [f"sent_cluster_{c}" for c in test_pivot.columns]
        
        # Ensure all clusters exist (fill missing with 0)
        for c in range(n_clusters):
            col = f"sent_cluster_{c}"
            if col not in test_pivot.columns:
                test_pivot[col] = 0.0
                
        # Merge with Stock/Factors for Test Day
        test_stock = stock_df[stock_df['date'] == target_date]
        test_factors = factors_df[factors_df['date'] == target_date]
        
        if test_stock.empty: continue
        
        test_reg = pd.merge(test_stock, test_pivot, left_on=['date', 'symbol'], right_on=['date', 'symbol_query'], how='left').fillna(0)
        test_reg = pd.merge(test_reg, test_factors, on='date', how='left')
        
        # Predict
        X_test = sm.add_constant(test_reg[features], has_constant='add')
        pred_ret = model.predict(X_test)
        
        # Save Predictions
        res = pd.DataFrame({
            'date': target_date,
            'symbol': test_reg['symbol'],
            'y_true': test_reg['log_ret'] - test_reg['RF'],
            'y_pred': pred_ret
        })
        results.append(res)
        
    # Save Outputs
    if results:
        full_res = pd.concat(results)
        full_res.to_csv(os.path.join(output_dir, 'dynamic_predictions.csv'), index=False)
        
        # Metrics
        mse = ((full_res['y_true'] - full_res['y_pred'])**2).mean()
        r2 = 1 - (mse / ((full_res['y_true'] - full_res['y_true'].mean())**2).mean())
        logger.info(f"Dynamic Model OOS R2: {r2:.4f}")
        
        # Save Topic History
        topic_df = pd.DataFrame(topic_history)
        topic_df.to_json(os.path.join(output_dir, 'topic_history.json'), orient='records', lines=True)
    else:
        logger.warning("No results generated.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_path", default="data/processed/mag7_embeddings.parquet")
    parser.add_argument("--stock_path", default="data/processed/mag7_yf_2021_2025.parquet")
    parser.add_argument("--factors_path", default="reports/fama_french/data/fama_french_factors.csv")
    args = parser.parse_args()
    
    emb, stock, fact = load_data(args.embeddings_path, args.stock_path, args.factors_path)
    run_dynamic_pipeline(emb, stock, fact)

if __name__ == "__main__":
    main()

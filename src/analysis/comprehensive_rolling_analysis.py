
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
import json
import gc

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(embedding_path, master_path):
    logger.info("Loading Data...")
    
    # 1. Master Data (FF5, Returns, Static Categories)
    # Columns expected: date, symbol, log_ret, RF, Mkt-RF, SMB, HML, RMW, CMA,
    # sent_earnings_magnitude, sent_growth_magnitude, sent_macro_magnitude, sent_risk_magnitude
    # day_sentiment, day_sentiment_magnitude
    df_master = pd.read_csv(master_path)
    df_master['date'] = pd.to_datetime(df_master['date'])
    
    # 2. Embeddings (Dynamic Topics)
    # Columns: date, symbol, embedding, text_for_nlp
    df_emb = pd.read_parquet(embedding_path)
    df_emb['date'] = pd.to_datetime(df_emb['date'])
    
    # Ensure embedding is a list or array
    # If it's a list in parquet, convert to list of arrays for stacking later? 
    # Actually MiniBatchKMeans needs 2D array.
    # We will stack them on the fly.
    
    return df_emb, df_master

def get_topic_name(texts, n_top=20):
    """
    Simulates AI Naming by extracting Top N TF-IDF terms.
    User request: 'find top 20 words and coming up with a topic name'
    """
    if not texts or len(texts) == 0:
        return "Topic_Empty"
    try:
        tfidf = TfidfVectorizer(stop_words='english', max_features=100)
        tfidf_matrix = tfidf.fit_transform(texts)
        sum_scores = tfidf_matrix.sum(axis=0)
        terms = tfidf.get_feature_names_out()
        
        scores = [(terms[i], sum_scores[0, i]) for i in range(len(terms))]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        top_terms = [x[0] for x in scores[:n_top]]
        return "_".join(top_terms[:5]) # Join top 5 for concise key, store full elsewhere?
    except:
        return "Topic_Unknown"

def run_rolling_analysis(df_emb, df_master, window_size, n_clusters=5):
    """
    Runs rolling analysis for a specific window size.
    """
    logger.info(f"--- Running Analysis for Window: {window_size} days ---")
    
    results = []
    topic_history = []  # Store topic names over time
    
    # Align dates
    dates = sorted(list(set(df_master['date'].unique()) & set(df_emb['date'].unique())))
    
    # Pre-compute static features list
    static_features = [
        'day_sentiment_magnitude', 
        'sent_earnings_magnitude', 
        'sent_growth_magnitude', 
        'sent_macro_magnitude', 
        'sent_risk_magnitude'
    ]
    # Check availability
    available_static = [f for f in static_features if f in df_master.columns]
    ff5_features = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    
    start_idx = window_size
    
    for i in tqdm(range(start_idx, len(dates))):
        target_date = dates[i]
        window_start = dates[i - window_size]
        
        # 1. Get Window Data
        # Dynamic: [T-W, T)
        mask_window = (df_emb['date'] >= window_start) & (df_emb['date'] < target_date)
        train_emb = df_emb[mask_window]
        
        if len(train_emb) < 50: # Minimum data check
            continue
            
        # 2. Dynamic Clustering (K-Means)
        # Stack embeddings
        X_train = np.vstack(train_emb['embedding'].values)
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024, n_init='auto')
        kmeans.fit(X_train)
        
        # 3. Name Topics (Simulate AI)
        train_emb = train_emb.copy()
        train_emb['cluster'] = kmeans.labels_
        
        topic_names = {}
        for c in range(n_clusters):
            texts = train_emb[train_emb['cluster'] == c]['text_for_nlp'].tolist()
            topic_names[c] = get_topic_name(texts, n_top=20)
            
        topic_history.append({
            'date': target_date.strftime('%Y-%m-%d'),
            'topics': topic_names
        })
            
        # 4. Construct Dynamic Factors (Topic Sentiment)
        # We need to project the static master data days onto these topics?
        # No, we need to regress Daily Return on Daily Topic Sentiment.
        # But Topic Sentiment varies by day.
        # So for every Day d in [T-W, T), we assign its news to the CURRENT clusters
        # and calculate sentiment.
        
        # Pivot: Index=[Date, Symbol], Columns=[Topic_0, Topic_1...]
        pivot = train_emb.pivot_table(
            index=['date', 'symbol_query'],
            columns='cluster',
            values='sentiment_finbert',
            aggfunc='mean'
        ).fillna(0)
        
        feature_cols = [f"Topic_{c}" for c in range(n_clusters)]
        pivot.columns = feature_cols
        
        # 5. Merge with Master Data (Target + Static Features)
        # Filter master to window
        mask_master = (df_master['date'] >= window_start) & (df_master['date'] < target_date)
        window_master = df_master[mask_master].copy()
        
        # Merge Dynamic Features
        # Note: df_master uses 'ticker_yf' or 'symbol'? 'symbol' matches 'symbol_query' usually
        # Let's assume 'symbol' in master maps to 'symbol_query' in embeddings
        reg_df = pd.merge(
            window_master, 
            pivot, 
            left_on=['date', 'symbol'], 
            right_on=['date', 'symbol_query'], 
            how='left'
        ).fillna(0) # Fill days with no news in a topic as 0
        
        # 6. Train Regression
        # Y = Excess Return
        # X = FF5 + Static + Dynamic
        
        y = reg_df['log_ret'] - reg_df['RF']
        X_cols = ff5_features + available_static + feature_cols
        X = sm.add_constant(reg_df[X_cols])
        
        try:
            model = sm.OLS(y, X).fit()
        except:
            continue
            
        # 7. Predict Target Day (OOS)
        # We need features for target_date
        target_master = df_master[df_master['date'] == target_date]
        if target_master.empty: continue
        
        target_emb = df_emb[df_emb['date'] == target_date]
        
        # Setup Target Dynamic Features
        day_topic_sent = {c: 0.0 for c in range(n_clusters)}
        
        if not target_emb.empty:
             X_target = np.vstack(target_emb['embedding'].values)
             labels = kmeans.predict(X_target)
             target_emb = target_emb.copy()
             target_emb['cluster'] = labels
             
             # Sent per cluster (mean of news in that cluster today)
             # Group by Symbol? Yes, regression is panel (symbol level)
             # But prediction loop here assumes we have rows per symbol.
             pass
        
        # We need to build the X_test for each symbol on target_date
        # 1. Get Static & FF from target_master (One row per symbol)
        # 2. Get Dynamic from target_emb (Aggregated per symbol)
        
        if not target_emb.empty:
            target_pivot = target_emb.pivot_table(
                index=['symbol_query'],
                columns='cluster',
                values='sentiment_finbert',
                aggfunc='mean'
            ).fillna(0)
            # Reindex to ensure all topics
            target_pivot = target_pivot.reindex(columns=range(n_clusters), fill_value=0)
            target_pivot.columns = feature_cols
        else:
            # No news today? All 0
            target_pivot = pd.DataFrame(columns=feature_cols)
            
        # Merge
        pred_df = pd.merge(
            target_master,
            target_pivot,
            left_on='symbol',
            right_index=True,
            how='left'
        ).fillna(0)
        
        if pred_df.empty: continue
        
        X_test = sm.add_constant(pred_df[X_cols], has_constant='add')
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Store
        daily_res = pd.DataFrame({
            'date': target_date,
            'window': window_size,
            'symbol': pred_df['symbol'],
            'y_true': pred_df['log_ret'] - pred_df['RF'],
            'y_pred': y_pred
        })
        results.append(daily_res)
        
    if not results:
        return pd.DataFrame()
        
    return pd.concat(results), topic_history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_path", default="data/processed/mag7_embeddings.parquet")
    parser.add_argument("--master_path", default="data/master_analysis_data.csv")
    parser.add_argument("--output_dir", default="reports/comprehensive_analysis")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load
    df_emb, df_master = load_data(args.emb_path, args.master_path)
    
    all_metrics = []
    
    # Run for different windows
    windows = [50, 100, 250]
    
    for w in windows:
        res_df, topic_history = run_rolling_analysis(df_emb, df_master, window_size=w, n_clusters=5)
        
        # Save Topic History
        with open(os.path.join(args.output_dir, f"topic_history_window_{w}.json"), "w") as f:
            json.dump(topic_history, f, indent=4)
        
        if not res_df.empty:
            # Save Predictions
            res_df.to_csv(os.path.join(args.output_dir, f"preds_window_{w}.csv"), index=False)
            
            # Calc Metrics
            mse = ((res_df['y_true'] - res_df['y_pred'])**2).mean()
            r2 = 1 - (mse / ((res_df['y_true'] - res_df['y_true'].mean())**2).mean())
            
            logger.info(f"Window {w} OOS R2: {r2:.4f}")
            
            all_metrics.append({
                'window': w,
                'R2': r2,
                'MSE': mse
            })
            
    # Load Baseline (Static) for Comparison context?
    # User asked for comparison. We can just list the new ones here.
    pd.DataFrame(all_metrics).to_csv(os.path.join(args.output_dir, "comparison_metrics.csv"), index=False)

if __name__ == "__main__":
    main()

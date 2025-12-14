
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

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(embedding_path, master_path, stock_path):
    logger.info("Loading Data...")
    
    # 1. Master Data (FF5, Returns, Static Categories)
    df_master = pd.read_csv(master_path)
    df_master['date'] = pd.to_datetime(df_master['date'])
    
    # 2. Embeddings (Dynamic Topics)
    df_emb = pd.read_parquet(embedding_path)
    df_emb['date'] = pd.to_datetime(df_emb['date'])
    
    # 3. Stock Data for Mag7 Index Construction
    df_stocks = pd.read_parquet(stock_path)
    df_stocks['date'] = pd.to_datetime(df_stocks['date'])
    
    # Construct Mag7 Index (Equal Weighted)
    # Group by date and mean of ret_1d (or ret_log_1d? FF factors use simple returns usually, then converted to excess.
    # But here our Y is log_ret.
    # Let's check df_master Y variable. It uses 'log_ret'. 
    # FF factors (Mkt-RF) are usually simple returns. 
    # To be consistent with "Market Return", we should probably use simple returns for the index, 
    # then subtract RF. 
    # However, if we are regressing Log Excess Return on Log Market Excess Return, that's one way. 
    # Typical standard is Simple Excess Return. 
    # But code in comprehensive_rolling_analysis.py uses: y = reg_df['log_ret'] - reg_df['RF']
    # And X includes 'Mkt-RF'. 
    # Assuming Mkt-RF from French library is simple return. 
    # Let's compute Mag7 Index as Mean of Log Returns for consistency if y is log ret?
    # Or Mean of Simple Returns? 
    # Let's stick to Simple Returns for index construction to match typical index methodology, 
    # but since the LHS is log returns, maybe we should use log returns for the index too to match units.
    # Let's use Mean of Log Returns to be safe and consistent with the dependent variable `log_ret`.
    
    mag7_index = df_stocks.groupby('date')['ret_log_1d'].mean().reset_index()
    mag7_index.rename(columns={'ret_log_1d': 'Mag7_Index'}, inplace=True)
    
    # Merge Mag7 Index into Master
    df_master = pd.merge(df_master, mag7_index, on='date', how='left')
    
    # Calculate Mag7-RF
    # Note: RF in df_master is likely simple or log? Usually RF is simple daily rate.
    # We will assume it's appropriate to subtract.
    df_master['Mag7-RF'] = df_master['Mag7_Index'] - df_master['RF']
    
    return df_emb, df_master

def get_topic_name(texts, n_top=20):
    """
    Simulates AI Naming by extracting Top N TF-IDF terms.
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
        return "_".join(top_terms[:5]) 
    except:
        return "Topic_Unknown"

def run_rolling_analysis(df_emb, df_master, window_size, n_clusters=5):
    """
    Runs rolling analysis for a specific window size.
    """
    logger.info(f"--- Running Analysis for Window: {window_size} days ---")
    
    results = []
    topic_history = [] 
    
    dates = sorted(list(set(df_master['date'].unique()) & set(df_emb['date'].unique())))
    
    static_features = [
        'day_sentiment_magnitude', 
        'sent_earnings_magnitude', 
        'sent_growth_magnitude', 
        'sent_macro_magnitude', 
        'sent_risk_magnitude'
    ]
    available_static = [f for f in static_features if f in df_master.columns]
    
    # MODIFIED: Use Mag7-RF instead of Mkt-RF
    # Keeping other factors to control for size/value etc within the tech sector?
    # SMB, HML might not be as relevant for Mag7 but RMW/CMA might.
    # Let's keep them as controls.
    ff5_features = ['Mag7-RF', 'SMB', 'HML', 'RMW', 'CMA']
    
    start_idx = window_size
    
    for i in tqdm(range(start_idx, len(dates))):
        target_date = dates[i]
        window_start = dates[i - window_size]
        
        # 1. Get Window Data
        mask_window = (df_emb['date'] >= window_start) & (df_emb['date'] < target_date)
        train_emb = df_emb[mask_window]
        
        if len(train_emb) < 50: 
            continue
            
        # 2. Dynamic Clustering (K-Means)
        X_train = np.vstack(train_emb['embedding'].values)
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024, n_init='auto')
        kmeans.fit(X_train)
        
        # 3. Name Topics
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
            
        # 4. Construct Dynamic Factors 
        pivot = train_emb.pivot_table(
            index=['date', 'symbol_query'],
            columns='cluster',
            values='sentiment_finbert',
            aggfunc='mean'
        ).fillna(0)
        
        feature_cols = [f"Topic_{c}" for c in range(n_clusters)]
        pivot.columns = feature_cols
        
        # 5. Merge with Master Data
        mask_master = (df_master['date'] >= window_start) & (df_master['date'] < target_date)
        window_master = df_master[mask_master].copy()
        
        reg_df = pd.merge(
            window_master, 
            pivot, 
            left_on=['date', 'symbol'], 
            right_on=['date', 'symbol_query'], 
            how='left'
        ).fillna(0)
        
        # 6. Train Regression
        # Y = Excess Return
        # X = Mag7-RF + Controls + Static + Dynamic
        
        y = reg_df['log_ret'] - reg_df['RF']
        # Check if Mag7-RF is nan
        if reg_df['Mag7-RF'].isnull().any():
             reg_df = reg_df.dropna(subset=['Mag7-RF'])
             y = reg_df['log_ret'] - reg_df['RF']
             
        X_cols = ff5_features + available_static + feature_cols
        X = sm.add_constant(reg_df[X_cols])
        
        try:
            model = sm.OLS(y, X).fit()
        except Exception as e:
            # logger.error(f"Regression failed on {target_date}: {e}")
            continue
            
        # 7. Predict Target Day (OOS)
        target_master = df_master[df_master['date'] == target_date]
        if target_master.empty: continue
        
        target_emb = df_emb[df_emb['date'] == target_date]
        
        # Target Dynamic Features
        if not target_emb.empty:
             X_target = np.vstack(target_emb['embedding'].values)
             labels = kmeans.predict(X_target)
             target_emb = target_emb.copy()
             target_emb['cluster'] = labels
             
             target_pivot = target_emb.pivot_table(
                index=['symbol_query'],
                columns='cluster',
                values='sentiment_finbert',
                aggfunc='mean'
             ).fillna(0)
             target_pivot = target_pivot.reindex(columns=range(n_clusters), fill_value=0)
             target_pivot.columns = feature_cols
        else:
            target_pivot = pd.DataFrame(columns=feature_cols)
            
        pred_df = pd.merge(
            target_master,
            target_pivot,
            left_on='symbol',
            right_index=True,
            how='left'
        ).fillna(0)
        
        if pred_df.empty: continue
        
        # Ensure cols exist
        start_cols = set(pred_df.columns)
        missing = set(X_cols) - start_cols
        for c in missing:
            pred_df[c] = 0.0
            
        X_test = sm.add_constant(pred_df[X_cols], has_constant='add')
        
        y_pred = model.predict(X_test)
        
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
    parser.add_argument("--stock_path", default="data/processed/mag7_yf_2021_2025.parquet")
    parser.add_argument("--output_dir", default="reports/mag7_benchmark_analysis")
    parser.add_argument("--window", type=int, help="Specific window size to run")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load
    df_emb, df_master = load_data(args.emb_path, args.master_path, args.stock_path)
    
    all_metrics = []
    
    if args.window:
        windows = [args.window]
    else:
        windows = [100, 252]

    
    for w in windows:
        res_df, topic_history = run_rolling_analysis(df_emb, df_master, window_size=w, n_clusters=5)
        
        with open(os.path.join(args.output_dir, f"topic_history_window_{w}.json"), "w") as f:
            json.dump(topic_history, f, indent=4)
        
        if not res_df.empty:
            res_df.to_csv(os.path.join(args.output_dir, f"preds_window_{w}.csv"), index=False)
            
            mse = ((res_df['y_true'] - res_df['y_pred'])**2).mean()
            # R2 formula: 1 - MSE / Var(y_true)
            r2 = 1 - (mse / ((res_df['y_true'] - res_df['y_true'].mean())**2).mean())
            
            logger.info(f"Window {w} OOS R2 (Mag7 Benchmark): {r2:.4f}")
            
            all_metrics.append({
                'window': w,
                'R2': r2,
                'MSE': mse
            })
            
    pd.DataFrame(all_metrics).to_csv(os.path.join(args.output_dir, "comparison_metrics.csv"), index=False)

if __name__ == "__main__":
    main()

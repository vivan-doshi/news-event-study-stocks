import pandas as pd
import numpy as np
import os
import logging
import json
import requests
import time
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load env
load_dotenv('config.env')
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')

def get_gemini_label(keywords, topic_id):
    if not GEMINI_API_KEY:
        logger.warning("No Gemini API Key found. Using fallback label.")
        return f"Topic {topic_id} (Unlabeled)"
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    
    prompt = f"""
    You are a financial news analyst. I will give you a list of keywords from a cluster of news articles.
    Your task is to provide a concise, professional 2-3 word topic label for this cluster.
    Examples: "Earnings Reports", "Geopolitical Risk", "Product Launch".
    
    Keywords: {', '.join(keywords)}
    
    Return ONLY the label. No quotes, no explanation.
    """
    
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            label = result['candidates'][0]['content']['parts'][0]['text'].strip().replace('"', '')
            return label
        else:
            logger.error(f"Gemini API Error: {response.text}")
            return f"Topic {topic_id} (API Error)"
    except Exception as e:
        logger.error(f"Request Error: {e}")
        return f"Topic {topic_id} (Error)"

def run_pipeline():
    # 1. Load Data
    logger.info("Loading embeddings...")
    df = pd.read_parquet('data/processed/mag7_embeddings.parquet')
    
    # Check embedding format
    # It might be an array or list in a column
    # We need a matrix for KMeans
    X = np.stack(df['embedding'].values)
    logger.info(f"Embedding Matrix Shape: {X.shape}")
    
    # 2. KMeans (k=5)
    logger.info("Running K-Means (k=5)...")
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['topic_id_kmeans'] = kmeans.fit_predict(X)
    
    # 3. Keyword Extraction
    logger.info("Extracting Keywords...")
    df['text_for_nlp'] = df['text_for_nlp'].fillna("")
    
    topic_keywords = {}
    
    # TF-IDF per cluster? or fit on all and rank?
    # Better: Fit TF-IDF on the whole corpus, then for each cluster avg the vectors?
    # Or: Class-based TF-IDF (c-TF-IDF)
    
    # Simple approach: Concatenate text for each cluster
    cluster_docs = df.groupby('topic_id_kmeans')['text_for_nlp'].apply(lambda x: " ".join(x)).reset_index()
    
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(cluster_docs['text_for_nlp'])
    feature_names = np.array(tfidf.get_feature_names_out())
    
    # Get top 20 words for each cluster
    label_map = {}
    
    for i, row in cluster_docs.iterrows():
        topic_id = row['topic_id_kmeans']
        
        # Get row from matrix
        row_vec = tfidf_matrix[i].toarray().flatten()
        
        # Top indices
        top_indices = row_vec.argsort()[-20:][::-1]
        top_words = feature_names[top_indices]
        
        topic_keywords[int(topic_id)] = top_words.tolist()
        logger.info(f"Topic {topic_id} Keywords: {top_words[:10]}")
        
        # 4. Labeling
        logger.info(f"Labeling Topic {topic_id} with Gemini...")
        label = get_gemini_label(top_words.tolist(), topic_id)
        label_map[str(topic_id)] = label # Key must be string for JSON
        logger.info(f"Label: {label}")
        time.sleep(1) # Rate limit safety
        
    # 5. Save Results
    # Column mapping for analysis script
    # analysis script uses 'topic_id_kmeans', so we are good.
    # But checking previous file: 'data/processed/mag7_news_with_sentiment_and_topics_labeledV2.parquet'
    
    # We will save to a NEW file to preserve history but use in build_master
    output_path = 'data/processed/mag7_news_5_topics_clean.parquet'
    logger.info(f"Saving labeled data to {output_path}...")
    df.to_parquet(output_path)
    
    # Save Map
    map_path = 'reports/clustering/topic_labels_map_5_clean.json'
    with open(map_path, 'w') as f:
        json.dump(label_map, f, indent=4)
        
    logger.info("Clustering & Labeling Complete.")

if __name__ == "__main__":
    run_pipeline()

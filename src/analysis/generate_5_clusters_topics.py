
# src/analysis/generate_5_clusters_topics.py

import os
import sys
import json
import re
import logging
import numpy as np
import pandas as pd
import nltk
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import google.generativeai as genai
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv("config.env")

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ==========================
# CONSTANTS & STOP WORDS
# ==========================

PLATFORM_STOP = {
    "yahoo", "yahoo finance", "benzinga", "bloomberg",
    "seekingalpha", "motley", "motley fool", "gurufocus",
    "researchandmarkets", "insider monkey", "monkey",
    "shutterstock", "wsj", "wsj com", "mt newswires",
    "investorplace", "zacks", "zacks investment research",
}

GENERIC_FIN_STOP = {
    "stock", "stocks", "share", "shares", "equity", "equities",
    "investor", "investors", "trader", "traders",
    "fund", "funds", "portfolio",
    "market", "markets", "exchange", "index", "indices",
    "wall street", "dow", "nasdaq", "s&p", "s p 500", "sp500",
    "buy", "sell", "hold", "rating", "ratings", "upgrade", "downgrade",
    "bullish", "bearish", "overweight", "underweight", "neutral",
    "price", "prices", "target", "price target",
    "return", "returns", "performance",
    "earnings", "results", "revenue", "sales", "profit", "profits",
    "loss", "losses", "guidance", "forecast",
    "quarter", "quarters", "q1", "q2", "q3", "q4",
    "full year", "fy24", "fy25",
    "today", "yesterday", "tomorrow",
    "week", "weeks", "month", "months", "year", "years",
}

TICKER_STOP = {
    "tesla", "nvidia", "apple", "amazon", "microsoft", "meta",
    "alphabet", "google", "goog", "googl",
    "palantir", "broadcom",
}

STEMMER = SnowballStemmer("english")

def get_custom_stop_words():
    # Combine all custom stop words and stem them
    all_stops = PLATFORM_STOP.union(GENERIC_FIN_STOP).union(TICKER_STOP)
    stemmed_stops = {STEMMER.stem(w) for w in all_stops}
    # Add unstemmed versions too just in case
    return stemmed_stops.union(all_stops)

# ==========================
# FUNCTIONS
# ==========================

def load_data(input_path):
    logger.info(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    if 'embedding' not in df.columns:
        raise ValueError("Column 'embedding' not found.")
    
    # Ensure embeddings are numpy array
    embeddings = np.stack(df['embedding'].values)
    return df, embeddings

def run_kmeans(embeddings, k=5):
    logger.info(f"Running K-Means (K={k})...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels

def preprocess_text_stemmed(text):
    if not isinstance(text, str):
        return ""
    # Basic cleanup
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text) # Keep only letters
    tokens = text.lower().split()
    # Stem
    stemmed = [STEMMER.stem(t) for t in tokens]
    return " ".join(stemmed)

def extract_top_terms(df, labels, top_n=20):
    logger.info("Extracting top terms with stemming and stop-word removal...")
    
    df['cluster'] = labels
    # We use the raw sanitized text column if available, else concat title+content
    # existing scripts used 'text_for_nlp'
    if 'text_for_nlp' not in df.columns:
        logger.warning("'text_for_nlp' not found, creating from title+content")
        df['text_for_nlp'] = df['title'].fillna("") + " " + df['content'].fillna("")

    # Preprocess (Stemming)
    logger.info("Applying stemming to all text (this might take a moment)...")
    df['stemmed_text'] = df['text_for_nlp'].apply(preprocess_text_stemmed)
    
    # Custom Stop Words
    custom_stops = list(get_custom_stop_words()) # sklearn needs list
    
    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words='english', # Standard english stops
        ngram_range=(1, 2), # Uni and Bigrams
        min_df=10,
        max_df=0.5
    )
    
    # We need to add our custom stemmed stops to the vectorizer AFTER or merge them
    # Since 'stop_words' arg takes a list, we can pass 'english' list + ours.
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    final_stops = list(ENGLISH_STOP_WORDS.union(custom_stops))
    vectorizer.stop_words = final_stops
    
    tfidf_matrix = vectorizer.fit_transform(df['stemmed_text'])
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    cluster_terms = {}
    
    for i in range(5):
        idx = df.index[df['cluster'] == i]
        if len(idx) == 0:
            continue
            
        cluster_tfidf = tfidf_matrix[idx]
        mean_tfidf = np.asarray(cluster_tfidf.mean(axis=0)).flatten()
        top_indices = mean_tfidf.argsort()[::-1][:top_n]
        top_words = feature_names[top_indices]
        cluster_terms[i] = top_words.tolist()
        
        logger.info(f"Cluster {i}: {', '.join(top_words)}")
        
    return cluster_terms

def generate_cluster_names(cluster_terms):
    api_key = os.getenv("GOOGLE_API_KEY") 
    # Fallback checking 
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
        
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in env.")
        # Return mock names if API fails
        return {k: f"Cluster {k} (Key Missing)" for k in cluster_terms}
        
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro') 
    
    prompt = f"""
    I have performed clustering on news articles. Here are the top 20 stemmed keywords for each of the 5 clusters.
    
    {json.dumps(cluster_terms, indent=2)}
    
    Please provide a short, descriptive Topic Name (max 3-5 words) for each cluster ID based on these keywords.
    Return the output as a valid JSON object mapping the cluster ID (as a string) to the Topic Name.
    Example: {{ "0": "Market Volatility", "1": "AI Innovation" }}
    """
    
    logger.info("Sending prompt to Gemini...")
    try:
        response = model.generate_content(prompt)
        text_response = response.text
        # Clean potential markdown wrapping
        text_response = text_response.replace("```json", "").replace("```", "").strip()
        names = json.loads(text_response)
        logger.info(f"Gemini Names: {names}")
        return names
    except Exception as e:
        logger.error(f"Gemini API Error: {e}")
        return {str(k): f"Cluster {k}" for k in cluster_terms}

def main():
    INPUT_PATH = "data/processed/mag7_embeddings.parquet"
    OUTPUT_JSON = "reports/clustering/topics_5_clusters.json"
    
    if not os.path.exists(INPUT_PATH):
        logger.error(f"Input {INPUT_PATH} not found.")
        sys.exit(1)
        
    df, embeddings = load_data(INPUT_PATH)
    
    # Run Clustering
    labels = run_kmeans(embeddings, k=5)
    
    # Extract Terms
    cluster_terms = extract_top_terms(df, labels, top_n=20)
    
    # Generate Names
    cluster_names = generate_cluster_names(cluster_terms)
    
    # Combine Results
    final_output = {}
    for cid, terms in cluster_terms.items():
        name = cluster_names.get(str(cid), f"Cluster {cid}")
        final_output[cid] = {
            "name": name,
            "terms": terms
        }
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(final_output, f, indent=4)
    logger.info(f"Saved results to {OUTPUT_JSON}")
    
    # Also print for logs
    print(json.dumps(final_output, indent=4))

if __name__ == "__main__":
    main()

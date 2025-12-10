# ================================================================
# STEP 1: IMPORT LIBRARIES
# ================================================================
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# ================================================================
# STEP 2: LOAD DATA
# ================================================================
df = pd.read_parquet("./eodhd_news_replication/data/raw/news_raw_20251031.parquet")

df = df[df['symbol_query'] == 'AAPL.US']

# Expected columns:
# ['news_article_id', 'published_date', 'news_title', 'news_content', 'stock_ticker_symbol']

# Combine title + content
df['text'] = df['news_title'].fillna('') + ' ' + df['news_content'].fillna('')

# Extract publication year
df['year'] = pd.to_datetime(df['published_date']).dt.year

df.head()

# ================================================================
# STEP 3: CLEAN TEXT
# ================================================================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', ' ', text.lower())
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(clean_text)

# ================================================================
# STEP 4: SELECT ENCODER MODEL BY YEAR
# ================================================================
# Map each year to a model trained BEFORE that year
model_versions = {
    2021: 'sentence-transformers/all-MiniLM-L6-v2',   # Released 2020 → safe for 2021 data
    2022: 'sentence-transformers/sentence-t5-base',   # Released 2021
    2023: 'sentence-transformers/all-distilroberta-v1', # Released 2022
    2024: 'sentence-transformers/multi-qa-mpnet-base-dot-v1', # Released 2023
    2025: 'sentence-transformers/all-mpnet-base-v2'   # Released 2024
}

# Function to encode text using the correct model per year
def encode_by_year(df, model_versions):
    embeddings = []
    for year, group in df.groupby('year'):
        model_name = model_versions.get(year, 'sentence-transformers/all-MiniLM-L6-v2')
        print(f"Encoding {len(group)} articles for {year} using model: {model_name}")
        model = SentenceTransformer(model_name)
        group_embeddings = model.encode(group['clean_text'].tolist(), show_progress_bar=True)
        embeddings.append(pd.DataFrame(group_embeddings, index=group.index))
    all_embeddings = pd.concat(embeddings).sort_index()
    return all_embeddings.values

X_embeddings = encode_by_year(df, model_versions)
print("Embeddings shape:", X_embeddings.shape)

# ================================================================
# STEP 5: BUILD AUTOENCODER
# ================================================================
input_dim = X_embeddings.shape[1]

input_layer = Input(shape=(input_dim,))
encoded = Dense(256, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
latent = Dense(16, activation='relu')(encoded)  # latent "topic" features

decoded = Dense(64, activation='relu')(latent)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, latent)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# ================================================================
# STEP 6: TRAIN AUTOENCODER
# ================================================================
autoencoder.fit(X_embeddings, X_embeddings,
                epochs=10, batch_size=256,
                shuffle=True, verbose=1)

# ================================================================
# STEP 7: EXTRACT LATENT FEATURES
# ================================================================
latent_features = encoder.predict(X_embeddings)
print("Latent feature shape:", latent_features.shape)

# ================================================================
# STEP 8: CLUSTER LATENT FEATURES
# ================================================================
num_clusters = 6  # Broad categories
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(latent_features)

# ================================================================
# STEP 9: INTERPRET CLUSTERS
# ================================================================
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = vectorizer.fit_transform(df['clean_text'])

cluster_keywords = {}
for c in range(num_clusters):
    idx = df[df['cluster'] == c].index
    cluster_mean = np.asarray(X_tfidf[idx].mean(axis=0)).flatten()
    top_indices = cluster_mean.argsort()[-10:][::-1]
    words = [vectorizer.get_feature_names_out()[i] for i in top_indices]
    cluster_keywords[c] = words
    print(f"\nCluster {c} → Top Words: {', '.join(words)}")

# ================================================================
# STEP 10: LABEL CATEGORIES (OPTIONAL MANUAL MAPPING)
# ================================================================
category_map = {
    0: 'Finance',
    1: 'Technology',
    2: 'Product/Launch',
    3: 'Market Movement',
    4: 'M&A',
    5: 'Other'
}

df['broad_category'] = df['cluster'].map(category_map)
df[['news_article_id', 'year', 'broad_category']].head()

# ================================================================
# STEP 11: SAVE RESULTS
# ================================================================
df.to_parquet("autoencoder_topic_output.parquet", index=False)
print("✅ Topic categories saved to 'autoencoder_topic_output.parquet'")

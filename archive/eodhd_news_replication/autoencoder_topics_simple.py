#!/usr/bin/env python
# ================================================================
# SIMPLIFIED AUTOENCODER TOPIC MODELING
# ================================================================

import os
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources silently
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

print("=" * 60)
print("AUTOENCODER TOPIC MODELING FOR NEWS ARTICLES")
print("=" * 60)

# ================================================================
# LOAD DATA
# ================================================================
print("\n1. Loading data...")
df = pd.read_parquet("./eodhd_news_replication/data/raw/news_raw_20251031.parquet")

# Filter for AAPL
df = df[df['symbol_query'] == 'AAPL.US']
print(f"   Found {len(df)} AAPL articles")

# Combine title + content
df['text'] = df['title'].fillna('') + ' ' + df['content'].fillna('')

# Extract publication year
df['year'] = pd.to_datetime(df['published_at']).dt.year
print(f"   Year range: {df['year'].min()} - {df['year'].max()}")

# ================================================================
# CLEAN TEXT
# ================================================================
print("\n2. Cleaning text...")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', ' ', text.lower())
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(clean_text)
print("   Text cleaning complete")

# ================================================================
# CREATE EMBEDDINGS USING TF-IDF
# ================================================================
print("\n3. Creating TF-IDF embeddings...")
from sklearn.feature_extraction.text import TfidfVectorizer

# Use TF-IDF as embeddings (simpler than sentence transformers)
vectorizer = TfidfVectorizer(max_features=500, min_df=5, max_df=0.95)
X_embeddings = vectorizer.fit_transform(df['clean_text']).toarray()
print(f"   Embeddings shape: {X_embeddings.shape}")

# ================================================================
# BUILD SIMPLE AUTOENCODER WITH SKLEARN
# ================================================================
print("\n4. Building autoencoder with PCA...")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_embeddings)

# Use PCA as a simple autoencoder (dimensionality reduction)
n_components = 16  # Latent dimensions
pca = PCA(n_components=n_components)
latent_features = pca.fit_transform(X_scaled)
print(f"   Latent features shape: {latent_features.shape}")
print(f"   Explained variance ratio: {pca.explained_variance_ratio_.sum():.2%}")

# ================================================================
# CLUSTER LATENT FEATURES
# ================================================================
print("\n5. Clustering articles...")
from sklearn.cluster import KMeans

num_clusters = 6
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(latent_features)

# Count articles per cluster
cluster_counts = df['cluster'].value_counts().sort_index()
print("\n   Articles per cluster:")
for cluster_id, count in cluster_counts.items():
    print(f"   - Cluster {cluster_id}: {count} articles")

# ================================================================
# INTERPRET CLUSTERS
# ================================================================
print("\n6. Interpreting clusters...")

# Get top words for each cluster using TF-IDF
feature_names = vectorizer.get_feature_names_out()
cluster_keywords = {}

for c in range(num_clusters):
    # Get articles in this cluster
    cluster_mask = df['cluster'] == c
    cluster_texts = df.loc[cluster_mask, 'clean_text']

    if len(cluster_texts) > 0:
        # Recompute TF-IDF for this cluster
        cluster_tfidf = vectorizer.transform(cluster_texts)
        cluster_mean = cluster_tfidf.mean(axis=0).A1

        # Get top words
        top_indices = cluster_mean.argsort()[-10:][::-1]
        words = [feature_names[i] for i in top_indices]
        cluster_keywords[c] = words

        print(f"\n   Cluster {c} keywords:")
        print(f"   {', '.join(words[:5])}")

# ================================================================
# LABEL CATEGORIES BASED ON KEYWORDS
# ================================================================
print("\n7. Assigning category labels...")

# Auto-assign categories based on keywords
def assign_category(cluster_id, keywords):
    keywords_str = ' '.join(keywords).lower()

    if any(word in keywords_str for word in ['earnings', 'revenue', 'profit', 'financial', 'quarter']):
        return 'Finance/Earnings'
    elif any(word in keywords_str for word in ['iphone', 'ipad', 'mac', 'product', 'device']):
        return 'Product/Hardware'
    elif any(word in keywords_str for word in ['stock', 'market', 'trade', 'share', 'price']):
        return 'Market/Trading'
    elif any(word in keywords_str for word in ['tech', 'software', 'ai', 'app', 'service']):
        return 'Technology/Software'
    elif any(word in keywords_str for word in ['legal', 'lawsuit', 'court', 'regulatory']):
        return 'Legal/Regulatory'
    else:
        return 'General News'

category_map = {}
for c, keywords in cluster_keywords.items():
    category_map[c] = assign_category(c, keywords)
    print(f"   Cluster {c} → {category_map[c]}")

df['broad_category'] = df['cluster'].map(category_map)

# ================================================================
# SHOW SAMPLE RESULTS
# ================================================================
print("\n8. Sample results:")
print("-" * 60)

# Show distribution of categories
category_dist = df['broad_category'].value_counts()
print("\nCategory distribution:")
for category, count in category_dist.items():
    print(f"   {category}: {count} articles ({count/len(df)*100:.1f}%)")

# Show sample articles from each category
print("\nSample articles by category:")
for category in df['broad_category'].unique():
    sample = df[df['broad_category'] == category].head(2)
    print(f"\n   {category}:")
    for _, row in sample.iterrows():
        title = row['title'][:60] + '...' if len(str(row['title'])) > 60 else row['title']
        print(f"   - {title}")

# ================================================================
# SAVE RESULTS
# ================================================================
print("\n9. Saving results...")
output_file = "autoencoder_topic_output.csv"
df.to_csv(output_file, index=False)
print(f"   ✅ Results saved to '{output_file}'")

# Show final summary
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("=" * 60)
print(f"Total articles processed: {len(df)}")
print(f"Number of topics discovered: {num_clusters}")
print(f"Output saved to: {output_file}")
print("\nColumns in output file:")
print("- article_id: Unique article identifier")
print("- published_at: Article publication date")
print("- title: Original article title")
print("- content: Original article content")
print("- year: Publication year")
print("- cluster: Cluster assignment (0-5)")
print("- broad_category: Human-readable category label")
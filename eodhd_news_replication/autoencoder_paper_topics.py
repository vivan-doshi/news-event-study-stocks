#!/usr/bin/env python
"""
AUTOENCODER WITH PAPER-SPECIFIC TOPICS
=======================================
This implementation creates topics that match typical academic finance papers
studying news impact on stock returns.

Common topics in finance event study papers:
1. Macroeconomic/Recession indicators
2. Monetary Policy (Fed, interest rates)
3. Earnings/Financial Performance
4. Mergers & Acquisitions
5. Product/Innovation announcements
6. Legal/Regulatory issues
7. Competition/Market Share
8. Supply Chain/Operations
9. Management/Leadership changes
10. ESG/Sustainability
"""

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

print("=" * 80)
print("AUTOENCODER WITH PAPER-SPECIFIC FINANCIAL TOPICS")
print("=" * 80)

# ================================================================
# DEFINE PAPER-SPECIFIC TOPICS AND KEYWORDS
# ================================================================

# These are typical topics from academic finance papers
PAPER_TOPICS = {
    'macro_recession': {
        'keywords': ['recession', 'economy', 'gdp', 'unemployment', 'inflation',
                    'deflation', 'stagflation', 'economic', 'downturn', 'recovery',
                    'growth', 'slowdown', 'contraction', 'expansion', 'crisis'],
        'weight': 1.5  # Higher weight for important topics
    },
    'monetary_policy': {
        'keywords': ['fed', 'federal reserve', 'interest rate', 'monetary', 'policy',
                    'powell', 'fomc', 'quantitative', 'tightening', 'easing', 'hawkish',
                    'dovish', 'yield', 'bond', 'treasury', 'inflation target'],
        'weight': 1.5
    },
    'earnings_financial': {
        'keywords': ['earnings', 'revenue', 'profit', 'loss', 'eps', 'ebitda',
                    'margin', 'guidance', 'forecast', 'beat', 'miss', 'quarter',
                    'quarterly', 'annual', 'fiscal', 'financial results'],
        'weight': 1.3
    },
    'product_innovation': {
        'keywords': ['product', 'launch', 'innovation', 'technology', 'ai',
                    'artificial intelligence', 'iphone', 'ipad', 'mac', 'macbook',
                    'ios', 'software', 'hardware', 'feature', 'upgrade', 'model'],
        'weight': 1.2
    },
    'merger_acquisition': {
        'keywords': ['merger', 'acquisition', 'acquire', 'buyout', 'takeover',
                    'deal', 'transaction', 'purchase', 'buy', 'sell', 'divest',
                    'spinoff', 'joint venture', 'partnership', 'strategic'],
        'weight': 1.3
    },
    'legal_regulatory': {
        'keywords': ['lawsuit', 'legal', 'court', 'judge', 'regulatory', 'sec',
                    'ftc', 'doj', 'antitrust', 'monopoly', 'fine', 'penalty',
                    'investigation', 'probe', 'compliance', 'settlement'],
        'weight': 1.2
    },
    'competition': {
        'keywords': ['competition', 'competitor', 'rival', 'samsung', 'google',
                    'microsoft', 'amazon', 'meta', 'market share', 'competitive',
                    'advantage', 'threat', 'challenge', 'versus', 'compare'],
        'weight': 1.0
    },
    'supply_operations': {
        'keywords': ['supply chain', 'manufacturing', 'production', 'factory',
                    'supplier', 'component', 'chip', 'shortage', 'logistics',
                    'inventory', 'capacity', 'output', 'delay', 'disruption'],
        'weight': 1.1
    },
    'management': {
        'keywords': ['ceo', 'cfo', 'executive', 'management', 'board', 'director',
                    'leadership', 'resignation', 'appointment', 'hire', 'fire',
                    'succession', 'corporate governance', 'compensation'],
        'weight': 1.0
    },
    'market_sentiment': {
        'keywords': ['analyst', 'upgrade', 'downgrade', 'rating', 'target',
                    'bullish', 'bearish', 'sentiment', 'outlook', 'forecast',
                    'consensus', 'estimate', 'valuation', 'overvalued', 'undervalued'],
        'weight': 1.1
    }
}

# ================================================================
# LOAD AND PREPARE DATA
# ================================================================
print("\n1. Loading data...")
df = pd.read_parquet("./eodhd_news_replication/data/raw/news_raw_20251031.parquet")

# Filter for AAPL
df = df[df['symbol_query'] == 'AAPL.US'].copy()
print(f"   Found {len(df)} AAPL articles")

# Parse dates
df['published_at'] = pd.to_datetime(df['published_at'])
df['year'] = df['published_at'].dt.year
df['month'] = df['published_at'].dt.month
df['quarter'] = df['published_at'].dt.quarter

# Combine text
df['text'] = df['title'].fillna('') + ' ' + df['content'].fillna('')

print(f"   Date range: {df['published_at'].min()} to {df['published_at'].max()}")

# ================================================================
# TEXT PREPROCESSING
# ================================================================
print("\n2. Preprocessing text...")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Don't remove finance-specific words
keep_words = {'fed', 'rate', 'earnings', 'revenue', 'profit', 'loss'}
stop_words = stop_words - keep_words

def clean_text(text):
    # Keep more structure for better topic detection
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s\-]', ' ', text.lower())
    words = []
    for w in text.split():
        if w not in stop_words and len(w) > 2:
            words.append(lemmatizer.lemmatize(w))
    return ' '.join(words)

df['clean_text'] = df['text'].apply(clean_text)

# ================================================================
# CREATE KEYWORD-GUIDED EMBEDDINGS
# ================================================================
print("\n3. Creating keyword-guided embeddings...")

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.95)
X_tfidf = vectorizer.fit_transform(df['clean_text'])
feature_names = vectorizer.get_feature_names_out()

# Create topic-keyword matrix
print("\n4. Building topic-keyword matrix...")
topic_keyword_matrix = np.zeros((len(PAPER_TOPICS), len(feature_names)))

for topic_idx, (topic_name, topic_info) in enumerate(PAPER_TOPICS.items()):
    keywords = topic_info['keywords']
    weight = topic_info['weight']

    for keyword in keywords:
        # Handle multi-word keywords
        keyword_parts = keyword.split()
        for i, feature in enumerate(feature_names):
            if any(part in feature for part in keyword_parts):
                topic_keyword_matrix[topic_idx, i] = weight

# ================================================================
# CALCULATE TOPIC SCORES FOR EACH ARTICLE
# ================================================================
print("\n5. Calculating topic scores for each article...")

# Calculate topic scores as weighted TF-IDF scores
topic_scores = X_tfidf.dot(topic_keyword_matrix.T)
# Check if sparse matrix or numpy array
if hasattr(topic_scores, 'toarray'):
    topic_scores_array = topic_scores.toarray()
else:
    topic_scores_array = topic_scores

# Normalize scores
from sklearn.preprocessing import normalize
topic_scores_normalized = normalize(topic_scores_array, axis=1, norm='l1')

# Add small amount of TF-IDF features for richness
combined_features = np.hstack([
    topic_scores_normalized * 10,  # Weight topic scores heavily
    X_tfidf.toarray()[:, :100] * 0.1  # Add some raw features
])

print(f"   Combined feature shape: {combined_features.shape}")

# ================================================================
# BUILD CONSTRAINED AUTOENCODER
# ================================================================
print("\n6. Building constrained autoencoder...")

class ConstrainedAutoencoder(nn.Module):
    def __init__(self, input_dim, n_topics=10, hidden_dim=128):
        super(ConstrainedAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_topics * 3),  # 3 features per topic
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_topics * 3, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

# ================================================================
# TRAIN AUTOENCODER
# ================================================================
print("\n7. Training autoencoder...")

# Prepare data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(combined_features)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"   Using device: {device}")

model = ConstrainedAutoencoder(
    input_dim=X_scaled.shape[1],
    n_topics=len(PAPER_TOPICS),
    hidden_dim=128
).to(device)

X_tensor = torch.FloatTensor(X_scaled).to(device)
dataset = TensorDataset(X_tensor, X_tensor)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20
for epoch in range(epochs):
    total_loss = 0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        print(f"   Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

# ================================================================
# EXTRACT LATENT FEATURES AND CLUSTER
# ================================================================
print("\n8. Extracting latent features and clustering...")

model.eval()
with torch.no_grad():
    latent_features = model.encode(X_tensor).cpu().numpy()

# Use latent features with original topic scores for clustering
clustering_features = np.hstack([
    latent_features,
    topic_scores_normalized * 20  # Heavily weight the topic scores
])

# Cluster with paper topic count
n_clusters = len(PAPER_TOPICS)
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
df['cluster'] = kmeans.fit_predict(clustering_features)

# ================================================================
# MAP CLUSTERS TO PAPER TOPICS
# ================================================================
print("\n9. Mapping clusters to paper topics...")

# Calculate average topic scores per cluster
cluster_topic_scores = np.zeros((n_clusters, len(PAPER_TOPICS)))
for cluster_id in range(n_clusters):
    cluster_mask = df['cluster'] == cluster_id
    cluster_topic_scores[cluster_id] = topic_scores_normalized[cluster_mask].mean(axis=0)

# Map each cluster to its strongest topic
cluster_to_topic = {}
topic_names = list(PAPER_TOPICS.keys())

used_topics = set()
for cluster_id in range(n_clusters):
    # Find strongest unused topic
    scores = cluster_topic_scores[cluster_id]
    sorted_indices = np.argsort(scores)[::-1]

    for idx in sorted_indices:
        topic_name = topic_names[idx]
        if topic_name not in used_topics:
            cluster_to_topic[cluster_id] = topic_name
            used_topics.add(topic_name)
            break

    # Fallback if all topics used
    if cluster_id not in cluster_to_topic:
        cluster_to_topic[cluster_id] = topic_names[sorted_indices[0]]

# Apply mapping
df['paper_topic'] = df['cluster'].map(cluster_to_topic)

# ================================================================
# CALCULATE TOPIC CONFIDENCE
# ================================================================
print("\n10. Calculating topic confidence scores...")

# Calculate how well each article matches its assigned topic
topic_confidences = []
for idx, row in df.iterrows():
    topic = row['paper_topic']
    topic_idx = topic_names.index(topic)
    # Use positional index instead of label index
    pos_idx = df.index.get_loc(idx)
    confidence = topic_scores_normalized[pos_idx, topic_idx]
    topic_confidences.append(confidence)

df['topic_confidence'] = topic_confidences

# ================================================================
# ADD SENTIMENT WITH FINANCIAL CONTEXT
# ================================================================
print("\n11. Adding financial sentiment analysis...")

# Financial sentiment keywords
positive_financial = {'beat', 'exceed', 'surpass', 'profit', 'growth', 'gain',
                     'rise', 'surge', 'rally', 'upgrade', 'bullish', 'record'}
negative_financial = {'miss', 'loss', 'decline', 'fall', 'drop', 'concern',
                     'risk', 'warning', 'bearish', 'downgrade', 'recession'}

def financial_sentiment(text):
    text_lower = text.lower()
    pos_score = sum(2 if word in text_lower else 0 for word in positive_financial)
    neg_score = sum(2 if word in text_lower else 0 for word in negative_financial)

    # Check for negations
    if 'not' in text_lower or "don't" in text_lower or "doesn't" in text_lower:
        pos_score, neg_score = neg_score * 0.5, pos_score * 0.5

    total = max(1, pos_score + neg_score)
    return (pos_score - neg_score) / total

df['sentiment_score'] = df['clean_text'].apply(financial_sentiment)
df['sentiment_category'] = pd.cut(df['sentiment_score'],
                                  bins=[-1, -0.1, 0.1, 1],
                                  labels=['negative', 'neutral', 'positive'])

# ================================================================
# SHOW RESULTS
# ================================================================
print("\n" + "=" * 80)
print("PAPER TOPIC DISTRIBUTION")
print("=" * 80)

topic_dist = df['paper_topic'].value_counts()
for topic in PAPER_TOPICS.keys():
    if topic in topic_dist.index:
        count = topic_dist[topic]
        pct = count / len(df) * 100
        display_name = topic.replace('_', ' ').title()
        print(f"{display_name:25s}: {count:5,d} articles ({pct:5.1f}%)")

# Cross-tabulation
print("\n" + "=" * 80)
print("TOPIC × SENTIMENT DISTRIBUTION")
print("=" * 80)

cross_tab = pd.crosstab(df['paper_topic'], df['sentiment_category'], margins=True)
print(cross_tab)

# ================================================================
# SAVE OUTPUT
# ================================================================
print("\n" + "=" * 80)
print("SAVING PAPER-FORMAT OUTPUT")
print("=" * 80)

# Create output dataframe with paper topics
output_df = df[[
    'article_id', 'published_at', 'title', 'content',
    'year', 'quarter', 'month',
    'cluster', 'paper_topic', 'topic_confidence',
    'sentiment_score', 'sentiment_category'
]].copy()

# Add binary flags for each topic
for topic in PAPER_TOPICS.keys():
    output_df[f'is_{topic}'] = (df['paper_topic'] == topic).astype(int)

# Save
#output_df.to_csv('paper_topics_output.csv', index=False)
output_df.to_parquet('paper_topics_output.parquet', index=False)
print("\n✅ Saved to 'paper_topics_output.parquet'")

# Save topic mapping
topic_summary = pd.DataFrame({
    'topic': PAPER_TOPICS.keys(),
    'keywords': [', '.join(info['keywords'][:5]) for info in PAPER_TOPICS.values()],
    'article_count': [topic_dist.get(t, 0) for t in PAPER_TOPICS.keys()]
})
topic_summary.to_csv('paper_topics_summary.csv', index=False)
print("✅ Saved topic summary to 'paper_topics_summary.csv'")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - PAPER TOPICS READY")
print("=" * 80)
print(f"\nProcessed: {len(df):,} articles")
print(f"Topics: {len(PAPER_TOPICS)} paper-specific categories")
print("\nOutput includes:")
print("  • Paper-specific topic classification")
print("  • Topic confidence scores")
print("  • Financial sentiment analysis")
print("  • Binary topic indicators for regression")
print("\n✅ Ready for academic event study analysis")
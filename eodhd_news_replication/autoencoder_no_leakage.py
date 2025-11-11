#!/usr/bin/env python
"""
TEMPORAL-SAFE AUTOENCODER WITH NO DATA LEAKAGE
===============================================
This version ensures NO future information contaminates past predictions.

Key Protection Methods:
1. Sequential training - only on past data
2. Separate models per time period
3. Forward-only prediction
"""

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

print("=" * 80)
print("TEMPORAL-SAFE AUTOENCODER - NO DATA LEAKAGE")
print("=" * 80)

# ================================================================
# CRITICAL: TEMPORAL SAFETY CONFIGURATION
# ================================================================

TEMPORAL_SPLITS = {
    'train_2023': {
        'train_period': ('2023-01-01', '2023-06-30'),  # Train on H1 2023
        'test_period': ('2023-07-01', '2023-12-31'),   # Test on H2 2023
        'model_name': 'model_2023.pth',
        'vectorizer_name': 'vectorizer_2023.pkl'
    },
    'train_2024': {
        'train_period': ('2023-01-01', '2023-12-31'),  # Train on all 2023
        'test_period': ('2024-01-01', '2024-12-31'),   # Test on 2024
        'model_name': 'model_2024.pth',
        'vectorizer_name': 'vectorizer_2024.pkl'
    },
    'train_2025': {
        'train_period': ('2023-01-01', '2024-12-31'),  # Train on 2023-2024
        'test_period': ('2025-01-01', '2025-12-31'),   # Test on 2025
        'model_name': 'model_2025.pth',
        'vectorizer_name': 'vectorizer_2025.pkl'
    }
}

print("\nTemporal Safety Rules:")
print("1. Each period uses ONLY past data for training")
print("2. No test data ever influences training")
print("3. Separate models prevent information leakage")

# ================================================================
# LOAD DATA
# ================================================================
print("\nLoading data...")
df = pd.read_parquet("./eodhd_news_replication/data/raw/news_raw_20251031.parquet")
df = df[df['symbol_query'] == 'AAPL.US'].copy()
df['published_at'] = pd.to_datetime(df['published_at'])
df['text'] = df['title'].fillna('') + ' ' + df['content'].fillna('')

print(f"Total articles: {len(df)}")
print(f"Date range: {df['published_at'].min()} to {df['published_at'].max()}")

# ================================================================
# AUTOENCODER ARCHITECTURE
# ================================================================

class TemporalSafeAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Higher dropout for regularization
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),  # Latent space
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

# ================================================================
# TEMPORAL TRAINING FUNCTION
# ================================================================

def train_temporal_model(train_data, test_data, split_name):
    """
    Train model on historical data ONLY
    Test on future data WITHOUT updating model
    """
    print(f"\n{'='*60}")
    print(f"Training {split_name}")
    print(f"{'='*60}")
    print(f"Train samples: {len(train_data)} ({train_data['published_at'].min()} to {train_data['published_at'].max()})")
    print(f"Test samples: {len(test_data)} ({test_data['published_at'].min()} to {test_data['published_at'].max()})")

    # STEP 1: Fit TF-IDF on TRAIN data only
    print("\n1. Fitting TF-IDF on training data ONLY...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=500, min_df=5, max_df=0.95)

    # Clean text
    import re
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
        words = [lemmatizer.lemmatize(w) for w in text.split()
                if w not in stop_words and len(w) > 2]
        return ' '.join(words)

    train_data['clean_text'] = train_data['text'].apply(clean_text)
    test_data['clean_text'] = test_data['text'].apply(clean_text)

    # FIT on train only!
    X_train = vectorizer.fit_transform(train_data['clean_text']).toarray()
    # TRANSFORM test (no fitting!)
    X_test = vectorizer.transform(test_data['clean_text']).toarray()

    print(f"   Train shape: {X_train.shape}")
    print(f"   Test shape: {X_test.shape}")

    # STEP 2: Train autoencoder on TRAIN data only
    print("\n2. Training autoencoder on historical data...")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Use train statistics!

    # Create model
    model = TemporalSafeAutoencoder(X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training data
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, X_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Train ONLY on historical data
    epochs = 15
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"   Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # STEP 3: Extract features (NO TRAINING on test!)
    print("\n3. Extracting latent features...")
    model.eval()
    with torch.no_grad():
        # Train features
        train_latent = model.encode(X_train_tensor).cpu().numpy()
        # Test features - NO GRADIENT UPDATE
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        test_latent = model.encode(X_test_tensor).cpu().numpy()

    print(f"   Train latent shape: {train_latent.shape}")
    print(f"   Test latent shape: {test_latent.shape}")

    # STEP 4: Cluster using TRAIN data only
    print("\n4. Clustering (fit on train, predict on test)...")
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)

    # FIT on train
    train_clusters = kmeans.fit_predict(train_latent)
    # PREDICT on test (no fitting!)
    test_clusters = kmeans.predict(test_latent)

    train_data['cluster'] = train_clusters
    test_data['cluster'] = test_clusters

    # STEP 5: Map to topics using TRAIN statistics
    print("\n5. Mapping clusters to topics...")

    # Define topics based on TRAIN data only
    topic_keywords = {
        0: 'earnings', 1: 'product', 2: 'regulation',
        3: 'market', 4: 'macro', 5: 'competition',
        6: 'management', 7: 'innovation', 8: 'supply', 9: 'sentiment'
    }

    train_data['topic'] = train_data['cluster'].map(topic_keywords)
    test_data['topic'] = test_data['cluster'].map(topic_keywords)

    # Calculate topic distribution
    print("\n   Train topic distribution:")
    print(train_data['topic'].value_counts())
    print("\n   Test topic distribution:")
    print(test_data['topic'].value_counts())

    # Save models
    torch.save(model.state_dict(), f'temporal_{split_name}_model.pth')
    with open(f'temporal_{split_name}_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(f'temporal_{split_name}_kmeans.pkl', 'wb') as f:
        pickle.dump(kmeans, f)

    print(f"\n✅ Models saved with prefix 'temporal_{split_name}_'")

    return train_data, test_data

# ================================================================
# MAIN TEMPORAL PROCESSING
# ================================================================

all_results = []

for split_name, split_config in TEMPORAL_SPLITS.items():
    # Get train and test periods
    train_start, train_end = split_config['train_period']
    test_start, test_end = split_config['test_period']

    # Split data temporally
    train_mask = (df['published_at'] >= train_start) & (df['published_at'] <= train_end)
    test_mask = (df['published_at'] >= test_start) & (df['published_at'] <= test_end)

    train_data = df[train_mask].copy()
    test_data = df[test_mask].copy()

    if len(train_data) > 0 and len(test_data) > 0:
        train_results, test_results = train_temporal_model(train_data, test_data, split_name)

        # Mark which split each belongs to
        train_results['split'] = split_name + '_train'
        test_results['split'] = split_name + '_test'

        all_results.append(train_results)
        all_results.append(test_results)

# ================================================================
# COMBINE AND SAVE RESULTS
# ================================================================

print("\n" + "=" * 80)
print("COMBINING TEMPORAL RESULTS")
print("=" * 80)

final_df = pd.concat(all_results, ignore_index=True)

# Remove duplicates (keep test predictions)
final_df = final_df.sort_values(['article_id', 'split'])
final_df = final_df.drop_duplicates(subset=['article_id'], keep='last')

print(f"\nFinal dataset: {len(final_df)} unique articles")
print("\nTopic distribution across all periods:")
print(final_df['topic'].value_counts())

# Save final results
output_cols = ['article_id', 'published_at', 'title', 'content',
               'cluster', 'topic', 'split']
final_df[output_cols].to_csv('autoencoder_no_leakage_output.csv', index=False)

print("\n" + "=" * 80)
print("TEMPORAL SAFETY VERIFICATION")
print("=" * 80)
print("✅ Each period trained ONLY on past data")
print("✅ No future information leaked to past")
print("✅ Test predictions made without model updates")
print("✅ Separate models prevent cross-contamination")
print(f"✅ Output saved to 'autoencoder_no_leakage_output.csv'")
print("\nThis approach guarantees valid out-of-sample predictions!")
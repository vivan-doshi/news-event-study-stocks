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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

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

print(f"Loaded {len(df)} articles")
print(df.head())

# ================================================================
# STEP 3: CLEAN TEXT
# ================================================================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', ' ', text.lower())
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)

print("Cleaning text...")
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

print("Encoding text with sentence transformers...")
X_embeddings = encode_by_year(df, model_versions)
print("Embeddings shape:", X_embeddings.shape)

# ================================================================
# STEP 5: BUILD PYTORCH AUTOENCODER
# ================================================================
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),  # Latent space
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_latent(self, x):
        return self.encoder(x)

# ================================================================
# STEP 6: TRAIN AUTOENCODER
# ================================================================
print("Training autoencoder...")
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

input_dim = X_embeddings.shape[1]
autoencoder = Autoencoder(input_dim).to(device)

# Prepare data
X_tensor = torch.FloatTensor(X_embeddings).to(device)
dataset = TensorDataset(X_tensor, X_tensor)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# Training
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters())

epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = autoencoder(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')

# ================================================================
# STEP 7: EXTRACT LATENT FEATURES
# ================================================================
print("Extracting latent features...")
autoencoder.eval()
with torch.no_grad():
    latent_features = autoencoder.get_latent(X_tensor).cpu().numpy()
print("Latent feature shape:", latent_features.shape)

# ================================================================
# STEP 8: CLUSTER LATENT FEATURES
# ================================================================
print("Clustering latent features...")
num_clusters = 6  # Broad categories
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(latent_features)

# ================================================================
# STEP 9: INTERPRET CLUSTERS
# ================================================================
from sklearn.feature_extraction.text import TfidfVectorizer

print("Interpreting clusters...")
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
print("\nSample results:")
print(df[['news_article_id', 'year', 'broad_category']].head())

# ================================================================
# STEP 11: SAVE RESULTS
# ================================================================
df.to_csv("autoencoder_topic_output.csv", index=False)
print("\n✅ Topic categories saved to 'autoencoder_topic_output.csv'")
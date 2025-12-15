import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

df = pd.read_parquet('data/processed/mag7_news_5_topics_clean.parquet')
df['text_for_nlp'] = df['text_for_nlp'].fillna("")

cluster_docs = df.groupby('topic_id_kmeans')['text_for_nlp'].apply(lambda x: " ".join(x)).reset_index()

tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(cluster_docs['text_for_nlp'])
feature_names = np.array(tfidf.get_feature_names_out())

print("=== TOPIC KEYWORDS ===")
for i, row in cluster_docs.iterrows():
    topic_id = row['topic_id_kmeans']
    row_vec = tfidf_matrix[i].toarray().flatten()
    top_indices = row_vec.argsort()[-20:][::-1]
    print(f"Topic {topic_id}: {feature_names[top_indices]}")

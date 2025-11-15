from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

df_emb = pd.read_parquet("/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/apple_news_with_features_chronobert.parquet")

def describe_topics_with_words(df_emb,
                               text_col="text_for_nlp",
                               topic_col="topic_id_kmeans",
                               max_features=20000,
                               top_n=20):
    """
    For each topic_id, print the top_n words/phrases (by mean TF-IDF).
    Returns: dict[int, list[str]] mapping topic_id -> list of top terms.
    """
    # 1) Build TF-IDF over all texts
    texts = df_emb[text_col].fillna("").tolist()
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",          # you can add your own financial stopwords
        ngram_range=(1, 2),            # unigrams + bigrams
        min_df=5,                      # ignore very rare terms
        max_df=0.5                     # ignore very common terms
    )
    X_tfidf = vectorizer.fit_transform(texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    topic_terms = {}
    unique_topics = sorted(df_emb[topic_col].unique())

    for t in unique_topics:
        idx = df_emb.index[df_emb[topic_col] == t]
        if len(idx) == 0:
            continue

        # rows of X_tfidf corresponding to this topic
        X_topic = X_tfidf[idx, :]

        # mean TF-IDF for each term within the cluster
        mean_tfidf = X_topic.mean(axis=0).A1  # (n_terms,)

        # top_n term indices
        top_idx = np.argsort(mean_tfidf)[::-1][:top_n]
        top_terms = feature_names[top_idx].tolist()
        topic_terms[t] = top_terms

        print(f"\n=== Topic {t} ===")
        print(", ".join(top_terms))

    return topic_terms


# Example usage after you build df_emb:
topic_terms = describe_topics_with_words(df_emb,
                                         text_col="text_for_nlp",
                                         topic_col="topic_id_kmeans",
                                         max_features=20000,
                                         top_n=20)

print("===============================")
print(topic_terms)
print("===============================")

#df_emb.to_parquet("apple_news_with_topics_describedv2.parquet", index=False)
"""
topic_listing.py

Step 3:
    - Load Parquet with KMeans topics (output of chronobert_kmeans.py,
      or the sentiment-augmented file from sentiment_finbert.py)
    - Build TF-IDF on text_for_nlp (already cleaned upstream)
    - Apply structural garbage filtering + chi2 feature selection
    - Compute top words per topic_id_kmeans
    - Build automatic labels per topic
    - Save:
        - topic_terms_kmeans.json
        - topic_labels_kmeans.json
        - news_with_topics_labeled.parquet (with topic_label_auto)
"""

# ==========================
# CONFIG
# ==========================

INPUT_PARQUET_PATH = "/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/mag7_news_with_topics_sentiment.parquet"  # or "news_with_topics_sentiment.parquet"
OUTPUT_PARQUET_PATH = "mag7_news_with_sentiment_and_topics_labeled.parquet"
TOPIC_TERMS_JSON_PATH = "topic_terms_kmeans.json"
TOPIC_LABELS_JSON_PATH = "topic_labels_kmeans.json"

MAX_FEATURES_TFIDF = 20000
TOP_N_WORDS = 20
CHI2_PERCENTILE = 50.0  # keep upper half of terms by chi2 score


# ==========================
# IMPORTS
# ==========================

import json
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2


# ==========================
# GENERIC TERM FILTERS
# ==========================

PLATFORM_STOP = {
    "yahoo", "yahoo finance", "benzinga", "bloomberg",
    "seekingalpha", "motley", "motley fool", "gurufocus",
    "researchandmarkets", "insider monkey", "monkey",
    "shutterstock", "wsj", "wsj com", "mt newswires",
}

SHORT_WHITELIST = {"ai", "fx", "eu", "us", "uk", "fed"}


def is_garbage_term(term: str) -> bool:
    """
    Generic structural/regex rules to filter out junk terms
    when building topic descriptors.
    """
    t = term.lower().strip()

    # URL / domain junk
    if "http" in t or "www" in t or ".com" in t:
        return True

    # Pure digits (years, round numbers)
    if t.isdigit():
        return True

    # Very short tokens that aren't in a whitelist
    if len(t) <= 2 and t not in SHORT_WHITELIST:
        return True

    # Obvious non-topical boilerplate
    if t in {
        "reading", "continue", "continue reading", "view", "comments",
        "view comments", "subscription", "upgrade", "premium", "article",
        "articles", "sign", "plan", "required", "access", "report",
        "reports", "today", "live", "quote", "quotes", "details",
        "start", "news", "video",
    }:
        return True

    # Publisher/platform names
    if t in PLATFORM_STOP:
        return True

    return False


# ==========================
# TOPIC DESCRIPTION
# ==========================

def describe_topics_with_words(df_emb,
                               text_col: str = "text_for_nlp",
                               topic_col: str = "topic_id_kmeans",
                               max_features: int = 20000,
                               top_n: int = 20,
                               chi2_percentile: float = 50.0):
    """
    For each topic_id, return:
      - top_n words/phrases (by mean TF-IDF) AFTER
        * structural garbage filtering
        * chi2-based filtering vs topic labels
      - an automatic short topic name (based on top 3 words)

    Returns:
        topic_terms: dict[int, list[str]]
        topic_labels: dict[int, str]
    """
    texts = df_emb[text_col].fillna("").tolist()
    y = df_emb[topic_col].values

    # 1) TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.5,
    )
    X_tfidf = vectorizer.fit_transform(texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    # 2) Structural filter
    basic_mask = np.array([not is_garbage_term(t) for t in feature_names])
    X_tfidf_basic = X_tfidf[:, basic_mask]
    feature_names_basic = feature_names[basic_mask]

    # 3) chi2 filter vs topic labels
    chi2_scores, pvals = chi2(X_tfidf_basic, y)
    thresh = np.percentile(chi2_scores, chi2_percentile)
    chi2_mask = chi2_scores >= thresh

    X_tfidf_filt = X_tfidf_basic[:, chi2_mask]
    feature_names_filt = feature_names_basic[chi2_mask]

    topic_terms: dict[int, list[str]] = {}
    topic_labels: dict[int, str] = {}

    unique_topics = sorted(df_emb[topic_col].unique())

    for t in unique_topics:
        idx = df_emb.index[df_emb[topic_col] == t]
        if len(idx) == 0:
            continue

        X_topic = X_tfidf_filt[idx, :]
        mean_tfidf = X_topic.mean(axis=0).A1  # (n_terms,)

        if np.all(mean_tfidf == 0):
            continue

        top_idx = np.argsort(mean_tfidf)[::-1][:top_n]
        top_terms = feature_names_filt[top_idx].tolist()
        topic_terms[int(t)] = top_terms

        # Auto label from top 3 unigrams if possible
        label_terms = [w for w in top_terms if " " not in w][:3]
        if len(label_terms) < 3:
            label_terms = top_terms[:3]
        topic_labels[int(t)] = " / ".join(label_terms)

        print(f"\n=== Topic {t} ===")
        print("Top terms:", ", ".join(top_terms))
        print("Auto label:", topic_labels[int(t)])

    return topic_terms, topic_labels


# ==========================
# MAIN
# ==========================

def main():
    print(f"[TOPICS] Loading news with topics from: {INPUT_PARQUET_PATH}")
    df = pd.read_parquet(INPUT_PARQUET_PATH)

    for col in ["text_for_nlp", "topic_id_kmeans"]:
        if col not in df.columns:
            raise ValueError(f"{col} column not found. Run chronobert_kmeans.py first.")

    print("[TOPICS] Building topic -> top words and labels...")
    topic_terms, topic_labels = describe_topics_with_words(
        df,
        text_col="text_for_nlp",
        topic_col="topic_id_kmeans",
        max_features=MAX_FEATURES_TFIDF,
        top_n=TOP_N_WORDS,
        chi2_percentile=CHI2_PERCENTILE,
    )

    df["topic_label_auto"] = df["topic_id_kmeans"].map(topic_labels)

    # Save JSONs
    try:
        with open(TOPIC_TERMS_JSON_PATH, "w") as f:
            json.dump(topic_terms, f, indent=2)
        with open(TOPIC_LABELS_JSON_PATH, "w") as f:
            json.dump(topic_labels, f, indent=2)
        print(f"[TOPICS] Saved topic_terms to {TOPIC_TERMS_JSON_PATH}")
        print(f"[TOPICS] Saved topic_labels to {TOPIC_LABELS_JSON_PATH}")
    except Exception as e:
        print(f"[TOPICS] Could not save topic dictionaries: {e}")

    # Save labeled Parquet
    print(f"[TOPICS] Saving news with auto topic labels to: {OUTPUT_PARQUET_PATH}")
    df.to_parquet(OUTPUT_PARQUET_PATH, index=False)

    print("[TOPICS] Done. Columns now include (sample):")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()

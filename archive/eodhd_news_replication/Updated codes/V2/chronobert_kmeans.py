"""
chronobert_kmeans.py

Step 1:
    - Load raw news
    - Filter to a symbol (e.g. AAPL.US)
    - Clean text (remove URLs / boilerplate / platform names / generic finance junk)
    - Mask company/ticker names to neutral token for better clustering
    - Build ChronoBERT embeddings (chronology-aware)
    - Run KMeans clustering on embeddings
    - Save Parquet with:
        - chrono_model
        - text_for_nlp (CLEANED)
        - topic_id_kmeans
"""

# ==========================
# CONFIG
# ==========================

NEWS_PARQUET_PATH = "/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/eodhd_news_replication/data/raw/news_raw_20251031.parquet"
OUTPUT_PARQUET_PATH = "mag7_news_with_topicsV2.parquet"
SYMBOL_FILTER = ["AAPL.US", "MSFT.US", "AMZN.US", "GOOGL.US", "TSLA.US", "META.US", "NVDA.US"]

N_TOPICS = 50
MAX_LENGTH = 256
BATCH_SIZE_EMB = 32


# ==========================
# IMPORTS
# ==========================

import re
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans


# ==========================
# DEVICE SELECTION
# ==========================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"[KMEANS] Using device: {device}")


# ==========================
# EXTRA DOMAIN FILTERS FOR CLUSTERING
# ==========================

# Very generic finance words that carry almost no cross-article signal
GENERIC_FIN_STOP = {
    # trading / generic finance
    "stock", "stocks", "share", "shares", "equity", "equities",
    "investor", "investors", "trader", "traders",
    "fund", "funds", "portfolio",
    "market", "markets", "exchange", "index", "indices",
    "wall street", "dow", "nasdaq", "s&p", "s p 500", "sp500",

    # generic action words
    "buy", "sell", "hold", "rating", "upgrade", "downgrade",
    "bullish", "bearish", "overweight", "underweight", "neutral",
    "price", "prices", "target", "price target",

    # earnings boilerplate
    "earnings", "results", "revenue", "sales", "profit", "profits",
    "loss", "losses", "guidance", "forecast",
    "quarter", "quarters", "q1", "q2", "q3", "q4",
    "full year", "fy24", "fy25",
}

# Company names / tickers you don't want to dominate clustering semantics
COMPANY_TOKENS = {
    # tickers
    "aapl", "msft", "amzn", "googl", "goog", "tsla", "meta", "nvda",
    # company names
    "apple", "microsoft", "amazon", "alphabet", "google", "tesla",
    "meta platforms", "meta platform", "nvidia",
}

# Important people you explicitly want to preserve everywhere
PERSON_WHITELIST = {"trump", "musk"}


# ==========================
# TEXT CLEANING (before embeddings)
# ==========================

def clean_text(text: str) -> str:
    """
    Clean raw article text before feeding into ChronoBERT:
      - remove URLs and bare domains
      - remove obvious UI / boilerplate phrases
      - remove some platform names
      - remove/normalize generic finance & company noise so clusters
        focus more on *narratives* than on "stock / ticker" chatter.
    """
    if not isinstance(text, str):
        return ""

    t = text

    # Remove URLs
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"www\.[^\s]+", " ", t)

    # Remove bare domains like something.com / something.co.uk
    t = re.sub(r"\b[\w\-]+\.(com|net|org|io|co|us|uk|ca)\b", " ", t, flags=re.IGNORECASE)

    # Remove common boilerplate / gating phrases (case-insensitive)
    boilerplate_phrases = [
        "continue reading",
        "view comments",
        "sign up",
        "subscription required",
        "upgrade your subscription",
        "all rights reserved",
        "this article was originally published",
        "click here",
    ]
    for phrase in boilerplate_phrases:
        t = re.sub(re.escape(phrase), " ", t, flags=re.IGNORECASE)

    # Remove some platform names that are not semantically interesting
    platforms = [
        "Yahoo Finance",
        "Benzinga",
        "Bloomberg",
        "The Motley Fool",
        "Motley Fool",
        "Seeking Alpha",
        "MarketWatch",
        "WSJ",
    ]
    for prov in platforms:
        t = re.sub(rf"\b{re.escape(prov)}\b", " ", t, flags=re.IGNORECASE)

    # --- DOMAIN-SPECIFIC NORMALIZATION FOR CLUSTERING ---

    # 1) Remove generic finance boilerplate words
    if GENERIC_FIN_STOP:
        pattern_fin = r"\b(" + "|".join(re.escape(w) for w in GENERIC_FIN_STOP) + r")\b"
        t = re.sub(pattern_fin, " ", t, flags=re.IGNORECASE)

    # 2) Replace company/ticker tokens with a neutral placeholder "company"
    for comp in COMPANY_TOKENS:
        # don't accidentally mask whitelisted people
        if comp in PERSON_WHITELIST:
            continue
        t = re.sub(rf"\b{re.escape(comp)}\b", " company ", t, flags=re.IGNORECASE)

    # 3) Normalize explicit ticker-like patterns (e.g. AAPL.US, TSLA.US)
    t = re.sub(r"\b[A-Z]{1,5}\.(US|NY|AX|TO|NS|L)\b", " company ", t)

    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()

    return t


# ==========================
# ChronoBERT MODEL SELECTION
# ==========================

def get_chronobert_name_for_date(dt: pd.Timestamp) -> str:
    """
    Map a publication date to the appropriate ChronoBERT checkpoint:
    use previous year-end model (t-1).
    """
    year = dt.year
    checkpoint_year = min(max(year - 1, 1999), 2024)
    return f"manelalab/chrono-bert-v1-{checkpoint_year}1231"


class ChronoBERTEncoder:
    """
    Cache-aware ChronoBERT encoder:
        - selects checkpoint per article date
        - embeds text via mean pooling over last hidden states
    """

    def __init__(self, max_length: int = 256, batch_size: int = 16):
        self.max_length = max_length
        self.batch_size = batch_size
        self._cache = {}  # model_name -> (tokenizer, model)

    def _load_model(self, model_name: str):
        if model_name not in self._cache:
            print(f"[KMEANS] Loading ChronoBERT model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(
                model_name,
                attn_implementation="sdpa",  # avoids flash-attn dependency
            ).to(device)
            model.eval()
            self._cache[model_name] = (tokenizer, model)
        return self._cache[model_name]

    def _encode_texts_with_model(self, texts, model_name: str) -> np.ndarray:
        tokenizer, model = self._load_model(model_name)
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(device)

                outputs = model(**inputs)
                last_hidden = outputs.last_hidden_state  # (B, L, H)
                attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (B, L, 1)

                masked = last_hidden * attention_mask
                summed = masked.sum(dim=1)
                counts = attention_mask.sum(dim=1).clamp(min=1)
                pooled = (summed / counts).cpu().numpy()  # (B, H)

                all_embeddings.append(pooled)

        return np.vstack(all_embeddings)

    def encode_dataframe(self, df: pd.DataFrame, text_col: str | None = None):
        """
        df must contain ['published_at', 'title', 'content'].
        Returns:
            embeddings: np.ndarray of shape (n_articles, hidden_dim)
            df_out: df with 'chrono_model' and 'text_for_nlp'
        """
        df = df.copy()
        df = df.reset_index(drop=True)
        df["published_at"] = pd.to_datetime(df["published_at"])

        df["chrono_model"] = df["published_at"].apply(get_chronobert_name_for_date)

        if text_col is None:
            # Build raw text then clean it once
            raw_text = (
                df["title"].fillna("") + ". " + df["content"].fillna("")
            ).astype(str)
            df["text_for_nlp"] = raw_text.apply(clean_text)
            text_col = "text_for_nlp"

        texts = df[text_col].tolist()
        embeddings = None

        for model_name, idx in df.groupby("chrono_model").groups.items():
            idx = list(idx)
            subset_texts = [texts[i] for i in idx]

            print(f"[KMEANS] Encoding {len(idx)} articles with {model_name}...")
            subset_emb = self._encode_texts_with_model(subset_texts, model_name)

            if embeddings is None:
                emb_dim = subset_emb.shape[1]
                embeddings = np.zeros((len(df), emb_dim), dtype=np.float32)

            embeddings[idx] = subset_emb

        return embeddings, df


# ==========================
# TOPIC MODEL (KMeans on embeddings)
# ==========================

def compute_topics_kmeans(embeddings: np.ndarray,
                          n_topics: int = 50,
                          random_state: int = 42):
    print(f"[KMEANS] Running KMeans with K={n_topics} on embeddings of shape {embeddings.shape}...")
    kmeans = KMeans(
        n_clusters=n_topics,
        random_state=random_state,
        n_init=10,
    )
    topic_ids = kmeans.fit_predict(embeddings)
    return topic_ids, kmeans


# ==========================
# MAIN
# ==========================

def main():
    print(f"[KMEANS] Loading news from: {NEWS_PARQUET_PATH}")
    df = pd.read_parquet(NEWS_PARQUET_PATH)
    print(f"[KMEANS] Filtering to symbols: {SYMBOL_FILTER}")
    df = df[df["symbol_query"].isin(SYMBOL_FILTER)].copy()
    print(f"[KMEANS] Found {len(df)} articles for symbols: {SYMBOL_FILTER}")

    required_cols = ["article_id", "published_at", "title", "content"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in news data: {missing}")

    encoder = ChronoBERTEncoder(max_length=MAX_LENGTH, batch_size=BATCH_SIZE_EMB)
    embeddings, df_emb = encoder.encode_dataframe(df)

    topic_ids, kmeans_model = compute_topics_kmeans(
        embeddings,
        n_topics=N_TOPICS,
        random_state=42,
    )
    df_emb["topic_id_kmeans"] = topic_ids

    print(f"[KMEANS] Saving news with topics to: {OUTPUT_PARQUET_PATH}")
    df_emb.to_parquet(OUTPUT_PARQUET_PATH, index=False)

    print("[KMEANS] Done. Columns include (sample):")
    print(df_emb.columns.tolist())


if __name__ == "__main__":
    main()

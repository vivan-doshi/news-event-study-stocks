"""
End-to-end ChronoBERT feature pipeline:
    - Chronology-aware embeddings
    - KMeans topics
    - ChronoBERT-based sentiment scores

Expected input:
    A CSV (or any DataFrame) with columns:
        ['article_id', 'published_at', 'title', 'content',
         'url', 'source', 'symbols', 'tags',
         'sentiment', 'symbol_query', 'fetch_date']

    We IGNORE the vendor 'sentiment' column and create our own.

How to use:
    1) Adjust the CONFIG section (paths, n_topics, sentiment model dir).
    2) Run:
        python chronobert_features.py

    It will:
        - load your news
        - embed with ChronoBERT
        - create KMeans topics
        - attach sentiment scores from a fine-tuned ChronoBERT classifier
        - save a Parquet file with features.
"""

# ==========================
# CONFIG
# ==========================

NEWS_PARQUET_PATH = "/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/eodhd_news_replication/data/raw/news_raw_20251031.parquet"          # path to your news data
OUTPUT_PARQUET_PATH = "news_with_features_chronobert.parquet"

# Number of KMeans topics
N_TOPICS = 50

# Path to a fine-tuned ChronoBERT sentiment model
# (run training once using the train_chronobert_sentiment() function below,
# or point this to any compatible HF path).
SENTIMENT_MODEL_DIR = "ProsusAI/finbert"

# Maximum sequence length and batch sizes
MAX_LENGTH = 256
BATCH_SIZE_EMB = 16     # embeddings
BATCH_SIZE_SENT = 32    # sentiment scoring


# ==========================
# IMPORTS
# ==========================

import os
import numpy as np
import pandas as pd

import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
)

from sklearn.cluster import KMeans
import torch.nn.functional as F


# ==========================
# DEVICE SELECTION (M3-friendly)
# ==========================

if torch.backends.mps.is_available():
    device = torch.device("mps")      # Apple Silicon GPU (M1/M2/M3)
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# ==========================
# ChronoBERT MODEL SELECTION
# ==========================

def get_chronobert_name_for_date(dt: pd.Timestamp) -> str:
    """
    Map a publication date to the appropriate ChronoBERT checkpoint:
    we use the previous year-end model (t-1).

    Example:
        2021-xx-xx -> manelalab/chrono-bert-v1-20201231

    ChronoBERT exists from 1999-12-31 to 2024-12-31.
    """
    year = dt.year
    checkpoint_year = min(max(year - 1, 1999), 2024)
    return f"manelalab/chrono-bert-v1-{checkpoint_year}1231"


class ChronoBERTEncoder:
    """
    Cache-aware ChronoBERT encoder that:
        - selects checkpoint per article date
        - embeds text via mean pooling over last hidden states
    """

    def __init__(self, max_length: int = 256, batch_size: int = 16):
        self.max_length = max_length
        self.batch_size = batch_size
        self._cache = {}  # model_name -> (tokenizer, model)

    def _load_model(self, model_name: str):
        if model_name not in self._cache:
            print(f"Loading ChronoBERT model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)
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
                    return_tensors="pt"
                ).to(device)

                outputs = model(**inputs)
                last_hidden = outputs.last_hidden_state  # (B, L, H)
                attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (B, L, 1)

                # mean pooling over non-padded tokens
                masked = last_hidden * attention_mask
                summed = masked.sum(dim=1)
                counts = attention_mask.sum(dim=1).clamp(min=1)
                pooled = (summed / counts).cpu().numpy()  # (B, H)

                all_embeddings.append(pooled)

        return np.vstack(all_embeddings)

    def encode_dataframe(self, df: pd.DataFrame, text_col: str | None = None):
        """
        df: must contain ['published_at', 'title', 'content']
        Returns:
            embeddings: np.ndarray of shape (n_articles, hidden_dim)
            df_out: df with extra columns: 'chrono_model', 'text_for_nlp'
        """
        df = df.copy()
        df = df.reset_index(drop=True)
        df["published_at"] = pd.to_datetime(df["published_at"])

        # choose ChronoBERT checkpoint for each row
        df["chrono_model"] = df["published_at"].apply(get_chronobert_name_for_date)

        # build text field if not provided: title + content
        if text_col is None:
            df["text_for_nlp"] = (
                df["title"].fillna("") + ". " + df["content"].fillna("")
            ).str.strip()
            text_col = "text_for_nlp"

        texts = df[text_col].tolist()

        embeddings = None

        # group by model to reuse loaded models
        for model_name, idx in df.groupby("chrono_model").groups.items():
            idx = list(idx)
            subset_texts = [texts[i] for i in idx]

            print(f"Encoding {len(idx)} articles with {model_name}...")
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
    """
    Cluster ChronoBERT embeddings into K topics.
    Returns:
        topic_ids: np.ndarray of shape (n_articles,)
        kmeans_model: fitted KMeans object
    """
    print(f"Running KMeans with K={n_topics} on embeddings of shape {embeddings.shape}...")
    kmeans = KMeans(
        n_clusters=n_topics,
        random_state=random_state,
        n_init=10
    )
    topic_ids = kmeans.fit_predict(embeddings)
    return topic_ids, kmeans


# ==========================
# SENTIMENT: TRAINING (OPTIONAL)
# ==========================

def train_chronobert_sentiment(sent_train_df: pd.DataFrame,
                               output_dir: str = "./chrono-sentiment-v1",
                               base_model: str = "manelalab/chrono-bert-v1-19991231",
                               num_labels: int = 3,
                               num_epochs: int = 3,
                               lr: float = 2e-5,
                               batch_size: int = 16):
    """
    One-time fine-tuning of ChronoBERT for sentiment.

    sent_train_df must have:
        'text'  : string
        'label' : int in {0, 1, 2} -> e.g. 0=neg, 1=neu, 2=pos

    Saves a classifier to output_dir that you can reuse later.
    """
    from transformers import TrainingArguments, Trainer
    from datasets import Dataset

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=num_labels
    )

    hf_ds = Dataset.from_pandas(
        sent_train_df[["text", "label"]].rename(columns={"label": "labels"})
    )

    def preprocess(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH
        )

    hf_ds = hf_ds.map(preprocess, batched=True)
    hf_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_ds,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Sentiment model saved to: {output_dir}")
    return output_dir


# ==========================
# SENTIMENT: INFERENCE
# ==========================

class ChronoBERTSentimentScorer:
    """
    Wraps a sentiment classifier.

    Assumes label order:
        0 = negative
        1 = neutral
        2 = positive

    Works with:
        - a local fine-tuned directory, OR
        - a Hugging Face hub model name (e.g. "ProsusAI/finbert").
    """

    def __init__(self, model_dir: str):
        # REMOVE the os.path.isdir(...) check block above

        print(f"Loading sentiment model from: {model_dir}")
        # ADD: load tokenizer and model directly from HF/local
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        self.model.eval()

        # (optional, just prints mapping so you can verify labels)
        try:  # ADD (optional)
            config = self.model.config
            print("Sentiment model id2label:", getattr(config, "id2label", None))
        except Exception:
            pass

    def score(self, texts, batch_size: int = 32, max_length: int = 256):
        """
        Returns:
            probs: np.ndarray of shape (n_texts, num_labels)
        """
        all_probs = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                ).to(device)

                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)


# ==========================
# MAIN PIPELINE
# ==========================

def main():
    # ---- 1. Load news ----
    print(f"Loading news from: {NEWS_PARQUET_PATH}")
    df = pd.read_parquet(NEWS_PARQUET_PATH)
    df = df[df['symbol_query'] == 'AAPL.US'].copy()
    print(f"   Found {len(df)} AAPL articles")

    # safeguard: ensure required columns exist
    required_cols = ["article_id", "published_at", "title", "content"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in news data: {missing}")

    # ---- 2. ChronoBERT embeddings ----
    encoder = ChronoBERTEncoder(max_length=MAX_LENGTH, batch_size=BATCH_SIZE_EMB)
    embeddings, df_emb = encoder.encode_dataframe(df)

    # ---- 3. Topic IDs via KMeans ----
    topic_ids, kmeans_model = compute_topics_kmeans(
        embeddings,
        n_topics=N_TOPICS,
        random_state=42
    )
    df_emb["topic_id_kmeans"] = topic_ids

    # ---- 4. ChronoBERT sentiment (using a fine-tuned classifier) ----
    # IMPORTANT:
    #   - Either pre-train the model with train_chronobert_sentiment()
    #     using your own labeled data, or point SENTIMENT_MODEL_DIR
    #     to a compatible pre-trained checkpoint.
    sentiment_scorer = ChronoBERTSentimentScorer(SENTIMENT_MODEL_DIR)

    texts_for_sent = df_emb["text_for_nlp"].tolist()
    print("Scoring sentiment...")
    sentiment_probs = sentiment_scorer.score(
        texts_for_sent,
        batch_size=BATCH_SIZE_SENT,
        max_length=MAX_LENGTH
    )

    # Attach sentiment probabilities
    df_emb[["sent_neg", "sent_neu", "sent_pos"]] = sentiment_probs

    # Continuous index: pos minus neg
    df_emb["sentiment_finbert"] = df_emb["sent_pos"] - df_emb["sent_neg"]

    # ---- 5. Clean up vendor sentiment name (optional) ----
    if "sentiment" in df_emb.columns:
        df_emb = df_emb.rename(columns={"sentiment": "sentiment_vendor"})

    # ---- 6. Save to Parquet ----
    print(f"Saving enriched news with features to: {OUTPUT_PARQUET_PATH}")
    df_emb.to_parquet(OUTPUT_PARQUET_PATH, index=False)

    print("Done. Columns now include:")
    print([
        "chrono_model",
        "text_for_nlp",
        "topic_id_kmeans",
        "sent_neg",
        "sent_neu",
        "sent_pos",
        "sentiment_finbert",
    ])


if __name__ == "__main__":
    main()

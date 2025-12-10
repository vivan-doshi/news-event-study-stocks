"""
sentiment_finbert.py

Step 2:
    - Load Parquet with topics (output of chronobert_kmeans.py)
    - Use FinBERT to compute sentiment scores per article
    - Save new Parquet with sentiment features
"""

# ==========================
# CONFIG
# ==========================

INPUT_PARQUET_PATH = "/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/mag7_news_with_topicsV2.parquet"
OUTPUT_PARQUET_PATH = "mag7_news_with_topics_sentimentV2.parquet"

SENTIMENT_MODEL_DIR = "ProsusAI/finbert"
MAX_LENGTH = 256
BATCH_SIZE_SENT = 64


# ==========================
# IMPORTS
# ==========================

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ==========================
# DEVICE SELECTION
# ==========================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"[SENT] Using device: {device}")


# ==========================
# SENTIMENT: INFERENCE (FinBERT)
# ==========================

class FinBERTSentimentScorer:
    """
    Wraps a sentiment classifier (FinBERT by default).
    Assumes labels: 0=neg, 1=neu, 2=pos.
    """

    def __init__(self, model_dir: str):
        print(f"[SENT] Loading sentiment model from: {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        self.model.eval()

        try:
            config = self.model.config
            print("[SENT] Sentiment model id2label:", getattr(config, "id2label", None))
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
                    return_tensors="pt",
                ).to(device)

                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)


# ==========================
# MAIN
# ==========================

def main():
    print(f"[SENT] Loading news with topics from: {INPUT_PARQUET_PATH}")
    df = pd.read_parquet(INPUT_PARQUET_PATH)

    if "text_for_nlp" not in df.columns:
        raise ValueError("text_for_nlp column not found. Run chronobert_kmeans.py first.")

    sentiment_scorer = FinBERTSentimentScorer(SENTIMENT_MODEL_DIR)

    texts_for_sent = df["text_for_nlp"].fillna("").tolist()
    print("[SENT] Scoring sentiment...")
    sentiment_probs = sentiment_scorer.score(
        texts_for_sent,
        batch_size=BATCH_SIZE_SENT,
        max_length=MAX_LENGTH,
    )

    df[["sent_neg", "sent_neu", "sent_pos"]] = sentiment_probs
    df["sentiment_finbert"] = df["sent_pos"] - df["sent_neg"]

    if "sentiment" in df.columns:
        df = df.rename(columns={"sentiment": "sentiment_vendor"})

    print(f"[SENT] Saving news with sentiment to: {OUTPUT_PARQUET_PATH}")
    df.to_parquet(OUTPUT_PARQUET_PATH, index=False)

    print("[SENT] Done. Columns now include (sample):")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()

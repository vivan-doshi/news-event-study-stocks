from transformers import AutoTokenizer
import pandas as pd

NEWS_PARQUET_PATH = "/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-event-study-stocks/eodhd_news_replication/data/raw/news_raw_20251031.parquet"
MODEL_NAME = "manelalab/chrono-bert-v1-20201231"  # any chrono checkpoint
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

df = pd.read_parquet(NEWS_PARQUET_PATH)
df = df[df["symbol_query"] == "AAPL.US"].copy()
texts = (df["title"].fillna("") + ". " + df["content"].fillna("")).astype(str).tolist()
lengths = [len(tokenizer.encode(t, truncation=False)) for t in texts]

import numpy as np
for p in [50, 75, 90, 95, 99]:
    print(f"{p}th percentile length: {np.percentile(lengths, p)}")

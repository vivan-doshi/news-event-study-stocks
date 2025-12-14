
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import os
import re
import argparse
import logging
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DEVICE SELECTION
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
logger.info(f"Using device: {device}")

# CONFIG
MAX_LENGTH = 256
BATCH_SIZE = 4

def get_chronobert_name_for_date(dt):
    year = dt.year
    checkpoint_year = min(max(year - 1, 1999), 2024)
    return f"manelalab/chrono-bert-v1-{checkpoint_year}1231"

def clean_text(text):
    if not isinstance(text, str): return ""
    t = text
    # Basic cleaning re-used from previous valid implementations
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"www\.[^\s]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

from transformers import AutoTokenizer, BertModel

class ChronoBERTEncoder:
    def __init__(self, max_length=256, batch_size=32):
        self.max_length = max_length
        self.batch_size = batch_size
        self._cache = {}

    def _load_model(self, model_name):
        if model_name not in self._cache:
            logger.info(f"Loading model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Use BertModel directly to avoid AutoModel inference bugs with new transformers
            model = BertModel.from_pretrained(model_name).to(device)
            model.eval()
            self._cache[model_name] = (tokenizer, model)
        return self._cache[model_name]

    def encode(self, texts, model_name):
        tokenizer, model = self._load_model(model_name)
        all_embs = []
        
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i+self.batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(device)
                outputs = model(**inputs)
                
                # Mean Pooling
                last_hidden = outputs.last_hidden_state
                mask = inputs['attention_mask'].unsqueeze(-1)
                summed = (last_hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                pooled = (summed / counts).cpu().tolist()
                all_embs.append(np.array(pooled))
                
        return np.vstack(all_embs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="data/processed/mag7_news_with_sentiment_and_topics_labeledV2.parquet")
    parser.add_argument("--output_path", default="data/processed/mag7_embeddings.parquet")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        # Fallback to raw if processed V2 not found?? 
        # Actually user said "Run chronobert embeddings according the way we have created just store the embeddings"
        # The file `mag7_news_with_sentiment_and_topics_labeledV2.parquet` has text but maybe not raw enough?
        # Let's check columns. It has 'title', 'content', 'text_for_nlp'.
        # We can use 'text_for_nlp' if it exists, roughly.
        pass

    logger.info(f"Loading data from {args.input_path}...")
    df = pd.read_parquet(args.input_path)

    # Full Run
    logger.info("Running on full dataset.")

    # Ensure date
    if 'published_at' in df.columns:
        df['date_obj'] = pd.to_datetime(df['published_at'])
    elif 'date' in df.columns:
         df['date_obj'] = pd.to_datetime(df['date'])
    else:
        raise ValueError("No date column found")

    # Determine Model Name per row
    df['chrono_model'] = df['date_obj'].apply(get_chronobert_name_for_date)
    
    # Prepare Text
    if 'text_for_nlp' not in df.columns:
        logger.info("Cleaning text...")
        df['text_for_nlp'] = (df['title'].fillna("") + ". " + df['content'].fillna("")).apply(clean_text)
    
    texts = df['text_for_nlp'].tolist()
    
    # Encode by Model Group
    encoder = ChronoBERTEncoder(max_length=MAX_LENGTH, batch_size=BATCH_SIZE)
    
    # Initialize embedding array
    embedding_dim = 768 # Standard BERT
    embeddings = np.zeros((len(df), embedding_dim), dtype=np.float32)
    
    groups = df.groupby('chrono_model')
    for model_name, indices in groups.groups.items():
        idx_list = list(indices)
        subset_texts = [texts[i] for i in idx_list]
        logger.info(f"Encoding {len(subset_texts)} articles with {model_name}...")
        
        full_embs = encoder.encode(subset_texts, model_name)
        embeddings[idx_list] = full_embs
        
    # Save
    # We save a parquet with Index (to join back) and Embedding Vectors (as list or massive cols?)
    # Saving as list in parquet is best for PyArrow.
    logger.info("Saving embeddings...")
    
    # Create a DataFrame with ID and Embedding
    # Attempt to keep relevant keys for joining
    out_df = df[['article_id', 'symbol_query', 'date_obj']].copy()
    out_df['embedding'] = list(embeddings) # Convert to list for parquet storage
    out_df['sentiment_finbert'] = df.get('sentiment_finbert', 0.0) # Pass through sentiment
    out_df['text_for_nlp'] = df['text_for_nlp'] # Pass through text for extraction
    
    # Rename date_obj back to suitable
    out_df = out_df.rename(columns={'date_obj': 'date'})
    out_df['date'] = out_df['date'].dt.strftime('%Y-%m-%d') # Standardize
    
    out_df.to_parquet(args.output_path)
    logger.info(f"Saved to {args.output_path}")

if __name__ == "__main__":
    main()

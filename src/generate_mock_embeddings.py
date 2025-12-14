
import pandas as pd
import numpy as np
import argparse
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default="data/processed/mag7_embeddings.parquet")
    args = parser.parse_args()
    
    # Load Source to get proper Dates and IDs
    source_path = "data/processed/mag7_news_with_sentiment_and_topics_labeledV2.parquet"
    logger.info(f"Loading source from {source_path}")
    df = pd.read_parquet(source_path)
    
    # Sample 1000
    df = df.sample(n=1000, random_state=42).reset_index(drop=True)
    
    # Generate Random Embeddings (768 dim)
    logger.info("Generating mock embeddings...")
    embeddings = np.random.rand(len(df), 768).astype(np.float32)
    
    out_df = df[['article_id', 'symbol_query', 'published_at', 'title', 'content', 'sentiment_finbert']].copy()
    out_df['embedding'] = list(embeddings)
    out_df['text_for_nlp'] = df['title'].fillna('') + " " + df['content'].fillna('')
    
    # Rename date
    out_df['date'] = pd.to_datetime(out_df['published_at']).dt.strftime('%Y-%m-%d')
    
    out_df.to_parquet(args.output_path)
    logger.info(f"Mock embeddings saved to {args.output_path}")

if __name__ == "__main__":
    main()

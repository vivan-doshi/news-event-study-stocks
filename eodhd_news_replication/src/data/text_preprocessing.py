"""
Text Normalization and Preprocessing Pipeline
Handles text cleaning, tokenization, and vocabulary construction
"""

import re
import logging
import json
from typing import List, Dict, Tuple, Set
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import yaml

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Text preprocessing and normalization pipeline"""

    def __init__(self, config_path: str = "conf/experiment.yaml"):
        """Initialize text preprocessor with configuration"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load stopwords
        self.stopwords = self._load_stopwords()
        self.lemmatizer = WordNetLemmatizer()

        # Text processing parameters
        self.min_text_length = self.config['text_processing']['min_text_length']
        self.min_tokens_per_doc = self.config['text_processing']['min_tokens_per_doc']
        self.duplicate_threshold = self.config['text_processing']['duplicate_similarity_threshold']
        self.duplicate_window = self.config['text_processing']['duplicate_window_days']

        # Vocabulary parameters
        self.unigram_min_df = self.config['vocabulary']['unigram_min_df']
        self.bigram_min_df = self.config['vocabulary']['bigram_min_df']
        self.max_vocab_size = self.config['vocabulary']['max_vocab_size']
        self.min_token_length = self.config['vocabulary']['min_token_length']

    def _load_stopwords(self) -> Set[str]:
        """Load stopwords from file and NLTK"""
        # Load custom stopwords
        custom_stopwords = set()
        stopwords_file = Path("conf/stopwords.txt")
        if stopwords_file.exists():
            with open(stopwords_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        custom_stopwords.add(line.lower())

        # Combine with NLTK stopwords
        nltk_stopwords = set(stopwords.words('english'))
        all_stopwords = custom_stopwords.union(nltk_stopwords)

        logger.info(f"Loaded {len(all_stopwords)} stopwords")
        return all_stopwords

    def clean_text(self, text: str) -> str:
        """Clean and normalize raw text"""
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)

        # Remove stock tickers (e.g., $AAPL, AAPL.US)
        text = re.sub(r'\$[A-Z]+', ' ', text)
        text = re.sub(r'\b[A-Z]+\.[A-Z]+\b', ' ', text)

        # Remove all-caps sections (often disclaimers)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Check if line is mostly uppercase
            if len(line) > 0:
                upper_ratio = sum(1 for c in line if c.isupper()) / len(line)
                if upper_ratio < 0.7:  # Keep lines that are less than 70% uppercase
                    cleaned_lines.append(line)
        text = ' '.join(cleaned_lines)

        # Remove numbers and special characters, keep only letters and spaces
        text = re.sub(r'[^a-z\s]', ' ', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize and lemmatize text"""
        # Tokenize
        tokens = word_tokenize(text)

        # Filter tokens
        processed_tokens = []
        for token in tokens:
            # Skip if token is a stopword
            if token in self.stopwords:
                continue

            # Skip if token is too short
            if len(token) < self.min_token_length:
                continue

            # Lemmatize
            lemma = self.lemmatizer.lemmatize(token, pos='v')  # Try as verb
            lemma = self.lemmatizer.lemmatize(lemma, pos='n')  # Then as noun

            processed_tokens.append(lemma)

        return processed_tokens

    def extract_ngrams(self, tokens: List[str], n: int = 2) -> List[str]:
        """Extract n-grams from token list"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = '_'.join(tokens[i:i + n])
            ngrams.append(ngram)
        return ngrams

    def process_documents(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process all documents in the dataframe"""
        logger.info(f"Starting text preprocessing for {len(df)} documents")

        # Filter by language (keep English only)
        if 'language' in df.columns:
            df = df[df['language'].isin(['en', 'english', 'EN', None, ''])]

        # Combine title and content
        df['full_text'] = df['title'].fillna('') + ' ' + df['content'].fillna('')

        # Filter by minimum text length
        df = df[df['full_text'].str.len() >= self.min_text_length]

        logger.info(f"After initial filtering: {len(df)} documents")

        # Clean text
        logger.info("Cleaning text...")
        df['cleaned_text'] = df['full_text'].apply(self.clean_text)

        # Tokenize and lemmatize
        logger.info("Tokenizing and lemmatizing...")
        df['tokens'] = df['cleaned_text'].apply(self.tokenize_and_lemmatize)

        # Filter by minimum token count
        df = df[df['tokens'].apply(len) >= self.min_tokens_per_doc]

        logger.info(f"After token filtering: {len(df)} documents")

        # Extract bigrams
        logger.info("Extracting bigrams...")
        df['bigrams'] = df['tokens'].apply(lambda x: self.extract_ngrams(x, 2))

        # Combine unigrams and bigrams
        df['all_terms'] = df['tokens'] + df['bigrams']

        # Remove near-duplicates
        df = self.remove_duplicates(df)

        logger.info(f"After duplicate removal: {len(df)} documents")

        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove near-duplicate articles within time window"""
        logger.info("Removing near-duplicates...")

        # Convert published_at to datetime
        df['published_date'] = pd.to_datetime(df['published_at'])

        # Sort by date and reset index
        df = df.sort_values('published_date').reset_index(drop=True)

        # Simplified approach: remove exact duplicates first
        initial_count = len(df)
        df = df.drop_duplicates(subset=['cleaned_text'], keep='first')
        logger.info(f"Removed {initial_count - len(df)} exact duplicates")

        # For large datasets, skip expensive near-duplicate detection
        # or process in smaller batches
        if len(df) > 50000:
            logger.info("Large dataset detected, using simplified deduplication")
            # Additional simple deduplication by title similarity
            df['title_lower'] = df['title'].str.lower().str.strip()
            # Check if 'symbol' column exists
            if 'symbol' in df.columns:
                df = df.drop_duplicates(subset=['title_lower', 'symbol'], keep='first')
            else:
                df = df.drop_duplicates(subset=['title_lower'], keep='first')
            df = df.drop(columns=['title_lower'])
        else:
            # Process smaller datasets with TF-IDF
            logger.info("Processing with TF-IDF similarity...")
            vectorizer = TfidfVectorizer(max_features=500, max_df=0.95)

            try:
                # Create TF-IDF matrix for all documents at once
                texts = df['cleaned_text'].fillna('')
                tfidf_matrix = vectorizer.fit_transform(texts)

                # Process in daily batches to find duplicates
                df['date_only'] = df['published_date'].dt.date
                keep_mask = pd.Series([True] * len(df))

                for date in df['date_only'].unique():
                    date_mask = df['date_only'] == date
                    date_indices = df[date_mask].index.tolist()

                    if len(date_indices) <= 1:
                        continue

                    # Check similarity within the same day
                    for i in range(len(date_indices)):
                        if not keep_mask[date_indices[i]]:
                            continue

                        for j in range(i + 1, len(date_indices)):
                            if not keep_mask[date_indices[j]]:
                                continue

                            similarity = cosine_similarity(
                                tfidf_matrix[date_indices[i]:date_indices[i] + 1],
                                tfidf_matrix[date_indices[j]:date_indices[j] + 1]
                            )[0, 0]

                            if similarity > self.duplicate_threshold:
                                keep_mask[date_indices[j]] = False

                df = df[keep_mask]
                df = df.drop(columns=['date_only'])

            except Exception as e:
                logger.warning(f"TF-IDF deduplication failed: {e}, keeping all documents")

        logger.info(f"After deduplication: {len(df)} documents remaining")

        return df

    def build_vocabulary(self, df: pd.DataFrame) -> Dict[str, int]:
        """Build vocabulary from processed documents"""
        logger.info("Building vocabulary...")

        # Count term frequencies
        unigram_counts = Counter()
        bigram_counts = Counter()

        for tokens, bigrams in zip(df['tokens'], df['bigrams']):
            unigram_counts.update(tokens)
            bigram_counts.update(bigrams)

        # Calculate document frequencies
        n_docs = len(df)
        min_unigram_docs = int(self.unigram_min_df * n_docs)
        min_bigram_docs = int(self.bigram_min_df * n_docs)

        # Filter by minimum document frequency
        filtered_unigrams = {
            term: count for term, count in unigram_counts.items()
            if count >= min_unigram_docs
        }

        filtered_bigrams = {
            term: count for term, count in bigram_counts.items()
            if count >= min_bigram_docs
        }

        # Combine and limit vocabulary size
        all_terms = list(filtered_unigrams.keys()) + list(filtered_bigrams.keys())

        # Sort by frequency and take top N
        term_counts = {**filtered_unigrams, **filtered_bigrams}
        sorted_terms = sorted(term_counts.keys(), key=lambda x: term_counts[x], reverse=True)

        vocab_terms = sorted_terms[:self.max_vocab_size]

        # Create vocabulary mapping
        vocabulary = {term: idx for idx, term in enumerate(vocab_terms)}

        logger.info(f"Built vocabulary with {len(vocabulary)} terms")
        logger.info(f"  - Unigrams: {len([t for t in vocabulary if '_' not in t])}")
        logger.info(f"  - Bigrams: {len([t for t in vocabulary if '_' in t])}")

        # Save vocabulary
        vocab_df = pd.DataFrame(list(vocabulary.items()), columns=['term', 'index'])
        vocab_df['count'] = vocab_df['term'].map(term_counts)
        vocab_df.to_csv("conf/vocabulary.csv", index=False)

        # Save vocabulary config
        vocab_config = {
            'n_documents': n_docs,
            'n_terms': len(vocabulary),
            'n_unigrams': len([t for t in vocabulary if '_' not in t]),
            'n_bigrams': len([t for t in vocabulary if '_' in t]),
            'min_unigram_docs': min_unigram_docs,
            'min_bigram_docs': min_bigram_docs
        }

        with open("conf/vocabulary.yaml", 'w') as f:
            yaml.dump(vocab_config, f)

        return vocabulary

    def process_pipeline(self, input_file: str = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Run the complete preprocessing pipeline"""
        # Load raw news data
        if input_file is None:
            # Find the most recent raw news file
            raw_files = list(Path("data/raw").glob("news_raw_*.parquet"))
            if not raw_files:
                raise FileNotFoundError("No raw news files found")
            input_file = str(sorted(raw_files)[-1])

        logger.info(f"Loading raw data from {input_file}")
        df = pd.read_parquet(input_file)

        # Process documents
        df_processed = self.process_documents(df)

        # Build vocabulary
        vocabulary = self.build_vocabulary(df_processed)

        # Save processed data
        output_file = "data/clean/news_clean.parquet"
        df_processed.to_parquet(output_file, compression='snappy')
        logger.info(f"Saved processed data to {output_file}")

        # Log statistics
        stats = {
            'input_documents': len(df),
            'output_documents': len(df_processed),
            'vocabulary_size': len(vocabulary),
            'avg_tokens_per_doc': df_processed['tokens'].apply(len).mean(),
            'avg_bigrams_per_doc': df_processed['bigrams'].apply(len).mean()
        }

        with open("logs/preprocessing_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Preprocessing complete: {stats}")

        return df_processed, vocabulary


def main():
    """Main function to run text preprocessing"""
    preprocessor = TextPreprocessor()
    df_processed, vocabulary = preprocessor.process_pipeline()
    logger.info("Text preprocessing pipeline complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
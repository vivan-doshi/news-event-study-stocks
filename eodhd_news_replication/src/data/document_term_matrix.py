"""
Document-Term Matrix Construction
Builds sparse matrix representation for topic modeling
"""

import logging
from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
import yaml

logger = logging.getLogger(__name__)


class DocumentTermMatrixBuilder:
    """Build document-term matrix from processed text"""

    def __init__(self, vocabulary: Dict[str, int] = None):
        """Initialize DTM builder with vocabulary"""
        self.vocabulary = vocabulary
        if vocabulary is None:
            self.vocabulary = self._load_vocabulary()

    def _load_vocabulary(self) -> Dict[str, int]:
        """Load vocabulary from file"""
        vocab_file = Path("conf/vocabulary.csv")
        if not vocab_file.exists():
            raise FileNotFoundError("Vocabulary file not found. Run text preprocessing first.")

        vocab_df = pd.read_csv(vocab_file)
        vocabulary = dict(zip(vocab_df['term'], vocab_df['index']))
        logger.info(f"Loaded vocabulary with {len(vocabulary)} terms")
        return vocabulary

    def build_dtm(self, df: pd.DataFrame) -> Tuple[csr_matrix, pd.DataFrame, pd.DataFrame]:
        """
        Build document-term matrix from processed documents

        Returns:
            dtm: Sparse document-term matrix
            doc_index: DataFrame mapping row indices to document IDs
            term_index: DataFrame mapping column indices to terms
        """
        logger.info(f"Building DTM for {len(df)} documents and {len(self.vocabulary)} terms")

        # Initialize sparse matrix components
        rows = []
        cols = []
        data = []

        # Build the matrix
        for doc_idx, (_, doc) in enumerate(df.iterrows()):
            if doc_idx % 1000 == 0:
                logger.info(f"Processing document {doc_idx}/{len(df)}")

            # Get all terms for this document
            all_terms = doc['all_terms'] if 'all_terms' in doc else doc['tokens'] + doc['bigrams']

            # Count term frequencies
            term_counts = {}
            for term in all_terms:
                if term in self.vocabulary:
                    term_idx = self.vocabulary[term]
                    term_counts[term_idx] = term_counts.get(term_idx, 0) + 1

            # Add to sparse matrix components
            for term_idx, count in term_counts.items():
                rows.append(doc_idx)
                cols.append(term_idx)
                data.append(count)

        # Create sparse matrix
        n_docs = len(df)
        n_terms = len(self.vocabulary)
        dtm = csr_matrix((data, (rows, cols)), shape=(n_docs, n_terms), dtype=np.int32)

        logger.info(f"Created DTM with shape {dtm.shape} and {dtm.nnz} non-zero elements")
        logger.info(f"Sparsity: {1 - dtm.nnz / (dtm.shape[0] * dtm.shape[1]):.4f}")

        # Create index mappings
        doc_index = pd.DataFrame({
            'row_idx': range(n_docs),
            'article_id': df['article_id'].values,
            'published_at': df['published_at'].values,
            'title': df['title'].values
        })

        term_index = pd.DataFrame(list(self.vocabulary.items()),
                                 columns=['term', 'col_idx']).sort_values('col_idx')

        return dtm, doc_index, term_index

    def save_dtm(self, dtm: csr_matrix, doc_index: pd.DataFrame, term_index: pd.DataFrame):
        """Save DTM and indices to disk"""
        # Create output directory if needed
        output_dir = Path("data/derived")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save sparse matrix
        dtm_file = output_dir / "dtm.npz"
        save_npz(dtm_file, dtm)
        logger.info(f"Saved DTM to {dtm_file}")

        # Save indices
        doc_index_file = output_dir / "doc_index.parquet"
        doc_index.to_parquet(doc_index_file, compression='snappy')
        logger.info(f"Saved document index to {doc_index_file}")

        term_index_file = output_dir / "term_index.parquet"
        term_index.to_parquet(term_index_file, compression='snappy')
        logger.info(f"Saved term index to {term_index_file}")

        # Save statistics
        stats = {
            'n_documents': dtm.shape[0],
            'n_terms': dtm.shape[1],
            'n_nonzero': dtm.nnz,
            'sparsity': float(1 - dtm.nnz / (dtm.shape[0] * dtm.shape[1])),
            'avg_terms_per_doc': float(dtm.sum(axis=1).mean()),
            'avg_docs_per_term': float(dtm.sum(axis=0).mean())
        }

        with open("logs/dtm_stats.json", 'w') as f:
            import json
            json.dump(stats, f, indent=2)

        logger.info(f"DTM statistics: {stats}")

    def calculate_tf_idf(self, dtm: csr_matrix) -> csr_matrix:
        """Calculate TF-IDF transformation of DTM"""
        logger.info("Calculating TF-IDF...")

        # Calculate term frequencies (normalize by document length)
        doc_lengths = np.array(dtm.sum(axis=1)).flatten()
        doc_lengths[doc_lengths == 0] = 1  # Avoid division by zero

        # Create diagonal matrix for normalization
        tf = dtm.multiply(1 / doc_lengths[:, np.newaxis])

        # Calculate inverse document frequencies
        n_docs = dtm.shape[0]
        doc_freq = np.array((dtm > 0).sum(axis=0)).flatten()
        idf = np.log(n_docs / (doc_freq + 1))  # Add 1 to avoid division by zero

        # Calculate TF-IDF
        tfidf = tf.multiply(idf)

        logger.info(f"Calculated TF-IDF matrix with shape {tfidf.shape}")

        return tfidf

    def get_top_terms(self, dtm: csr_matrix, term_index: pd.DataFrame, n_terms: int = 50) -> pd.DataFrame:
        """Get most frequent terms across corpus"""
        # Calculate term frequencies
        term_freqs = np.array(dtm.sum(axis=0)).flatten()

        # Get top term indices
        top_indices = np.argsort(term_freqs)[-n_terms:][::-1]

        # Get term information
        top_terms_df = pd.DataFrame({
            'term': term_index.iloc[top_indices]['term'].values,
            'frequency': term_freqs[top_indices],
            'document_frequency': np.array((dtm[:, top_indices] > 0).sum(axis=0)).flatten()
        })

        return top_terms_df

    def filter_extreme_terms(self, dtm: csr_matrix, term_index: pd.DataFrame,
                            min_df: float = 0.001, max_df: float = 0.95) -> Tuple[csr_matrix, pd.DataFrame]:
        """Filter out extremely rare or common terms"""
        n_docs = dtm.shape[0]

        # Calculate document frequencies
        doc_freqs = np.array((dtm > 0).sum(axis=0)).flatten() / n_docs

        # Find terms to keep
        keep_mask = (doc_freqs >= min_df) & (doc_freqs <= max_df)
        keep_indices = np.where(keep_mask)[0]

        # Filter DTM and term index
        dtm_filtered = dtm[:, keep_indices]
        term_index_filtered = term_index.iloc[keep_indices].reset_index(drop=True)

        logger.info(f"Filtered DTM from {dtm.shape[1]} to {dtm_filtered.shape[1]} terms")

        return dtm_filtered, term_index_filtered


def build_dtm_pipeline(input_file: str = None) -> Tuple[csr_matrix, pd.DataFrame, pd.DataFrame]:
    """Complete pipeline to build document-term matrix"""

    # Load processed data
    if input_file is None:
        input_file = "data/clean/news_clean.parquet"

    logger.info(f"Loading processed data from {input_file}")
    df = pd.read_parquet(input_file)

    # Initialize DTM builder
    builder = DocumentTermMatrixBuilder()

    # Build DTM
    dtm, doc_index, term_index = builder.build_dtm(df)

    # Save DTM and indices
    builder.save_dtm(dtm, doc_index, term_index)

    # Calculate and save TF-IDF (for future use)
    tfidf = builder.calculate_tf_idf(dtm)
    save_npz("data/derived/tfidf.npz", tfidf)

    # Get and save top terms
    top_terms = builder.get_top_terms(dtm, term_index, n_terms=100)
    top_terms.to_csv("data/derived/top_terms.csv", index=False)
    logger.info(f"Top 10 terms:\n{top_terms.head(10)}")

    return dtm, doc_index, term_index


def main():
    """Main function to build DTM"""
    logging.basicConfig(level=logging.INFO)
    dtm, doc_index, term_index = build_dtm_pipeline()
    logger.info("Document-term matrix construction complete!")


if __name__ == "__main__":
    main()
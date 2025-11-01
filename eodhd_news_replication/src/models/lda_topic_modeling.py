"""
LDA Topic Modeling with Hyperparameter Search
Implements both batch and online LDA for topic discovery
"""

import logging
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import KFold
import yaml
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class LDATopicModeling:
    """LDA topic modeling with cross-validation and hyperparameter search"""

    def __init__(self, config_path: str = "conf/experiment.yaml"):
        """Initialize LDA modeling with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # LDA parameters
        self.k_values = self.config['lda']['k_values']
        self.alpha = self.config['lda']['alpha']
        self.beta = self.config['lda']['beta']
        self.iterations = self.config['lda']['iterations']
        self.random_seed = self.config['lda']['random_seed']
        self.cv_folds = self.config['lda']['cv_folds']

        # Set random seeds
        np.random.seed(self.random_seed)

    def load_data(self) -> Tuple[csr_matrix, pd.DataFrame, pd.DataFrame]:
        """Load DTM and indices"""
        dtm = load_npz("data/derived/dtm.npz")
        doc_index = pd.read_parquet("data/derived/doc_index.parquet")
        term_index = pd.read_parquet("data/derived/term_index.parquet")

        logger.info(f"Loaded DTM with shape {dtm.shape}")

        return dtm, doc_index, term_index

    def train_lda(self, dtm: csr_matrix, n_topics: int,
                  learning_method: str = 'batch') -> LatentDirichletAllocation:
        """Train a single LDA model"""
        logger.info(f"Training LDA with K={n_topics} topics using {learning_method} method")

        model = LatentDirichletAllocation(
            n_components=n_topics,
            doc_topic_prior=self.alpha / n_topics,  # Symmetric Dirichlet prior
            topic_word_prior=self.beta / dtm.shape[1],  # Symmetric Dirichlet prior
            learning_method=learning_method,
            max_iter=self.iterations,
            random_state=self.random_seed,
            n_jobs=-1,
            verbose=0
        )

        # Fit the model
        model.fit(dtm)

        return model

    def calculate_perplexity(self, model: LatentDirichletAllocation,
                            dtm: csr_matrix) -> float:
        """Calculate perplexity for model evaluation"""
        return model.perplexity(dtm)

    def calculate_topic_coherence(self, model: LatentDirichletAllocation,
                                 dtm: csr_matrix,
                                 term_index: pd.DataFrame,
                                 top_n: int = 10) -> float:
        """Calculate topic coherence using PMI-based metric"""
        n_topics = model.n_components
        coherence_scores = []

        # Get term co-occurrence matrix (simplified version)
        doc_term_binary = (dtm > 0).astype(int)
        cooccurrence = doc_term_binary.T @ doc_term_binary
        n_docs = dtm.shape[0]

        for topic_idx in range(n_topics):
            # Get top words for this topic
            top_word_indices = model.components_[topic_idx].argsort()[-top_n:][::-1]

            # Calculate coherence for this topic
            topic_coherence = 0
            pairs = 0

            for i in range(len(top_word_indices)):
                for j in range(i + 1, len(top_word_indices)):
                    wi = top_word_indices[i]
                    wj = top_word_indices[j]

                    # PMI calculation
                    p_wi = doc_term_binary[:, wi].sum() / n_docs
                    p_wj = doc_term_binary[:, wj].sum() / n_docs
                    p_wi_wj = cooccurrence[wi, wj] / n_docs

                    if p_wi_wj > 0:
                        pmi = np.log(p_wi_wj / (p_wi * p_wj + 1e-10) + 1e-10)
                        topic_coherence += pmi
                        pairs += 1

            if pairs > 0:
                coherence_scores.append(topic_coherence / pairs)

        return np.mean(coherence_scores)

    def cross_validate_lda(self, dtm: csr_matrix, n_topics: int) -> Dict:
        """Cross-validate LDA model"""
        logger.info(f"Cross-validating LDA with K={n_topics}")

        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)

        perplexities = []
        log_likelihoods = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(dtm)):
            logger.info(f"  Fold {fold + 1}/{self.cv_folds}")

            # Split data
            dtm_train = dtm[train_idx]
            dtm_val = dtm[val_idx]

            # Train model
            model = self.train_lda(dtm_train, n_topics)

            # Evaluate
            perplexity = self.calculate_perplexity(model, dtm_val)
            log_likelihood = model.score(dtm_val) / dtm_val.shape[0]

            perplexities.append(perplexity)
            log_likelihoods.append(log_likelihood)

        results = {
            'n_topics': n_topics,
            'mean_perplexity': np.mean(perplexities),
            'std_perplexity': np.std(perplexities),
            'mean_log_likelihood': np.mean(log_likelihoods),
            'std_log_likelihood': np.std(log_likelihoods)
        }

        logger.info(f"  Results: perplexity={results['mean_perplexity']:.2f}, "
                   f"log_likelihood={results['mean_log_likelihood']:.4f}")

        return results

    def hyperparameter_search(self, dtm: csr_matrix) -> pd.DataFrame:
        """Search for optimal number of topics"""
        logger.info(f"Starting hyperparameter search over K={self.k_values}")

        results = []
        for k in self.k_values:
            cv_results = self.cross_validate_lda(dtm, k)
            results.append(cv_results)

        results_df = pd.DataFrame(results)

        # Save results
        results_df.to_csv("models/lda_cv_results.csv", index=False)

        # Find best K (highest log likelihood)
        best_k = results_df.loc[results_df['mean_log_likelihood'].idxmax(), 'n_topics']
        logger.info(f"Best K={int(best_k)} with log_likelihood="
                   f"{results_df.loc[results_df['mean_log_likelihood'].idxmax(), 'mean_log_likelihood']:.4f}")

        return results_df

    def train_final_model(self, dtm: csr_matrix, doc_index: pd.DataFrame,
                         term_index: pd.DataFrame, n_topics: int) -> Dict:
        """Train final LDA model with selected K"""
        logger.info(f"Training final LDA model with K={n_topics}")

        # Train model
        model = self.train_lda(dtm, n_topics, learning_method='batch')

        # Get document-topic distributions
        doc_topics = model.transform(dtm)

        # Get topic-term distributions
        topic_terms = model.components_

        # Normalize to get probabilities
        topic_terms_norm = topic_terms / topic_terms.sum(axis=1, keepdims=True)

        # Calculate metrics
        perplexity = self.calculate_perplexity(model, dtm)
        coherence = self.calculate_topic_coherence(model, dtm, term_index)

        # Extract top terms for each topic
        top_terms = self.get_top_terms(model, term_index, n_terms=15)

        # Create output directory
        model_dir = Path(f"models/lda_k={n_topics:03d}")
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        with open(model_dir / "model.pkl", 'wb') as f:
            pickle.dump(model, f)

        # Save document-topic matrix (theta)
        theta_df = pd.DataFrame(
            doc_topics,
            columns=[f"topic_{i:03d}" for i in range(n_topics)]
        )
        theta_df['article_id'] = doc_index['article_id'].values
        theta_df.to_parquet(model_dir / "theta.parquet", compression='snappy')

        # Save topic-term matrix (phi)
        phi_df = pd.DataFrame(
            topic_terms_norm.T,
            index=term_index['term'].values,
            columns=[f"topic_{i:03d}" for i in range(n_topics)]
        )
        phi_df.to_parquet(model_dir / "phi.parquet", compression='snappy')

        # Save top terms
        top_terms_df = pd.DataFrame(top_terms)
        top_terms_df.to_csv(model_dir / "top_terms.csv", index=False)

        # Save metrics
        metrics = {
            'n_topics': n_topics,
            'n_documents': dtm.shape[0],
            'n_terms': dtm.shape[1],
            'perplexity': float(perplexity),
            'coherence': float(coherence),
            'log_likelihood': float(model.score(dtm) / dtm.shape[0])
        }

        with open(model_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved model to {model_dir}")
        logger.info(f"Metrics: {metrics}")

        return {
            'model': model,
            'doc_topics': doc_topics,
            'topic_terms': topic_terms_norm,
            'top_terms': top_terms,
            'metrics': metrics
        }

    def get_top_terms(self, model: LatentDirichletAllocation,
                     term_index: pd.DataFrame,
                     n_terms: int = 15) -> Dict[str, List[str]]:
        """Extract top terms for each topic"""
        top_terms = {}

        for topic_idx in range(model.n_components):
            # Get top term indices
            top_indices = model.components_[topic_idx].argsort()[-n_terms:][::-1]

            # Get terms
            terms = term_index.iloc[top_indices]['term'].tolist()
            top_terms[f"topic_{topic_idx:03d}"] = terms

        return top_terms

    def train_online_lda(self, dtm: csr_matrix, doc_index: pd.DataFrame,
                        n_topics: int, batch_size: int = 128) -> Tuple[np.ndarray, LatentDirichletAllocation]:
        """Train online LDA for no-look-ahead protocol"""
        logger.info(f"Training online LDA with K={n_topics}")

        # Sort documents by date
        doc_index['published_date'] = pd.to_datetime(doc_index['published_at'])
        sorted_indices = doc_index.sort_values('published_date').index.values

        # Initialize online LDA
        model = LatentDirichletAllocation(
            n_components=n_topics,
            doc_topic_prior=self.alpha / n_topics,
            topic_word_prior=self.beta / dtm.shape[1],
            learning_method='online',
            batch_size=batch_size,
            max_iter=1,  # Single pass for online learning
            random_state=self.random_seed,
            n_jobs=-1
        )

        # Train incrementally
        doc_topics_online = np.zeros((dtm.shape[0], n_topics))

        for i in range(0, len(sorted_indices), batch_size):
            batch_indices = sorted_indices[i:i + batch_size]

            if i == 0:
                # First batch: initialize the model
                model.fit(dtm[batch_indices])
            else:
                # Subsequent batches: partial fit
                model.partial_fit(dtm[batch_indices])

            # Transform documents with current model state
            doc_topics_online[batch_indices] = model.transform(dtm[batch_indices])

            if i % 1000 == 0:
                logger.info(f"  Processed {i}/{len(sorted_indices)} documents")

        logger.info("Online LDA training complete")

        return doc_topics_online, model

    def create_topic_labels(self, top_terms: Dict[str, List[str]]) -> Dict[str, str]:
        """Create human-readable labels for topics"""
        topic_labels = {}

        for topic_id, terms in top_terms.items():
            # Simple label: concatenate top 3 terms
            label = "_".join(terms[:3])
            topic_labels[topic_id] = label

        return topic_labels


def run_lda_pipeline(n_topics: Optional[int] = None) -> Dict:
    """Complete LDA topic modeling pipeline"""
    logger.info("Starting LDA topic modeling pipeline")

    # Initialize LDA modeling
    lda = LDATopicModeling()

    # Load data
    dtm, doc_index, term_index = lda.load_data()

    # Hyperparameter search if n_topics not specified
    if n_topics is None:
        cv_results = lda.hyperparameter_search(dtm)
        n_topics = int(cv_results.loc[cv_results['mean_log_likelihood'].idxmax(), 'n_topics'])

    # Train final model
    results = lda.train_final_model(dtm, doc_index, term_index, n_topics)

    # Train online version for forecasting
    logger.info("Training online LDA for forecasting")
    doc_topics_online, online_model = lda.train_online_lda(dtm, doc_index, n_topics)

    # Save online results
    online_dir = Path(f"models/lda_k={n_topics:03d}_online")
    online_dir.mkdir(parents=True, exist_ok=True)

    # Save online document-topics
    theta_online_df = pd.DataFrame(
        doc_topics_online,
        columns=[f"topic_{i:03d}" for i in range(n_topics)]
    )
    theta_online_df['article_id'] = doc_index['article_id'].values
    theta_online_df.to_parquet(online_dir / "theta_online.parquet", compression='snappy')

    # Save online model
    with open(online_dir / "model_online.pkl", 'wb') as f:
        pickle.dump(online_model, f)

    # Create topic labels
    topic_labels = lda.create_topic_labels(results['top_terms'])
    with open(f"models/lda_k={n_topics:03d}/topic_labels.json", 'w') as f:
        json.dump(topic_labels, f, indent=2)

    logger.info("LDA pipeline complete!")

    return results


def main():
    """Main function to run LDA modeling"""
    logging.basicConfig(level=logging.INFO)
    results = run_lda_pipeline()
    print("\nTop terms for first 5 topics:")
    for topic_id in list(results['top_terms'].keys())[:5]:
        print(f"{topic_id}: {', '.join(results['top_terms'][topic_id][:10])}")


if __name__ == "__main__":
    main()
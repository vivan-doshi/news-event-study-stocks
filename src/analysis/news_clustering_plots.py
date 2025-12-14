
# src/analysis/news_clustering_plots.py

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_embeddings(input_path: str):
    logger.info(f"Loading embeddings from {input_path}...")
    df = pd.read_parquet(input_path)
    
    if 'embedding' not in df.columns:
        raise ValueError(f"Column 'embedding' not found in {input_path}")
        
    # Convert list column to numpy matrix
    embeddings = np.stack(df['embedding'].values)
    logger.info(f"Embeddings loaded. Shape: {embeddings.shape}")
    return embeddings

def plot_scree(embeddings: np.ndarray, output_path: str, max_k: int = 50):
    logger.info(f"Generating Scree Plot (K=2 to {max_k})...")
    
    inertias = []
    k_values = range(2, max_k + 1)
    
    # Use full dataset for Scree Plot
    n_samples = embeddings.shape[0]
    logger.info(f"Using full dataset for Scree Plot (Total: {n_samples} samples)")
    data_for_scree = embeddings
        
    for k in k_values:
        # Standard KMeans
        model = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
        model.fit(data_for_scree)
        inertias.append(model.inertia_)
        logger.info(f"  ... K={k} done (Inertia: {model.inertia_:.2f})")
            
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertias, 'bo-', markersize=8)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
    plt.title('Scree Plot / Elbow Method for News Embeddings (Full Data)')
    plt.grid(True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    logger.info(f"Scree plot saved to {output_path}")
    plt.close()

def plot_dendrogram(embeddings: np.ndarray, output_path: str, sample_size: int = 15000):
    logger.info(f"Generating Dendrogram (Sample Size: {sample_size})...")
    
    n_samples = embeddings.shape[0]
    if n_samples > sample_size:
        logger.info(f"Dataset too large for clear dendrogram. Sampling {sample_size} random points.")
        indices = np.random.choice(n_samples, sample_size, replace=False)
        data_for_dendro = embeddings[indices]
    else:
        data_for_dendro = embeddings

    # Hierarchical Clustering (Ward linkage)
    logger.info("Computing linkage matrix...")
    Z = linkage(data_for_dendro, method='ward')
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Hierarchical Clustering Dendrogram (Sample N={len(data_for_dendro)})')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    
    dendrogram(
        Z,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=50,                   # show only the last 50 merges
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,   # to get a distribution impression in truncated branches
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    logger.info(f"Dendrogram saved to {output_path}")
    plt.close()

def main():
    INPUT_PATH = "data/processed/mag7_embeddings.parquet"
    if not os.path.exists(INPUT_PATH):
        logger.error(f"Input file not found: {INPUT_PATH}")
        sys.exit(1)
        
    embeddings = load_embeddings(INPUT_PATH)
    
    # Generate Plots
    plot_scree(embeddings, "reports/clustering/scree_plot.png", max_k=50) 
    plot_dendrogram(embeddings, "reports/clustering/dendrogram.png", sample_size=15000)

if __name__ == "__main__":
    main()

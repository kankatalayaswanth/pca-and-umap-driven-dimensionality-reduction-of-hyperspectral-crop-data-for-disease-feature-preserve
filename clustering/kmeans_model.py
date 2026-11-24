"""
Independent K-Means Clustering for Raw, PCA, and UMAP Feature Spaces
Course: INT396 - Unsupervised Learning
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def fit_single_kmeans(
    X: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
    n_init: int = 10,
) -> dict:
    """
    Fit K-Means clustering model independently on a specific feature representation.
    """
    n_samples = X.shape[0]
    safe_k = max(2, min(int(n_clusters), n_samples - 1))
    
    kmeans = KMeans(
        n_clusters=safe_k,
        random_state=random_state,
        n_init=n_init,
    )
    labels = kmeans.fit_predict(X)
    
    # Calculate cluster size counts
    unique, counts = np.unique(labels, return_counts=True)
    cluster_distribution = {f"Cluster {u}": int(c) for u, c in zip(unique, counts)}
    
    return {
        "model": kmeans,
        "labels": labels,
        "inertia": float(kmeans.inertia_),
        "n_iter": int(kmeans.n_iter_),
        "cluster_centers": kmeans.cluster_centers_,
        "cluster_distribution": cluster_distribution,
        "n_clusters": safe_k,
    }


def run_tri_representation_clustering(
    X_raw: np.ndarray,
    X_pca: np.ndarray,
    X_umap: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
    n_init: int = 10,
) -> dict:
    """
    Execute K-Means independently on:
    1. Preprocessed Raw features
    2. PCA-reduced features
    3. UMAP-reduced features
    
    Returns independent models and label assignments.
    """
    raw_res = fit_single_kmeans(X_raw, n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    pca_res = fit_single_kmeans(X_pca, n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    umap_res = fit_single_kmeans(X_umap, n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    
    return {
        "raw": raw_res,
        "pca": pca_res,
        "umap": umap_res,
        "n_clusters": n_clusters,
        "random_state": random_state,
        "n_init": n_init,
    }

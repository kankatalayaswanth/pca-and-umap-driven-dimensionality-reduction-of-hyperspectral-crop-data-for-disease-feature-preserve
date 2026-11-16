"""
Clustering Algorithms Module for Hyperspectral Unsupervised Learning.
Implements:
1. K-Means
2. Agglomerative Hierarchical Clustering
3. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
4. Gaussian Mixture Models (GMM)
Supports benchmarking across Raw, PCA, and UMAP representations.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture


def run_kmeans(
    X: np.ndarray,
    n_clusters: int = 4,
    init: str = 'k-means++',
    n_init: int = 10,
    random_state: int = 42
) -> Tuple[KMeans, np.ndarray]:
    """
    Fits K-Means clustering algorithm.
    Returns:
        model: Fitted KMeans instance
        labels: Assigned cluster labels for each sample
    """
    model = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        random_state=random_state
    )
    labels = model.fit_predict(X)
    return model, labels


def run_agglomerative(
    X: np.ndarray,
    n_clusters: int = 4,
    linkage: str = 'ward',
    metric: str = 'euclidean'
) -> Tuple[AgglomerativeClustering, np.ndarray]:
    """
    Fits Agglomerative Hierarchical Clustering.
    Note: For 'ward' linkage, metric must be 'euclidean'.
    """
    if linkage == 'ward':
        metric = 'euclidean'
        
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        metric=metric
    )
    labels = model.fit_predict(X)
    return model, labels


def run_dbscan(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = 'euclidean'
) -> Tuple[DBSCAN, np.ndarray]:
    """
    Fits DBSCAN density-based clustering.
    Identifies core samples, border samples, and noise points (label = -1).
    """
    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric
    )
    labels = model.fit_predict(X)
    return model, labels


def run_gmm(
    X: np.ndarray,
    n_components: int = 4,
    covariance_type: str = 'full',
    random_state: int = 42
) -> Tuple[GaussianMixture, np.ndarray]:
    """Fits Gaussian Mixture Model for probabilistic clustering."""
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state
    )
    labels = model.fit_predict(X)
    return model, labels


def fit_clustering_model(
    algorithm: str,
    X: np.ndarray,
    params: Dict[str, Any]
) -> Tuple[Any, np.ndarray]:
    """
    Universal factory method to run specified clustering algorithm.
    """
    algo = algorithm.lower().strip()
    if algo == 'kmeans' or algo == 'k-means':
        k = params.get('n_clusters', 4)
        init = params.get('init', 'k-means++')
        n_init = params.get('n_init', 10)
        seed = params.get('random_state', 42)
        return run_kmeans(X, n_clusters=k, init=init, n_init=n_init, random_state=seed)
        
    elif algo == 'agglomerative' or algo == 'hierarchical':
        k = params.get('n_clusters', 4)
        linkage = params.get('linkage', 'ward')
        metric = params.get('metric', 'euclidean')
        return run_agglomerative(X, n_clusters=k, linkage=linkage, metric=metric)
        
    elif algo == 'dbscan':
        eps = params.get('eps', 0.5)
        min_samples = params.get('min_samples', 5)
        metric = params.get('metric', 'euclidean')
        return run_dbscan(X, eps=eps, min_samples=min_samples, metric=metric)
        
    elif algo == 'gmm':
        k = params.get('n_components', 4)
        cov = params.get('covariance_type', 'full')
        seed = params.get('random_state', 42)
        return run_gmm(X, n_components=k, covariance_type=cov, random_state=seed)
        
    else:
        raise ValueError(f"Unknown clustering algorithm '{algorithm}'. Supported: kmeans, agglomerative, dbscan, gmm")


def compute_cluster_centroids_original_space(
    X_original: np.ndarray,
    labels: np.ndarray
) -> Dict[int, np.ndarray]:
    """
    Calculates the mean spectral profile (centroid) in the original high-dimensional
    spectral feature space for each cluster assignment.
    """
    centroids = {}
    unique_clusters = [c for c in np.unique(labels) if c != -1]  # Exclude noise
    for c in unique_clusters:
        mask = (labels == c)
        centroids[int(c)] = np.mean(X_original[mask], axis=0)
    return centroids


def save_clustering_model(model: Any, path: str):
    """Saves clustering model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_clustering_model(path: str) -> Any:
    """Loads clustering model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Clustering model file not found: {path}")
    return joblib.load(path)

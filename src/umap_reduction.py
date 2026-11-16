"""
UMAP (Uniform Manifold Approximation and Projection) Module for Hyperspectral Data.
Implements non-linear manifold dimensionality reduction, neighborhood modeling,
and low-dimensional embeddings for hyperspectral crop features.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List
import umap


def fit_umap(
    X_train: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean',
    random_state: int = 42
) -> Tuple[umap.UMAP, np.ndarray]:
    """
    Fits UMAP reducer on input features and transforms X_train.
    
    Parameters:
        X_train: Preprocessed spectral features (scaled or normalized)
        n_components: Target embedding dimension (usually 2 or 3)
        n_neighbors: Size of local neighborhood (controls balance of local vs global structure)
        min_dist: Minimum distance between points in embedding (controls clustering tightness)
        metric: Distance metric on high-dimensional manifold ('euclidean', 'cosine', 'manhattan')
        random_state: Random seed for reproducible stochastic gradient descent
        
    Returns:
        reducer (umap.UMAP): Fitted UMAP instance
        embedding (np.ndarray): Low-dimensional projection coordinates
    """
    # Guard n_neighbors against very small sample sizes
    effective_neighbors = min(n_neighbors, max(2, X_train.shape[0] - 1))
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_jobs=1  # Deterministic execution
    )
    
    embedding = reducer.fit_transform(X_train)
    return reducer, embedding


def transform_umap(reducer: umap.UMAP, X: np.ndarray) -> np.ndarray:
    """Transforms new or test samples into existing UMAP embedding space."""
    return reducer.transform(X)


def get_umap_coordinates_df(
    embedding: np.ndarray,
    target_series: Optional[pd.Series] = None,
    index_labels: Optional[List[Any]] = None
) -> pd.DataFrame:
    """Formats low-dimensional UMAP embedding into a clean DataFrame for plotting."""
    n_dims = embedding.shape[1]
    cols = [f"UMAP{i+1}" for i in range(n_dims)]
    df = pd.DataFrame(embedding, columns=cols, index=index_labels)
    if target_series is not None:
        df['Target'] = target_series.values if hasattr(target_series, 'values') else target_series
    return df


def save_umap_model(reducer: umap.UMAP, path: str):
    """Saves fitted UMAP model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(reducer, path)


def load_umap_model(path: str) -> umap.UMAP:
    """Loads saved UMAP model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"UMAP model file not found: {path}")
    return joblib.load(path)

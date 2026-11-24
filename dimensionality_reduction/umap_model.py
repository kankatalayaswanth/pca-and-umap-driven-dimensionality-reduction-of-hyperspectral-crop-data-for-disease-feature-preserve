"""
Uniform Manifold Approximation and Projection (UMAP)
Non-linear topological dimensionality reduction preserving local and global structure.
"""

import numpy as np
import pandas as pd


def run_umap_reduction(
    X: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    n_components: int = 2,
    random_state: int = 42,
) -> dict:
    """
    Fit and apply UMAP on the preprocessed hyperspectral feature matrix.
    
    Parameters:
    - X: (N, D) preprocessed feature matrix
    - n_neighbors: size of local neighborhood (balance between local vs global structure)
    - min_dist: minimum distance between points in low-dimensional space
    - metric: distance metric in input space ('euclidean', 'cosine', 'manhattan', 'correlation')
    - n_components: dimension of embedded space (2 or 3)
    - random_state: integer seed for reproducibility
    
    Returns:
    dict with model, transformed coordinates, DataFrame, and stats.
    """
    # Defer heavy UMAP/Numba import to execution time for snappy page loading
    import umap
    n_samples, n_features = X.shape
    
    # Adjust n_neighbors if larger than sample count
    safe_neighbors = min(int(n_neighbors), max(2, n_samples - 1))
    
    # Validate metric
    valid_metrics = ["euclidean", "cosine", "manhattan", "correlation"]
    if metric not in valid_metrics:
        metric = "euclidean"
        
    safe_components = max(2, min(int(n_components), 3))
    
    # Instantiate and fit UMAP
    reducer = umap.UMAP(
        n_neighbors=safe_neighbors,
        min_dist=float(min_dist),
        metric=metric,
        n_components=safe_components,
        random_state=int(random_state) if random_state is not None else None,
        n_jobs=1 if random_state is not None else None,
        compatibility_layout=True,
        verbose=False,
    )
    
    X_umap = reducer.fit_transform(X)
    
    # Create column names
    col_names = [f"UMAP{i+1}" for i in range(safe_components)]
    umap_df = pd.DataFrame(X_umap, columns=col_names)
    
    dim_reduction_pct = float(1.0 - (safe_components / n_features)) * 100.0
    
    stats = {
        "original_features": n_features,
        "n_components": safe_components,
        "n_neighbors": safe_neighbors,
        "min_dist": min_dist,
        "metric": metric,
        "random_state": random_state,
        "dimensionality_reduction_pct": round(dim_reduction_pct, 2),
        "variance_note": (
            "UMAP preserves local and fuzzy topological manifold structures. "
            "Unlike linear PCA, UMAP does not define an orthogonal variance-retention metric."
        ),
    }
    
    return {
        "model": reducer,
        "transformed": X_umap,
        "umap_df": umap_df,
        "stats": stats,
    }

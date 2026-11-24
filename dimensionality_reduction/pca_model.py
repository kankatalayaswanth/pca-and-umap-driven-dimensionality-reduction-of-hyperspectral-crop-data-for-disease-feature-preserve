"""
Principal Component Analysis (PCA) Dimensionality Reduction
Linear orthogonal transformation maximizing global variance retention.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def run_pca_reduction(
    X: np.ndarray,
    feature_names: list,
    mode: str = "variance_threshold",  # "variance_threshold" or "manual"
    variance_threshold: float = 0.95,
    n_components_manual: int = 2,
) -> dict:
    """
    Fit and apply PCA on the preprocessed hyperspectral feature matrix.
    
    Parameters:
    - X: (N, D) preprocessed feature matrix
    - feature_names: original spectral band names
    - mode: 'variance_threshold' or 'manual'
    - variance_threshold: desired cumulative explained variance (e.g. 0.95)
    - n_components_manual: manual component count if mode == 'manual'
    
    Returns:
    dict with model, transformed coordinates, DataFrame, stats, and loadings.
    """
    n_samples, n_features = X.shape
    max_components = min(n_samples - 1, n_features)
    
    if max_components < 1:
        raise ValueError("Insufficient dimensions/samples for PCA.")
        
    if mode == "variance_threshold":
        # First fit a full-rank PCA to inspect cumulative variance profile
        full_pca = PCA(n_components=max_components)
        full_pca.fit(X)
        cum_var = np.cumsum(full_pca.explained_variance_ratio_)
        
        # Find index where cumulative variance reaches or exceeds target
        matching_indices = np.where(cum_var >= variance_threshold)[0]
        if len(matching_indices) > 0:
            k = int(matching_indices[0]) + 1
        else:
            k = max_components
            
        # Ensure at least 2 components for 2D visualization if possible
        if k < 2 and max_components >= 2:
            k = 2
            
    else:  # manual
        k = max(1, min(int(n_components_manual), max_components))
        
    # Fit final PCA with k components
    pca = PCA(n_components=k)
    X_pca = pca.fit_transform(X)
    
    exp_var = pca.explained_variance_ratio_
    cum_exp_var = np.cumsum(exp_var)
    total_var_retained = float(cum_exp_var[-1]) * 100.0
    dim_reduction_pct = float(1.0 - (k / n_features)) * 100.0
    
    # Create column names
    col_names = [f"PC{i+1}" for i in range(k)]
    pca_df = pd.DataFrame(X_pca, columns=col_names)
    
    # Calculate Component Loadings (eigenvectors * sqrt(eigenvalues))
    # Shape: (k, n_features)
    loadings = pd.DataFrame(
        pca.components_,
        columns=feature_names,
        index=[f"PC{i+1}" for i in range(k)]
    )
    
    # Identify top 5 contributing bands for PC1 and PC2
    top_pc1_bands = loadings.loc["PC1"].abs().nlargest(min(5, len(feature_names))).to_dict()
    top_pc2_bands = loadings.loc["PC2"].abs().nlargest(min(5, len(feature_names))).to_dict() if k >= 2 else {}
    
    stats = {
        "original_features": n_features,
        "n_components": k,
        "max_possible_components": max_components,
        "mode": mode,
        "target_threshold": variance_threshold if mode == "variance_threshold" else None,
        "explained_variance_ratio": [float(v) for v in exp_var],
        "cumulative_explained_variance": [float(v) for v in cum_exp_var],
        "total_variance_retained": round(total_var_retained, 2),
        "dimensionality_reduction_pct": round(dim_reduction_pct, 2),
        "singular_values": [float(s) for s in pca.singular_values_],
        "top_pc1_bands": top_pc1_bands,
        "top_pc2_bands": top_pc2_bands,
    }
    
    return {
        "model": pca,
        "transformed": X_pca,
        "pca_df": pca_df,
        "loadings": loadings,
        "stats": stats,
    }

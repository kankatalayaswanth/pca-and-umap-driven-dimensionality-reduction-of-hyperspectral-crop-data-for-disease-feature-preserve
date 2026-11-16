"""
Principal Component Analysis (PCA) Module for Hyperspectral Data.
Implements linear dimensionality reduction, variance analysis, scree plots,
component loadings, and reconstruction fidelity checks.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
from sklearn.decomposition import PCA


def fit_pca(
    X_train: np.ndarray,
    n_components: Optional[int] = None,
    variance_threshold: Optional[float] = None,
    random_state: int = 42
) -> Tuple[PCA, int]:
    """
    Fits PCA on training data.
    Allows specifying exact integer number of components, or a variance threshold (e.g. 0.95).
    
    Returns:
        pca_model (PCA): Fitted Scikit-learn PCA instance
        effective_n_components (int): Number of components retained
    """
    if variance_threshold is not None and 0.0 < variance_threshold <= 1.0:
        pca = PCA(n_components=variance_threshold, random_state=random_state)
    elif n_components is not None:
        max_comps = min(X_train.shape[0], X_train.shape[1])
        k = min(n_components, max_comps)
        pca = PCA(n_components=k, random_state=random_state)
    else:
        pca = PCA(n_components=min(10, X_train.shape[1]), random_state=random_state)
        
    pca.fit(X_train)
    return pca, pca.n_components_


def compute_scree_data(pca_model: PCA) -> pd.DataFrame:
    """
    Computes individual explained variance ratio and cumulative explained variance.
    """
    exp_var = pca_model.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)
    comp_names = [f"PC{i+1}" for i in range(len(exp_var))]
    
    return pd.DataFrame({
        'Component': comp_names,
        'Explained_Variance_Ratio': exp_var,
        'Cumulative_Variance_Ratio': cum_var,
        'Eigenvalue': pca_model.explained_variance_
    })


def get_components_for_thresholds(
    pca_full: PCA,
    thresholds: List[float] = [0.80, 0.90, 0.95, 0.99]
) -> Dict[str, int]:
    """Determines minimum components needed to reach given cumulative variance thresholds."""
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    res = {}
    for th in thresholds:
        indices = np.where(cum_var >= th)[0]
        count = int(indices[0] + 1) if len(indices) > 0 else len(cum_var)
        res[f"{int(th*100)}%"] = count
    return res


def compute_loadings(
    pca_model: PCA,
    feature_names: List[str],
    wavelengths: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Extracts the PCA loadings matrix (eigenvectors): components_ (n_components, n_features).
    Returns a DataFrame with features/wavelengths as rows and PC1..PCk as columns.
    """
    loadings = pca_model.components_.T  # Shape: (n_features, n_components)
    comp_cols = [f"PC{i+1}" for i in range(loadings.shape[1])]
    
    df = pd.DataFrame(loadings, columns=comp_cols, index=feature_names)
    if wavelengths is not None and len(wavelengths) == len(feature_names):
        df.insert(0, 'Wavelength_nm', wavelengths)
    return df


def get_top_contributing_bands(
    loadings_df: pd.DataFrame,
    pc_column: str = 'PC1',
    top_n: int = 10
) -> pd.DataFrame:
    """Identifies the bands with the highest absolute loading for a specific PC."""
    if pc_column not in loadings_df.columns:
        raise ValueError(f"Column '{pc_column}' not found in loadings matrix.")
        
    df_sorted = loadings_df.copy()
    df_sorted['Abs_Loading'] = df_sorted[pc_column].abs()
    top_bands = df_sorted.sort_values(by='Abs_Loading', ascending=False).head(top_n)
    
    cols = ['Wavelength_nm', pc_column, 'Abs_Loading'] if 'Wavelength_nm' in df_sorted.columns else [pc_column, 'Abs_Loading']
    return top_bands[cols].reset_index().rename(columns={'index': 'Band'})


def transform_pca(pca_model: PCA, X: np.ndarray) -> np.ndarray:
    """Transforms feature matrix into PCA projection space."""
    return pca_model.transform(X)


def compute_reconstruction_error(pca_model: PCA, X_scaled: np.ndarray) -> float:
    """
    Measures information loss by reconstructing X from reduced components:
    Reconstructed = X_pca * Components + Mean
    Returns Mean Squared Error (MSE).
    """
    X_proj = pca_model.transform(X_scaled)
    X_recon = pca_model.inverse_transform(X_proj)
    mse = float(np.mean((X_scaled - X_recon) ** 2))
    return mse


def save_pca_model(pca_model: PCA, path: str):
    """Saves fitted PCA model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pca_model, path)


def load_pca_model(path: str) -> PCA:
    """Loads fitted PCA model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"PCA model file not found: {path}")
    return joblib.load(path)

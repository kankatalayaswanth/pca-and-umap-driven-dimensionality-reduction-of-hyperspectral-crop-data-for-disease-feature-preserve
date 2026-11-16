"""
Tests for Dimensionality Reduction Modules (PCA and UMAP).
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.data_loader import load_dataset
from src.preprocessing import build_preprocessing_pipeline, fit_transform_pipeline
from src.pca import (
    fit_pca,
    compute_scree_data,
    get_components_for_thresholds,
    compute_loadings,
    get_top_contributing_bands,
    compute_reconstruction_error
)
from src.umap_reduction import fit_umap, get_umap_coordinates_df


@pytest.fixture(scope="module")
def scaled_data():
    X, y, _, bands, wls = load_dataset('potato_dai70_vnir')
    pipe = build_preprocessing_pipeline('standard')
    X_sc, _, act_bands = fit_transform_pipeline(pipe, X, band_names=bands)
    return X, y, X_sc, act_bands, wls


def test_pca_variance_and_scree(scaled_data):
    X, y, X_sc, bands, wls = scaled_data
    pca, k = fit_pca(X_sc, n_components=10)
    assert k == 10
    
    scree = compute_scree_data(pca)
    assert len(scree) == 10
    assert scree['Explained_Variance_Ratio'].sum() <= 1.0
    assert (scree['Cumulative_Variance_Ratio'].diff().dropna() >= 0).all()  # Monotonic
    
    thresh_map = get_components_for_thresholds(pca, [0.80, 0.90])
    assert '80%' in thresh_map
    assert '90%' in thresh_map
    assert thresh_map['80%'] <= thresh_map['90%']


def test_pca_loadings_and_reconstruction(scaled_data):
    X, y, X_sc, bands, wls = scaled_data
    pca, k = fit_pca(X_sc, n_components=5)
    loadings = compute_loadings(pca, bands, wls)
    
    assert loadings.shape == (160, 6)  # Wavelength + PC1..PC5
    top_bands = get_top_contributing_bands(loadings, 'PC1', top_n=5)
    assert len(top_bands) == 5
    assert 'Abs_Loading' in top_bands.columns
    
    # Reconstruction MSE decreases with higher components
    mse_5 = compute_reconstruction_error(pca, X_sc)
    pca_20, _ = fit_pca(X_sc, n_components=20)
    mse_20 = compute_reconstruction_error(pca_20, X_sc)
    assert mse_20 < mse_5


def test_umap_embedding(scaled_data):
    X, y, X_sc, bands, wls = scaled_data
    # Fast UMAP test with small components
    reducer, emb = fit_umap(X_sc, n_components=2, n_neighbors=5, min_dist=0.1)
    assert emb.shape == (70, 2)
    assert not np.isnan(emb).any()
    
    coord_df = get_umap_coordinates_df(emb, target_series=y)
    assert 'UMAP1' in coord_df.columns
    assert 'UMAP2' in coord_df.columns
    assert 'Target' in coord_df.columns

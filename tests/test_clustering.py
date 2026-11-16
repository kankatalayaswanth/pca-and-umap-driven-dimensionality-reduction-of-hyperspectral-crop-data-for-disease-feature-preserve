"""
Tests for Clustering and Evaluation Modules.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_loader import load_dataset
from src.preprocessing import build_preprocessing_pipeline, fit_transform_pipeline
from src.pca import fit_pca, transform_pca
from src.umap_reduction import fit_umap
from src.clustering import run_kmeans, run_agglomerative, run_dbscan, compute_cluster_centroids_original_space
from src.evaluation import evaluate_clustering, build_contingency_matrix, compute_sample_silhouette_values


@pytest.fixture(scope="module")
def prepared_representations():
    X, y, _, bands, wls = load_dataset('potato_dai70_vnir')
    pipe = build_preprocessing_pipeline('standard')
    X_scaled, _, _ = fit_transform_pipeline(pipe, X)
    
    # 1. Raw scaled features
    X_raw = X_scaled
    
    # 2. PCA features (5 components)
    pca, _ = fit_pca(X_scaled, n_components=5)
    X_pca = transform_pca(pca, X_scaled)
    
    # 3. UMAP features (2 components)
    _, X_umap = fit_umap(X_scaled, n_components=2, n_neighbors=10, min_dist=0.1)
    
    return {
        'raw_df': X,
        'y': y,
        'wls': wls,
        'X_raw': X_raw,
        'X_pca': X_pca,
        'X_umap': X_umap
    }


def test_kmeans_on_all_representations(prepared_representations):
    data = prepared_representations
    y = data['y']
    
    for rep_name in ['X_raw', 'X_pca', 'X_umap']:
        feat = data[rep_name]
        km, labels = run_kmeans(feat, n_clusters=4, random_state=42)
        assert len(labels) == 70
        assert len(np.unique(labels)) == 4
        
        eval_res = evaluate_clustering(feat, labels, y_true=y)
        assert eval_res['is_evaluable'] is True
        im = eval_res['internal_metrics']
        assert -1.0 <= im['silhouette_score'] <= 1.0
        assert im['calinski_harabasz_score'] > 0
        assert im['davies_bouldin_score'] > 0
        
        em = eval_res['external_metrics']
        assert 'adjusted_rand_index' in em
        assert 'normalized_mutual_info' in em


def test_agglomerative_clustering(prepared_representations):
    data = prepared_representations
    feat = data['X_pca']
    model, labels = run_agglomerative(feat, n_clusters=3, linkage='ward')
    assert len(labels) == 70
    assert len(np.unique(labels)) == 3


def test_dbscan_clustering(prepared_representations):
    data = prepared_representations
    feat = data['X_pca']
    model, labels = run_dbscan(feat, eps=2.0, min_samples=3)
    assert len(labels) == 70
    # DBSCAN may assign noise (-1)
    assert isinstance(labels, np.ndarray)


def test_contingency_and_centroids(prepared_representations):
    data = prepared_representations
    feat = data['X_pca']
    y = data['y']
    km, labels = run_kmeans(feat, n_clusters=4, random_state=42)
    
    ct = build_contingency_matrix(labels, y)
    assert not ct.empty
    assert ct.shape[0] == 4  # 4 clusters
    
    centroids = compute_cluster_centroids_original_space(data['raw_df'].values, labels)
    assert len(centroids) == 4
    for c_id, vec in centroids.items():
        assert len(vec) == 160  # Length of original spectral bands

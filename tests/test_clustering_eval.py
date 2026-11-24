"""
Unit tests for independent K-Means clustering and evaluation metrics.
"""

import pytest
import numpy as np
import pandas as pd
from clustering.kmeans_model import run_tri_representation_clustering
from evaluation.metrics import compute_internal_metrics, compute_external_metrics
from evaluation.comparator import generate_comparison_benchmark


def test_independent_kmeans():
    np.random.seed(42)
    X_raw = np.random.randn(60, 20)
    X_pca = np.random.randn(60, 5)
    X_umap = np.random.randn(60, 2)
    
    res = run_tri_representation_clustering(X_raw, X_pca, X_umap, n_clusters=3, random_state=42)
    assert len(res["raw"]["labels"]) == 60
    assert len(res["pca"]["labels"]) == 60
    assert len(res["umap"]["labels"]) == 60
    assert len(np.unique(res["raw"]["labels"])) == 3


def test_evaluation_metrics():
    np.random.seed(42)
    X = np.random.randn(40, 4)
    labels = np.array([0] * 20 + [1] * 20)
    
    internal = compute_internal_metrics(X, labels)
    assert internal["silhouette"] is not None
    assert internal["davies_bouldin"] is not None
    assert internal["calinski_harabasz"] is not None
    
    # Perfect external match
    external_perf = compute_external_metrics(labels, labels)
    assert external_perf["ari"] == 1.0
    assert external_perf["nmi"] == 1.0
    
    # None ground truth
    external_none = compute_external_metrics(None, labels)
    assert external_none["has_ground_truth"] is False
    assert external_none["ari"] is None

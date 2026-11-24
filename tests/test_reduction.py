"""
Unit tests for PCA and UMAP dimensionality reduction modules.
"""

import pytest
import numpy as np
from dimensionality_reduction.pca_model import run_pca_reduction
from dimensionality_reduction.umap_model import run_umap_reduction


def test_pca_reduction():
    np.random.seed(42)
    X = np.random.randn(50, 10)
    feat_names = [f"band_{i}" for i in range(10)]
    
    # Variance threshold mode
    res_thresh = run_pca_reduction(X, feat_names, mode="variance_threshold", variance_threshold=0.90)
    assert res_thresh["stats"]["total_variance_retained"] >= 90.0
    assert res_thresh["transformed"].shape[0] == 50
    assert res_thresh["transformed"].shape[1] >= 2
    
    # Manual mode
    res_man = run_pca_reduction(X, feat_names, mode="manual", n_components_manual=3)
    assert res_man["stats"]["n_components"] == 3
    assert res_man["transformed"].shape == (50, 3)


def test_umap_reduction():
    np.random.seed(42)
    X = np.random.randn(40, 8)
    res = run_umap_reduction(X, n_neighbors=10, min_dist=0.1, n_components=2, random_state=42)
    assert res["transformed"].shape == (40, 2)
    assert res["stats"]["n_components"] == 2

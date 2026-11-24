"""
Clustering Evaluation Metrics: Internal and External Validation
Course: INT396 - Unsupervised Learning
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def compute_internal_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute unsupervised internal validation metrics (ground-truth independent).
    - Silhouette Score: [-1, +1], higher is better
    - Davies-Bouldin Index: [0, inf), lower is better
    - Calinski-Harabasz Index: [0, inf), higher is better
    """
    unique_labels = np.unique(labels)
    n_samples = X.shape[0]
    
    # Validation: need at least 2 distinct clusters and at least 3 samples
    if len(unique_labels) < 2 or len(unique_labels) >= n_samples:
        return {
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
        }
        
    try:
        # If dataset is large, sample for silhouette computation efficiency
        if n_samples > 3000:
            sil = float(silhouette_score(X, labels, sample_size=1500, random_state=42))
        else:
            sil = float(silhouette_score(X, labels))
    except Exception:
        sil = None
        
    try:
        db = float(davies_bouldin_score(X, labels))
    except Exception:
        db = None
        
    try:
        ch = float(calinski_harabasz_score(X, labels))
    except Exception:
        ch = None
        
    return {
        "silhouette": round(sil, 4) if sil is not None else None,
        "davies_bouldin": round(db, 4) if db is not None else None,
        "calinski_harabasz": round(ch, 2) if ch is not None else None,
    }


def compute_external_metrics(y_true, y_pred) -> dict:
    """
    Compute external validation metrics comparing cluster partitions to genuine ground truth.
    - Adjusted Rand Index (ARI): [-1, +1], higher is better
    - Normalized Mutual Information (NMI): [0, 1], higher is better
    - Contingency Matrix (Cross-tabulation)
    """
    if y_true is None:
        return {
            "has_ground_truth": False,
            "ari": None,
            "nmi": None,
            "contingency_matrix": None,
            "contingency_df": None,
        }
        
    # Convert to standard arrays / series
    y_true_clean = pd.Series(y_true).astype(str).values
    y_pred_clean = np.array(y_pred)
    
    try:
        ari = float(adjusted_rand_score(y_true_clean, y_pred_clean))
    except Exception:
        ari = None
        
    try:
        nmi = float(normalized_mutual_info_score(y_true_clean, y_pred_clean))
    except Exception:
        nmi = None
        
    # Build Contingency Cross-tabulation
    try:
        contingency_df = pd.crosstab(
            pd.Series(y_pred_clean, name="Cluster"),
            pd.Series(y_true_clean, name="Ground Truth Class")
        )
    except Exception:
        contingency_df = None
        
    return {
        "has_ground_truth": True,
        "ari": round(ari, 4) if ari is not None else None,
        "nmi": round(nmi, 4) if nmi is not None else None,
        "contingency_df": contingency_df,
    }


def evaluate_representation(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    y_true = None,
    representation_name: str = "Feature Space",
) -> dict:
    """
    Run comprehensive evaluation (both internal and external) for a given feature space.
    """
    internal = compute_internal_metrics(X, cluster_labels)
    external = compute_external_metrics(y_true, cluster_labels)
    
    return {
        "representation": representation_name,
        **internal,
        **external,
    }

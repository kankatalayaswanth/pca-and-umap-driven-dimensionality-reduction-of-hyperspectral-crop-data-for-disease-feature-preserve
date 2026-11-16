"""
Clustering Evaluation Module for Hyperspectral Unsupervised Learning.
Implements:
1. Internal Unsupervised Metrics:
   - Silhouette Score
   - Calinski-Harabasz Index
   - Davies-Bouldin Index
2. External Ground-Truth Validation Metrics:
   - Adjusted Rand Index (ARI)
   - Normalized Mutual Information (NMI)
   - Homogeneity, Completeness, V-Measure
3. Error and Cluster Quality Analysis:
   - Per-sample Silhouette values
   - Contingency / Cross-tabulation matrices
   - Cluster size distributions and noise counts
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure
)


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    y_true: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    Evaluates clustering performance using both internal metrics and
    optional external ground-truth metrics.
    
    Handles noise points (-1) correctly by excluding them from internal distance metrics.
    """
    results: Dict[str, Any] = {
        'is_evaluable': False,
        'n_clusters': 0,
        'noise_count': 0,
        'internal_metrics': {},
        'external_metrics': {},
        'notes': []
    }
    
    unique_labels = np.unique(labels)
    n_noise = int(np.sum(labels == -1))
    valid_mask = (labels != -1)
    valid_labels = labels[valid_mask]
    n_valid_clusters = len(np.unique(valid_labels))
    
    results['n_clusters'] = int(n_valid_clusters)
    results['noise_count'] = n_noise
    
    # Internal metrics require at least 2 distinct clusters and at least (n_clusters + 1) samples
    if n_valid_clusters < 2:
        results['notes'].append(f"Cannot compute internal metrics: requires at least 2 valid clusters, found {n_valid_clusters}.")
        return results
        
    X_valid = X[valid_mask]
    if len(X_valid) <= n_valid_clusters:
        results['notes'].append(f"Cannot compute internal metrics: number of valid samples ({len(X_valid)}) must exceed cluster count ({n_valid_clusters}).")
        return results
        
    # 1. Internal Unsupervised Metrics
    try:
        sil = float(silhouette_score(X_valid, valid_labels))
    except Exception as e:
        sil = float('nan')
        results['notes'].append(f"Silhouette calculation error: {str(e)}")
        
    try:
        ch = float(calinski_harabasz_score(X_valid, valid_labels))
    except Exception as e:
        ch = float('nan')
        results['notes'].append(f"Calinski-Harabasz calculation error: {str(e)}")
        
    try:
        db = float(davies_bouldin_score(X_valid, valid_labels))
    except Exception as e:
        db = float('nan')
        results['notes'].append(f"Davies-Bouldin calculation error: {str(e)}")
        
    results['internal_metrics'] = {
        'silhouette_score': sil,
        'calinski_harabasz_score': ch,
        'davies_bouldin_score': db
    }
    results['is_evaluable'] = True
    
    # 2. External Metrics (when ground-truth disease labels exist)
    if y_true is not None and len(y_true) == len(labels):
        try:
            # Map string classes to integer codes if necessary
            y_arr = np.array(y_true)
            ari = float(adjusted_rand_score(y_arr, labels))
            nmi = float(normalized_mutual_info_score(y_arr, labels))
            homo, comp, v_meas = homogeneity_completeness_v_measure(y_arr, labels)
            
            results['external_metrics'] = {
                'adjusted_rand_index': ari,
                'normalized_mutual_info': nmi,
                'homogeneity': float(homo),
                'completeness': float(comp),
                'v_measure': float(v_meas)
            }
        except Exception as e:
            results['notes'].append(f"External metric error: {str(e)}")
            
    return results


def compute_sample_silhouette_values(X: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Computes individual silhouette score for every sample for cluster thickness plotting."""
    valid_mask = (labels != -1)
    if len(np.unique(labels[valid_mask])) < 2:
        return pd.DataFrame()
        
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]
    sample_sil = silhouette_samples(X_valid, labels_valid)
    
    df = pd.DataFrame({
        'Sample_Index': np.where(valid_mask)[0],
        'Cluster': labels_valid,
        'Silhouette_Value': sample_sil
    })
    return df


def build_contingency_matrix(labels: np.ndarray, y_true: pd.Series) -> pd.DataFrame:
    """
    Constructs cross-tabulation table comparing cluster assignments
    with ground-truth crop disease classes.
    """
    if y_true is None or len(labels) != len(y_true):
        return pd.DataFrame()
        
    ct = pd.crosstab(
        pd.Series(labels, name="Assigned Cluster"),
        pd.Series(y_true.values, name="True Disease Class")
    )
    return ct


def get_metric_descriptions() -> Dict[str, Dict[str, str]]:
    """Returns academic definitions and interpretations for all metrics."""
    return {
        'Silhouette Score': {
            'range': '[-1.0, 1.0]',
            'ideal': 'Closer to +1.0',
            'meaning': 'Measures how similar a sample is to its own cluster compared to other clusters. Values near 1 indicate dense, well-separated clusters; values near 0 indicate overlapping clusters; negative values indicate misassigned points.'
        },
        'Calinski-Harabasz Index': {
            'range': '[0.0, ∞)',
            'ideal': 'Higher is better',
            'meaning': 'Variance Ratio Criterion: ratio of the between-cluster dispersion matrix to the within-cluster dispersion matrix. Higher values signify clusters that are compact and far apart.'
        },
        'Davies-Bouldin Index': {
            'range': '[0.0, ∞)',
            'ideal': 'Lower is better (0 is optimal)',
            'meaning': 'Average similarity ratio between each cluster and its most similar one. Lower values indicate better clustering with minimal inter-cluster similarity.'
        },
        'Adjusted Rand Index (ARI)': {
            'range': '[-1.0, 1.0]',
            'ideal': '+1.0 indicates perfect agreement',
            'meaning': 'External metric comparing cluster groupings against ground-truth disease classes, adjusted for random chance. Expected value for random clustering is 0.0.'
        },
        'Normalized Mutual Information (NMI)': {
            'range': '[0.0, 1.0]',
            'ideal': '+1.0 indicates perfect correlation',
            'meaning': 'Information-theoretic score measuring the shared information between assigned clusters and true disease classes, normalized by cluster entropies.'
        }
    }

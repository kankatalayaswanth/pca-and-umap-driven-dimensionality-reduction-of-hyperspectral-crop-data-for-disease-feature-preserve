"""
Hyperparameter Optimization Module for Hyperspectral Unsupervised Learning.
Implements reproducible parameter grid searches and metric-driven selection:
1. K-Means (Elbow method & Silhouette optimization over k)
2. UMAP (n_neighbors, min_dist, metric grid search)
3. DBSCAN (eps, min_samples search)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.cluster import KMeans, DBSCAN
from src.evaluation import evaluate_clustering


def optimize_kmeans_k(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 8,
    y_true: Optional[pd.Series] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, int]:
    """
    Evaluates K-Means clustering over a range of k values.
    Returns:
        metrics_df: Table of k, inertia, silhouette, CH, DB, and ARI/NMI (if y_true given)
        best_k: Value of k that maximizes the Silhouette Score
    """
    records = []
    effective_k_max = min(k_max, max(2, len(X) - 1))
    
    for k in range(k_min, effective_k_max + 1):
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        eval_res = evaluate_clustering(X, labels, y_true=y_true)
        
        im = eval_res.get('internal_metrics', {})
        em = eval_res.get('external_metrics', {})
        
        row = {
            'k': k,
            'Inertia': float(km.inertia_),
            'Silhouette': im.get('silhouette_score', np.nan),
            'Calinski_Harabasz': im.get('calinski_harabasz_score', np.nan),
            'Davies_Bouldin': im.get('davies_bouldin_score', np.nan)
        }
        if y_true is not None:
            row['ARI'] = em.get('adjusted_rand_index', np.nan)
            row['NMI'] = em.get('normalized_mutual_info', np.nan)
            
        records.append(row)
        
    df = pd.DataFrame(records)
    
    # Best k based on highest Silhouette Score
    valid_sil = df.dropna(subset=['Silhouette'])
    if not valid_sil.empty:
        best_k = int(valid_sil.loc[valid_sil['Silhouette'].idxmax()]['k'])
    else:
        best_k = k_min
        
    return df, best_k


def optimize_umap_parameters(
    X: np.ndarray,
    n_neighbors_list: List[int] = [5, 10, 15],
    min_dist_list: List[float] = [0.0, 0.1, 0.25],
    metrics_list: List[str] = ['euclidean', 'cosine'],
    n_clusters: int = 4,
    y_true: Optional[pd.Series] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs grid search across UMAP hyperparameters by evaluating the quality of
    downstream K-Means clustering on the resulting 2D UMAP embedding.
    """
    from src.umap_reduction import fit_umap
    records = []
    best_sil = -1.0
    best_config = {}
    
    for metric in metrics_list:
        for n_nbrs in n_neighbors_list:
            if n_nbrs >= len(X):
                continue
            for md in min_dist_list:
                try:
                    reducer, emb = fit_umap(
                        X,
                        n_components=2,
                        n_neighbors=n_nbrs,
                        min_dist=md,
                        metric=metric,
                        random_state=random_state
                    )
                    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
                    labels = km.fit_predict(emb)
                    eval_res = evaluate_clustering(emb, labels, y_true=y_true)
                    
                    im = eval_res.get('internal_metrics', {})
                    em = eval_res.get('external_metrics', {})
                    sil = im.get('silhouette_score', np.nan)
                    
                    row = {
                        'n_neighbors': n_nbrs,
                        'min_dist': md,
                        'metric': metric,
                        'Silhouette': sil,
                        'Calinski_Harabasz': im.get('calinski_harabasz_score', np.nan),
                        'Davies_Bouldin': im.get('davies_bouldin_score', np.nan)
                    }
                    if y_true is not None:
                        row['ARI'] = em.get('adjusted_rand_index', np.nan)
                        row['NMI'] = em.get('normalized_mutual_info', np.nan)
                        
                    records.append(row)
                    
                    if not np.isnan(sil) and sil > best_sil:
                        best_sil = sil
                        best_config = {
                            'n_neighbors': n_nbrs,
                            'min_dist': md,
                            'metric': metric,
                            'silhouette': sil
                        }
                except Exception:
                    continue
                    
    df = pd.DataFrame(records)
    if not df.empty and 'Silhouette' in df.columns:
        df = df.sort_values(by='Silhouette', ascending=False).reset_index(drop=True)
    return df, best_config


def optimize_dbscan_parameters(
    X: np.ndarray,
    eps_list: List[float] = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
    min_samples_list: List[int] = [3, 5, 8],
    y_true: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates DBSCAN parameter combinations over eps and min_samples.
    """
    records = []
    best_config = {}
    best_sil = -1.0
    
    for eps in eps_list:
        for ms in min_samples_list:
            db = DBSCAN(eps=eps, min_samples=ms)
            labels = db.fit_predict(X)
            eval_res = evaluate_clustering(X, labels, y_true=y_true)
            
            im = eval_res.get('internal_metrics', {})
            em = eval_res.get('external_metrics', {})
            n_clusters = eval_res.get('n_clusters', 0)
            noise_cnt = eval_res.get('noise_count', 0)
            noise_pct = (noise_cnt / len(X)) * 100
            
            sil = im.get('silhouette_score', np.nan)
            
            row = {
                'eps': eps,
                'min_samples': ms,
                'Clusters_Found': n_clusters,
                'Noise_Count': noise_cnt,
                'Noise_Percent': round(noise_pct, 1),
                'Silhouette': sil,
                'Davies_Bouldin': im.get('davies_bouldin_score', np.nan)
            }
            if y_true is not None:
                row['ARI'] = em.get('adjusted_rand_index', np.nan)
                row['NMI'] = em.get('normalized_mutual_info', np.nan)
                
            records.append(row)
            if n_clusters >= 2 and not np.isnan(sil) and sil > best_sil:
                best_sil = sil
                best_config = {'eps': eps, 'min_samples': ms, 'clusters': n_clusters, 'silhouette': sil}
                
    df = pd.DataFrame(records)
    return df, best_config

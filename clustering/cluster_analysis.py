"""
Cluster Number Analysis: Elbow Method & Silhouette Score Profiling
Course: INT396 - Unsupervised Learning
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def compute_k_analysis(
    X: np.ndarray,
    name: str = "Feature Space",
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42,
) -> dict:
    """
    Compute Inertia (WCSS) and Silhouette Score across candidate K values for a single representation.
    """
    n_samples = X.shape[0]
    safe_k_max = min(k_max, n_samples - 1)
    
    k_values = list(range(k_min, safe_k_max + 1))
    inertias = []
    silhouettes = []
    
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=5)
        labels = km.fit_predict(X)
        inertias.append(float(km.inertia_))
        
        # Silhouette requires at least 2 distinct clusters
        if len(np.unique(labels)) > 1:
            # Subsample for silhouette computation efficiency to prevent UI lag on high dimensions
            if n_samples > 800:
                sil = float(silhouette_score(X, labels, sample_size=min(800, n_samples), random_state=random_state))
            else:
                sil = float(silhouette_score(X, labels))
        else:
            sil = 0.0
            
        silhouettes.append(sil)
        
    df_results = pd.DataFrame({
        "K": k_values,
        "Inertia": inertias,
        "Silhouette": silhouettes,
        "Representation": name,
    })
    
    # Identify optimal K by silhouette
    best_k_idx = int(np.argmax(silhouettes))
    best_k = k_values[best_k_idx]
    best_sil = silhouettes[best_k_idx]
    
    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "best_k": best_k,
        "best_silhouette": round(best_sil, 4),
        "results_df": df_results,
    }


def compute_multi_representation_k_analysis(
    X_raw: np.ndarray,
    X_pca: np.ndarray,
    X_umap: np.ndarray,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42,
) -> dict:
    """
    Compute Elbow and Silhouette metrics across K for Raw, PCA, and UMAP.
    """
    raw_analysis = compute_k_analysis(X_raw, name="Raw Features", k_min=k_min, k_max=k_max, random_state=random_state)
    pca_analysis = compute_k_analysis(X_pca, name="PCA Features", k_min=k_min, k_max=k_max, random_state=random_state)
    umap_analysis = compute_k_analysis(X_umap, name="UMAP Features", k_min=k_min, k_max=k_max, random_state=random_state)
    
    combined_df = pd.concat([
        raw_analysis["results_df"],
        pca_analysis["results_df"],
        umap_analysis["results_df"],
    ], ignore_index=True)
    
    return {
        "raw": raw_analysis,
        "pca": pca_analysis,
        "umap": umap_analysis,
        "combined_df": combined_df,
    }

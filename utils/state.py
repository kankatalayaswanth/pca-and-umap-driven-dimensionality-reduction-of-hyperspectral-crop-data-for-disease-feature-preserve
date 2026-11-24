"""
Session State Management for Antigravity Hyperspectral ML Pipeline
Preserves experiment pipeline state across pages without loss.
"""

import streamlit as st
import pandas as pd
import numpy as np
from config.settings import PAGES


def init_session_state():
    """Initialize all persistent state variables if they do not exist."""
    defaults = {
        # Current navigation page (Default to Step 0: project overview dashboard)
        "current_page": "dashboard",
        
        # Dataset storage
        "dataset_loaded": False,
        "raw_file_name": None,
        "raw_df": None,
        "feature_cols": [],
        "label_col": None,
        "id_cols": [],
        "wavelength_meta": {},
        
        # Preprocessing state
        "preprocessing_done": False,
        "preprocessing_config": {
            "imputation_strategy": "mean",
            "scaling_method": "standard",
            "remove_constant": True,
            "remove_duplicates": True,
        },
        "preprocessed_df": None,
        "X_clean": None,           # Preprocessed features as numpy array
        "X_clean_df": None,        # Preprocessed features as DataFrame
        "y_clean": None,           # Ground truth series if label_col exists
        "preprocessing_stats": {},
        
        # PCA state
        "pca_done": False,
        "pca_config": {
            "mode": "variance_threshold",  # "variance_threshold" or "manual"
            "variance_threshold": 0.95,
            "n_components": 2,
        },
        "pca_model": None,
        "pca_transformed": None,    # Transformed coordinates (N, k)
        "pca_df": None,             # DataFrame of PC1, PC2...
        "pca_loadings": None,       # Eigenvector loading matrix
        "pca_stats": {},
        
        # UMAP state
        "umap_done": False,
        "umap_config": {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "metric": "euclidean",
            "n_components": 2,
            "random_state": 42,
        },
        "umap_model": None,
        "umap_transformed": None,   # Transformed coordinates (N, k)
        "umap_df": None,            # DataFrame of UMAP1, UMAP2...
        "umap_stats": {},
        
        # Clustering state
        "clustering_done": False,
        "kmeans_config": {
            "n_clusters": 3,
            "random_state": 42,
            "n_init": 10,
        },
        "kmeans_raw_labels": None,
        "kmeans_pca_labels": None,
        "kmeans_umap_labels": None,
        "kmeans_raw_model": None,
        "kmeans_pca_model": None,
        "kmeans_umap_model": None,
        "kmeans_tri_results": None,
        "elbow_analysis_done": False,
        "elbow_results": None,
        
        # Evaluation & Benchmark
        "evaluation_done": False,
        "comparison_table": None,
        "detailed_metrics": {},
        "dynamic_interpretations": [],
        
        # Final milestone flags
        "visualization_viewed": False,
        "results_viewed": False,
    }

    # Clean up any legacy radio keys from prior sessions
    st.session_state.pop("nav_radio", None)

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def is_dataset_ready() -> bool:
    """Return True if dataset is uploaded and feature columns are selected."""
    return (
        st.session_state.get("dataset_loaded", False)
        and st.session_state.get("raw_df") is not None
        and len(st.session_state.get("feature_cols", [])) >= 2
    )


def is_preprocessing_ready() -> bool:
    """Return True if preprocessing has been successfully executed."""
    return (
        is_dataset_ready()
        and st.session_state.get("preprocessing_done", False)
        and st.session_state.get("X_clean") is not None
    )


def is_pca_ready() -> bool:
    """Return True if PCA dimensionality reduction is completed."""
    return (
        is_preprocessing_ready()
        and st.session_state.get("pca_done", False)
        and st.session_state.get("pca_transformed") is not None
    )


def is_umap_ready() -> bool:
    """Return True if UMAP dimensionality reduction is completed."""
    return (
        is_preprocessing_ready()
        and st.session_state.get("umap_done", False)
        and st.session_state.get("umap_transformed") is not None
    )


def is_clustering_ready() -> bool:
    """Return True if K-Means has been run on Raw, PCA, and UMAP."""
    return (
        is_pca_ready()
        and is_umap_ready()
        and st.session_state.get("clustering_done", False)
        and st.session_state.get("kmeans_raw_labels") is not None
        and st.session_state.get("kmeans_pca_labels") is not None
        and st.session_state.get("kmeans_umap_labels") is not None
    )


def is_evaluation_ready() -> bool:
    """Return True if evaluation benchmark table has been generated."""
    return (
        is_clustering_ready()
        and st.session_state.get("evaluation_done", False)
        and st.session_state.get("comparison_table") is not None
    )


def is_visualization_ready() -> bool:
    """Return True if user has inspected visualizations and clustering is completed."""
    return is_clustering_ready() and st.session_state.get("visualization_viewed", False)


def is_results_ready() -> bool:
    """Return True if all pipeline stages and final results report are compiled."""
    return is_evaluation_ready() and st.session_state.get("results_viewed", False)


def get_pipeline_stage_status() -> list:
    """
    Return comprehensive status across all 8 active workflow milestones (Steps 1 to 8).
    Ensures that 100% completion is accurately reached only on Step 8 (Results).
    """
    raw_df = st.session_state.get("raw_df")
    prep_stats = st.session_state.get("preprocessing_stats", {})
    pca_stats = st.session_state.get("pca_stats", {})
    umap_stats = st.session_state.get("umap_stats", {})
    kmeans_cfg = st.session_state.get("kmeans_config", {})

    stages = [
        {
            "id": "dataset",
            "name": "Dataset",
            "icon": "📁",
            "ready": is_dataset_ready(),
            "badge": f"{len(st.session_state.get('feature_cols', []))} bands" if is_dataset_ready() else ("Loaded" if st.session_state.get("dataset_loaded", False) else "Pending"),
        },
        {
            "id": "preprocessing",
            "name": "Preprocess",
            "icon": "⚙️",
            "ready": is_preprocessing_ready(),
            "badge": prep_stats.get("scaling_method", "").split()[0] if is_preprocessing_ready() else "Pending",
        },
        {
            "id": "pca",
            "name": "PCA",
            "icon": "📉",
            "ready": is_pca_ready(),
            "badge": f"{pca_stats.get('n_components')} PCs" if is_pca_ready() else "Pending",
        },
        {
            "id": "umap",
            "name": "UMAP",
            "icon": "🌐",
            "ready": is_umap_ready(),
            "badge": f"{umap_stats.get('n_components')}D Manifold" if is_umap_ready() else "Pending",
        },
        {
            "id": "clustering",
            "name": "Clustering",
            "icon": "🎯",
            "ready": is_clustering_ready(),
            "badge": f"K={kmeans_cfg.get('n_clusters', 3)}" if is_clustering_ready() else "Pending",
        },
        {
            "id": "comparison",
            "name": "Benchmark",
            "icon": "⚖️",
            "ready": is_evaluation_ready(),
            "badge": "Evaluated" if is_evaluation_ready() else "Pending",
        },
        {
            "id": "visualization",
            "name": "Visualizations",
            "icon": "🔍",
            "ready": is_visualization_ready(),
            "badge": "Inspected" if is_visualization_ready() else "Pending",
        },
        {
            "id": "results",
            "name": "Results",
            "icon": "📑",
            "ready": is_results_ready(),
            "badge": "100% Done" if is_results_ready() else "Pending",
        },
    ]
    return stages


def get_pipeline_progress_pct() -> int:
    """Calculate exact pipeline completion percentage (0% to 100%)."""
    stages = get_pipeline_stage_status()
    completed = sum(1 for s in stages if s["ready"])
    return int((completed / len(stages)) * 100)


def get_stage_readiness(stage_id: str) -> bool:
    """Check whether a specific page stage is completed."""
    if stage_id == "dashboard":
        return True
    if stage_id == "dataset":
        return is_dataset_ready()
    if stage_id == "preprocessing":
        return is_preprocessing_ready()
    if stage_id == "pca":
        return is_pca_ready()
    if stage_id == "umap":
        return is_umap_ready()
    if stage_id == "clustering":
        return is_clustering_ready()
    if stage_id == "comparison":
        return is_evaluation_ready()
    if stage_id == "visualization":
        return is_visualization_ready()
    if stage_id == "results":
        return is_results_ready()
    return False


def reset_downstream_from_dataset():
    """Reset all downstream stages when a new dataset is uploaded or feature columns change."""
    st.session_state["preprocessing_done"] = False
    st.session_state["preprocessed_df"] = None
    st.session_state["X_clean"] = None
    st.session_state["X_clean_df"] = None
    st.session_state["y_clean"] = None
    st.session_state["preprocessing_stats"] = {}
    st.session_state.pop("cached_spec_fig", None)
    st.session_state.pop("cached_corr_fig", None)
    reset_downstream_from_preprocessing()


def reset_downstream_from_preprocessing():
    """Reset PCA, UMAP, clustering, evaluation and results when preprocessing changes."""
    st.session_state["pca_done"] = False
    st.session_state["pca_model"] = None
    st.session_state["pca_transformed"] = None
    st.session_state["pca_df"] = None
    st.session_state["pca_loadings"] = None
    st.session_state["pca_stats"] = {}
    
    st.session_state["umap_done"] = False
    st.session_state["umap_model"] = None
    st.session_state["umap_transformed"] = None
    st.session_state["umap_df"] = None
    st.session_state["umap_stats"] = {}
    
    reset_downstream_from_reduction()


def reset_downstream_from_reduction():
    """Reset clustering and evaluation when PCA or UMAP is re-run."""
    st.session_state["clustering_done"] = False
    st.session_state["kmeans_raw_labels"] = None
    st.session_state["kmeans_pca_labels"] = None
    st.session_state["kmeans_umap_labels"] = None
    st.session_state["kmeans_raw_model"] = None
    st.session_state["kmeans_pca_model"] = None
    st.session_state["kmeans_umap_model"] = None
    st.session_state["kmeans_tri_results"] = None
    st.session_state["elbow_analysis_done"] = False
    st.session_state["elbow_results"] = None
    
    st.session_state["evaluation_done"] = False
    st.session_state["comparison_table"] = None
    st.session_state["detailed_metrics"] = {}
    st.session_state["dynamic_interpretations"] = []
    st.session_state["visualization_viewed"] = False
    st.session_state["results_viewed"] = False


def reset_full_experiment():
    """Completely wipe all uploaded dataset and models to start a fresh experiment."""
    st.session_state["dataset_loaded"] = False
    st.session_state["raw_file_name"] = None
    st.session_state["raw_df"] = None
    st.session_state["feature_cols"] = []
    st.session_state["label_col"] = None
    st.session_state["id_cols"] = []
    st.session_state["wavelength_meta"] = {}
    st.session_state["visualization_viewed"] = False
    st.session_state["results_viewed"] = False
    st.session_state.pop("cached_spec_fig", None)
    st.session_state.pop("cached_corr_fig", None)
    reset_downstream_from_dataset()
    set_page("dataset")


PAGE_ORDER = ["dashboard", "dataset", "preprocessing", "pca", "umap", "clustering", "comparison", "visualization", "results"]


def get_next_page(current_page: str) -> str:
    """Get the next sequential page in the workflow."""
    if current_page in PAGE_ORDER:
        idx = PAGE_ORDER.index(current_page)
        if idx < len(PAGE_ORDER) - 1:
            return PAGE_ORDER[idx + 1]
    return PAGE_ORDER[0]


def get_previous_page(current_page: str) -> str:
    """Get the previous sequential page in the workflow."""
    if current_page in PAGE_ORDER:
        idx = PAGE_ORDER.index(current_page)
        if idx > 0:
            return PAGE_ORDER[idx - 1]
    return PAGE_ORDER[0]


def set_page(page_id: str):
    """Set active page in session state."""
    st.session_state["current_page"] = page_id

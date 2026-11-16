"""
Hyperspectral Disease Feature Analyzer
PCA & UMAP Dimensionality Reduction for Crop Disease Analysis
Course: INT396 - Unsupervised Learning
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Hyperspectral Disease Feature Analyzer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom academic styling (clean light theme, simple borders, subtle accents, no AI glow)
st.markdown("""
<style>
    /* Academic styling */
    .main-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 0.75rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #0f172a;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 0.3rem;
    }
    .info-box {
        background-color: #f0fdf4;
        border-left: 4px solid #16a34a;
        padding: 0.75rem 1rem;
        border-radius: 0 4px 4px 0;
        margin-bottom: 1rem;
        color: #166534;
        font-size: 0.9rem;
    }
    .warning-box {
        background-color: #fffbeb;
        border-left: 4px solid #d97706;
        padding: 0.75rem 1rem;
        border-radius: 0 4px 4px 0;
        margin-bottom: 1rem;
        color: #92400e;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Add root to sys.path
sys.path.append(os.path.dirname(__file__))

from src.data_loader import get_available_datasets, load_dataset, load_custom_file
from src.data_validation import validate_hyperspectral_data
from src.eda import (
    calculate_feature_statistics,
    calculate_class_distribution,
    compute_spectral_signatures_by_class,
    compute_band_variance_ranking,
    detect_outliers_iqr,
    compute_correlation_matrix
)
from src.preprocessing import (
    SCALER_REGISTRY,
    build_preprocessing_pipeline,
    create_train_test_split,
    fit_transform_pipeline,
    compare_before_after
)
from src.pca import (
    fit_pca,
    compute_scree_data,
    get_components_for_thresholds,
    compute_loadings,
    get_top_contributing_bands,
    transform_pca,
    compute_reconstruction_error
)
from src.clustering import (
    run_kmeans,
    run_agglomerative,
    run_dbscan,
    fit_clustering_model,
    compute_cluster_centroids_original_space
)
from src.evaluation import (
    evaluate_clustering,
    compute_sample_silhouette_values,
    build_contingency_matrix,
    get_metric_descriptions
)
from src.optimization import (
    optimize_kmeans_k,
    optimize_umap_parameters,
    optimize_dbscan_parameters
)
from src.interpretation import (
    interpret_top_pca_bands,
    find_cluster_exemplars,
    compute_centroid_spectral_curves
)
from src.experiment_tracker import (
    log_experiment,
    get_all_experiments,
    clear_experiments
)

# Cached computation wrappers
@st.cache_data
def cached_load_dataset(key: str):
    return load_dataset(key)

@st.cache_data
def cached_fit_pca(X: np.ndarray, n_components: int):
    pca, k = fit_pca(X, n_components=n_components)
    X_pca = transform_pca(pca, X)
    return pca, k, X_pca

@st.cache_data
def cached_fit_umap(X: np.ndarray, n_components: int, n_neighbors: int, min_dist: float, metric: str, random_state: int):
    from src.umap_reduction import fit_umap
    reducer, emb = fit_umap(
        X,
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    return emb

# Initialize session state variables
if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = 'potato_dai70_vnir'
if 'scaler_type' not in st.session_state:
    st.session_state.scaler_type = 'standard'
if 'pipeline_fitted' not in st.session_state:
    st.session_state.pipeline_fitted = False
if 'experiments_cache' not in st.session_state:
    st.session_state.experiments_cache = []

# Sidebar Navigation
st.sidebar.markdown("### Navigation")
menu = [
    "1. Home / Project Overview",
    "2. Dataset Selection & Inspection",
    "3. Data Quality & EDA",
    "4. Preprocessing Pipeline",
    "5. PCA Analysis",
    "6. UMAP Analysis",
    "7. Clustering Studio",
    "8. Model Comparison",
    "9. Hyperparameter Optimization",
    "10. Explainability & Interpretation",
    "11. Error & Quality Analysis",
    "12. Experiment Results Tracker",
    "13. About / Viva Defense Guide"
]
choice = st.sidebar.radio("Go to Module:", menu)

st.sidebar.markdown("---")
st.sidebar.markdown("**Course:** INT396 (Unsupervised Learning)")
st.sidebar.markdown("**Role:** ML Student Researcher")


# -------------------------------------------------------------------------------------------------
# 1. HOME / PROJECT OVERVIEW
# -------------------------------------------------------------------------------------------------
if choice == "1. Home / Project Overview":
    st.markdown('<div class="main-title">Hyperspectral Disease Feature Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">PCA & UMAP-Driven Dimensionality Reduction of Hyperspectral Crop Data for Disease-Relevant Feature Preservation</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Problem Statement
        Modern hyperspectral sensors capture hundreds of contiguous spectral bands across the visible, near-infrared (VNIR),
        and short-wave infrared (SWIR) spectrum. While this rich spectral data captures subtle biochemical and cellular changes
        induced by crop pathogens and insect pests (such as potato cyst nematodes or leaf rust), high dimensionality introduces:
        - **The Curse of Dimensionality**: High feature-to-sample ratio, sparse feature spaces, and distance metric degradation.
        - **Severe Spectral Multicollinearity**: Highly adjacent bands share strong mutual correlation.
        - **Computational Inefficiency**: Large memory overhead for downstream field classification and robotics.

        ### Central Research Question
        > **"Does linear (PCA) or non-linear (UMAP) dimensionality reduction preserve, enhance, or degrade disease-relevant cluster structure compared to original raw hyperspectral bands?"**

        ### Project Objectives
        1. **Preprocess high-dimensional spectral signatures** without data leakage using Scikit-learn Pipelines.
        2. **Implement PCA (Linear) and UMAP (Nonlinear Manifold)** dimensionality reduction and analyze explained variance vs local topological preservation.
        3. **Benchmark downstream clustering (K-Means, Agglomerative, DBSCAN)** across:
           - **Raw Hyperspectral Bands (Baseline)**
           - **PCA-Reduced Features**
           - **UMAP-Reduced Features**
        4. **Evaluate using both internal unsupervised metrics** (Silhouette, Calinski-Harabasz, Davies-Bouldin) and **external ground-truth validation** (ARI, NMI).
        5. **Physically interpret top spectral wavelengths** linking mathematical loadings to physiological plant biology (Red Edge, Chlorophyll Absorption, NIR Mesophyll Plateau).
        """)

    with col2:
        st.markdown("### System Architecture")
        st.markdown("""
        ```
        Hyperspectral Data
               │
               ▼
        Data Validation
               │
               ▼
        EDA & Signatures
               │
               ▼
        Preprocessing Pipeline
               │
               ▼
        ┌─────────────────────┐
        │  Feature Spaces:    │
        │  1. Raw Baseline    │
        │  2. PCA Reduced     │
        │  3. UMAP Embedding  │
        └──────────┬──────────┘
                   │
                   ▼
        Clustering Benchmarking
        (K-Means, Agglomerative, DBSCAN)
                   │
                   ▼
        Unsupervised & Ground-Truth
        Metric Evaluation (Sil, CH, DB, ARI)
                   │
                   ▼
        Biophysical Interpretation & Tracking
        ```
        """)

    st.markdown("---")
    st.markdown("### Current Workspace Dataset Status")
    
    # Load dataset status
    try:
        X, y, meta, bands, wls = cached_load_dataset(st.session_state.selected_dataset)
        val_res = validate_hyperspectral_data(X, y)
        
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Active Dataset", st.session_state.selected_dataset)
        with m2:
            st.metric("Total Samples", len(X))
        with m3:
            st.metric("Spectral Bands", len(bands))
        with m4:
            st.metric("Disease Classes", y.nunique() if y is not None else "Unlabeled")
        with m5:
            st.metric("Data Integrity", "Validated (Valid)" if val_res['is_valid'] else "Issues Found")
    except Exception as e:
        st.warning(f"Dataset status check: {str(e)}")


# -------------------------------------------------------------------------------------------------
# 2. DATASET MODULE
# -------------------------------------------------------------------------------------------------
elif choice == "2. Dataset Selection & Inspection":
    st.markdown('<div class="main-title">Dataset Selection & Inspection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Authentic Peer-Reviewed Hyperspectral Datasets</div>', unsafe_allow_html=True)

    available_ds = get_available_datasets()
    
    col_sel, col_src = st.columns([1, 1])
    with col_sel:
        ds_choice = st.selectbox(
            "Select Benchmark Dataset:",
            options=list(available_ds.keys()),
            format_func=lambda k: available_ds[k]['name'],
            index=list(available_ds.keys()).index(st.session_state.selected_dataset)
        )
        if ds_choice != st.session_state.selected_dataset:
            st.session_state.selected_dataset = ds_choice
            st.session_state.pipeline_fitted = False
            st.rerun()

    info = available_ds[st.session_state.selected_dataset]
    with col_src:
        st.info(f"**Citation:** {info['citation']}\n\n**Spectral Range:** {info['spectral_type']}")

    st.markdown(f"**Description:** {info['description']}")

    # Load and display dataset
    X, y, meta, bands, wls = cached_load_dataset(st.session_state.selected_dataset)
    
    tab1, tab2, tab3 = st.tabs(["Spectral Features (X)", "Target Labels (y)", "Metadata & Experimental Setup"])
    with tab1:
        st.markdown(f"**Spectral Feature Matrix (Shape: {X.shape[0]} samples × {X.shape[1]} bands):**")
        st.dataframe(X.head(10), use_container_width=True)
    with tab2:
        if y is not None:
            st.markdown(f"**Target Class Series ({len(y)} samples, {y.nunique()} unique categories):**")
            y_df = pd.DataFrame({'Sample_Index': y.index, 'Class': y.values})
            st.dataframe(y_df.head(15), use_container_width=True)
        else:
            st.write("No ground truth labels provided for this dataset.")
    with tab3:
        st.markdown("**Experimental Metadata & Treatments:**")
        st.dataframe(meta.head(15), use_container_width=True)

    st.markdown("---")
    st.markdown("### Upload Custom Spectral Dataset (.CSV / .PARQUET)")
    uploaded_file = st.file_uploader("Upload your own hyperspectral table:", type=['csv', 'parquet'])
    if uploaded_file is not None:
        target_input = st.text_input("Target/Class column name (leave blank if unlabeled):", value="")
        if st.button("Load Custom Dataset"):
            try:
                X_custom, y_custom, meta_custom, bands_custom, wls_custom = load_custom_file(
                    uploaded_file, target_col=target_input if target_input.strip() else None
                )
                st.session_state.custom_X = X_custom
                st.session_state.custom_y = y_custom
                st.session_state.custom_meta = meta_custom
                st.session_state.custom_bands = bands_custom
                st.session_state.custom_wls = wls_custom
                st.success(f"Custom dataset loaded successfully! Shape: {X_custom.shape}")
            except Exception as e:
                st.error(f"Failed to load custom dataset: {str(e)}")


# -------------------------------------------------------------------------------------------------
# 3. DATA QUALITY & EDA
# -------------------------------------------------------------------------------------------------
elif choice == "3. Data Quality & EDA":
    st.markdown('<div class="main-title">Data Quality & Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Statistical Distributions, Quality Verification, and Mean Spectral Reflectance Signatures</div>', unsafe_allow_html=True)

    X, y, meta, bands, wls = cached_load_dataset(st.session_state.selected_dataset)
    val_report = validate_hyperspectral_data(X, y)

    st.markdown("### 1. Data Quality Verification Report")
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    with q_col1:
        st.metric("Missing Values", val_report['statistics']['missing_values'])
    with q_col2:
        st.metric("Duplicate Rows", val_report['statistics']['duplicate_rows'])
    with q_col3:
        st.metric("Min Reflectance", f"{val_report['statistics']['min_value']:.4f}")
    with q_col4:
        st.metric("Max Reflectance", f"{val_report['statistics']['max_value']:.4f}")

    if val_report['is_valid']:
        st.markdown('<div class="info-box">✔ Data passed all structural and numeric integrity checks. No missing values or infinite features detected.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warning-box">⚠ Validation warnings detected: {val_report["issues"]}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 2. Class Distribution & Balance")
    if y is not None:
        c_dist = calculate_class_distribution(y)
        c_col1, c_col2 = st.columns([1, 1])
        with c_col1:
            st.dataframe(c_dist, use_container_width=True)
        with c_col2:
            fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='white')
            sns.barplot(data=c_dist, x='Class', y='Count', color='#2563eb', ax=ax)
            ax.set_title("Class Frequency Distribution", fontsize=11, fontweight='bold')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
            ax.set_ylabel("Number of Samples")
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("---")
    st.markdown("### 3. Spectral Reflectance Signatures")
    st.markdown("""
    **Physical Context:** Healthy vegetative tissue exhibits low reflectance in the visible region (400-700 nm) due to chlorophyll absorption,
    a sharp transition known as the **Red Edge (700-740 nm)**, and a elevated **Near-Infrared (NIR) Plateau (750-1000 nm)** caused by spongy mesophyll cellular scattering.
    Pathogen infections and water deficit cause cell collapse and pigment degradation, altering these spectral curves.
    """)

    sigs = compute_spectral_signatures_by_class(X, y, wls)
    
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor='white')
    colors = sns.color_palette("tab10", len(sigs))
    
    for (cls_name, sig), col in zip(sigs.items(), colors):
        mean_curve = sig['mean']
        std_curve = sig['std']
        wl_arr = sig['wavelengths']
        ax.plot(wl_arr, mean_curve, label=cls_name, color=col, linewidth=2)
        ax.fill_between(wl_arr, mean_curve - std_curve, mean_curve + std_curve, color=col, alpha=0.15)

    ax.set_xlabel("Wavelength (nm)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Spectral Reflectance", fontsize=10, fontweight='bold')
    ax.set_title("Mean Spectral Reflectance Signatures (±1 Std Dev Ribbon)", fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.markdown("### 4. Spectral Band Variance & Multicollinearity")
    var_col, corr_col = st.columns([1, 1])
    with var_col:
        st.markdown("**Top Variable Spectral Wavelengths:**")
        var_df = compute_band_variance_ranking(X, wls)
        st.dataframe(var_df.head(10), use_container_width=True)
    with corr_col:
        st.markdown("**Inter-Band Pearson Correlation Subsample:**")
        corr_mat, sel_cols = compute_correlation_matrix(X, max_bands=25)
        fig_corr, ax_corr = plt.subplots(figsize=(5.5, 4.5), facecolor='white')
        sns.heatmap(corr_mat, cmap='Blues', ax=ax_corr, cbar=True, square=True, xticklabels=False, yticklabels=False)
        ax_corr.set_title("Inter-Band Correlation Matrix", fontsize=11, fontweight='bold')
        st.pyplot(fig_corr)
        plt.close(fig_corr)


# -------------------------------------------------------------------------------------------------
# 4. PREPROCESSING MODULE
# -------------------------------------------------------------------------------------------------
elif choice == "4. Preprocessing Pipeline":
    st.markdown('<div class="main-title">Preprocessing Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Scikit-Learn Pipeline: Imputation, Variance Filtering, Scaling, and Train/Test Split (No Data Leakage)</div>', unsafe_allow_html=True)

    X, y, meta, bands, wls = cached_load_dataset(st.session_state.selected_dataset)

    col_cfg, col_split = st.columns([1, 1])
    with col_cfg:
        st.markdown("### Pipeline Configuration")
        scaler_choice = st.selectbox(
            "Select Feature Scaler:",
            options=list(SCALER_REGISTRY.keys()),
            format_func=lambda k: SCALER_REGISTRY[k][0],
            index=list(SCALER_REGISTRY.keys()).index(st.session_state.scaler_type)
        )
        st.session_state.scaler_type = scaler_choice

        var_thresh = st.number_input(
            "Variance Threshold for Constant Band Removal:",
            min_value=0.0, max_value=0.05, value=0.0, step=0.001, format="%.4f"
        )
        impute_method = st.selectbox("Imputation Strategy:", ["median", "mean", "most_frequent"])

    with col_split:
        st.markdown("### Data Leakage Prevention")
        st.markdown(r"""
        In rigorous scientific workflows, transformers that learn parameters (such as `StandardScaler` mean $\mu$ and variance $\sigma^2$)
        must be fitted **strictly on the training split** and then applied to test data to prevent data leakage.
        """)
        split_data = st.checkbox("Enable Train / Validation Split (for leakage-free verification)", value=True)
        test_ratio = st.slider("Test Split Proportion:", min_value=0.1, max_value=0.4, value=0.25, step=0.05) if split_data else 0.0

    if st.button("Build and Execute Preprocessing Pipeline"):
        pipeline = build_preprocessing_pipeline(
            scaler_type=scaler_choice,
            variance_threshold=var_thresh,
            impute_strategy=impute_method
        )
        if split_data:
            split_dict = create_train_test_split(X, y, test_size=test_ratio, random_state=42)
            X_tr_proc, X_te_proc, act_bands = fit_transform_pipeline(
                pipeline, split_dict['X_train'], split_dict['X_test'], band_names=bands
            )
            # Full dataset transformed for downstream exploration
            X_full_proc = pipeline.transform(X)
            st.session_state.split_dict = split_dict
        else:
            X_full_proc, _, act_bands = fit_transform_pipeline(pipeline, X, band_names=bands)
            X_tr_proc, X_te_proc = X_full_proc, None

        st.session_state.pipeline = pipeline
        st.session_state.X_proc = X_full_proc
        st.session_state.active_bands = act_bands
        st.session_state.pipeline_fitted = True
        st.success(f"Pipeline executed! Output shape: {X_full_proc.shape[0]} samples × {X_full_proc.shape[1]} features.")

    if st.session_state.pipeline_fitted:
        st.markdown("---")
        st.markdown("### Before vs. After Preprocessing Inspection")
        X_proc = st.session_state.X_proc
        act_bands = st.session_state.active_bands
        
        comp_df = compare_before_after(X, X_proc, act_bands)
        st.dataframe(comp_df.head(10), use_container_width=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), facecolor='white')
        ax1.hist(X.iloc[:, :5].values.flatten(), bins=30, color='#64748b', alpha=0.7)
        ax1.set_title("Before Scaling (Raw Reflectance)", fontsize=10, fontweight='bold')
        ax1.set_xlabel("Value")
        ax1.set_ylabel("Count")

        ax2.hist(X_proc[:, :5].flatten(), bins=30, color='#2563eb', alpha=0.7)
        ax2.set_title(f"After {SCALER_REGISTRY[st.session_state.scaler_type][0]}", fontsize=10, fontweight='bold')
        ax2.set_xlabel("Transformed Value")
        ax2.set_ylabel("Count")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# -------------------------------------------------------------------------------------------------
# 5. PCA MODULE
# -------------------------------------------------------------------------------------------------
elif choice == "5. PCA Analysis":
    st.markdown('<div class="main-title">Principal Component Analysis (PCA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Linear Dimensionality Reduction, Scree Plot, Explained Variance, and Component Loadings</div>', unsafe_allow_html=True)

    X, y, meta, bands, wls = cached_load_dataset(st.session_state.selected_dataset)
    if not st.session_state.pipeline_fitted:
        pipe = build_preprocessing_pipeline(st.session_state.scaler_type)
        st.session_state.X_proc, _, st.session_state.active_bands = fit_transform_pipeline(pipe, X, band_names=bands)
        st.session_state.pipeline_fitted = True

    X_proc = st.session_state.X_proc
    act_bands = st.session_state.active_bands

    max_avail_comp = min(X_proc.shape[0], X_proc.shape[1])

    c_cfg1, c_cfg2 = st.columns([1, 1])
    with c_cfg1:
        n_comp = st.slider("Number of Principal Components (k):", min_value=2, max_value=min(30, max_avail_comp), value=5)
    with c_cfg2:
        st.markdown(f"**Dataset Dimensions:** {X_proc.shape[0]} samples × {X_proc.shape[1]} bands")

    pca_model, k, X_pca = cached_fit_pca(X_proc, n_components=n_comp)
    st.session_state.pca_model = pca_model
    st.session_state.X_pca = X_pca

    scree_df = compute_scree_data(pca_model)
    cum_var_total = scree_df['Cumulative_Variance_Ratio'].iloc[-1] * 100
    recon_mse = compute_reconstruction_error(pca_model, X_proc)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Retained Components", k)
    with m2:
        st.metric("Cumulative Variance Explained", f"{cum_var_total:.2f}%")
    with m3:
        st.metric("Reconstruction MSE", f"{recon_mse:.5f}")

    st.markdown("---")
    st.markdown("### Scree Plot (Explained & Cumulative Variance)")
    fig_scree, ax1 = plt.subplots(figsize=(9, 4), facecolor='white')
    ax2 = ax1.twinx()

    x_indices = range(1, k + 1)
    bars = ax1.bar(x_indices, scree_df['Explained_Variance_Ratio'], color='#93c5fd', alpha=0.8, label='Individual Variance')
    line = ax2.plot(x_indices, scree_df['Cumulative_Variance_Ratio'], color='#1d4ed8', marker='o', linewidth=2, label='Cumulative Variance')

    ax1.set_xlabel("Principal Component Index", fontweight='bold')
    ax1.set_ylabel("Explained Variance Ratio", color='#1e3a8a', fontweight='bold')
    ax2.set_ylabel("Cumulative Explained Variance", color='#1d4ed8', fontweight='bold')
    ax2.set_ylim([0, 1.05])
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels([f"PC{i}" for i in x_indices], rotation=45)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax2.axhline(0.90, color='gray', linestyle=':', label='90% Threshold')
    ax2.axhline(0.95, color='orange', linestyle=':', label='95% Threshold')

    plt.title("PCA Scree Plot with Variance Retention Thresholds", fontsize=11, fontweight='bold')
    st.pyplot(fig_scree)
    plt.close(fig_scree)

    st.markdown("---")
    st.markdown("### 2D Projection Space (PC1 vs PC2)")
    fig_pca2d, ax_pca = plt.subplots(figsize=(8, 4.5), facecolor='white')
    pca_df = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
    if y is not None:
        pca_df['Target'] = y.values
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Target', palette='tab10', s=70, alpha=0.85, ax=ax_pca)
        ax_pca.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    else:
        ax_pca.scatter(pca_df['PC1'], pca_df['PC2'], color='#2563eb', alpha=0.7, s=60)

    ax_pca.set_xlabel(f"PC1 ({scree_df['Explained_Variance_Ratio'].iloc[0]*100:.1f}% Variance)", fontweight='bold')
    ax_pca.set_ylabel(f"PC2 ({scree_df['Explained_Variance_Ratio'].iloc[1]*100:.1f}% Variance)", fontweight='bold')
    ax_pca.set_title("First Two Principal Components Projection", fontsize=11, fontweight='bold')
    ax_pca.grid(True, linestyle='--', alpha=0.4)
    st.pyplot(fig_pca2d)
    plt.close(fig_pca2d)


# -------------------------------------------------------------------------------------------------
# 6. UMAP MODULE
# -------------------------------------------------------------------------------------------------
elif choice == "6. UMAP Analysis":
    st.markdown('<div class="main-title">UMAP Manifold Dimensionality Reduction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Non-linear Manifold Approximation, Fuzzy Simplicial Sets, and Low-Dimensional Topological Embedding</div>', unsafe_allow_html=True)

    X, y, meta, bands, wls = cached_load_dataset(st.session_state.selected_dataset)
    if not st.session_state.pipeline_fitted:
        pipe = build_preprocessing_pipeline(st.session_state.scaler_type)
        st.session_state.X_proc, _, st.session_state.active_bands = fit_transform_pipeline(pipe, X, band_names=bands)
        st.session_state.pipeline_fitted = True

    X_proc = st.session_state.X_proc

    st.markdown("""
    > **Mathematical Note for Viva:** Unlike PCA which maximizes orthogonal global variance ($Var(Xv)$),
    > UMAP models the underlying Riemannian manifold using fuzzy simplicial sets and minimizes cross-entropy between
    > high-dimensional fuzzy topological representations and low-dimensional representations.
    """)

    u_c1, u_c2, u_c3, u_c4 = st.columns(4)
    with u_c1:
        n_neighbors = st.slider("n_neighbors (Local vs Global balance):", min_value=3, max_value=min(40, len(X_proc)-1), value=10)
    with u_c2:
        min_dist = st.slider("min_dist (Embedding cluster tightness):", min_value=0.0, max_value=0.9, value=0.1, step=0.05)
    with u_c3:
        metric = st.selectbox("Distance Metric:", ['euclidean', 'cosine', 'manhattan'])
    with u_c4:
        seed = st.number_input("Random Seed:", value=42, step=1)

    if st.button("Generate UMAP Embedding") or 'X_umap' not in st.session_state:
        with st.spinner("Calculating UMAP manifold projection..."):
            t0 = time.time()
            X_umap = cached_fit_umap(X_proc, 2, n_neighbors, min_dist, metric, seed)
            st.session_state.X_umap = X_umap
            st.session_state.umap_params = {'n_neighbors': n_neighbors, 'min_dist': min_dist, 'metric': metric}
            st.success(f"UMAP completed in {time.time()-t0:.2f}s! Shape: {X_umap.shape}")

    if 'X_umap' in st.session_state:
        X_umap = st.session_state.X_umap
        fig_umap, ax_u = plt.subplots(figsize=(8, 4.5), facecolor='white')
        umap_df = pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2'])
        if y is not None:
            umap_df['Target'] = y.values
            sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='Target', palette='tab10', s=70, alpha=0.85, ax=ax_u)
            ax_u.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        else:
            ax_u.scatter(umap_df['UMAP1'], umap_df['UMAP2'], color='#059669', alpha=0.7, s=60)

        ax_u.set_xlabel("UMAP Dimension 1", fontweight='bold')
        ax_u.set_ylabel("UMAP Dimension 2", fontweight='bold')
        ax_u.set_title("2D UMAP Manifold Embedding", fontsize=11, fontweight='bold')
        ax_u.grid(True, linestyle='--', alpha=0.4)
        st.pyplot(fig_umap)
        plt.close(fig_umap)


# -------------------------------------------------------------------------------------------------
# 7. CLUSTERING STUDIO
# -------------------------------------------------------------------------------------------------
elif choice == "7. Clustering Studio":
    st.markdown('<div class="main-title">Clustering Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Benchmark K-Means, Agglomerative Hierarchical, and DBSCAN on Raw, PCA, and UMAP Representations</div>', unsafe_allow_html=True)

    X, y, meta, bands, wls = cached_load_dataset(st.session_state.selected_dataset)
    if not st.session_state.pipeline_fitted:
        pipe = build_preprocessing_pipeline(st.session_state.scaler_type)
        st.session_state.X_proc, _, st.session_state.active_bands = fit_transform_pipeline(pipe, X, band_names=bands)
        st.session_state.pipeline_fitted = True

    X_proc = st.session_state.X_proc
    if 'X_pca' not in st.session_state:
        _, _, st.session_state.X_pca = cached_fit_pca(X_proc, 5)
    if 'X_umap' not in st.session_state:
        st.session_state.X_umap = cached_fit_umap(X_proc, 2, 10, 0.1, 'euclidean', 42)

    c1, c2, c3 = st.columns(3)
    with c1:
        rep_choice = st.selectbox(
            "Feature Representation:",
            ["Raw Hyperspectral Bands (Baseline)", "PCA Reduced Features", "UMAP Embedding"]
        )
    with c2:
        algo_choice = st.selectbox(
            "Clustering Algorithm:",
            ["K-Means", "Agglomerative Hierarchical", "DBSCAN"]
        )
    with c3:
        n_clusters = st.slider("Target Clusters (k):", min_value=2, max_value=8, value=4) if algo_choice != "DBSCAN" else 0

    # Select feature matrix
    if "Raw" in rep_choice:
        feat_matrix = X_proc
        rep_label = "Raw"
    elif "PCA" in rep_choice:
        feat_matrix = st.session_state.X_pca
        rep_label = "PCA"
    else:
        feat_matrix = st.session_state.X_umap
        rep_label = "UMAP"

    # Algorithm hyperparameters
    params = {}
    if algo_choice == "K-Means":
        params = {'n_clusters': n_clusters, 'random_state': 42}
    elif algo_choice == "Agglomerative Hierarchical":
        linkage = st.selectbox("Linkage Method:", ['ward', 'complete', 'average'])
        params = {'n_clusters': n_clusters, 'linkage': linkage}
    elif algo_choice == "DBSCAN":
        db_eps = st.slider("eps (Neighborhood Radius):", 0.1, 5.0, 1.5, 0.1)
        db_min = st.slider("min_samples:", 2, 10, 3)
        params = {'eps': db_eps, 'min_samples': db_min}

    if st.button("Run Clustering"):
        model, labels = fit_clustering_model(algo_choice, feat_matrix, params)
        eval_res = evaluate_clustering(feat_matrix, labels, y_true=y)
        st.session_state.current_cluster_labels = labels
        st.session_state.current_eval_res = eval_res
        st.session_state.current_rep_label = rep_label
        st.session_state.current_algo_label = algo_choice

        # Auto-log to experiments
        log_experiment(
            dataset_name=st.session_state.selected_dataset,
            representation=rep_label,
            algorithm=algo_choice,
            preprocessing_config={'scaler': st.session_state.scaler_type},
            hyperparameters=params,
            metrics=eval_res,
            random_seed=42
        )

    if 'current_cluster_labels' in st.session_state:
        labels = st.session_state.current_cluster_labels
        eval_res = st.session_state.current_eval_res

        st.markdown("---")
        st.markdown("### Clustering Performance Metrics")
        im = eval_res.get('internal_metrics', {})
        em = eval_res.get('external_metrics', {})

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Silhouette Score", f"{im.get('silhouette_score', np.nan):.3f}" if not np.isnan(im.get('silhouette_score', np.nan)) else "N/A")
        with m2:
            st.metric("Calinski-Harabasz", f"{im.get('calinski_harabasz_score', np.nan):.1f}" if not np.isnan(im.get('calinski_harabasz_score', np.nan)) else "N/A")
        with m3:
            st.metric("Davies-Bouldin", f"{im.get('davies_bouldin_score', np.nan):.3f}" if not np.isnan(im.get('davies_bouldin_score', np.nan)) else "N/A")
        with m4:
            st.metric("Adjusted Rand Index (ARI)", f"{em.get('adjusted_rand_index', np.nan):.3f}" if not np.isnan(em.get('adjusted_rand_index', np.nan)) else "N/A")
        with m5:
            st.metric("Mutual Info (NMI)", f"{em.get('normalized_mutual_info', np.nan):.3f}" if not np.isnan(em.get('normalized_mutual_info', np.nan)) else "N/A")

        st.markdown("### Cluster Visual Assignments")
        # Plot clusters in 2D space (using PCA or UMAP coordinates for display)
        fig_cl, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), facecolor='white')
        
        # Left: Cluster assignments
        df_plot = pd.DataFrame(st.session_state.X_pca[:, :2], columns=['PC1', 'PC2'])
        df_plot['Cluster'] = [f"Cluster {c}" if c != -1 else "Noise (-1)" for c in labels]
        sns.scatterplot(data=df_plot, x='PC1', y='PC2', hue='Cluster', palette='tab10', s=70, ax=ax1)
        ax1.set_title(f"{st.session_state.current_algo_label} on {st.session_state.current_rep_label} (Clusters)", fontsize=10, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.3)

        # Right: True Disease classes (if available)
        if y is not None:
            df_plot['Ground_Truth'] = y.values
            sns.scatterplot(data=df_plot, x='PC1', y='PC2', hue='Ground_Truth', palette='Set2', s=70, ax=ax2)
            ax2.set_title("Ground-Truth Disease Classes", fontsize=10, fontweight='bold')
            ax2.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig_cl)
        plt.close(fig_cl)


# -------------------------------------------------------------------------------------------------
# 8. MODEL COMPARISON
# -------------------------------------------------------------------------------------------------
elif choice == "8. Model Comparison":
    st.markdown('<div class="main-title">Multi-Model Benchmarking Matrix</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Systematic Benchmark: Raw vs PCA vs UMAP across K-Means, Agglomerative, and DBSCAN</div>', unsafe_allow_html=True)

    X, y, meta, bands, wls = cached_load_dataset(st.session_state.selected_dataset)
    if not st.session_state.pipeline_fitted:
        pipe = build_preprocessing_pipeline(st.session_state.scaler_type)
        st.session_state.X_proc, _, st.session_state.active_bands = fit_transform_pipeline(pipe, X, band_names=bands)
        st.session_state.pipeline_fitted = True

    X_proc = st.session_state.X_proc
    if 'X_pca' not in st.session_state:
        _, _, st.session_state.X_pca = cached_fit_pca(X_proc, 5)
    if 'X_umap' not in st.session_state:
        st.session_state.X_umap = cached_fit_umap(X_proc, 2, 10, 0.1, 'euclidean', 42)

    k_val = st.slider("Benchmark Cluster Count (k):", 2, 8, 4)

    if st.button("Execute Full Benchmark Matrix"):
        with st.spinner("Benchmarking all representations and clustering algorithms..."):
            representations = {
                'Raw Features (Baseline)': X_proc,
                'PCA Features': st.session_state.X_pca,
                'UMAP Embedding': st.session_state.X_umap
            }
            algorithms = {
                'K-Means': lambda feat: run_kmeans(feat, n_clusters=k_val, random_state=42),
                'Agglomerative': lambda feat: run_agglomerative(feat, n_clusters=k_val, linkage='ward')
            }

            comparison_records = []
            for rep_name, feat_mat in representations.items():
                for algo_name, run_func in algorithms.items():
                    model, labels = run_func(feat_mat)
                    res = evaluate_clustering(feat_mat, labels, y_true=y)
                    im = res.get('internal_metrics', {})
                    em = res.get('external_metrics', {})

                    comparison_records.append({
                        'Representation': rep_name,
                        'Clustering Algorithm': algo_name,
                        'Silhouette Score': im.get('silhouette_score', np.nan),
                        'Calinski-Harabasz': im.get('calinski_harabasz_score', np.nan),
                        'Davies-Bouldin': im.get('davies_bouldin_score', np.nan),
                        'ARI (Ground-Truth)': em.get('adjusted_rand_index', np.nan),
                        'NMI (Mutual Info)': em.get('normalized_mutual_info', np.nan)
                    })

            comp_df = pd.DataFrame(comparison_records)
            st.session_state.comparison_matrix_df = comp_df

    if 'comparison_matrix_df' in st.session_state:
        comp_df = st.session_state.comparison_matrix_df
        st.dataframe(
            comp_df.style.highlight_max(subset=['Silhouette Score', 'Calinski-Harabasz', 'ARI (Ground-Truth)', 'NMI (Mutual Info)'], color='#bbf7d0')
                   .highlight_min(subset=['Davies-Bouldin'], color='#bbf7d0'),
            use_container_width=True
        )

        st.markdown("### Comparative Performance Charts")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), facecolor='white')

        # Silhouette Comparison
        sns.barplot(data=comp_df, x='Representation', y='Silhouette Score', hue='Clustering Algorithm', palette='Blues', ax=ax1)
        ax1.set_title("Silhouette Score (Higher is Better)", fontsize=10, fontweight='bold')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=15)
        ax1.grid(axis='y', linestyle='--', alpha=0.4)

        # ARI Comparison
        if 'ARI (Ground-Truth)' in comp_df.columns and not comp_df['ARI (Ground-Truth)'].isna().all():
            sns.barplot(data=comp_df, x='Representation', y='ARI (Ground-Truth)', hue='Clustering Algorithm', palette='Greens', ax=ax2)
            ax2.set_title("Adjusted Rand Index (ARI vs Disease Classes)", fontsize=10, fontweight='bold')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=15)
            ax2.grid(axis='y', linestyle='--', alpha=0.4)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# -------------------------------------------------------------------------------------------------
# 9. HYPERPARAMETER OPTIMIZATION
# -------------------------------------------------------------------------------------------------
elif choice == "9. Hyperparameter Optimization":
    st.markdown('<div class="main-title">Hyperparameter Optimization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Systematic Metric-Driven Grid Searches for K-Means and DBSCAN</div>', unsafe_allow_html=True)

    X, y, meta, bands, wls = cached_load_dataset(st.session_state.selected_dataset)
    if not st.session_state.pipeline_fitted:
        pipe = build_preprocessing_pipeline(st.session_state.scaler_type)
        st.session_state.X_proc, _, st.session_state.active_bands = fit_transform_pipeline(pipe, X, band_names=bands)
        st.session_state.pipeline_fitted = True

    X_proc = st.session_state.X_proc

    opt_tab1, opt_tab2 = st.tabs(["K-Means Optimal k Search", "DBSCAN Density Parameter Grid"])
    with opt_tab1:
        st.markdown("### K-Means Elbow & Silhouette Curve")
        k_max = st.slider("Maximum k to evaluate:", 4, 10, 7)
        if st.button("Run K-Means k Search"):
            with st.spinner("Searching optimal k..."):
                km_opt_df, best_k = optimize_kmeans_k(X_proc, k_min=2, k_max=k_max, y_true=y)
                st.dataframe(km_opt_df, use_container_width=True)

                fig, ax1 = plt.subplots(figsize=(8, 3.5), facecolor='white')
                ax2 = ax1.twinx()

                ax1.plot(km_opt_df['k'], km_opt_df['Inertia'], 'b-o', label='Inertia (Elbow)')
                ax2.plot(km_opt_df['k'], km_opt_df['Silhouette'], 'g-s', label='Silhouette Score')

                ax1.set_xlabel("Number of Clusters (k)", fontweight='bold')
                ax1.set_ylabel("Inertia (Sum of Squared Distances)", color='b')
                ax2.set_ylabel("Silhouette Score", color='g')
                ax1.grid(True, linestyle='--', alpha=0.3)
                plt.title("Elbow Method & Silhouette Optimization Curve", fontsize=11, fontweight='bold')
                st.pyplot(fig)
                plt.close(fig)

                st.success(f"Optimal cluster count based on maximum Silhouette Score: k = {best_k}")

    with opt_tab2:
        st.markdown("### DBSCAN eps & min_samples Grid Search")
        if st.button("Run DBSCAN Grid Search"):
            with st.spinner("Evaluating density parameters..."):
                db_df, best_db = optimize_dbscan_parameters(
                    X_proc, eps_list=[0.5, 1.0, 2.0, 3.0, 5.0], min_samples_list=[2, 3, 5], y_true=y
                )
                st.dataframe(db_df, use_container_width=True)
                if best_db:
                    st.success(f"Best DBSCAN configuration: eps={best_db['eps']}, min_samples={best_db['min_samples']} (Clusters: {best_db['clusters']}, Silhouette: {best_db['silhouette']:.3f})")


# -------------------------------------------------------------------------------------------------
# 10. EXPLAINABILITY & INTERPRETATION
# -------------------------------------------------------------------------------------------------
elif choice == "10. Explainability & Interpretation":
    st.markdown('<div class="main-title">Explainability & Feature Interpretation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Biophysical Interpretation of PCA Loadings, Diagnostic Wavelengths, and Cluster Centroid Profiles</div>', unsafe_allow_html=True)

    X, y, meta, bands, wls = cached_load_dataset(st.session_state.selected_dataset)
    if not st.session_state.pipeline_fitted:
        pipe = build_preprocessing_pipeline(st.session_state.scaler_type)
        st.session_state.X_proc, _, st.session_state.active_bands = fit_transform_pipeline(pipe, X, band_names=bands)
        st.session_state.pipeline_fitted = True

    X_proc = st.session_state.X_proc
    act_bands = st.session_state.active_bands

    if 'pca_model' not in st.session_state:
        st.session_state.pca_model, _, st.session_state.X_pca = cached_fit_pca(X_proc, 5)

    pca_model = st.session_state.pca_model
    loadings_df = compute_loadings(pca_model, act_bands, wls)

    st.markdown("### 1. Principal Component Spectral Loadings Matrix ($V^T$)")
    st.markdown("""
    Loadings quantify the contribution of each individual physical wavelength (nm) to each Principal Component.
    Peaks and dips in loading vectors highlight diagnostic biochemical absorption bands.
    """)

    fig_load, ax_l = plt.subplots(figsize=(10, 4), facecolor='white')
    if 'Wavelength_nm' in loadings_df.columns:
        ax_l.plot(loadings_df['Wavelength_nm'], loadings_df['PC1'], label='PC1 Loadings', color='#1d4ed8', linewidth=1.8)
        ax_l.plot(loadings_df['Wavelength_nm'], loadings_df['PC2'], label='PC2 Loadings', color='#059669', linewidth=1.8)
        if 'PC3' in loadings_df.columns:
            ax_l.plot(loadings_df['Wavelength_nm'], loadings_df['PC3'], label='PC3 Loadings', color='#d97706', linewidth=1.5)
        ax_l.set_xlabel("Wavelength (nm)", fontweight='bold')
    else:
        ax_l.plot(loadings_df['PC1'], label='PC1 Loadings', color='#1d4ed8')
        ax_l.plot(loadings_df['PC2'], label='PC2 Loadings', color='#059669')
        ax_l.set_xlabel("Band Index", fontweight='bold')

    ax_l.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax_l.set_ylabel("Eigenvector Loading Weight", fontweight='bold')
    ax_l.set_title("Spectral Band Contribution across Principal Axes", fontsize=11, fontweight='bold')
    ax_l.legend(loc='upper right', frameon=True)
    ax_l.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig_load)
    plt.close(fig_load)

    st.markdown("---")
    st.markdown("### 2. Biophysical Domain Interpretation of Top Bands")
    pc_target = st.selectbox("Inspect Top Contributing Bands for:", [f"PC{i+1}" for i in range(min(5, pca_model.n_components_))])
    top_interpreted = interpret_top_pca_bands(loadings_df, pc_name=pc_target, top_n=5)
    
    for item in top_interpreted:
        wl_str = f"**{item['wavelength_nm']} nm**" if item['wavelength_nm'] else f"**{item['band']}**"
        st.markdown(f"- {wl_str} | Loading: `{item['loading']:+.4f}` ({item['direction']})")
        st.markdown(f"  *Biological Significance:* {item['interpretation']}")

    st.markdown("---")
    st.markdown("### 3. Cluster Centroid Signatures in Original Spectral Space")
    if 'current_cluster_labels' in st.session_state:
        lbls = st.session_state.current_cluster_labels
        centroid_df = compute_centroid_spectral_curves(X, lbls, wls)
        
        fig_c, ax_c = plt.subplots(figsize=(10, 4), facecolor='white')
        for col in centroid_df.columns:
            if col != 'Wavelength_nm':
                ax_c.plot(centroid_df['Wavelength_nm'], centroid_df[col], label=col, linewidth=2)
        ax_c.set_xlabel("Wavelength (nm)", fontweight='bold')
        ax_c.set_ylabel("Mean Raw Reflectance", fontweight='bold')
        ax_c.set_title("Reconstructed Cluster Centroid Reflectance Profiles", fontsize=11, fontweight='bold')
        ax_c.legend()
        ax_c.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig_c)
        plt.close(fig_c)
    else:
        st.info("Run clustering in the Clustering Studio module to display reconstructed cluster centroid curves.")


# -------------------------------------------------------------------------------------------------
# 11. ERROR & QUALITY ANALYSIS
# -------------------------------------------------------------------------------------------------
elif choice == "11. Error & Quality Analysis":
    st.markdown('<div class="main-title">Clustering Quality & Error Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Per-Sample Silhouette Thickness, Confusion Matrix with Ground Truth, and Cluster Overlap</div>', unsafe_allow_html=True)

    X, y, meta, bands, wls = cached_load_dataset(st.session_state.selected_dataset)
    if 'current_cluster_labels' in st.session_state:
        labels = st.session_state.current_cluster_labels
        feat_used = st.session_state.X_pca if st.session_state.current_rep_label == 'PCA' else (st.session_state.X_umap if st.session_state.current_rep_label == 'UMAP' else st.session_state.X_proc)

        col_sil, col_ct = st.columns([1, 1])
        with col_sil:
            st.markdown("### 1. Silhouette Samples Distribution")
            sil_df = compute_sample_silhouette_values(feat_used, labels)
            if not sil_df.empty:
                fig_sil, ax_s = plt.subplots(figsize=(5, 3.5), facecolor='white')
                sns.boxplot(data=sil_df, x='Cluster', y='Silhouette_Value', palette='Blues', ax=ax_s)
                ax_s.axhline(0, color='red', linestyle='--', alpha=0.7, label='Zero Separation')
                ax_s.set_title("Silhouette Thickness per Cluster", fontsize=10, fontweight='bold')
                ax_s.set_ylabel("Silhouette Coefficient")
                ax_s.grid(axis='y', linestyle='--', alpha=0.3)
                st.pyplot(fig_sil)
                plt.close(fig_sil)

        with col_ct:
            st.markdown("### 2. Contingency Cross-Tabulation")
            if y is not None:
                ct = build_contingency_matrix(labels, y)
                st.dataframe(ct, use_container_width=True)
                st.caption("Rows: Discovered Unsupervised Clusters | Columns: Ground-Truth Disease Classes")

        st.markdown("---")
        st.markdown("### 3. Failure Mode & Boundary Analysis")
        st.markdown("""
        **Academic Viva Defense Explanation:**
        - **Why do clusters overlap?** Plant diseases manifest as a continuum of cellular stress rather than discrete disjoint boundaries.
          Early biotic infection (nematode feeding) causes localized root damage that mimics water stress before leaf chlorosis becomes visible.
        - **Why is ARI lower than supervised accuracy?** Unsupervised clustering groups samples by physical reflectance similarity
          (such as canopy density and moisture), whereas supervised classifiers are explicitly trained with class labels.
        """)
    else:
        st.info("Please execute clustering in '7. Clustering Studio' first to generate error and silhouette profiles.")


# -------------------------------------------------------------------------------------------------
# 12. EXPERIMENT RESULTS TRACKER
# -------------------------------------------------------------------------------------------------
elif choice == "12. Experiment Results Tracker":
    st.markdown('<div class="main-title">Experiment Results Tracker</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Audit Trail of Benchmarked Models, Hyperparameters, and Evaluation Metrics</div>', unsafe_allow_html=True)

    exp_df = get_all_experiments()
    if not exp_df.empty:
        st.markdown(f"**Total Logged Experiments:** {len(exp_df)}")
        display_cols = [c for c in ['experiment_id', 'timestamp', 'representation', 'algorithm', 'n_clusters', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'ari', 'nmi'] if c in exp_df.columns]
        st.dataframe(exp_df[display_cols], use_container_width=True)

        csv_data = exp_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Experiment History (CSV)",
            data=csv_data,
            file_name="hyperspectral_experiments_log.csv",
            mime="text/csv"
        )

        if st.button("Clear All Logged Experiments"):
            clear_experiments()
            st.success("Experiments cleared.")
            st.rerun()
    else:
        st.info("No experiments logged yet. Run models in 'Clustering Studio' or 'Model Comparison' to automatically track results.")


# -------------------------------------------------------------------------------------------------
# 13. ABOUT / METHODOLOGY
# -------------------------------------------------------------------------------------------------
elif choice == "13. About / Viva Defense Guide":
    st.markdown('<div class="main-title">Methodology & Viva Defense Guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Theoretical Foundations, Mathematical Equations, and Research Defense</div>', unsafe_allow_html=True)

    st.markdown(r"""
    ### 1. Mathematical Grounding

    #### Principal Component Analysis (PCA)
    Given centered data matrix $X \in \mathbb{R}^{n \times p}$, PCA finds orthogonal directions $v_1, \dots, v_k$ maximizing variance:
    $$\max_v \frac{1}{n} v^T X^T X v \quad \text{subject to} \quad \|v\|_2 = 1$$
    This is equivalent to the eigendecomposition of the sample covariance matrix $\Sigma = \frac{1}{n} X^T X = V \Lambda V^T$.
    The explained variance ratio for component $i$ is:
    $$\text{EVR}_i = \frac{\lambda_i}{\sum_{j=1}^p \lambda_j}$$

    #### UMAP (Uniform Manifold Approximation and Projection)
    UMAP assumes data lies on a Riemannian manifold. It models pairwise fuzzy membership:
    $$p_{i|j} = \exp\left(-\frac{\max(0, d(x_i, x_j) - \rho_i)}{\sigma_i}\right)$$
    and finds a low-dimensional layout $Y$ by minimizing the fuzzy set cross-entropy:
    $$C(P, Q) = \sum_{e} \left[ p_e \log \frac{p_e}{q_e} + (1 - p_e) \log \frac{1 - p_e}{1 - q_e} \right]$$

    #### Unsupervised Evaluation Metrics
    - **Silhouette Score**: $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$, where $a(i)$ is mean intra-cluster distance and $b(i)$ is mean nearest-cluster distance.
    - **Calinski-Harabasz Index**: $\text{CH} = \frac{\text{Tr}(B_k)}{\text{Tr}(W_k)} \times \frac{n - k}{k - 1}$
    - **Davies-Bouldin Index**: $\text{DB} = \frac{1}{k} \sum_{i=1}^k \max_{j \neq i} \left( \frac{s_i + s_j}{d(c_i, c_j)} \right)$
    - **Adjusted Rand Index (ARI)**: Corrects Rand Index for chance: $\text{ARI} = \frac{\text{Index} - \text{Expected}}{\text{Max} - \text{Expected}}$

    ---
    ### 2. Viva Defense Questions & Key Answers

    **Q1: Why treat raw spectral bands as the baseline?**
    *Answer:* To objectively test the central research question: *Does dimensionality reduction preserve or lose disease-relevant information?*
    If clustering on PCA or UMAP outperforms raw bands, dimensionality reduction removed noise and collinear redundancy. If raw performs better, reduction discarded discriminative variance.

    **Q2: Why not just use supervised classification?**
    *Answer:* In real precision agriculture and drone scouting, ground-truth pathogen lab assays (PCR or nematode cyst extraction) are expensive and sparse.
    Unsupervised learning enables discovery of emerging stress clusters without prior labels.

    **Q3: Why can't UMAP embeddings be used to interpret original feature loadings?**
    *Answer:* UMAP applies non-linear manifold optimization where coordinates represent relative neighborhood topology, not linear combinations of original bands.
    PCA loadings provide direct physical interpretability because each component is an explicit linear eigenvector of the original wavelengths.
    """)

"""
Page 0: Academic Project Overview & Theoretical Framework
Course: INT396 - Unsupervised Learning
"""

import streamlit as st
import pandas as pd
from config.settings import APP_TITLE, APP_SUBTITLE, COURSE_CODE, PROJECT_TYPE, PAGES
from data.loader import get_curated_datasets, load_curated_dataset, inspect_dataset
from data.validator import validate_dataset
from utils.state import (
    is_dataset_ready, is_preprocessing_ready, is_pca_ready,
    is_umap_ready, is_clustering_ready, is_evaluation_ready,
    is_visualization_ready, is_results_ready,
    set_page, reset_full_experiment,
    reset_downstream_from_dataset
)


def _load_and_store_curated(selected_key: str):
    """Load curated dataset into session state directly from dashboard."""
    curated_dict = get_curated_datasets()
    meta = curated_dict[selected_key]
    df = load_curated_dataset(selected_key)
    inspection = inspect_dataset(df, meta["file_name"])

    all_cols = df.columns.tolist()
    num_cols = inspection["numerical_cols"]
    detected_label = meta.get("default_label")
    detected_ids = [c for c in all_cols if any(t in c.lower() for t in ["id", "sample", "leaf", "pot", "stage"]) and c != detected_label]
    detected_features = [c for c in num_cols if c != detected_label and c not in detected_ids]

    st.session_state["raw_df"] = df
    st.session_state["raw_file_name"] = meta["file_name"]
    st.session_state["dataset_loaded"] = True
    st.session_state["dataset_inspection"] = inspection
    st.session_state["wavelength_meta"] = inspection["wavelength_meta"]
    st.session_state["feature_cols"] = detected_features
    st.session_state["label_col"] = detected_label
    st.session_state["id_cols"] = detected_ids
    reset_downstream_from_dataset()
    return df, detected_features, detected_label, detected_ids


def render():
    """Render the academic project overview and theoretical foundation."""
    # Academic Header Banner
    st.markdown(
        f"""
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #0D9488; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.35rem;">
                {PROJECT_TYPE}
            </div>
            <h1 style="font-size: 1.65rem; font-weight: 700; color: #0F172A; margin: 0 0 0.5rem 0; letter-spacing: -0.02em;">
                Hyperspectral Crop Disease Feature Analysis
            </h1>
            <p style="font-size: 0.95rem; color: #475569; margin: 0; line-height: 1.5;">
                {APP_SUBTITLE}. Benchmark raw high-dimensional reflectance bands against orthogonal linear PCA components and non-linear UMAP manifold embeddings.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1. Active Experiment State Callout (Productive resume / start fresh)
    dataset_loaded = st.session_state.get("dataset_loaded", False) and st.session_state.get("raw_df") is not None

    if dataset_loaded:
        df = st.session_state["raw_df"]
        n_bands = len(st.session_state.get("feature_cols", []))
        has_label = st.session_state.get("label_col") is not None
        fname = st.session_state.get("raw_file_name", "Uploaded Dataset")

        # Determine next logical step
        if not is_preprocessing_ready():
            next_step_name = "Preprocessing"
            next_step_id = "preprocessing"
        elif not is_pca_ready():
            next_step_name = "PCA Reduction"
            next_step_id = "pca"
        elif not is_umap_ready():
            next_step_name = "UMAP Embedding"
            next_step_id = "umap"
        elif not is_clustering_ready():
            next_step_name = "K-Means Clustering"
            next_step_id = "clustering"
        elif not is_evaluation_ready():
            next_step_name = "Benchmarking"
            next_step_id = "comparison"
        elif not is_visualization_ready():
            next_step_name = "Visualizations"
            next_step_id = "visualization"
        else:
            next_step_name = "Results & Report"
            next_step_id = "results"

        st.markdown(
            f"""
            <div style="background: #F0FDF4; border: 1.5px solid #86EFAC; border-radius: 8px; padding: 1.15rem 1.35rem; margin-bottom: 1.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.75rem;">
                    <div>
                        <div style="font-weight: 700; font-size: 1.05rem; color: #166534;">
                            📁 Active Dataset Loaded: <strong>{fname}</strong>
                        </div>
                        <div style="font-size: 0.85rem; color: #15803D; margin-top: 0.25rem;">
                            {len(df):,} pixel samples • {n_bands} active spectral bands • Label: {'Present' if has_label else 'Purely Unsupervised'}
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_act1, col_act2 = st.columns([1.5, 1])
        with col_act1:
            if st.button(f"➡️ Continue Active Step ({next_step_name})", type="primary", use_container_width=True):
                set_page(next_step_id)
                st.rerun()
        with col_act2:
            if st.button("🔄 Start Fresh", type="secondary", use_container_width=True):
                reset_full_experiment()
                st.toast("Started fresh experiment! Select or upload a dataset.", icon="🔄")
                st.rerun()

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    else:
        st.markdown(
            """
            <div style="background: #F8FAFC; border: 1.5px solid #CBD5E1; border-radius: 8px; padding: 1.25rem 1.45rem; margin-bottom: 1.5rem;">
                <div style="font-weight: 700; font-size: 1.05rem; color: #1E293B; margin-bottom: 0.35rem;">
                    🚀 Quick Start: Select a Curated Hyperspectral Benchmark or Upload Data
                </div>
                <div style="font-size: 0.86rem; color: #64748B; margin-bottom: 1rem;">
                    Jump right into unsupervised learning using one of our verified agricultural hyperspectral datasets, or upload your own file.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        curated_dict = get_curated_datasets()
        col_sel, col_desc = st.columns([1.5, 2])
        with col_sel:
            selected_ds = st.selectbox(
                "Choose Curated Benchmark Dataset:",
                options=list(curated_dict.keys()),
                format_func=lambda k: curated_dict[k]["title"],
                key="dash_curated_select"
            )
        with col_desc:
            ds_meta = curated_dict[selected_ds]
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.55rem 0.85rem; font-size: 0.82rem; color: #475569; margin-top: 0.2rem;">
                    {ds_meta['description']} (<strong>{ds_meta['samples']} samples</strong>, <strong>{ds_meta['bands']} bands</strong>)
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        col_btn1, col_btn2 = st.columns([1.5, 1])
        with col_btn1:
            if st.button("🚀 Load Selected Dataset & Begin", type="primary", use_container_width=True, help="Loads dataset and navigates to Dataset exploration step."):
                _load_and_store_curated(selected_ds)
                set_page("dataset")
                st.rerun()

        with col_btn2:
            if st.button("📤 Upload Custom File ➡️", type="secondary", use_container_width=True):
                set_page("dataset")
                st.rerun()

        st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # 2. Academic Problem Statement & Research Foundation
    st.markdown(
        """
        <div class="sci-card">
            <div class="sci-card-header">
                <span>🔬</span> Scientific Problem Statement & Theoretical Motivation
            </div>
            <div style="font-size: 0.88rem; color: #334155; line-height: 1.6;">
                <p>
                    Hyperspectral imaging sensors capture narrow, contiguous spectral bands spanning the visible and near-infrared 
                    (VNIR: 400–1000 nm) domains. While this provides detailed biochemical signatures of foliar tissue (chlorophyll absorption, 
                    cell wall scattering, water content), it introduces severe mathematical challenges for unsupervised machine learning:
                </p>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 0.85rem; margin-top: 0.75rem;">
                    <div style="background: #F8FAFC; padding: 0.85rem; border-radius: 6px; border: 1px solid #E2E8F0;">
                        <strong style="color: #991B1B;">1. High Inter-Band Collinearity</strong>
                        <p style="margin: 0.25rem 0 0 0; color: #64748B; font-size: 0.82rem;">
                            Adjacent spectral wavelengths exhibit correlation coefficients exceeding 0.95, resulting in massive feature redundancy and ill-conditioned covariance matrices.
                        </p>
                    </div>
                    <div style="background: #F8FAFC; padding: 0.85rem; border-radius: 6px; border: 1px solid #E2E8F0;">
                        <strong style="color: #9A3412;">2. Curse of Dimensionality (Hughes Effect)</strong>
                        <p style="margin: 0.25rem 0 0 0; color: #64748B; font-size: 0.82rem;">
                            As band count increases without proportional exponential sample scaling, the high-dimensional Euclidean space becomes sparse, distorting distance-based clustering.
                        </p>
                    </div>
                    <div style="background: #F8FAFC; padding: 0.85rem; border-radius: 6px; border: 1px solid #E2E8F0;">
                        <strong style="color: #166534;">3. Disease Feature Preservation</strong>
                        <p style="margin: 0.25rem 0 0 0; color: #64748B; font-size: 0.82rem;">
                            Dimensionality reduction must compress the spectral space while preserving subtle physiological deviations caused by fungal, viral, or bacterial crop pathogens.
                        </p>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 3. Comparative Framework Cards
    st.markdown("### ⚖️ Dimensionality Reduction Paradigms Under Study")
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        st.markdown(
            """
            <div class="sci-card" style="height: 100%;">
                <div class="sci-card-header">
                    <span>🌐</span> Raw Spectral Baseline
                </div>
                <div style="font-size: 0.84rem; color: #475569; line-height: 1.5;">
                    <strong>Nature:</strong> Full preprocessed spectral channels.<br>
                    <strong>Hypothesis:</strong> Serves as the uncompressed ground-truth control. Suffers from high dimensionality and noise in non-informative water-absorption channels.<br>
                    <strong>Target:</strong> Baseline for downstream K-Means clustering quality.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_f2:
        st.markdown(
            """
            <div class="sci-card" style="height: 100%;">
                <div class="sci-card-header">
                    <span>📉</span> Linear Orthogonal PCA
                </div>
                <div style="font-size: 0.84rem; color: #475569; line-height: 1.5;">
                    <strong>Nature:</strong> Global variance maximization.<br>
                    <strong>Mechanism:</strong> Eigendecomposition of the sample covariance matrix \\( \\mathbf{C} = \\frac{1}{n} \\mathbf{X}^T \\mathbf{X} \\). Uncorrelated orthogonal principal axes.<br>
                    <strong>Strengths:</strong> Fast, deterministic, captures dominant global reflectance variance.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_f3:
        st.markdown(
            """
            <div class="sci-card" style="height: 100%;">
                <div class="sci-card-header">
                    <span>🌐</span> Non-Linear UMAP Manifold
                </div>
                <div style="font-size: 0.84rem; color: #475569; line-height: 1.5;">
                    <strong>Nature:</strong> Riemannian manifold approximation.<br>
                    <strong>Mechanism:</strong> Fuzzy simplicial sets + cross-entropy optimization. Models the non-linear curvature of spectral reflectance variations.<br>
                    <strong>Strengths:</strong> Separates non-linearly clustered pathology classes and compacts intra-class variance.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 4. Sequential Scientific Workflow Diagram
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown("### 📋 Sequential Workflow Architecture")
    
    workflow_steps = [
        {"icon": "📁", "title": "Dataset Upload", "desc": "Upload CSV/Excel/Parquet and designate spectral band columns."},
        {"icon": "⚙️", "title": "Preprocessing", "desc": "Sanitize data: impute NaNs, eliminate zero-variance bands, normalize."},
        {"icon": "📉", "title": "PCA Reduction", "desc": "Fit orthogonal principal components retaining 90-99% cumulative variance."},
        {"icon": "🌐", "title": "UMAP Embedding", "desc": "Construct fuzzy simplicial sets and project into 2D/3D topological manifolds."},
        {"icon": "🎯", "title": "K-Means Clustering", "desc": "Independently partition Raw, PCA, and UMAP feature spaces."},
        {"icon": "⚖️", "title": "Benchmarking", "desc": "Calculate Silhouette, Davies-Bouldin, Calinski-Harabasz, ARI, and NMI."},
        {"icon": "🔍", "title": "Visualizations", "desc": "Inspect 2D/3D projections, Scree plots, loadings, and disease preservation."},
        {"icon": "📑", "title": "Results & Exports", "desc": "Synthesize empirical academic conclusions and download data artifacts."},
    ]

    col_w1, col_w2 = st.columns(2)
    for idx, s in enumerate(workflow_steps):
        target_col = col_w1 if idx % 2 == 0 else col_w2
        with target_col:
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 0.65rem; display: flex; align-items: flex-start; gap: 0.75rem;">
                    <span style="display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px; border-radius: 6px; background: #EEF2FF; font-size: 0.95rem; flex-shrink: 0;">
                        {s['icon']}
                    </span>
                    <div>
                        <div style="font-weight: 600; font-size: 0.88rem; color: #1E293B;">{s['title']}</div>
                        <div style="font-size: 0.8rem; color: #64748B; margin-top: 0.15rem;">{s['desc']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

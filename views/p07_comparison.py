"""
Page 7: Comparative Benchmarking (Raw vs PCA vs UMAP)
Course: INT396 - Unsupervised Learning
"""

import streamlit as st
import pandas as pd
import numpy as np
from components.status_tracker import render_pipeline_header, render_pipeline_dependency_warning, render_step_footer
from evaluation.comparator import generate_comparison_benchmark
from visualization.evaluation_plots import (
    plot_benchmark_metric_bars,
    plot_external_benchmark_bars,
    plot_contingency_heatmap,
)
from config.settings import METRIC_DEFINITIONS
from utils.state import is_clustering_ready, is_evaluation_ready, set_page


def render():
    """Render the Comparative Benchmark page."""
    render_pipeline_header(
        "Comparative Benchmarking: Raw vs PCA vs UMAP",
        "Rigorous quantitative comparison of downstream clustering quality across preprocessed raw bands, linear PCA components, and non-linear UMAP embeddings."
    )

    if not is_clustering_ready():
        render_pipeline_dependency_warning("K-Means Clustering", "clustering", "Clustering")
        return

    X_raw = st.session_state["X_clean"]
    X_pca = st.session_state["pca_transformed"]
    X_umap = st.session_state["umap_transformed"]
    
    raw_labels = st.session_state["kmeans_raw_labels"]
    pca_labels = st.session_state["kmeans_pca_labels"]
    umap_labels = st.session_state["kmeans_umap_labels"]
    
    label_series = st.session_state.get("y_clean")
    has_labels = (label_series is not None)
    
    raw_dim = X_raw.shape[1]
    pca_dim = X_pca.shape[1]
    umap_dim = X_umap.shape[1]
    n_clusters = st.session_state["kmeans_config"].get("n_clusters", 3)

    # Compute or retrieve benchmark results
    if (
        not st.session_state.get("evaluation_done", False)
        or st.session_state.get("comparison_table") is None
    ):
        with st.spinner("Calculating internal and external clustering validation metrics..."):
            benchmark_res = generate_comparison_benchmark(
                X_raw=X_raw,
                raw_labels=raw_labels,
                raw_dim=raw_dim,
                X_pca=X_pca,
                pca_labels=pca_labels,
                pca_dim=pca_dim,
                X_umap=X_umap,
                umap_labels=umap_labels,
                umap_dim=umap_dim,
                n_clusters=n_clusters,
                y_true=label_series,
            )
            st.session_state["comparison_table"] = benchmark_res["benchmark_df"]
            st.session_state["detailed_metrics"] = benchmark_res["detailed_metrics"]
            st.session_state["dynamic_interpretations"] = benchmark_res["interpretation"]
            st.session_state["evaluation_done"] = True

    benchmark_df = st.session_state["comparison_table"]
    detailed_metrics = st.session_state["detailed_metrics"]

    # Success Banner
    st.markdown(
        f"""
        <div style="background: #ECFDF5; border: 1.5px solid #A7F3D0; border-radius: 8px; padding: 1.15rem 1.35rem; margin-top: 1rem; margin-bottom: 1.25rem;">
            <div style="font-weight: 700; color: #065F46; font-size: 1.05rem;">
                ✅ Tri-Space Clustering Benchmark Successfully Computed!
            </div>
            <div style="font-size: 0.85rem; color: #047857; margin-top: 0.2rem;">
                Compared Raw ({raw_dim} bands) vs PCA ({pca_dim} PCs) vs UMAP ({umap_dim}D). Internal validation indices and compression evaluated. Inspect the comparative charts below and proceed to Visualizations at the bottom.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1. Primary Benchmark Table
    st.markdown('<div class="sci-card">', unsafe_allow_html=True)
    st.markdown('<div class="sci-card-header"><span>⚖️</span> Unified Clustering Benchmark Table</div>', unsafe_allow_html=True)
    st.dataframe(benchmark_df, use_container_width=True, hide_index=True)
    
    if not has_labels:
        st.caption("ℹ️ *Note: External metrics (ARI, NMI) are displayed as 'N/A' because no ground-truth disease label was designated in the dataset. Values are never fabricated.*")
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. Metric Definition Reference Accordion
    with st.expander("📚 Metric Definitions & Evaluation Criteria (Academic Reference)", expanded=False):
        for metric_name, desc in METRIC_DEFINITIONS.items():
            st.markdown(f"**{metric_name}**: {desc}")

    # 3. Comparative Benchmark Visualizations
    st.markdown("### 📊 Benchmark Visualizations")
    tab_int, tab_ext = st.tabs(["📐 Internal Cluster Quality (Silhouette & Davies-Bouldin)", "🎯 External Disease Signal Preservation (ARI & NMI)"])

    with tab_int:
        fig_int = plot_benchmark_metric_bars(detailed_metrics)
        st.plotly_chart(fig_int, use_container_width=True)

    with tab_ext:
        if has_labels:
            fig_ext = plot_external_benchmark_bars(detailed_metrics)
            st.plotly_chart(fig_ext, use_container_width=True)

            # Contingency matrices
            st.markdown("#### Contingency Cross-Tabulations (Clusters vs Disease Labels)")
            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1:
                contingency_raw = detailed_metrics["raw"].get("contingency_df")
                fig_c_raw = plot_contingency_heatmap(contingency_raw, title="Raw Features")
                st.plotly_chart(fig_c_raw, use_container_width=True)
            with col_c2:
                contingency_pca = detailed_metrics["pca"].get("contingency_df")
                fig_c_pca = plot_contingency_heatmap(contingency_pca, title="PCA Reduced")
                st.plotly_chart(fig_c_pca, use_container_width=True)
            with col_c3:
                contingency_umap = detailed_metrics["umap"].get("contingency_df")
                fig_c_umap = plot_contingency_heatmap(contingency_umap, title="UMAP Embedded")
                st.plotly_chart(fig_c_umap, use_container_width=True)
        else:
            st.markdown(
                """
                <div class="sci-callout sci-callout-info">
                    ℹ️ <strong>External Validation Requires Ground-Truth Labels:</strong><br>
                    Because this dataset is operating in purely unsupervised mode without ground-truth disease classes,
                    Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) cannot be calculated.
                    All clustering benchmarking is rigorously evaluated through unsupervised internal geometric indices.
                </div>
                """,
                unsafe_allow_html=True
            )

    # 4. Dimension Compression Comparison Metric
    st.markdown("### 📉 Feature Space Compression")
    c_red1, c_red2, c_red3 = st.columns(3)
    with c_red1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Raw Hyperspectral Baseline</div>
                <div class="metric-val">{raw_dim} bands</div>
                <div class="metric-sub">0% compression (full spectrum)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c_red2:
        pca_pct = round((1 - pca_dim / raw_dim) * 100, 1)
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">PCA Compressed Space</div>
                <div class="metric-val">{pca_dim} components</div>
                <div class="metric-sub">{pca_pct}% dimensionality reduction</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c_red3:
        umap_pct = round((1 - umap_dim / raw_dim) * 100, 1)
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">UMAP Embedded Space</div>
                <div class="metric-val">{umap_dim} dimensions</div>
                <div class="metric-sub">{umap_pct}% dimensionality reduction</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Standardized Bidirectional Navigation Footer
    render_step_footer(
        current_page_id="comparison",
        next_page_id="visualization",
        prev_page_id="clustering",
        can_proceed=True,
    )

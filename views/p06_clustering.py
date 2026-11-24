"""
Page 6: K-Means Clustering on Raw, PCA, and UMAP Representations
Course: INT396 - Unsupervised Learning
"""

import streamlit as st
import pandas as pd
import numpy as np
from components.status_tracker import render_pipeline_header, render_pipeline_dependency_warning, render_step_footer
from clustering.kmeans_model import run_tri_representation_clustering
from clustering.cluster_analysis import compute_multi_representation_k_analysis
from visualization.evaluation_plots import plot_elbow_and_silhouette_curves, plot_elbow_inertia_curves
from visualization.projection_plots import plot_projection_2d
from utils.state import (
    is_preprocessing_ready, is_pca_ready, is_umap_ready, is_clustering_ready,
    reset_downstream_from_reduction, set_page
)


def render():
    """Render K-Means clustering configuration, Elbow analysis, and execution."""
    render_pipeline_header(
        "K-Means Clustering & K Selection Analysis",
        "Perform independent K-Means partitioning across Raw, PCA, and UMAP representations, and conduct Elbow & Silhouette score profiling to determine optimal cluster counts."
    )

    if not is_preprocessing_ready():
        render_pipeline_dependency_warning("Preprocessing Pipeline", "preprocessing", "Preprocessing")
        return
    if not is_pca_ready():
        render_pipeline_dependency_warning("PCA Dimensionality Reduction", "pca", "PCA")
        return
    if not is_umap_ready():
        render_pipeline_dependency_warning("UMAP Manifold Learning", "umap", "UMAP")
        return

    X_raw = st.session_state["X_clean"]
    X_pca = st.session_state["pca_transformed"]
    X_umap = st.session_state["umap_transformed"]
    label_series = st.session_state.get("y_clean")
    n_samples = X_raw.shape[0]

    # 1. Cluster Number Analysis (Elbow & Silhouette)
    st.markdown('<div class="sci-card">', unsafe_allow_html=True)
    st.markdown('<div class="sci-card-header"><span>📐</span> Cluster Number Analysis (Elbow & Silhouette Methods)</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <p style="font-size: 0.85rem; color: #475569; margin-bottom: 0.75rem;">
            Run empirical profiling over candidate cluster numbers (K = 2 to 8) across Raw, PCA, and UMAP spaces.
            Inspect the point of inflection (Elbow) and the peak Silhouette Score to inform your selection of K.
        </p>
        """,
        unsafe_allow_html=True
    )

    col_el1, col_el2 = st.columns([1, 2])
    with col_el1:
        run_elbow_btn = st.button("🔍 Run Elbow & Silhouette Analysis", use_container_width=True)

    if run_elbow_btn:
        with st.spinner("Computing WCSS inertia and Silhouette scores across K = 2...8 for all 3 feature spaces..."):
            try:
                analysis_results = compute_multi_representation_k_analysis(
                    X_raw=X_raw,
                    X_pca=X_pca,
                    X_umap=X_umap,
                    k_min=2,
                    k_max=min(8, n_samples - 1),
                    random_state=st.session_state["kmeans_config"].get("random_state", 42),
                )
                st.session_state["elbow_results"] = analysis_results
                st.session_state["elbow_analysis_done"] = True
                st.toast("Cluster number analysis computed across all representations!", icon="✅")
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")

    if st.session_state.get("elbow_analysis_done", False) and st.session_state.get("elbow_results"):
        analysis = st.session_state["elbow_results"]
        tab_sil, tab_wcss = st.tabs(["📈 Silhouette Score vs K", "📉 Elbow Inertia (WCSS) vs K"])
        with tab_sil:
            fig_sil = plot_elbow_and_silhouette_curves(analysis)
            st.plotly_chart(fig_sil, use_container_width=True)
        with tab_wcss:
            fig_wcss = plot_elbow_inertia_curves(analysis)
            st.plotly_chart(fig_wcss, use_container_width=True)

        best_pca_k = analysis["pca"]["best_k"]
        best_umap_k = analysis["umap"]["best_k"]
        st.markdown(
            f"""
            <div style="font-size: 0.82rem; color: #334155; background: #F8FAFC; padding: 0.6rem 0.85rem; border-radius: 6px; border: 1px solid #E2E8F0;">
                💡 <em>Empirical Silhouette Peak:</em> Optimal K in PCA space = <strong>{best_pca_k}</strong> (score: {analysis['pca']['best_silhouette']});
                in UMAP space = <strong>{best_umap_k}</strong> (score: {analysis['umap']['best_silhouette']}).
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # 2. K-Means Configuration Panel
    st.markdown('<div class="sci-card">', unsafe_allow_html=True)
    st.markdown('<div class="sci-card-header"><span>🎯</span> K-Means Hyperparameter Configuration</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        max_possible_k = min(15, n_samples - 1)
        current_k = st.session_state["kmeans_config"].get("n_clusters", 3)
        safe_k = max(2, min(current_k, max_possible_k))
        
        selected_k = st.slider(
            f"Number of Clusters (K: 2 to {max_possible_k}):",
            min_value=2,
            max_value=max_possible_k,
            value=safe_k,
            help="Number of centroids to fit across all three representations."
        )

    with col2:
        n_init = st.slider(
            "Number of Initializations (n_init):",
            min_value=1,
            max_value=25,
            value=st.session_state["kmeans_config"].get("n_init", 10),
            help="Number of times the K-Means algorithm runs with different centroid seeds to avoid local minima."
        )

    with col3:
        random_state = st.number_input(
            "Random State Seed:",
            min_value=0,
            max_value=99999,
            value=st.session_state["kmeans_config"].get("random_state", 42),
            help="Controls centroid initialization for reproducibility."
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Execution Action Button
    col_b1, col_b2, col_b3 = st.columns([1, 1.4, 1])
    with col_b2:
        is_done = st.session_state.get("clustering_done", False)
        btn_label = f"🔄 Re-run K-Means on Raw, PCA & UMAP (K={selected_k})" if is_done else f"▶️ Fit K-Means on Raw, PCA & UMAP (K={selected_k})"
        run_clustering_btn = st.button(btn_label, type="primary", use_container_width=True)

    if run_clustering_btn:
        with st.spinner(f"Fitting independent K-Means (K={selected_k}) on Raw, PCA, and UMAP feature spaces..."):
            try:
                cluster_results = run_tri_representation_clustering(
                    X_raw=X_raw,
                    X_pca=X_pca,
                    X_umap=X_umap,
                    n_clusters=selected_k,
                    random_state=random_state,
                    n_init=n_init,
                )

                # Store models and label assignments in session state
                st.session_state["kmeans_raw_labels"] = cluster_results["raw"]["labels"]
                st.session_state["kmeans_pca_labels"] = cluster_results["pca"]["labels"]
                st.session_state["kmeans_umap_labels"] = cluster_results["umap"]["labels"]
                
                st.session_state["kmeans_raw_model"] = cluster_results["raw"]["model"]
                st.session_state["kmeans_pca_model"] = cluster_results["pca"]["model"]
                st.session_state["kmeans_umap_model"] = cluster_results["umap"]["model"]
                
                st.session_state["kmeans_tri_results"] = cluster_results
                st.session_state["clustering_done"] = True
                st.session_state["kmeans_config"] = {
                    "n_clusters": selected_k,
                    "random_state": random_state,
                    "n_init": n_init,
                }

                # Reset downstream evaluation
                st.session_state["evaluation_done"] = False
                st.session_state["comparison_table"] = None
                st.toast(f"K-Means (K={selected_k}) independently fitted on Raw, PCA, and UMAP!", icon="✅")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Clustering execution failed: {str(e)}")
                return

    # Render Clustering Output and Visualizations if completed
    if st.session_state.get("clustering_done", False) and st.session_state.get("kmeans_tri_results"):
        tri = st.session_state["kmeans_tri_results"]
        raw_dist = tri["raw"]["cluster_distribution"]
        pca_dist = tri["pca"]["cluster_distribution"]
        umap_dist = tri["umap"]["cluster_distribution"]

        st.markdown(
            f"""
            <div style="background: #ECFDF5; border: 1.5px solid #A7F3D0; border-radius: 8px; padding: 1.15rem 1.35rem; margin-top: 1.25rem; margin-bottom: 1.25rem;">
                <div style="font-weight: 700; color: #065F46; font-size: 1.05rem;">
                    ✅ Tri-Space K-Means (K={selected_k}) Successfully Fitted!
                </div>
                <div style="font-size: 0.85rem; color: #047857; margin-top: 0.2rem;">
                    Generated independent unsupervised partitions across Raw ({X_raw.shape[1]} bands), PCA ({X_pca.shape[1]} PCs), and UMAP ({X_umap.shape[1]}D). Inspect the cluster distributions and projections below, then proceed to Comparative Benchmarking at the bottom.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.markdown("### 📊 Independent Cluster Size Distributions")

        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.markdown('<div class="sci-card">', unsafe_allow_html=True)
            st.markdown('<div class="sci-card-header"><span>🌐</span> Raw Space Clusters</div>', unsafe_allow_html=True)
            for c_name, count in raw_dist.items():
                pct = (count / n_samples) * 100
                st.write(f"• **{c_name}**: {count:,} samples ({pct:.1f}%)")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_d2:
            st.markdown('<div class="sci-card">', unsafe_allow_html=True)
            st.markdown('<div class="sci-card-header"><span>📉</span> PCA Space Clusters</div>', unsafe_allow_html=True)
            for c_name, count in pca_dist.items():
                pct = (count / n_samples) * 100
                st.write(f"• **{c_name}**: {count:,} samples ({pct:.1f}%)")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_d3:
            st.markdown('<div class="sci-card">', unsafe_allow_html=True)
            st.markdown('<div class="sci-card-header"><span>🌐</span> UMAP Space Clusters</div>', unsafe_allow_html=True)
            for c_name, count in umap_dist.items():
                pct = (count / n_samples) * 100
                st.write(f"• **{c_name}**: {count:,} samples ({pct:.1f}%)")
            st.markdown('</div>', unsafe_allow_html=True)

        # Cluster Partition Projections
        st.markdown("### 🗺️ Discovered Cluster Partition Projections")
        tab_pca_clust, tab_umap_clust = st.tabs(["PCA Projection (Colored by PCA Clusters)", "UMAP Projection (Colored by UMAP Clusters)"])
        
        with tab_pca_clust:
            fig_pca_c = plot_projection_2d(
                coords=X_pca,
                dim1_name="PC1",
                dim2_name="PC2",
                title=f"PCA Space Colored by PCA-Derived K-Means Clusters (K={selected_k})",
                color_by="cluster",
                cluster_labels=st.session_state["kmeans_pca_labels"],
            )
            st.plotly_chart(fig_pca_c, use_container_width=True)

        with tab_umap_clust:
            fig_umap_c = plot_projection_2d(
                coords=X_umap,
                dim1_name="UMAP1",
                dim2_name="UMAP2",
                title=f"UMAP Space Colored by UMAP-Derived K-Means Clusters (K={selected_k})",
                color_by="cluster",
                cluster_labels=st.session_state["kmeans_umap_labels"],
            )
            st.plotly_chart(fig_umap_c, use_container_width=True)

    else:
        st.markdown(
            """
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.85rem 1.15rem; margin-top: 1.25rem; color: #475569; font-size: 0.85rem;">
                ⏳ <strong>Status:</strong> Ready to cluster. Configure number of clusters (K) and initializations above, then click <strong>Fit K-Means on Raw, PCA & UMAP</strong>.
            </div>
            """,
            unsafe_allow_html=True
        )

    # Standardized Bidirectional Navigation Footer
    render_step_footer(
        current_page_id="clustering",
        next_page_id="comparison",
        prev_page_id="umap",
        can_proceed=is_clustering_ready(),
        proceed_hint="Click 'Fit K-Means on Raw, PCA & UMAP' above before proceeding to Benchmarking.",
    )

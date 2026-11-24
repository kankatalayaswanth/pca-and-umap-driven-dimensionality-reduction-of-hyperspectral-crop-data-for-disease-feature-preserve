"""
Page 8: Multi-Representation Visualizations & Disease Signal Preservation
Course: INT396 - Unsupervised Learning
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from components.status_tracker import render_pipeline_header, render_pipeline_dependency_warning, render_step_footer
from visualization.projection_plots import plot_projection_2d, plot_projection_3d
from config.settings import QUALITATIVE_PALETTE
from utils.state import is_clustering_ready, set_page


def render():
    """Render interactive projection and disease signal preservation visualizations."""
    # Mark visualizations as active/viewed
    st.session_state["visualization_viewed"] = True

    render_pipeline_header(
        "Interactive Visualizations & Signal Preservation",
        "Explore multi-dimensional projections side-by-side: contrast unsupervised K-Means partitions against true crop pathology labels in PCA and UMAP manifolds."
    )

    if not is_clustering_ready():
        render_pipeline_dependency_warning("K-Means Clustering", "clustering", "Clustering")
        return

    # Top Status Banner
    st.markdown(
        """
        <div style="background: #ECFDF5; border: 1.5px solid #A7F3D0; border-radius: 8px; padding: 1.15rem 1.35rem; margin-top: 1rem; margin-bottom: 1.25rem;">
            <div style="font-weight: 700; color: #065F46; font-size: 1.05rem;">
                ✅ Multi-Representation Manifolds & Disease Signals Active
            </div>
            <div style="font-size: 0.85rem; color: #047857; margin-top: 0.2rem;">
                Explore 2D and 3D projections below. When finished, proceed to Results & Report at the bottom to review the complete scientific report and download data artifacts.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    X_pca = st.session_state["pca_transformed"]
    X_umap = st.session_state["umap_transformed"]
    pca_labels = st.session_state["kmeans_pca_labels"]
    umap_labels = st.session_state["kmeans_umap_labels"]
    label_series = st.session_state.get("y_clean")
    has_labels = (label_series is not None)
    pca_stats = st.session_state["pca_stats"]

    # 1. Disease / Pest Signal Preservation Section (if labels present)
    if has_labels:
        st.markdown("### 🌾 Crop Disease / Pathology Structure Preservation")
        
        col_l1, col_l2 = st.columns([1, 2.5])
        with col_l1:
            st.markdown('<div class="sci-card">', unsafe_allow_html=True)
            st.markdown('<div class="sci-card-header"><span>🏷️</span> Disease Class Distribution</div>', unsafe_allow_html=True)
            counts = label_series.value_counts()
            for cls_name, cnt in counts.items():
                pct = (cnt / len(label_series)) * 100
                st.write(f"• **{cls_name}**: {cnt:,} ({pct:.1f}%)")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_l2:
            st.markdown('<div class="sci-card">', unsafe_allow_html=True)
            st.markdown('<div class="sci-card-header"><span>📊</span> Class Prevalence Bar Chart</div>', unsafe_allow_html=True)
            dist_df = pd.DataFrame({"Class": counts.index.astype(str), "Count": counts.values})
            fig_bar = px.bar(
                dist_df, x="Class", y="Count",
                color="Class", color_discrete_sequence=QUALITATIVE_PALETTE,
                template="simple_white", height=240
            )
            fig_bar.update_layout(showlegend=False, margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Side-by-side Disease Ground Truth Projections
        st.markdown("#### Side-by-Side: PCA vs UMAP Colored by True Crop Disease")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            fig_pca_gt = plot_projection_2d(
                coords=X_pca,
                dim1_name="PC1",
                dim2_name="PC2",
                title="PCA Projection (Colored by True Disease)",
                color_by="ground_truth",
                true_labels=label_series,
            )
            st.plotly_chart(fig_pca_gt, use_container_width=True)

        with col_p2:
            fig_umap_gt = plot_projection_2d(
                coords=X_umap,
                dim1_name="UMAP1",
                dim2_name="UMAP2",
                title="UMAP Embedding (Colored by True Disease)",
                color_by="ground_truth",
                true_labels=label_series,
            )
            st.plotly_chart(fig_umap_gt, use_container_width=True)

    else:
        st.markdown(
            """
            <div class="sci-callout sci-callout-info">
                ℹ️ <strong>External Disease Validation Note:</strong><br>
                The uploaded dataset has no ground-truth disease or pest classification label designated.
                Projections are colored strictly by unsupervised K-Means cluster assignments.
            </div>
            """,
            unsafe_allow_html=True
        )

    # 2. Side-by-Side Discovered K-Means Clusters
    st.markdown("---")
    st.markdown("### 🎯 Unsupervised K-Means Cluster Partitions")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        fig_pca_cl = plot_projection_2d(
            coords=X_pca,
            dim1_name="PC1",
            dim2_name="PC2",
            title="PCA Space (Colored by PCA K-Means Clusters)",
            color_by="cluster",
            cluster_labels=pca_labels,
        )
        st.plotly_chart(fig_pca_cl, use_container_width=True)

    with col_c2:
        fig_umap_cl = plot_projection_2d(
            coords=X_umap,
            dim1_name="UMAP1",
            dim2_name="UMAP2",
            title="UMAP Space (Colored by UMAP K-Means Clusters)",
            color_by="cluster",
            cluster_labels=umap_labels,
        )
        st.plotly_chart(fig_umap_cl, use_container_width=True)

    # 3. Interactive 3D Exploration Tab (if available)
    if X_pca.shape[1] >= 3 or X_umap.shape[1] >= 3:
        st.markdown("---")
        st.markdown("### 🌐 Interactive 3D Rotational Manifolds")
        
        tab_3d_pca, tab_3d_umap = st.tabs(["PCA 3D Manifold", "UMAP 3D Manifold"])
        with tab_3d_pca:
            if X_pca.shape[1] >= 3:
                fig_3d_pca = plot_projection_3d(
                    coords=X_pca,
                    dim1_name="PC1",
                    dim2_name="PC2",
                    dim3_name="PC3",
                    title="Interactive PCA 3D Orbit (PC1 × PC2 × PC3)",
                    color_by="cluster",
                    cluster_labels=pca_labels,
                )
                st.plotly_chart(fig_3d_pca, use_container_width=True)
            else:
                st.info("PCA was computed with fewer than 3 components.")
                
        with tab_3d_umap:
            if X_umap.shape[1] >= 3:
                fig_3d_umap = plot_projection_3d(
                    coords=X_umap,
                    dim1_name="UMAP1",
                    dim2_name="UMAP2",
                    dim3_name="UMAP3",
                    title="Interactive UMAP 3D Orbit (UMAP1 × UMAP2 × UMAP3)",
                    color_by="cluster",
                    cluster_labels=umap_labels,
                )
                st.plotly_chart(fig_3d_umap, use_container_width=True)
            else:
                st.info("UMAP was fitted with n_components=2. To view 3D manifold, configure n_components=3 on the UMAP page.")

    # Standardized Bidirectional Navigation Footer
    render_step_footer(
        current_page_id="visualization",
        next_page_id="results",
        prev_page_id="comparison",
        can_proceed=True,
    )

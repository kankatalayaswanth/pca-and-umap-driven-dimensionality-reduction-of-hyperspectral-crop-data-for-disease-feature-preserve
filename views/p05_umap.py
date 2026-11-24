"""
Page 5: Uniform Manifold Approximation and Projection (UMAP)
Course: INT396 - Unsupervised Learning
"""

import streamlit as st
import pandas as pd
import numpy as np
from components.status_tracker import render_pipeline_header, render_pipeline_dependency_warning, render_step_footer
from dimensionality_reduction.umap_model import run_umap_reduction
from visualization.projection_plots import plot_projection_2d, plot_projection_3d
from utils.state import is_preprocessing_ready, is_umap_ready, reset_downstream_from_reduction, set_page


def render():
    """Render UMAP Manifold Learning page."""
    render_pipeline_header(
        "Uniform Manifold Approximation and Projection (UMAP)",
        "Apply Riemannian geometry and fuzzy simplicial set theory to construct a low-dimensional manifold embedding that preserves non-linear local neighborhood topologies of spectral signatures."
    )

    if not is_preprocessing_ready():
        render_pipeline_dependency_warning("Preprocessing Pipeline", "preprocessing", "Preprocessing")
        return

    X = st.session_state["X_clean"]
    label_series = st.session_state.get("y_clean")
    n_samples, n_features = X.shape

    # UMAP Configuration Form
    st.markdown('<div class="sci-card">', unsafe_allow_html=True)
    st.markdown('<div class="sci-card-header"><span>🌐</span> UMAP Hyperparameter Configuration</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # n_neighbors
        current_neighbors = st.session_state["umap_config"].get("n_neighbors", 15)
        safe_max_neighbors = max(3, min(100, n_samples - 1))
        safe_val_neighbors = max(2, min(current_neighbors, safe_max_neighbors))

        n_neighbors = st.slider(
            f"Number of Neighbors (n_neighbors: 2 to {safe_max_neighbors}):",
            min_value=2,
            max_value=safe_max_neighbors,
            value=safe_val_neighbors,
            help="Controls how UMAP balances local versus global structure. Smaller values emphasize fine-grained local spectral clusters; larger values capture broader macroscopic trends."
        )

        # min_dist
        current_min_dist = st.session_state["umap_config"].get("min_dist", 0.1)
        min_dist = st.slider(
            "Minimum Distance (min_dist: 0.0 to 0.99):",
            min_value=0.0,
            max_value=0.99,
            value=float(current_min_dist),
            step=0.05,
            help="Controls how tightly UMAP packs points together. Low values create dense clumped clusters; higher values prevent overcrowding and preserve broad topological spread."
        )

    with col2:
        # Distance Metric
        metrics_dict = {
            "euclidean": "Euclidean (L2 distance in spectral reflectance space)",
            "cosine": "Cosine (Spectral angle distance — widely used in remote sensing)",
            "manhattan": "Manhattan (L1 taxicab distance)",
            "correlation": "Correlation (Pearson correlation distance between spectra)",
        }
        current_metric = st.session_state["umap_config"].get("metric", "euclidean")
        metric_keys = list(metrics_dict.keys())
        metric_idx = metric_keys.index(current_metric) if current_metric in metric_keys else 0

        metric = st.selectbox(
            "Distance Metric:",
            metric_keys,
            format_func=lambda k: metrics_dict[k],
            index=metric_idx,
            help="Distance metric used to evaluate pair-wise distance between hyperspectral pixels."
        )

        # Components and Random State
        c_sub1, c_sub2 = st.columns(2)
        with c_sub1:
            current_comp = st.session_state["umap_config"].get("n_components", 2)
            n_components = st.radio(
                "Embedded Dimension:",
                [2, 3],
                index=0 if current_comp == 2 else 1,
                horizontal=True,
                help="2D or 3D manifold embedding coordinates."
            )
        with c_sub2:
            random_state = st.number_input(
                "Random Seed:",
                min_value=0,
                max_value=99999,
                value=st.session_state["umap_config"].get("random_state", 42),
                help="Ensures reproducible stochastic gradient descent optimization."
            )

    st.markdown('</div>', unsafe_allow_html=True)

    # Academic Alert on UMAP variance
    st.markdown(
        """
        <div class="sci-callout sci-callout-info">
            💡 <strong>Academic Note on Manifold Learning:</strong><br>
            UMAP is a non-linear manifold learning algorithm that optimizes fuzzy topological representations.
            Unlike PCA, UMAP <em>does not possess orthogonal eigenvalues or an explained variance ratio</em>.
            Instead, UMAP preserves neighborhood connectivity and is evaluated downstream through clustering quality 
            and external disease signal preservation.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Execution Action Button
    col_b1, col_b2, col_b3 = st.columns([1, 1.3, 1])
    with col_b2:
        is_done = st.session_state.get("umap_done", False)
        btn_label = "🔄 Re-run UMAP Embedding" if is_done else "▶️ Fit & Embed UMAP Manifold"
        run_umap_btn = st.button(btn_label, type="primary", use_container_width=True)

    if run_umap_btn:
        with st.spinner("Constructing high-dimensional fuzzy simplicial set and optimizing low-dimensional embedding..."):
            try:
                res = run_umap_reduction(
                    X=X,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric=metric,
                    n_components=n_components,
                    random_state=random_state,
                )

                st.session_state["umap_model"] = res["model"]
                st.session_state["umap_transformed"] = res["transformed"]
                st.session_state["umap_df"] = res["umap_df"]
                st.session_state["umap_stats"] = res["stats"]
                st.session_state["umap_done"] = True
                st.session_state["umap_config"] = {
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
                    "metric": metric,
                    "n_components": n_components,
                    "random_state": random_state,
                }

                reset_downstream_from_reduction()
                st.toast(f"UMAP {n_components}D embedding generated successfully!", icon="✅")
                st.rerun()

            except Exception as e:
                st.error(f"❌ UMAP execution failed: {str(e)}")
                return

    # Render UMAP Results and Visualizations if completed
    if st.session_state.get("umap_done", False) and st.session_state.get("umap_stats"):
        stats = st.session_state["umap_stats"]
        coords = st.session_state["umap_transformed"]
        dim = stats["n_components"]

        st.markdown(
            f"""
            <div style="background: #ECFDF5; border: 1.5px solid #A7F3D0; border-radius: 8px; padding: 1.15rem 1.35rem; margin-top: 1.25rem; margin-bottom: 1.25rem;">
                <div style="font-weight: 700; color: #065F46; font-size: 1.05rem;">
                    ✅ UMAP Manifold Embedding Successfully Generated!
                </div>
                <div style="font-size: 0.85rem; color: #047857; margin-top: 0.2rem;">
                    Constructed <strong>{dim}D non-linear manifold</strong> preserving local topological neighborhoods (compressed by <strong>{stats['dimensionality_reduction_pct']:.1f}%</strong>). Review manifold diagnostics below and proceed to K-Means clustering at the bottom.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.markdown("### 📊 UMAP Embedding Diagnostics")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Embedded Dimensions</div>
                    <div class="metric-val">{dim}D</div>
                    <div class="metric-sub">Manifold coordinates</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m2:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Local Neighborhood</div>
                    <div class="metric-val">{stats['n_neighbors']}</div>
                    <div class="metric-sub">Nearest neighbors</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m3:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Distance Metric</div>
                    <div class="metric-val">{stats['metric'].capitalize()}</div>
                    <div class="metric-sub">Input spectral distance</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m4:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Dim. Reduction Ratio</div>
                    <div class="metric-val">{stats['dimensionality_reduction_pct']}%</div>
                    <div class="metric-sub">Feature space compressed</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # UMAP Visualizations
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        if dim == 3:
            tab_u2d, tab_u3d = st.tabs(["🗺️ UMAP 2D Projection (UMAP1 vs UMAP2)", "🌐 UMAP 3D Projection (UMAP1 × UMAP2 × UMAP3)"])
            with tab_u2d:
                fig_u2d = plot_projection_2d(
                    coords=coords,
                    dim1_name="UMAP1",
                    dim2_name="UMAP2",
                    title="UMAP 2D Projection of Hyperspectral Crop Data",
                    color_by="ground_truth" if label_series is not None else "none",
                    true_labels=label_series,
                )
                st.plotly_chart(fig_u2d, use_container_width=True)
            with tab_u3d:
                fig_u3d = plot_projection_3d(
                    coords=coords,
                    dim1_name="UMAP1",
                    dim2_name="UMAP2",
                    dim3_name="UMAP3",
                    title="Interactive UMAP 3D Manifold Embedding",
                    color_by="ground_truth" if label_series is not None else "none",
                    true_labels=label_series,
                )
                st.plotly_chart(fig_u3d, use_container_width=True)
        else:
            fig_u2d = plot_projection_2d(
                coords=coords,
                dim1_name="UMAP1",
                dim2_name="UMAP2",
                title="UMAP 2D Projection of Hyperspectral Crop Data",
                color_by="ground_truth" if label_series is not None else "none",
                true_labels=label_series,
            )
            st.plotly_chart(fig_u2d, use_container_width=True)

    else:
        st.markdown(
            """
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.85rem 1.15rem; margin-top: 1.25rem; color: #475569; font-size: 0.85rem;">
                ⏳ <strong>Status:</strong> Ready to fit. Configure number of neighbors, minimum distance, and metric above, then click <strong>Fit & Embed UMAP Manifold</strong>.
            </div>
            """,
            unsafe_allow_html=True
        )

    # Standardized Bidirectional Navigation Footer
    render_step_footer(
        current_page_id="umap",
        next_page_id="clustering",
        prev_page_id="pca",
        can_proceed=is_umap_ready(),
        proceed_hint="Click 'Fit & Embed UMAP Manifold' above before proceeding to K-Means.",
    )

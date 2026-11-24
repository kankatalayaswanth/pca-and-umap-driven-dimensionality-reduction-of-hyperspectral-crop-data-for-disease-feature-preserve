"""
Page 4: Principal Component Analysis (PCA) Dimensionality Reduction
Course: INT396 - Unsupervised Learning
"""

import streamlit as st
import pandas as pd
import numpy as np
from components.status_tracker import render_pipeline_header, render_pipeline_dependency_warning, render_step_footer
from dimensionality_reduction.pca_model import run_pca_reduction
from visualization.evaluation_plots import plot_pca_variance_diagnostics, plot_pca_spectral_loadings
from visualization.projection_plots import plot_projection_2d, plot_projection_3d
from utils.state import is_preprocessing_ready, is_pca_ready, reset_downstream_from_reduction, set_page


def render():
    """Render PCA Dimensionality Reduction page."""
    render_pipeline_header(
        "Principal Component Analysis (PCA)",
        "Perform linear orthogonal transformation to project high-dimensional hyperspectral reflectance bands onto uncorrelated principal axes maximizing global variance retention."
    )

    if not is_preprocessing_ready():
        render_pipeline_dependency_warning("Preprocessing Pipeline", "preprocessing", "Preprocessing")
        return

    X = st.session_state["X_clean"]
    feature_names = st.session_state["preprocessing_stats"].get("feature_names", [f"Band_{i}" for i in range(X.shape[1])])
    label_series = st.session_state.get("y_clean")
    n_samples, n_features = X.shape
    max_k = min(n_samples - 1, n_features)

    # PCA Configuration Panel
    st.markdown('<div class="sci-card">', unsafe_allow_html=True)
    st.markdown('<div class="sci-card-header"><span>📉</span> PCA Hyperparameter Configuration</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        mode_options = {
            "variance_threshold": "Variance Retention Threshold (Automatic component count)",
            "manual": "Manual Component Count (Specify exact number of PCs)",
        }
        current_mode = st.session_state["pca_config"].get("mode", "variance_threshold")
        mode_keys = list(mode_options.keys())
        mode_idx = mode_keys.index(current_mode) if current_mode in mode_keys else 0

        selected_mode = st.selectbox(
            "Component Selection Mode:",
            mode_keys,
            format_func=lambda k: mode_options[k],
            index=mode_idx,
            help="Choose whether PCA automatically determines the minimum components needed to preserve target variance or uses a fixed number of components."
        )

    with c2:
        if selected_mode == "variance_threshold":
            threshold_options = [0.85, 0.90, 0.95, 0.98, 0.99]
            current_th = st.session_state["pca_config"].get("variance_threshold", 0.95)
            th_idx = threshold_options.index(current_th) if current_th in threshold_options else 2

            selected_threshold = st.select_slider(
                "Target Retained Variance Threshold:",
                options=threshold_options,
                value=current_th,
                format_func=lambda x: f"{int(x * 100)}% Variance",
                help="PCA will automatically compute the smallest k components satisfying cumulative explained variance >= target threshold."
            )
            selected_k = 2
        else:
            current_k = st.session_state["pca_config"].get("n_components", 2)
            safe_default_k = max(1, min(current_k, max_k))
            selected_k = st.slider(
                f"Number of Principal Components (1 to {max_k}):",
                min_value=1,
                max_value=max_k,
                value=safe_default_k,
                help="Explicit number of principal components to extract."
            )
            selected_threshold = 0.95

    st.markdown('</div>', unsafe_allow_html=True)

    # Execution Action Button
    col_b1, col_b2, col_b3 = st.columns([1, 1.3, 1])
    with col_b2:
        is_done = st.session_state.get("pca_done", False)
        btn_label = "🔄 Re-run PCA Reduction" if is_done else "▶️ Fit & Compute Principal Component Analysis"
        run_pca_btn = st.button(btn_label, type="primary", use_container_width=True)

    if run_pca_btn:
        with st.spinner("Computing covariance matrix and spectral eigenvectors..."):
            try:
                res = run_pca_reduction(
                    X=X,
                    feature_names=feature_names,
                    mode=selected_mode,
                    variance_threshold=selected_threshold,
                    n_components_manual=selected_k,
                )

                st.session_state["pca_model"] = res["model"]
                st.session_state["pca_transformed"] = res["transformed"]
                st.session_state["pca_df"] = res["pca_df"]
                st.session_state["pca_loadings"] = res["loadings"]
                st.session_state["pca_stats"] = res["stats"]
                st.session_state["pca_done"] = True
                st.session_state["pca_config"] = {
                    "mode": selected_mode,
                    "variance_threshold": selected_threshold,
                    "n_components": res["stats"]["n_components"],
                }

                reset_downstream_from_reduction()
                st.toast(f"PCA fitted successfully ({res['stats']['n_components']} components retained {res['stats']['total_variance_retained']}% variance)!", icon="✅")
                st.rerun()

            except Exception as e:
                st.error(f"❌ PCA execution failed: {str(e)}")
                return

    # Render PCA Results and Visualizations if completed
    if st.session_state.get("pca_done", False) and st.session_state.get("pca_stats"):
        stats = st.session_state["pca_stats"]
        coords = st.session_state["pca_transformed"]
        k = stats["n_components"]
        exp_vars = stats["explained_variance_ratio"]

        st.markdown(
            f"""
            <div style="background: #ECFDF5; border: 1.5px solid #A7F3D0; border-radius: 8px; padding: 1.15rem 1.35rem; margin-top: 1.25rem; margin-bottom: 1.25rem;">
                <div style="font-weight: 700; color: #065F46; font-size: 1.05rem;">
                    ✅ PCA Reduction Successfully Computed!
                </div>
                <div style="font-size: 0.85rem; color: #047857; margin-top: 0.2rem;">
                    Extracted <strong>{k} principal components</strong> capturing <strong>{stats['total_variance_retained']:.1f}% cumulative variance</strong> (compressed dimensionality by <strong>{stats['dimensionality_reduction_pct']:.1f}%</strong>). Review variance diagnostics and loadings below, then proceed to UMAP embedding at the bottom.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.markdown("### 📊 PCA Dimensionality & Variance Metrics")

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Original Bands (D)</div>
                    <div class="metric-val">{stats['original_features']}</div>
                    <div class="metric-sub">Input spectral channels</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m2:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">PCA Components (k)</div>
                    <div class="metric-val">{k}</div>
                    <div class="metric-sub">Orthogonal eigenvectors</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m3:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Retained Variance</div>
                    <div class="metric-val">{stats['total_variance_retained']}%</div>
                    <div class="metric-sub">Cumulative information</div>
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
        with m5:
            pc1_pct = exp_vars[0] * 100.0 if len(exp_vars) > 0 else 0
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">PC1 Dominance</div>
                    <div class="metric-val">{pc1_pct:.1f}%</div>
                    <div class="metric-sub">Primary variance axis</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # PCA Visualizations Tabs
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        tab_scree, tab_2d, tab_3d, tab_loadings = st.tabs([
            "📉 Scree & Cumulative Variance",
            "🗺️ PCA 2D Projection",
            "🌐 PCA 3D Projection",
            "🔍 Spectral Loadings & Feature Weights"
        ])

        with tab_scree:
            scree_fig = plot_pca_variance_diagnostics(
                stats["explained_variance_ratio"],
                stats["cumulative_explained_variance"],
                threshold=stats.get("target_threshold")
            )
            st.plotly_chart(scree_fig, use_container_width=True)

        with tab_2d:
            color_mode = "ground_truth" if label_series is not None else "none"
            var1 = exp_vars[0] * 100.0 if len(exp_vars) > 0 else None
            var2 = exp_vars[1] * 100.0 if len(exp_vars) > 1 else None

            fig_2d = plot_projection_2d(
                coords=coords,
                dim1_name="PC1",
                dim2_name="PC2",
                title=f"PCA 2D Projection: PC1 vs PC2 ({stats['total_variance_retained']:.1f}% Retained)",
                color_by=color_mode,
                true_labels=label_series,
                dim1_var_pct=var1,
                dim2_var_pct=var2,
            )
            st.plotly_chart(fig_2d, use_container_width=True)

        with tab_3d:
            if k >= 3:
                var_pcts = [v * 100.0 for v in exp_vars[:3]]
                fig_3d = plot_projection_3d(
                    coords=coords,
                    dim1_name="PC1",
                    dim2_name="PC2",
                    dim3_name="PC3",
                    title="Interactive PCA 3D Projection (PC1 × PC2 × PC3)",
                    color_by="ground_truth" if label_series is not None else "none",
                    true_labels=label_series,
                    var_pcts=var_pcts,
                )
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.info(f"ℹ️ PCA was fitted with k={k} components. 3D projection requires at least 3 principal components.")

        with tab_loadings:
            st.markdown(
                """
                <p style="font-size: 0.85rem; color: #64748B;">
                    PCA loadings represent the eigenvector weights assigned to each spectral band.
                    Bands with higher absolute magnitude dominate the variance captured by that component.
                </p>
                """,
                unsafe_allow_html=True
            )
            if "pca_loadings" in st.session_state:
                load_fig = plot_pca_spectral_loadings(st.session_state["pca_loadings"])
                st.plotly_chart(load_fig, use_container_width=True)

    else:
        st.markdown(
            """
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.85rem 1.15rem; margin-top: 1.25rem; color: #475569; font-size: 0.85rem;">
                ⏳ <strong>Status:</strong> Ready to compute. Configure the variance threshold or manual components above, then click <strong>Fit & Compute Principal Component Analysis</strong>.
            </div>
            """,
            unsafe_allow_html=True
        )

    # Standardized Bidirectional Navigation Footer
    render_step_footer(
        current_page_id="pca",
        next_page_id="umap",
        prev_page_id="preprocessing",
        can_proceed=is_pca_ready(),
        proceed_hint="Click 'Fit & Compute Principal Component Analysis' above before proceeding to UMAP.",
    )

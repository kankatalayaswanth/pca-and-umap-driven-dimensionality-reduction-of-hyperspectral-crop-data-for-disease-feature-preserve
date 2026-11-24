"""
Page 3: Preprocessing Pipeline Configuration & Execution
Course: INT396 - Unsupervised Learning
"""

import streamlit as st
import pandas as pd
import numpy as np
from components.status_tracker import render_pipeline_header, render_empty_dataset_warning, render_step_footer
from preprocessing.pipeline import run_preprocessing_pipeline
from utils.state import is_dataset_ready, is_preprocessing_ready, reset_downstream_from_preprocessing, set_page


def render():
    """Render the Preprocessing configuration and execution page."""
    render_pipeline_header(
        "Preprocessing Pipeline",
        "Configure and execute data sanitization: resolve missing values, eliminate zero-variance bands, remove duplicates, and apply spectral feature normalization."
    )

    if not is_dataset_ready():
        render_empty_dataset_warning("Preprocessing")
        return

    df = st.session_state["raw_df"]
    feature_cols = st.session_state["feature_cols"]
    label_col = st.session_state.get("label_col")
    id_cols = st.session_state.get("id_cols", [])

    # Preprocessing Configuration Form
    st.markdown('<div class="sci-card">', unsafe_allow_html=True)
    st.markdown('<div class="sci-card-header"><span>⚙️</span> Preprocessing Options</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        imputation_options = {
            "mean": "Mean Imputation (Fill missing values with band mean)",
            "median": "Median Imputation (Fill missing values with band median)",
            "drop_rows": "Remove Incomplete Rows (Drop samples containing NaNs)",
        }
        current_imp = st.session_state["preprocessing_config"].get("imputation_strategy", "mean")
        imp_keys = list(imputation_options.keys())
        imp_idx = imp_keys.index(current_imp) if current_imp in imp_keys else 0

        selected_imp = st.selectbox(
            "Missing-Value Handling Strategy:",
            imp_keys,
            format_func=lambda k: imputation_options[k],
            index=imp_idx,
            help="Replace missing or infinite spectral readings with central tendencies or discard incomplete samples."
        )

        remove_constant = st.checkbox(
            "Remove Constant / Zero-Variance Spectral Bands",
            value=st.session_state["preprocessing_config"].get("remove_constant", True),
            help="Bands with zero variance offer no discriminative information and cause mathematical instability in covariance matrix inversion."
        )

    with col2:
        scaling_options = {
            "standard": "StandardScaler (Z-Score: Zero mean, unit variance — recommended for PCA)",
            "minmax": "MinMaxScaler (Normalize band intensities to range [0, 1])",
            "none": "No Scaling (Preserve raw sensor reflectance values)",
        }
        current_scale = st.session_state["preprocessing_config"].get("scaling_method", "standard")
        scale_keys = list(scaling_options.keys())
        scale_idx = scale_keys.index(current_scale) if current_scale in scale_keys else 0

        selected_scale = st.selectbox(
            "Feature Scaling Normalization:",
            scale_keys,
            format_func=lambda k: scaling_options[k],
            index=scale_idx,
            help="Feature scaling standardizes the dynamic range of reflectance channels. StandardScaler is essential for PCA."
        )

        remove_duplicates = st.checkbox(
            "Remove Duplicate Pixel Rows",
            value=st.session_state["preprocessing_config"].get("remove_duplicates", True),
            help="Drop redundant spectral vectors that have identical reflectance readings across all selected features."
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Execution Action Button
    col_b1, col_b2, col_b3 = st.columns([1, 1.3, 1])
    with col_b2:
        is_done = st.session_state.get("preprocessing_done", False)
        btn_label = "🔄 Re-run Preprocessing Pipeline" if is_done else "▶️ Run Preprocessing Pipeline"
        execute_button = st.button(btn_label, type="primary", use_container_width=True)

    if execute_button:
        with st.spinner("Executing preprocessing transformations..."):
            try:
                res = run_preprocessing_pipeline(
                    df=df,
                    feature_cols=feature_cols,
                    label_col=label_col,
                    id_cols=id_cols,
                    imputation_strategy=selected_imp,
                    scaling_method=selected_scale,
                    remove_constant=remove_constant,
                    remove_duplicates=remove_duplicates,
                )

                # Store in session state
                st.session_state["preprocessed_df"] = res["preprocessed_df"]
                st.session_state["X_clean"] = res["X_clean"]
                st.session_state["X_clean_df"] = res["X_clean_df"]
                st.session_state["y_clean"] = res["y_clean"]
                st.session_state["preprocessing_stats"] = res["stats"]
                st.session_state["preprocessing_done"] = True
                st.session_state["preprocessing_config"] = {
                    "imputation_strategy": selected_imp,
                    "scaling_method": selected_scale,
                    "remove_constant": remove_constant,
                    "remove_duplicates": remove_duplicates,
                }

                reset_downstream_from_preprocessing()
                st.toast("Preprocessing pipeline completed successfully!", icon="✅")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Preprocessing failed: {str(e)}")
                return

    # Display Before vs After Comparison if Preprocessing has been performed
    if st.session_state.get("preprocessing_done", False) and st.session_state.get("preprocessing_stats"):
        stats = st.session_state["preprocessing_stats"]
        
        st.markdown(
            f"""
            <div style="background: #ECFDF5; border: 1.5px solid #A7F3D0; border-radius: 8px; padding: 1.15rem 1.35rem; margin-top: 1.25rem; margin-bottom: 1.25rem;">
                <div style="font-weight: 700; color: #065F46; font-size: 1.05rem;">
                    ✅ Preprocessing Pipeline Executed Successfully!
                </div>
                <div style="font-size: 0.85rem; color: #047857; margin-top: 0.2rem;">
                    Retained <strong>{stats['final_samples']:,} samples</strong> and <strong>{stats['final_features']} spectral bands</strong> using <em>{stats['scaling_method']}</em>. Review the transformation audit below and proceed to PCA reduction at the bottom.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.markdown("### 📊 Before vs After Preprocessing Audit")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Sample Count</div>
                    <div class="metric-val">{stats['final_samples']:,}</div>
                    <div class="metric-sub">Before: {stats['initial_samples']:,} (-{stats['removed_rows']:,})</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Active Feature Bands</div>
                    <div class="metric-val">{stats['final_features']}</div>
                    <div class="metric-sub">Before: {stats['initial_features']} (-{stats['removed_features_count']})</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Missing Values Remaining</div>
                    <div class="metric-val">{stats['final_missing_count']}</div>
                    <div class="metric-sub">Before: {stats['initial_missing_count']} resolved</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col4:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Sample Retention</div>
                    <div class="metric-val">{stats['sample_retention_pct']}%</div>
                    <div class="metric-sub">100% of retained data valid</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Detailed Modification Audit Table
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        audit_rows = [
            {"Parameter": "Missing-Value Handling Strategy", "Value": stats["imputation_strategy"].capitalize()},
            {"Parameter": "Scaling / Normalization Transform", "Value": stats["scaling_method"]},
            {"Parameter": "Duplicate Rows Dropped", "Value": f"{stats['removed_duplicates']:,} rows"},
            {"Parameter": "Rows Dropped Due to Missing Values", "Value": f"{stats['removed_na_rows']:,} rows"},
            {"Parameter": "Infinite Values Corrected (±inf)", "Value": f"{stats['inf_count_replaced']:,} entries"},
            {"Parameter": "Constant / Zero-Variance Bands Removed", "Value": f"{stats['removed_features_count']} bands" + (f" ({', '.join(stats['removed_features_list'][:4])})" if stats['removed_features_list'] else "")},
        ]
        audit_df = pd.DataFrame(audit_rows)
        st.dataframe(audit_df, use_container_width=True, hide_index=True)

        # Preview preprocessed features
        with st.expander("🔍 Preview Preprocessed Baseline Feature Matrix", expanded=False):
            st.dataframe(st.session_state["X_clean_df"].head(50), use_container_width=True)

    else:
        st.markdown(
            """
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.85rem 1.15rem; margin-top: 1.25rem; color: #475569; font-size: 0.85rem;">
                ⏳ <strong>Status:</strong> Ready to execute. Configure the missing-value strategy and normalization above, then click <strong>Run Preprocessing Pipeline</strong>.
            </div>
            """,
            unsafe_allow_html=True
        )

    # Standardized Bidirectional Navigation Footer
    render_step_footer(
        current_page_id="preprocessing",
        next_page_id="pca",
        prev_page_id="dataset",
        can_proceed=is_preprocessing_ready(),
        proceed_hint="Click 'Run Preprocessing Pipeline' above before proceeding to PCA.",
    )

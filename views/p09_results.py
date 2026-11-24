"""
Page 9: Experiment Results, Dynamic Scientific Interpretation & Download Center
Course: INT396 - Unsupervised Learning
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
from components.status_tracker import render_pipeline_header, render_pipeline_dependency_warning, render_step_footer
from utils.exporter import generate_full_experiment_csv, generate_experiment_report_markdown
from utils.state import is_evaluation_ready, set_page


def render():
    """Render the comprehensive Results page."""
    # Mark results & visualization as viewed for true 100% pipeline milestone completion
    st.session_state["visualization_viewed"] = True
    st.session_state["results_viewed"] = True

    render_pipeline_header(
        "Final Results, Scientific Interpretation & Exports",
        "Synthesized report summarizing the complete unsupervised hyperspectral dimensionality reduction experiment, empirical findings, and exportable data artifacts."
    )

    if not is_evaluation_ready():
        render_pipeline_dependency_warning("Comparative Benchmarking", "comparison", "Benchmarking")
        return

    # Top 100% Completion Celebration Banner
    st.markdown(
        """
        <div style="background: #ECFDF5; border: 1.5px solid #10B981; border-radius: 8px; padding: 1.25rem 1.5rem; margin-bottom: 1.25rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.75rem;">
            <div>
                <div style="color: #065F46; font-size: 1.15rem; font-weight: 700;">
                    🏆 Experiment 100% Complete: All Pipeline Stages Successfully Executed!
                </div>
                <div style="font-size: 0.88rem; color: #047857; margin-top: 0.25rem;">
                    Full unsupervised dimensionality reduction & clustering benchmark completed. Synthesized findings, quantitative comparisons, and exportable artifacts are compiled below.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_top1, col_top2 = st.columns([2, 1])
    with col_top2:
        from utils.state import reset_full_experiment
        if st.button("🔄 Start Fresh Process (New Dataset)", type="secondary", use_container_width=True, help="Wipes all current results and restarts cleanly with a new dataset upload."):
            reset_full_experiment()
            st.toast("Started fresh experiment! Please upload a dataset.", icon="🔄")
            st.rerun()

    # Reconfiguration / Status Checklist (solves having to visit dashboard!)
    with st.expander("🛠️ Inspect Stage Status & Change Any Configuration", expanded=False):
        st.markdown(
            """
            <p style="font-size: 0.84rem; color: #475569; margin-bottom: 0.75rem;">
                Need to fine-tune any stage? Jump directly to that page without returning to the dashboard:
            </p>
            """,
            unsafe_allow_html=True
        )
        c_rf1, c_rf2, c_rf3, c_rf4, c_rf5 = st.columns(5)
        with c_rf1:
            if st.button("📁 Edit Columns/Label", use_container_width=True):
                set_page("dataset")
                st.rerun()
        with c_rf2:
            if st.button("⚙️ Edit Preprocessing", use_container_width=True):
                set_page("preprocessing")
                st.rerun()
        with c_rf3:
            if st.button("📉 Edit PCA Settings", use_container_width=True):
                set_page("pca")
                st.rerun()
        with c_rf4:
            if st.button("🌐 Edit UMAP Settings", use_container_width=True):
                set_page("umap")
                st.rerun()
        with c_rf5:
            if st.button("🎯 Change K Clusters", use_container_width=True):
                set_page("clustering")
                st.rerun()

    # Retrieve all pipeline objects from session state
    file_name = st.session_state.get("raw_file_name", "Hyperspectral_Dataset")
    raw_df = st.session_state["raw_df"]
    preprocessed_df = st.session_state["preprocessed_df"]
    prep_stats = st.session_state["preprocessing_stats"]
    
    pca_df = st.session_state["pca_df"]
    pca_stats = st.session_state["pca_stats"]
    
    umap_df = st.session_state["umap_df"]
    umap_stats = st.session_state["umap_stats"]
    
    kmeans_config = st.session_state["kmeans_config"]
    raw_labels = st.session_state["kmeans_raw_labels"]
    pca_labels = st.session_state["kmeans_pca_labels"]
    umap_labels = st.session_state["kmeans_umap_labels"]
    
    label_series = st.session_state.get("y_clean")
    label_col_name = st.session_state.get("label_col", "Ground_Truth")
    
    benchmark_df = st.session_state["comparison_table"]
    detailed_metrics = st.session_state["detailed_metrics"]
    interpretations = st.session_state.get("dynamic_interpretations", [])

    # 1. Pipeline Summary Architecture Cards
    st.markdown("### 📋 Experiment Architecture & Configuration Recap")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="sci-card">', unsafe_allow_html=True)
        st.markdown('<div class="sci-card-header"><span>📁</span> Dataset</div>', unsafe_allow_html=True)
        st.write(f"• **Source:** `{file_name}`")
        st.write(f"• **Analyzed Rows:** {prep_stats['final_samples']:,}")
        st.write(f"• **Original Bands:** {prep_stats['initial_features']}")
        st.write(f"• **Active Bands:** {prep_stats['final_features']}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="sci-card">', unsafe_allow_html=True)
        st.markdown('<div class="sci-card-header"><span>⚙️</span> Preprocessing</div>', unsafe_allow_html=True)
        st.write(f"• **Imputation:** {prep_stats['imputation_strategy'].capitalize()}")
        st.write(f"• **Scaling:** {prep_stats['scaling_method']}")
        st.write(f"• **Zero-Var Dropped:** {prep_stats['removed_features_count']}")
        st.write(f"• **Duplicates Dropped:** {prep_stats['removed_duplicates']}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="sci-card">', unsafe_allow_html=True)
        st.markdown('<div class="sci-card-header"><span>📉</span> PCA Model</div>', unsafe_allow_html=True)
        st.write(f"• **Components:** {pca_stats['n_components']}")
        st.write(f"• **Variance Retained:** {pca_stats['total_variance_retained']}%")
        st.write(f"• **Dimension Reduction:** {pca_stats['dimensionality_reduction_pct']}%")
        st.write(f"• **Mode:** {pca_stats['mode']}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="sci-card">', unsafe_allow_html=True)
        st.markdown('<div class="sci-card-header"><span>🌐</span> UMAP Model</div>', unsafe_allow_html=True)
        st.write(f"• **Embedding Dim:** {umap_stats['n_components']}D")
        st.write(f"• **Neighbors:** {umap_stats['n_neighbors']}")
        st.write(f"• **Min Distance:** {umap_stats['min_dist']}")
        st.write(f"• **Metric:** {umap_stats['metric']}")
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. Benchmark Evaluation Table
    st.markdown("### ⚖️ Final Comparative Benchmark Table")
    st.dataframe(benchmark_df, use_container_width=True, hide_index=True)

    # 3. Dynamic Scientific Interpretation
    st.markdown("### 🧠 Dynamic Academic Findings & Scientific Interpretation")
    st.markdown(
        """
        <p style="font-size: 0.85rem; color: #475569; margin-bottom: 0.75rem;">
            The findings below are dynamically synthesized from the empirical calculations of your uploaded dataset.
            In compliance with INT396 project rules, no static or fabricated assertions are used.
        </p>
        """,
        unsafe_allow_html=True
    )

    for item in interpretations:
        st.markdown(
            f"""
            <div class="sci-card" style="margin-bottom: 0.75rem; padding: 1rem 1.25rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.35rem;">
                    <span class="status-badge badge-info">{item['badge']}</span>
                    <strong style="color: #0F172A; font-size: 0.95rem;">{item['category']}</strong>
                </div>
                <div style="font-size: 0.88rem; color: #334155; line-height: 1.55;">
                    {item['text']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 4. Download Center
    st.markdown("---")
    st.markdown("### 📥 Download Results & Artifacts")
    st.markdown(
        """
        <p style="font-size: 0.85rem; color: #475569; margin-bottom: 1rem;">
            Download actual generated datasets, reduced embeddings, cluster assignments, and formatted experiment reports.
        </p>
        """,
        unsafe_allow_html=True
    )

    down_col1, down_col2, down_col3 = st.columns(3)

    # 1. Consolidated Master Experiment Dataset CSV
    with down_col1:
        full_csv_str = generate_full_experiment_csv(
            preprocessed_df=preprocessed_df,
            pca_df=pca_df,
            umap_df=umap_df,
            kmeans_raw_labels=raw_labels,
            kmeans_pca_labels=pca_labels,
            kmeans_umap_labels=umap_labels,
            label_series=label_series,
            label_col_name=label_col_name if label_series is not None else "Ground_Truth",
        )
        st.download_button(
            label="📊 Download Full Experiment Dataset (CSV)",
            data=full_csv_str,
            file_name="hyperspectral_complete_experiment.csv",
            mime="text/csv",
            use_container_width=True,
            help="Includes preprocessed raw bands, PCA coordinates, UMAP embeddings, and all 3 independent K-Means cluster labels."
        )

    # 2. Benchmark Comparison Table CSV
    with down_col2:
        benchmark_csv = benchmark_df.to_csv(index=False)
        st.download_button(
            label="📋 Download Benchmark Table (CSV)",
            data=benchmark_csv,
            file_name="clustering_benchmark_table.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download the comparative evaluation table (Raw vs PCA vs UMAP)."
        )

    # 3. Comprehensive Markdown Report
    with down_col3:
        report_md = generate_experiment_report_markdown(
            file_name=file_name,
            n_samples=prep_stats["final_samples"],
            n_raw_features=prep_stats["initial_features"],
            prep_stats=prep_stats,
            pca_stats=pca_stats,
            umap_stats=umap_stats,
            clustering_config=kmeans_config,
            benchmark_df=benchmark_df,
            interpretations=interpretations,
        )
        st.download_button(
            label="📑 Download Academic Report (Markdown)",
            data=report_md,
            file_name="INT396_Hyperspectral_Report.md",
            mime="text/markdown",
            use_container_width=True,
            help="Comprehensive academic lab report detailing the entire experiment, methodology, results, and interpretation."
        )

    # Additional individual embeddings download
    with st.expander("📦 Additional Individual Embeddings & Configurations (JSON & CSV)", expanded=False):
        sub_c1, sub_c2, sub_c3 = st.columns(3)
        with sub_c1:
            st.download_button(
                label="PCA Coordinates CSV",
                data=pca_df.to_csv(index=False),
                file_name="pca_components.csv",
                mime="text/csv",
                use_container_width=True
            )
        with sub_c2:
            st.download_button(
                label="UMAP Coordinates CSV",
                data=umap_df.to_csv(index=False),
                file_name="umap_embeddings.csv",
                mime="text/csv",
                use_container_width=True
            )
        with sub_c3:
            # Full experiment configuration JSON
            config_dump = {
                "dataset_file": file_name,
                "n_samples": prep_stats["final_samples"],
                "n_initial_features": prep_stats["initial_features"],
                "preprocessing": st.session_state["preprocessing_config"],
                "pca": pca_stats,
                "umap": umap_stats,
                "clustering": kmeans_config,
                "metrics_summary": detailed_metrics,
            }
            # Convert non-serializable objects (like DataFrames or Series)
            def sanitize(obj):
                if isinstance(obj, (np.integer, int)):
                    return int(obj)
                if isinstance(obj, (np.floating, float)):
                    return float(obj) if not np.isnan(obj) else None
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, pd.DataFrame):
                    return obj.to_dict()
                if isinstance(obj, pd.Series):
                    return obj.to_dict()
                return str(obj)

            config_json = json.dumps(config_dump, default=sanitize, indent=2)
            st.download_button(
                label="Experiment Config JSON",
                data=config_json,
                file_name="experiment_configuration.json",
                mime="application/json",
                use_container_width=True
            )

    # Standardized Navigation Footer
    render_step_footer(
        current_page_id="results",
        prev_page_id="visualization",
        next_page_id="dashboard",
        next_label="Back to Dashboard 🏠",
        can_proceed=True,
    )

    # Final Action Bar: Start Fresh
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🔄 Start a New Experiment")
    st.markdown(
        """
        <p style="font-size: 0.88rem; color: #64748B;">
            All analysis results and artifacts have been generated above. To analyze another hyperspectral crop dataset, start fresh:
        </p>
        """,
        unsafe_allow_html=True
    )
    c_bot1, c_bot2, c_bot3 = st.columns([1, 1.4, 1])
    with c_bot2:
        from utils.state import reset_full_experiment
        if st.button("🔄 Start Fresh Process (Upload New Dataset)", type="primary", key="results_start_fresh_bottom", use_container_width=True):
            reset_full_experiment()
            st.toast("Started fresh experiment! Please upload a new dataset.", icon="🔄")
            st.rerun()

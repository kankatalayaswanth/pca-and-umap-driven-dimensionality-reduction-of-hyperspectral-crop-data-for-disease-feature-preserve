"""
Page 2: Dataset Upload, Inspection, Column Mapping & Hyperspectral Exploration
Course: INT396 - Unsupervised Learning
"""

import streamlit as st
import pandas as pd
import numpy as np
from components.status_tracker import render_pipeline_header, render_step_footer
from data.loader import (
    load_uploaded_file,
    inspect_dataset,
    extract_wavelength_metadata,
    get_curated_datasets,
    load_curated_dataset,
)
from data.validator import validate_dataset, validate_feature_selection
from visualization.spectral_plots import (
    plot_spectral_signatures,
    plot_band_correlation_heatmap,
    plot_band_distribution,
)
from utils.state import reset_downstream_from_dataset, set_page


def _process_and_store_dataset(df: pd.DataFrame, file_name: str, file_size: int = 0, forced_label: str = None) -> bool:
    """Internal helper to validate, inspect, auto-detect column roles, and save dataset to session state."""
    val_res = validate_dataset(df)
    if not val_res["valid"]:
        for err in val_res["errors"]:
            st.error(f"❌ {err}")
        return False

    inspection = inspect_dataset(df, file_name, file_size)
    all_cols = df.columns.tolist()
    num_cols = inspection["numerical_cols"]

    # 1. Detect candidate label column
    detected_label = forced_label if (forced_label and forced_label in all_cols) else None
    if not detected_label:
        for col in all_cols:
            c_lower = str(col).lower()
            if any(term in c_lower for term in ["crop_class", "disease_class", "disease", "pathology", "diagnosis", "health", "class", "label", "status"]):
                if not pd.api.types.is_numeric_dtype(df[col]) or df[col].nunique() <= 20:
                    detected_label = col
                    break

    # 2. Detect candidate ID / metadata columns to exclude
    detected_ids = []
    for col in all_cols:
        c_lower = str(col).lower()
        if col != detected_label and any(term in c_lower for term in ["sample_id", "leaf_id", "class_id", "pot", "id", "plot", "stage", "setup", "watering", "nematode", "inoculum", "pf_cyst", "date", "time"]):
            detected_ids.append(col)

    # 3. Detect candidate spectral features (numeric excluding label & ID)
    detected_features = [c for c in num_cols if c != detected_label and c not in detected_ids]

    # Save to state
    st.session_state["raw_df"] = df
    st.session_state["raw_file_name"] = file_name
    st.session_state["dataset_loaded"] = True
    st.session_state["dataset_inspection"] = inspection
    st.session_state["wavelength_meta"] = inspection["wavelength_meta"]
    st.session_state["feature_cols"] = detected_features
    st.session_state["label_col"] = detected_label
    st.session_state["id_cols"] = detected_ids

    reset_downstream_from_dataset()
    msg = f"Loaded '{file_name}' ({len(df):,} samples, {len(detected_features)} spectral bands)."
    if detected_label:
        msg += f" Target Label: '{detected_label}'."
    st.toast(msg, icon="✅")
    return True


def render():
    """Render Dataset Upload, Inspection, and Exploration interface."""
    render_pipeline_header(
        "Dataset Selection & Hyperspectral Exploration",
        "Select an authentic agricultural benchmark dataset or upload your own hyperspectral data (CSV, XLSX, Parquet), inspect spectral distributions, and designate feature bands."
    )

    # 1. Dataset Selection / Upload Options
    st.markdown('<div class="sci-card">', unsafe_allow_html=True)
    st.markdown('<div class="sci-card-header"><span>📥</span> Choose Hyperspectral / Multispectral Dataset</div>', unsafe_allow_html=True)

    tab_curated, tab_upload = st.tabs([
        "🌾 Select Benchmark Dataset (Instant Load)",
        "📤 Upload Custom Dataset File (CSV, XLSX, Parquet)"
    ])

    curated_dict = get_curated_datasets()

    with tab_curated:
        st.markdown(
            """
            <p style="font-size: 0.88rem; color: #475569; margin-bottom: 0.75rem;">
                Select from real, calibrated hyperspectral crop pathology and agriculture datasets to begin immediate analysis without uploading:
            </p>
            """,
            unsafe_allow_html=True
        )
        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            selected_key = st.selectbox(
                "Select Preloaded Benchmark Dataset:",
                options=list(curated_dict.keys()),
                format_func=lambda k: curated_dict[k]["title"],
                key="curated_dataset_select",
                help="Curated datasets have verified spectral bands, calibrated reflectance values, and authentic crop pathology ground truth."
            )
            meta = curated_dict[selected_key]
            st.markdown(
                f"""
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.85rem; margin-top: 0.5rem; font-size: 0.84rem; color: #334155;">
                    <div><strong>Description:</strong> {meta['description']}</div>
                    <div style="margin-top: 0.35rem; display: flex; gap: 1.25rem; color: #64748B;">
                        <span>• Samples: <strong>{meta['samples']}</strong></span>
                        <span>• Bands: <strong>{meta['bands']}</strong></span>
                        <span>• Spectral Domain: <strong>{meta['domain']}</strong></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col_c2:
            st.markdown("<div style='height: 1.85rem;'></div>", unsafe_allow_html=True)
            if st.button("🚀 Load Selected Dataset", type="primary", key="btn_load_curated", use_container_width=True):
                with st.spinner(f"Loading '{meta['title']}'..."):
                    df = load_curated_dataset(selected_key)
                    if _process_and_store_dataset(df, meta["file_name"], forced_label=meta["default_label"]):
                        st.rerun()

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload your spectral dataset file",
            type=["csv", "xlsx", "xls", "parquet", "pq"],
            help="Upload tabular crop reflectance data where columns represent contiguous spectral bands.",
            key="main_file_uploader"
        )
        if uploaded_file is not None:
            if (
                not st.session_state.get("dataset_loaded", False)
                or st.session_state.get("raw_file_name") != uploaded_file.name
            ):
                with st.spinner(f"Reading and inspecting '{uploaded_file.name}'..."):
                    try:
                        df = load_uploaded_file(uploaded_file)
                        if _process_and_store_dataset(df, uploaded_file.name, uploaded_file.size):
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to parse uploaded file: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

    # If no dataset has been uploaded/selected yet:
    if not st.session_state.get("dataset_loaded", False) or st.session_state.get("raw_df") is None:
        st.markdown(
            """
            <div class="sci-callout sci-callout-info">
                ℹ️ <strong>No dataset loaded.</strong><br>
                Please choose a preloaded benchmark dataset above or upload your own file to start the unsupervised learning pipeline.
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # Dataset is loaded: Retrieve from session state
    df = st.session_state["raw_df"]
    inspection = st.session_state.get("dataset_inspection", inspect_dataset(df, st.session_state["raw_file_name"]))
    
    # 2. Dataset Inspection Metrics Overview
    st.markdown("### 📊 Dataset Diagnostic Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Total Samples (Rows)</div>
                <div class="metric-val">{inspection['n_rows']:,}</div>
                <div class="metric-sub">Pixel spectra</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Total Columns</div>
                <div class="metric-val">{inspection['n_cols']}</div>
                <div class="metric-sub">{inspection['n_numerical']} num, {inspection['n_non_numerical']} cat</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Missing Values</div>
                <div class="metric-val">{inspection['total_missing']:,}</div>
                <div class="metric-sub">{inspection['missing_pct']}% of cells</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Duplicate Rows</div>
                <div class="metric-val">{inspection['n_duplicates']:,}</div>
                <div class="metric-sub">Exact duplicate rows</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col5:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Memory Footprint</div>
                <div class="metric-val">{inspection['memory_size']}</div>
                <div class="metric-sub">In-memory DataFrame</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 3. Interactive Data Preview Table
    with st.expander("🔍 View Raw Dataset Table (First 100 Rows)", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)

    # 4. Column Role Assignment (Features, Label, ID)
    st.markdown("### 🎛️ Column Role Designation")
    st.markdown(
        """
        <p style="font-size: 0.88rem; color: #475569; margin-bottom: 0.75rem;">
            Assign roles to columns. The system guarantees that ID and ground-truth label columns are 
            <strong>never</strong> used as features during unsupervised dimensionality reduction or clustering.
        </p>
        """,
        unsafe_allow_html=True
    )

    all_columns = df.columns.tolist()
    num_cols = inspection["numerical_cols"]
    cat_cols = inspection["non_numerical_cols"]

    col_map1, col_map2 = st.columns([1.2, 1])

    with col_map2:
        st.markdown('<div class="sci-card">', unsafe_allow_html=True)
        st.markdown('<div class="sci-card-header"><span>🏷️</span> Ground Truth & ID Columns</div>', unsafe_allow_html=True)
        
        # Ground Truth Label Column
        label_options = ["None (Unsupervised / No Ground Truth)"] + all_columns
        current_label = st.session_state.get("label_col")
        curr_label_idx = label_options.index(current_label) if current_label in label_options else 0
        
        chosen_label = st.selectbox(
            "Select Ground-Truth Disease / Pest Label (Optional):",
            label_options,
            index=curr_label_idx,
            help="If your dataset contains disease classifications (e.g., 'Healthy', 'Rust', 'Blight'), select it here for external ARI/NMI validation."
        )
        new_label_col = None if chosen_label.startswith("None") else chosen_label
        
        # ID Columns
        current_ids = st.session_state.get("id_cols", [])
        chosen_ids = st.multiselect(
            "Select Identifier / Metadata Columns to Exclude:",
            [c for c in all_columns if c != new_label_col],
            default=[c for c in current_ids if c in all_columns and c != new_label_col],
            help="Columns such as Sample_ID, Plot_Number, Timestamp that must be excluded from ML calculations."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_map1:
        st.markdown('<div class="sci-card">', unsafe_allow_html=True)
        st.markdown('<div class="sci-card-header"><span>🌈</span> Spectral Feature Bands (Input Space)</div>', unsafe_allow_html=True)
        
        # Available feature candidate columns: must be numeric and not label or ID
        available_feature_cols = [c for c in num_cols if c != new_label_col and c not in chosen_ids]
        
        # Quick selection buttons
        btn_c1, btn_c2, btn_c3 = st.columns(3)
        with btn_c1:
            if st.button("Select All Numerical", use_container_width=True):
                st.session_state["feature_cols"] = list(available_feature_cols)
                st.rerun()
        with btn_c2:
            if st.button("Clear Selection", use_container_width=True):
                st.session_state["feature_cols"] = []
                st.rerun()
        with btn_c3:
            # Range filter if columns have numeric names
            wl_meta = st.session_state.get("wavelength_meta", {})
            has_numeric_bands = len(wl_meta) > 0
            if has_numeric_bands:
                st.caption(f"✨ Detected {len(wl_meta)} bands with physical wavelength metadata (nm).")
                
        current_features = st.session_state.get("feature_cols", available_feature_cols)
        # Filter current_features to only those in available
        valid_current_features = [c for c in current_features if c in available_feature_cols]
        
        chosen_features = st.multiselect(
            "Spectral Feature Columns for ML Processing:",
            available_feature_cols,
            default=valid_current_features,
            help="Select the spectral bands (reflectance channels) to include in PCA and UMAP dimensionality reduction."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Validate the current selection
    val_selection = validate_feature_selection(df, chosen_features, new_label_col, chosen_ids)
    if not val_selection["valid"]:
        for err in val_selection["errors"]:
            st.error(f"❌ {err}")
    for warn in val_selection["warnings"]:
        st.warning(f"⚠️ {warn}")

    if new_label_col:
        st.markdown(
            f"""
            <div class="sci-callout sci-callout-success" style="margin-top: 0.5rem;">
                🎯 <strong>Ground-Truth Label Active:</strong> <strong>'{new_label_col}'</strong> is selected.
                Downstream evaluation will calculate <strong>Adjusted Rand Index (ARI)</strong>, 
                <strong>Normalized Mutual Information (NMI)</strong>, and contingency heatmaps to benchmark disease signal preservation.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="sci-callout sci-callout-info" style="margin-top: 0.5rem;">
                ℹ️ <strong>Unsupervised Mode:</strong> No disease label is currently selected. 
                If your dataset contains disease/pest classes (such as <strong>Disease_Class</strong>), select it in the dropdown above to enable external accuracy benchmarking (ARI & NMI).
            </div>
            """,
            unsafe_allow_html=True
        )

    # Save confirmed selections
    if chosen_features != st.session_state.get("feature_cols") or new_label_col != st.session_state.get("label_col") or chosen_ids != st.session_state.get("id_cols"):
        st.session_state["feature_cols"] = chosen_features
        st.session_state["label_col"] = new_label_col
        st.session_state["id_cols"] = chosen_ids
        st.session_state.pop("cached_spec_fig", None)
        st.session_state.pop("cached_corr_fig", None)
        reset_downstream_from_dataset()

    # 5. Hyperspectral Data Exploration Visualizations
    if val_selection["valid"] and len(chosen_features) >= 2:
        st.markdown("---")
        st.markdown("### 🔬 Spectral Profile Exploration")
        
        tab_spec, tab_corr, tab_dist, tab_stats = st.tabs([
            "📈 Spectral Reflectance Profile",
            "🔥 Band Correlation Heatmap",
            "📊 Feature Distributions",
            "📋 Band Descriptive Statistics"
        ])
        
        label_series = df[new_label_col] if new_label_col else None
        
        with tab_spec:
            st.markdown(
                """
                <p style="font-size: 0.85rem; color: #64748B;">
                    Displays the mean spectral reflectance curve across the selected bands. 
                    If a disease label column is assigned, distinct signature curves are plotted for each crop pathology class.
                </p>
                """,
                unsafe_allow_html=True
            )
            if "cached_spec_fig" not in st.session_state:
                st.session_state["cached_spec_fig"] = plot_spectral_signatures(
                    df[chosen_features],
                    y=label_series,
                    wavelength_meta=st.session_state.get("wavelength_meta", {})
                )
            st.plotly_chart(st.session_state["cached_spec_fig"], use_container_width=True)
            
        with tab_corr:
            st.markdown(
                """
                <p style="font-size: 0.85rem; color: #64748B;">
                    Contiguous hyperspectral bands exhibit strong collinearity. The heatmap displays Pearson correlation coefficients
                    across representative spectral channels.
                </p>
                """,
                unsafe_allow_html=True
            )
            if "cached_corr_fig" not in st.session_state:
                st.session_state["cached_corr_fig"] = plot_band_correlation_heatmap(df[chosen_features])
            st.plotly_chart(st.session_state["cached_corr_fig"], use_container_width=True)
            
        with tab_dist:
            col_d1, col_d2 = st.columns([1, 3])
            with col_d1:
                selected_band = st.selectbox("Select Band to Inspect:", chosen_features, index=0)
            with col_d2:
                dist_fig = plot_band_distribution(df[chosen_features], selected_band, y=label_series)
                st.plotly_chart(dist_fig, use_container_width=True)
                
        with tab_stats:
            stats_df = df[chosen_features].describe().T[["mean", "std", "min", "50%", "max"]]
            stats_df.columns = ["Mean", "Std. Dev", "Min", "Median", "Max"]
            stats_df["Skewness"] = df[chosen_features].skew()
            st.dataframe(stats_df.round(4), use_container_width=True)

    # Standardized Footer Navigation
    render_step_footer(
        current_page_id="dataset",
        next_page_id="preprocessing",
        can_proceed=val_selection["valid"],
        proceed_hint="Select at least 2 valid spectral features above to proceed.",
    )

"""
Sidebar Navigation and Experiment Status Component
Course: INT396 - Unsupervised Learning
"""

import streamlit as st
from config.settings import PAGES, APP_TITLE, COURSE_CODE
from utils.state import (
    get_stage_readiness,
    reset_full_experiment, set_page
)


def render_sidebar():
    """Render the persistent left navigation sidebar with deterministic button navigation."""
    with st.sidebar:
        # Sequential Navigation Menu
        st.markdown("<p style='font-size: 0.76rem; font-weight: 700; text-transform: uppercase; color: #64748B; margin: 0.5rem 0 0.5rem 0; letter-spacing: 0.05em;'>🌾 Workflow Navigation</p>", unsafe_allow_html=True)

        current_page = st.session_state.get("current_page", "dataset")

        for p in PAGES:
            is_active = (p["id"] == current_page)
            is_ready = get_stage_readiness(p["id"])
            status_suffix = "  ✓" if (is_ready and p["id"] != "dashboard") else ""
            btn_label = f"{p['icon']}  {p['title']}{status_suffix}"

            if st.button(
                btn_label,
                key=f"sidebar_btn_{p['id']}",
                type="primary" if is_active else "secondary",
                use_container_width=True
            ):
                if current_page != p["id"]:
                    set_page(p["id"])
                    st.rerun()

        st.markdown("<hr style='margin: 0.85rem 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)

        # 3. Active Dataset Summary Card
        if st.session_state.get("dataset_loaded", False) and st.session_state.get("raw_df") is not None:
            filename = st.session_state.get("raw_file_name", "Uploaded Data")
            if len(filename) > 22:
                filename = filename[:19] + "..."
            samples = len(st.session_state["raw_df"])
            features = len(st.session_state.get("feature_cols", []))
            has_label = st.session_state.get("label_col") is not None
            
            st.markdown(
                f"""
                <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.65rem 0.75rem; font-size: 0.76rem; margin-bottom: 0.75rem;">
                    <div style="font-weight: 600; color: #1E293B; margin-bottom: 0.25rem;">
                        📄 {filename}
                    </div>
                    <div style="color: #64748B; line-height: 1.35;">
                        <div>• Samples: <strong>{samples:,}</strong></div>
                        <div>• Active Bands: <strong>{features}</strong></div>
                        <div>• Disease Label: <strong>{'Yes' if has_label else 'Unsupervised'}</strong></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Prominent "Start Fresh Experiment" Button
            if st.button("🔄 Start Fresh (New Dataset)", key="sidebar_start_fresh", use_container_width=True, help="Wipes current dataset, PCA, UMAP and clusters to start over with a new dataset."):
                reset_full_experiment()
                st.toast("Started fresh experiment! Ready for a new dataset.", icon="🔄")
                st.rerun()
        else:
            st.markdown(
                """
                <div style="background-color: #FFFBEB; border: 1px solid #FDE68A; border-radius: 6px; padding: 0.6rem 0.75rem; font-size: 0.74rem; color: #92400E;">
                    ⚠️ No dataset loaded.<br>Begin at <strong>Dataset Upload</strong>.
                </div>
                """,
                unsafe_allow_html=True
            )

        # Academic citation footer
        st.markdown(
            """
            <div style="margin-top: 1.25rem; font-size: 0.68rem; color: #94A3B8; text-align: center; border-top: 1px solid #E2E8F0; padding-top: 0.5rem;">
                INT396 Machine Learning Project<br>
                Hyperspectral Feature Preservation
            </div>
            """,
            unsafe_allow_html=True
        )

"""
ANTIGRAVITY — Hyperspectral Crop Disease Feature Preservation
PCA and UMAP-Driven Dimensionality Reduction & Downstream Clustering Benchmark
Course: INT396 – Unsupervised Learning
"""

import streamlit as st
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.settings import APP_TITLE, APP_SUBTITLE, COURSE_CODE
from components.styles import apply_custom_styles
from components.sidebar import render_sidebar
from utils.state import init_session_state
from views import (
    p01_dashboard,
    p02_dataset,
    p03_preprocessing,
    p04_pca,
    p05_umap,
    p06_clustering,
    p07_comparison,
    p08_visualization,
    p09_results,
)

# Streamlit Page Configuration
st.set_page_config(
    page_title=f"{APP_TITLE} | {COURSE_CODE}",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Main application lifecycle and page router."""
    # 1. Initialize persistent session state
    init_session_state()

    # 2. Inject restrained scientific CSS
    apply_custom_styles()

    # 3. Render persistent sidebar navigation & status
    render_sidebar()

    # 4. Route to active page
    current_page = st.session_state.get("current_page", "dashboard")

    page_map = {
        "dashboard": p01_dashboard.render,
        "dataset": p02_dataset.render,
        "preprocessing": p03_preprocessing.render,
        "pca": p04_pca.render,
        "umap": p05_umap.render,
        "clustering": p06_clustering.render,
        "comparison": p07_comparison.render,
        "visualization": p08_visualization.render,
        "results": p09_results.render,
    }

    render_func = page_map.get(current_page, p01_dashboard.render)

    try:
        render_func()
    except Exception as e:
        st.markdown(
            f"""
            <div class="sci-callout sci-callout-danger">
                <strong>An error occurred while rendering this page:</strong><br>
                <code>{str(e)}</code><br><br>
                <em>Recommendation:</em> If this occurred after modifying inputs or data, try resetting the current stage or returning to the Dataset Upload page.
            </div>
            """,
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()

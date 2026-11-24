"""
Status Banner, Pipeline Navigation Breadcrumbs and Persistent Navigation Controls
Course: INT396 - Unsupervised Learning
"""

import streamlit as st
from config.settings import PAGES
from utils.state import (
    is_dataset_ready, is_preprocessing_ready, is_pca_ready,
    is_umap_ready, is_clustering_ready, is_evaluation_ready,
    get_pipeline_stage_status,
    get_next_page, get_previous_page, set_page, reset_full_experiment
)


def render_pipeline_status_ribbon():
    """
    Render a compact, persistent horizontal status ribbon at the top of every page.
    Allows user to inspect the current state of all pipeline stages.
    """
    stages = get_pipeline_stage_status()
    current_page = st.session_state.get("current_page", "dashboard")
    
    # Build compact HTML pills
    pills_html = []
    for s in stages:
        is_cur = (s["id"] == current_page)
        border_style = "border: 2px solid #0D9488;" if is_cur else "border: 1px solid #A7F3D0;" if s["ready"] else "border: 1px solid #E2E8F0;"
        bg_style = "background: #CCFBF1;" if is_cur else "background: #ECFDF5;" if s["ready"] else "background: #F8FAFC;"
        color_style = "color: #0F766E; font-weight: 700;" if is_cur else "color: #065F46; font-weight: 600;" if s["ready"] else "color: #64748B;"
        
        pill = f"""<span style="display: inline-flex; align-items: center; gap: 0.25rem; {bg_style} {border_style} {color_style} padding: 0.2rem 0.55rem; border-radius: 4px; font-size: 0.74rem;">
            <span>{s['icon']}</span> <span>{s['name']}{(' (' + s['badge'] + ')') if s['ready'] else ''}</span>
        </span>"""
        pills_html.append(pill)
        
    all_pills_str = " <span style='color: #CBD5E1; font-size: 0.75rem;'>→</span> ".join(pills_html)
    
    st.markdown(
        f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.55rem 0.85rem; margin-bottom: 1.15rem; display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-start; gap: 0.5rem;">
            <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem; overflow-x: auto;">
                <span style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #475569; letter-spacing: 0.04em; margin-right: 0.2rem;">Pipeline:</span>
                {all_pills_str}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_pipeline_header(current_step_name: str, step_description: str):
    """Render a clean scientific section header with contextual description and status ribbon."""
    st.markdown(
        f"""
        <div style="margin-bottom: 0.85rem;">
            <h2 style="font-size: 1.55rem; font-weight: 700; color: #0F172A; margin: 0 0 0.25rem 0; letter-spacing: -0.02em;">
                {current_step_name}
            </h2>
            <p style="font-size: 0.92rem; color: #475569; margin: 0; line-height: 1.45;">
                {step_description}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Always render the live status ribbon directly below the title
    render_pipeline_status_ribbon()


def render_step_footer(
    current_page_id: str,
    next_page_id: str = None,
    prev_page_id: str = None,
    next_label: str = None,
    prev_label: str = None,
    can_proceed: bool = True,
    proceed_hint: str = None,
    show_quick_run: bool = False,
):
    """
    Standardized interactive footer with paired Back / Continue buttons.
    Eliminates confusing navigation and gives users seamless bidirectional flow.
    """
    target_prev = prev_page_id or get_previous_page(current_page_id)
    target_next = next_page_id or get_next_page(current_page_id)

    # Find titles
    page_lookup = {p["id"]: p["title"] for p in PAGES}
    default_prev_title = page_lookup.get(target_prev, "Previous Step")
    default_next_title = page_lookup.get(target_next, "Next Step")

    btn_prev_text = prev_label or f"⬅️ Back: {default_prev_title}"
    btn_next_text = next_label or f"Continue: {default_next_title} ➡️"

    st.markdown("<hr style='margin: 2rem 0 1.25rem 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)

    col_prev, col_space, col_next = st.columns([1.2, 1.6, 1.2])

    with col_prev:
        if target_prev:
            if st.button(btn_prev_text, key=f"footer_prev_{current_page_id}", type="secondary", use_container_width=True):
                set_page(target_prev)
                st.rerun()

    with col_next:
        if target_next:
            if can_proceed:
                if st.button(btn_next_text, key=f"footer_next_{current_page_id}", type="primary", use_container_width=True):
                    set_page(target_next)
                    st.rerun()
            else:
                hint_str = proceed_hint or "Complete this step above to proceed"
                st.button(f"{btn_next_text} (Incomplete)", key=f"footer_next_disabled_{current_page_id}", disabled=True, use_container_width=True, help=hint_str)


def render_empty_dataset_warning(page_name: str = "this page"):
    """Standard empty state when user accesses a page without uploading a dataset."""
    st.markdown(
        f"""
        <div class="sci-card" style="text-align: center; padding: 2.5rem 1.5rem; margin-top: 1rem;">
            <div style="font-size: 2.5rem; margin-bottom: 0.75rem;">📂</div>
            <h3 style="font-size: 1.2rem; font-weight: 600; color: #1E293B; margin-bottom: 0.5rem;">
                No Dataset Loaded
            </h3>
            <p style="font-size: 0.9rem; color: #64748B; max-width: 520px; margin: 0 auto 1.5rem auto; line-height: 1.5;">
                Select a curated hyperspectral agricultural benchmark or upload your own dataset on the Dataset page to begin.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        if st.button("➡️ Go to Dataset Page", type="primary", use_container_width=True):
            set_page("dataset")
            st.rerun()


def render_pipeline_dependency_warning(missing_stage: str, target_page_id: str, action_label: str):
    """Display a warning when a prerequisite pipeline stage has not been executed."""
    st.markdown(
        f"""
        <div class="sci-card" style="text-align: center; padding: 2rem 1.5rem; margin-top: 1rem;">
            <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">⚠️</div>
            <h3 style="font-size: 1.15rem; font-weight: 600; color: #92400E; margin-bottom: 0.5rem;">
                Prerequisite Incomplete: {missing_stage}
            </h3>
            <p style="font-size: 0.88rem; color: #64748B; max-width: 520px; margin: 0 auto 1.25rem auto;">
                Downstream unsupervised machine learning requires the completion of <strong>{missing_stage}</strong>.
                Please execute that stage to continue the analysis workflow.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        if st.button(f"➡️ Go to {action_label}", type="primary", use_container_width=True):
            set_page(target_page_id)
            st.rerun()

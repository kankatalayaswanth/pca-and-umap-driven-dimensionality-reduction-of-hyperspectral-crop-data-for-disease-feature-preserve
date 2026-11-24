"""
Professional Scientific UI/UX Styling
Restrained, high-contrast, clean academic visual theme.
Prioritizes functionality, clarity, and scientific professionalism.
"""

import streamlit as st


def apply_custom_styles():
    """Inject custom CSS for clean scientific data analysis interface."""
    css = """
    <style>
    /* Global scientific typography and surfaces */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #0F172A;
    }

    /* Main container clean spacing */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1400px;
    }

    /* Scientific Card Container */
    .sci-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }

    .sci-card-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 0.5rem;
    }

    /* Metric Card */
    .metric-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 0.85rem 1rem;
        text-align: left;
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748B;
        margin-bottom: 0.25rem;
    }
    .metric-val {
        font-size: 1.45rem;
        font-weight: 700;
        color: #0F172A;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 0.2rem;
    }

    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-success {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    .badge-warning {
        background-color: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    .badge-info {
        background-color: #EFF6FF;
        color: #1E40AF;
        border: 1px solid #BFDBFE;
    }
    .badge-neutral {
        background-color: #F1F5F9;
        color: #475569;
        border: 1px solid #CBD5E1;
    }

    /* Alert / Callout Boxes */
    .sci-callout {
        border-radius: 6px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 1rem;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    .sci-callout-info {
        background-color: #F0F9FF;
        border-left: 4px solid #0284C7;
        color: #0C4A6E;
    }
    .sci-callout-success {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
        color: #14532D;
    }
    .sci-callout-warning {
        background-color: #FFFBEB;
        border-left: 4px solid #D97706;
        color: #78350F;
    }
    .sci-callout-danger {
        background-color: #FEF2F2;
        border-left: 4px solid #DC2626;
        color: #7F1D1D;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }

    /* Completely hide Streamlit's default multi-page auto-navigation list */
    [data-testid="stSidebarNav"],
    [data-testid="stSidebarNavItems"],
    [data-testid="stSidebarNavSeparator"],
    div:has(> ul[data-testid="stSidebarNavItems"]) {
        display: none !important;
    }

    .sidebar-brand {
        padding: 0.75rem 0.25rem 1.25rem 0.25rem;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    .sidebar-brand-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.01em;
        line-height: 1.3;
    }
    .sidebar-brand-sub {
        font-size: 0.75rem;
        font-weight: 600;
        color: #0D9488;
        margin-top: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Pipeline Step Tracker in Sidebar */
    .step-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.4rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-bottom: 0.25rem;
    }
    .step-done {
        color: #065F46;
        background-color: #ECFDF5;
        font-weight: 500;
    }
    .step-pending {
        color: #64748B;
    }

    /* Sidebar Navigation Menu Buttons */
    section[data-testid="stSidebar"] div.stButton > button {
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 0.45rem 0.75rem !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        border-radius: 6px !important;
        margin-bottom: 0.2rem !important;
        border: 1px solid #E2E8F0 !important;
        background-color: #FFFFFF !important;
        color: #334155 !important;
        width: 100% !important;
        display: flex !important;
    }
    section[data-testid="stSidebar"] div.stButton > button:hover {
        border-color: #0D9488 !important;
        color: #0D9488 !important;
        background-color: #F0FDFA !important;
    }
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background-color: #1E3A8A !important;
        border-color: #1E3A8A !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"]:hover {
        background-color: #1E40AF !important;
        border-color: #1E40AF !important;
        color: #FFFFFF !important;
    }

    /* Streamlit buttons refinement */
    div.stButton > button {
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.88rem;
        transition: all 0.15s ease-in-out;
    }
    div.stButton > button:hover {
        border-color: #0D9488;
        color: #0D9488;
    }

    /* Primary buttons */
    div.stButton > button[kind="primary"] {
        background-color: #1E3A8A;
        border-color: #1E3A8A;
        color: #FFFFFF;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #1E40AF;
        border-color: #1E40AF;
        color: #FFFFFF;
    }

    /* Dataframe styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        overflow: hidden;
    }

    /* Code & pre blocks */
    code {
        font-family: 'JetBrains Mono', monospace;
        background-color: #F1F5F9;
        padding: 0.15rem 0.35rem;
        border-radius: 4px;
        font-size: 0.85em;
        color: #0F172A;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

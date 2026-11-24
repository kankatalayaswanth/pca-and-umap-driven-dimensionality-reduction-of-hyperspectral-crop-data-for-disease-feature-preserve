"""
Application Configuration and Constants
Course: INT396 - Unsupervised Learning
Project: PCA and UMAP-Driven Dimensionality Reduction of Hyperspectral Crop Data
         for Disease-Relevant Feature Preservation
"""

APP_TITLE = "Hyperspectral Crop Disease Feature Preservation"
APP_SUBTITLE = "PCA and UMAP-Driven Dimensionality Reduction & Downstream Clustering Benchmark"
COURSE_CODE = "INT396 – Unsupervised Learning"
PROJECT_TYPE = "Unsupervised Machine Learning & Dimensionality Reduction"

# Navigation Pages: Clean icon and proper name (no 'Step X' in menu)
PAGES = [
    {"id": "dashboard", "title": "Project Overview", "step": 0, "icon": "📊", "desc": "Research objectives, theory & architecture"},
    {"id": "dataset", "title": "Dataset Upload", "step": 1, "icon": "📁", "desc": "Upload data & designate spectral bands"},
    {"id": "preprocessing", "title": "Preprocessing", "step": 2, "icon": "⚙️", "desc": "Normalization & variance filtering"},
    {"id": "pca", "title": "PCA Reduction", "step": 3, "icon": "📉", "desc": "Linear orthogonal variance maximization"},
    {"id": "umap", "title": "UMAP Embedding", "step": 4, "icon": "🌐", "desc": "Non-linear manifold learning"},
    {"id": "clustering", "title": "K-Means Clustering", "step": 5, "icon": "🎯", "desc": "Independent tri-space partitioning"},
    {"id": "comparison", "title": "Benchmarking", "step": 6, "icon": "⚖️", "desc": "Raw vs PCA vs UMAP evaluation"},
    {"id": "visualization", "title": "Visualizations", "step": 7, "icon": "🔍", "desc": "2D/3D projections & disease signal"},
    {"id": "results", "title": "Results & Report", "step": 8, "icon": "📑", "desc": "Scientific conclusions & exports"},
]

# Color Schemes for Scientific Visualization
COLORS = {
    "primary": "#1E3A8A",      # Deep Slate Blue
    "secondary": "#0D9488",    # Teal
    "accent": "#6366F1",       # Indigo
    "success": "#10B981",      # Emerald Green
    "warning": "#F59E0B",      # Amber
    "danger": "#EF4444",       # Crimson Red
    "background": "#F8FAFC",   # Very Light Slate
    "surface": "#FFFFFF",      # Pure White
    "border": "#E2E8F0",       # Light Gray Border
    "text": "#0F172A",         # Deep Slate Charcoal
    "muted": "#64748B",        # Slate Gray
}

# Qualitative color palette for disease classes and clusters (accessible, distinct)
QUALITATIVE_PALETTE = [
    "#2563EB", "#059669", "#D97706", "#DC2626", "#7C3AED",
    "#0891B2", "#DB2777", "#4F46E5", "#16A34A", "#CA8A04",
    "#9333EA", "#0284C7", "#EA580C", "#65A30D", "#475569"
]

# Metric definitions for tooltips and academic explanation
METRIC_DEFINITIONS = {
    "Silhouette Score": (
        "Internal metric measuring how similar an object is to its own cluster compared to other clusters. "
        "Range: [-1, +1]. Higher is better. Values near +1 indicate dense, well-separated clusters; "
        "values near 0 indicate overlapping clusters; negative values suggest misassigned samples."
    ),
    "Davies-Bouldin Index": (
        "Internal metric representing the average similarity measure between each cluster and its most similar one. "
        "Range: [0, +inf). Lower is better. A lower score signifies tighter clusters with greater inter-cluster separation."
    ),
    "Calinski-Harabasz Index": (
        "Internal metric defined as the ratio of between-cluster dispersion to within-cluster dispersion. "
        "Range: [0, +inf). Higher is better. Higher values indicate denser, more distinct clusters (Variance Ratio Criterion)."
    ),
    "Adjusted Rand Index (ARI)": (
        "External metric measuring the similarity between the discovered clusters and ground-truth disease labels, "
        "adjusted for the chance grouping of elements. Range: [-1, +1]. 1.0 indicates perfect agreement; "
        "0.0 indicates agreement expected by chance."
    ),
    "Normalized Mutual Information (NMI)": (
        "External metric quantifying the shared mutual information between cluster assignments and true disease classes, "
        "normalized to the range [0, 1]. 1.0 denotes complete agreement; 0.0 denotes independent distributions."
    )
}

"""
Evaluation and Diagnostic Visualizations
Scree plot, cumulative variance curve, elbow/silhouette curves, benchmark comparisons, and contingency heatmap.
Course: INT396 - Unsupervised Learning
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config.settings import COLORS, QUALITATIVE_PALETTE


def plot_pca_variance_diagnostics(
    explained_variance_ratio: list,
    cumulative_variance: list,
    threshold: float = None,
) -> go.Figure:
    """
    Dual-axis diagnostic chart:
    - Bar chart of individual explained variance ratio per principal component (Scree plot)
    - Line chart of cumulative explained variance with target threshold indicator
    """
    n_comp = len(explained_variance_ratio)
    comp_labels = [f"PC{i+1}" for i in range(n_comp)]
    exp_var_pct = [v * 100.0 for v in explained_variance_ratio]
    cum_var_pct = [v * 100.0 for v in cumulative_variance]
    
    fig = go.Figure()
    
    # Scree Plot Bars
    fig.add_trace(go.Bar(
        x=comp_labels,
        y=exp_var_pct,
        name="Individual Explained Variance",
        marker_color="#3B82F6",
        text=[f"{v:.1f}%" for v in exp_var_pct],
        textposition="auto",
        hovertemplate="Component: %{x}<br>Explained Variance: %{y:.2f}%<extra></extra>"
    ))
    
    # Cumulative Variance Line (secondary axis)
    fig.add_trace(go.Scatter(
        x=comp_labels,
        y=cum_var_pct,
        name="Cumulative Variance Retained",
        mode="lines+markers",
        line=dict(color="#0D9488", width=3),
        marker=dict(size=7, color="#0D9488"),
        yaxis="y2",
        hovertemplate="Component: %{x}<br>Cumulative Variance: %{y:.2f}%<extra></extra>"
    ))
    
    # Target Threshold Line (if provided)
    if threshold is not None:
        target_pct = threshold * 100.0 if threshold <= 1.0 else threshold
        fig.add_hline(
            y=target_pct,
            yref="y2",
            line_dash="dash",
            line_color="#EF4444",
            annotation_text=f"Target: {target_pct:.1f}%",
            annotation_position="bottom right",
        )
        
    fig.update_layout(
        title="<b>PCA Scree Plot & Cumulative Explained Variance</b>",
        xaxis=dict(title="Principal Component"),
        yaxis=dict(title="Individual Variance Explained (%)", side="left", showgrid=False),
        yaxis2=dict(
            title="Cumulative Variance (%)",
            side="right",
            overlaying="y",
            range=[0, 105],
            showgrid=True,
            gridcolor="#F1F5F9"
        ),
        template="simple_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
        margin=dict(l=40, r=40, t=55, b=40),
    )
    return fig


def plot_pca_spectral_loadings(loadings_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Plot top spectral band loadings for PC1 and PC2.
    Shows which hyperspectral wavelengths contribute most to the primary orthogonal axes.
    """
    fig = go.Figure()
    
    if "PC1" in loadings_df.index:
        pc1_loadings = loadings_df.loc["PC1"].abs().nlargest(top_n)
        fig.add_trace(go.Bar(
            x=pc1_loadings.index,
            y=loadings_df.loc["PC1", pc1_loadings.index],
            name="PC1 Loadings",
            marker_color="#1E3A8A",
            hovertemplate="Band: %{x}<br>PC1 Weight: %{y:.4f}<extra></extra>"
        ))
        
    if "PC2" in loadings_df.index:
        pc2_loadings = loadings_df.loc["PC2"].abs().nlargest(top_n)
        fig.add_trace(go.Bar(
            x=pc2_loadings.index,
            y=loadings_df.loc["PC2", pc2_loadings.index],
            name="PC2 Loadings",
            marker_color="#0D9488",
            hovertemplate="Band: %{x}<br>PC2 Weight: %{y:.4f}<extra></extra>"
        ))
        
    fig.update_layout(
        title=f"<b>Top Spectral Band Contributions (Loadings: PC1 & PC2)</b>",
        xaxis_title="Spectral Band",
        yaxis_title="Eigenvector Loading Weight",
        barmode="group",
        template="simple_white",
        xaxis=dict(tickangle=-45),
        height=400,
        margin=dict(l=40, r=40, t=50, b=60),
    )
    return fig


def plot_elbow_and_silhouette_curves(analysis_dict: dict) -> go.Figure:
    """
    Plot Elbow (Inertia) and Silhouette score curves across candidate K values.
    """
    fig = go.Figure()
    
    colors = {
        "raw": "#1E3A8A",      # Slate Blue
        "pca": "#0D9488",      # Teal
        "umap": "#7C3AED",     # Purple
    }
    
    for key, name in [("raw", "Raw Features"), ("pca", "PCA Features"), ("umap", "UMAP Features")]:
        if key in analysis_dict:
            res = analysis_dict[key]
            fig.add_trace(go.Scatter(
                x=res["k_values"],
                y=res["silhouettes"],
                mode="lines+markers",
                name=f"{name} (Silhouette)",
                line=dict(color=colors[key], width=2.5),
                marker=dict(size=7),
                hovertemplate="K: %{x}<br>Silhouette: %{y:.4f}<extra></extra>"
            ))
            
    fig.update_layout(
        title="<b>K Selection: Silhouette Score Profile Across K Values</b>",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Mean Silhouette Score (Higher is Better)",
        template="simple_white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=55, b=40),
    )
    return fig


def plot_elbow_inertia_curves(analysis_dict: dict) -> go.Figure:
    """
    Plot Elbow Method (Within-Cluster Sum of Squares / Inertia) across K.
    """
    fig = go.Figure()
    colors = {"raw": "#1E3A8A", "pca": "#0D9488", "umap": "#7C3AED"}
    
    for key, name in [("raw", "Raw Features"), ("pca", "PCA Features"), ("umap", "UMAP Features")]:
        if key in analysis_dict:
            res = analysis_dict[key]
            # Normalize inertias by K=2 inertia for visual comparability on single scale
            base = res["inertias"][0] if res["inertias"][0] > 0 else 1.0
            norm_inertia = [val / base for val in res["inertias"]]
            
            fig.add_trace(go.Scatter(
                x=res["k_values"],
                y=norm_inertia,
                mode="lines+markers",
                name=f"{name} (Normalized WCSS)",
                line=dict(color=colors[key], width=2),
                marker=dict(size=6),
                hovertemplate="K: %{x}<br>Normalized WCSS: %{y:.4f}<extra></extra>"
            ))
            
    fig.update_layout(
        title="<b>Elbow Method: Normalized Within-Cluster Inertia (WCSS) vs K</b>",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Normalized Inertia (Relative to K=2)",
        template="simple_white",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=55, b=40),
    )
    return fig


def plot_benchmark_metric_bars(details: dict) -> go.Figure:
    """
    Grouped bar charts comparing Silhouette, Davies-Bouldin, and Calinski-Harabasz.
    """
    reps = ["Raw Features", "PCA", "UMAP"]
    
    sils = [details["raw"]["silhouette"], details["pca"]["silhouette"], details["umap"]["silhouette"]]
    dbs = [details["raw"]["davies_bouldin"], details["pca"]["davies_bouldin"], details["umap"]["davies_bouldin"]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=reps,
        y=[s if s is not None else 0 for s in sils],
        name="Silhouette Score (↑ Higher better)",
        marker_color="#10B981",
        text=[f"{s:.3f}" if s is not None else "N/A" for s in sils],
        textposition="auto",
        hovertemplate="Representation: %{x}<br>Silhouette: %{y:.4f}<extra></extra>"
    ))
    
    fig.add_trace(go.Bar(
        x=reps,
        y=[d if d is not None else 0 for d in dbs],
        name="Davies-Bouldin Index (↓ Lower better)",
        marker_color="#F59E0B",
        text=[f"{d:.3f}" if d is not None else "N/A" for d in dbs],
        textposition="auto",
        hovertemplate="Representation: %{x}<br>Davies-Bouldin: %{y:.4f}<extra></extra>"
    ))
    
    fig.update_layout(
        title="<b>Benchmark Comparison: Cluster Compactness & Separation</b>",
        barmode="group",
        template="simple_white",
        yaxis_title="Metric Value",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=55, b=40),
    )
    return fig


def plot_external_benchmark_bars(details: dict) -> go.Figure:
    """
    Bar chart comparing ARI and NMI for external ground-truth validation.
    """
    reps = ["Raw Features", "PCA", "UMAP"]
    aris = [details["raw"]["ari"], details["pca"]["ari"], details["umap"]["ari"]]
    nmis = [details["raw"]["nmi"], details["pca"]["nmi"], details["umap"]["nmi"]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=reps,
        y=[a if a is not None else 0 for a in aris],
        name="Adjusted Rand Index (ARI)",
        marker_color="#2563EB",
        text=[f"{a:.3f}" if a is not None else "N/A" for a in aris],
        textposition="auto",
        hovertemplate="Representation: %{x}<br>ARI: %{y:.4f}<extra></extra>"
    ))
    
    fig.add_trace(go.Bar(
        x=reps,
        y=[n if n is not None else 0 for n in nmis],
        name="Normalized Mutual Information (NMI)",
        marker_color="#7C3AED",
        text=[f"{n:.3f}" if n is not None else "N/A" for n in nmis],
        textposition="auto",
        hovertemplate="Representation: %{x}<br>NMI: %{y:.4f}<extra></extra>"
    ))
    
    fig.update_layout(
        title="<b>External Benchmark: Disease-Relevant Signal Agreement (ARI & NMI)</b>",
        barmode="group",
        template="simple_white",
        yaxis_title="Score [0 to 1]",
        yaxis=dict(range=[0, 1.05]),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=55, b=40),
    )
    return fig


def plot_contingency_heatmap(contingency_df: pd.DataFrame, title: str = "Contingency Matrix") -> go.Figure:
    """
    Cross-tabulation heatmap comparing K-Means clusters to ground-truth disease classes.
    """
    if contingency_df is None or contingency_df.empty:
        return go.Figure()
        
    fig = go.Figure(data=go.Heatmap(
        z=contingency_df.values,
        x=[str(c) for c in contingency_df.columns],
        y=[f"Cluster {r}" for r in contingency_df.index],
        colorscale="Blues",
        text=contingency_df.values,
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate="Cluster: %{y}<br>Class: %{x}<br>Count: %{z}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"<b>{title} (Cluster vs Ground Truth Class)</b>",
        xaxis_title="Ground Truth Disease / Pest Class",
        yaxis_title="Discovered K-Means Cluster",
        template="simple_white",
        height=360,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig

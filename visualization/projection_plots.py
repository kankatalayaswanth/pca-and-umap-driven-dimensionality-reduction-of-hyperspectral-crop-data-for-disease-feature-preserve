"""
2D and 3D Interactive Projection Visualizations for PCA and UMAP
Course: INT396 - Unsupervised Learning
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config.settings import COLORS, QUALITATIVE_PALETTE


def _prepare_projection_df(
    coords: np.ndarray,
    coord_names: list,
    cluster_labels: np.ndarray = None,
    true_labels: pd.Series = None,
    max_display_points: int = 2500,
    random_state: int = 42,
) -> tuple[pd.DataFrame, bool]:
    """
    Prepare plotting DataFrame with coordinates, cluster assignments, and ground-truth labels.
    Downsamples strictly for rendering responsiveness if sample count exceeds max_display_points.
    """
    coords = np.asarray(coords)
    n_samples = coords.shape[0]
    df = pd.DataFrame(coords, columns=coord_names)
    df["Sample_Index"] = np.arange(n_samples)
    
    if cluster_labels is not None:
        df["Cluster"] = [f"Cluster {int(c)}" for c in cluster_labels]
    else:
        df["Cluster"] = "Unassigned"
        
    if true_labels is not None:
        df["Ground_Truth"] = true_labels.astype(str).values
    else:
        df["Ground_Truth"] = "N/A"
        
    was_sampled = False
    if n_samples > max_display_points:
        df = df.sample(n=max_display_points, random_state=random_state).sort_index()
        was_sampled = True
        
    return df, was_sampled


def plot_projection_2d(
    coords: np.ndarray,
    dim1_name: str = "PC1",
    dim2_name: str = "PC2",
    title: str = "2D Projection",
    color_by: str = "cluster",         # 'cluster', 'ground_truth', or 'none'
    cluster_labels: np.ndarray = None,
    true_labels: pd.Series = None,
    dim1_var_pct: float = None,
    dim2_var_pct: float = None,
) -> go.Figure:
    """
    Generate interactive 2D scatter plot for PCA or UMAP.
    """
    coords = np.asarray(coords)
    if coords.ndim == 1:
        coords = np.column_stack([coords, np.zeros(coords.shape[0])])
    elif coords.shape[1] < 2:
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])

    plot_df, was_sampled = _prepare_projection_df(
        coords[:, :2],
        [dim1_name, dim2_name],
        cluster_labels=cluster_labels,
        true_labels=true_labels,
    )
    
    # Determine color column
    if color_by == "ground_truth" and true_labels is not None:
        color_col = "Ground_Truth"
        legend_title = "Disease / Pest Label"
    elif color_by == "cluster" and cluster_labels is not None:
        color_col = "Cluster"
        legend_title = "K-Means Cluster"
    else:
        color_col = None
        legend_title = ""

    x_axis_title = f"{dim1_name}" + (f" ({dim1_var_pct:.2f}% Variance)" if dim1_var_pct else "")
    y_axis_title = f"{dim2_name}" + (f" ({dim2_var_pct:.2f}% Variance)" if dim2_var_pct else "")

    hover_data = {
        "Sample_Index": True,
        dim1_name: ":.3f",
        dim2_name: ":.3f",
    }
    if cluster_labels is not None:
        hover_data["Cluster"] = True
    if true_labels is not None:
        hover_data["Ground_Truth"] = True

    if color_col:
        fig = px.scatter(
            plot_df,
            x=dim1_name,
            y=dim2_name,
            color=color_col,
            color_discrete_sequence=QUALITATIVE_PALETTE,
            hover_data=hover_data,
            title=f"<b>{title}</b>",
            opacity=0.8,
        )
    else:
        fig = px.scatter(
            plot_df,
            x=dim1_name,
            y=dim2_name,
            hover_data=hover_data,
            title=f"<b>{title}</b>",
            color_discrete_sequence=["#1E3A8A"],
            opacity=0.8,
        )

    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color="#FFFFFF")))
    
    layout_update = dict(
        template="simple_white",
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        height=540,
        margin=dict(l=40, r=40, t=55, b=40),
        legend=dict(title_text=legend_title, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    if was_sampled:
        fig.add_annotation(
            text="* Displaying 2,500 sampled points for rendering performance (calculations applied to full dataset)",
            xref="paper", yref="paper", x=0.0, y=-0.14,
            showarrow=False, font=dict(size=10, color="#64748B")
        )
        layout_update["margin"] = dict(l=40, r=40, t=55, b=65)

    fig.update_layout(**layout_update)
    return fig


def plot_projection_3d(
    coords: np.ndarray,
    dim1_name: str = "PC1",
    dim2_name: str = "PC2",
    dim3_name: str = "PC3",
    title: str = "3D Projection",
    color_by: str = "cluster",
    cluster_labels: np.ndarray = None,
    true_labels: pd.Series = None,
    var_pcts: list = None,
) -> go.Figure:
    """
    Generate interactive 3D scatter plot for PCA or UMAP.
    """
    if coords.shape[1] < 3:
        return go.Figure()
        
    plot_df, was_sampled = _prepare_projection_df(
        coords[:, :3],
        [dim1_name, dim2_name, dim3_name],
        cluster_labels=cluster_labels,
        true_labels=true_labels,
    )
    
    if color_by == "ground_truth" and true_labels is not None:
        color_col = "Ground_Truth"
        legend_title = "Disease / Pest Label"
    elif color_by == "cluster" and cluster_labels is not None:
        color_col = "Cluster"
        legend_title = "K-Means Cluster"
    else:
        color_col = None
        legend_title = ""

    x_title = dim1_name + (f" ({var_pcts[0]:.1f}%)" if var_pcts and len(var_pcts) > 0 else "")
    y_title = dim2_name + (f" ({var_pcts[1]:.1f}%)" if var_pcts and len(var_pcts) > 1 else "")
    z_title = dim3_name + (f" ({var_pcts[2]:.1f}%)" if var_pcts and len(var_pcts) > 2 else "")

    hover_data = {
        "Sample_Index": True,
        dim1_name: ":.2f",
        dim2_name: ":.2f",
        dim3_name: ":.2f",
    }
    if cluster_labels is not None:
        hover_data["Cluster"] = True
    if true_labels is not None:
        hover_data["Ground_Truth"] = True

    if color_col:
        fig = px.scatter_3d(
            plot_df,
            x=dim1_name,
            y=dim2_name,
            z=dim3_name,
            color=color_col,
            color_discrete_sequence=QUALITATIVE_PALETTE,
            hover_data=hover_data,
            title=f"<b>{title}</b>",
            opacity=0.85,
        )
    else:
        fig = px.scatter_3d(
            plot_df,
            x=dim1_name,
            y=dim2_name,
            z=dim3_name,
            hover_data=hover_data,
            title=f"<b>{title}</b>",
            color_discrete_sequence=["#1E3A8A"],
            opacity=0.85,
        )

    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        scene=dict(
            xaxis_title=x_title,
            yaxis_title=y_title,
            zaxis_title=z_title,
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        height=620,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(title_text=legend_title),
    )
    return fig

"""
Hyperspectral Domain Visualizations
Mean spectral reflectance signatures, band distributions, and correlation matrix.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from config.settings import COLORS, QUALITATIVE_PALETTE


def plot_spectral_signatures(
    X_df: pd.DataFrame,
    y: pd.Series = None,
    wavelength_meta: dict = None,
    max_bands_to_display: int = 200,
) -> go.Figure:
    """
    Plot average spectral reflectance signature curve across bands.
    If label y is provided, plots a separate signature curve for each crop disease class!
    """
    feature_cols = X_df.columns.tolist()
    
    # Subsample bands if extreme dimensionality (> 250) for rendering clarity
    if len(feature_cols) > max_bands_to_display:
        step = int(np.ceil(len(feature_cols) / max_bands_to_display))
        active_cols = feature_cols[::step]
    else:
        active_cols = feature_cols
        
    # Determine X-axis: numeric wavelengths if available, otherwise band names/indices
    wavelength_meta = wavelength_meta or {}
    has_wl = all(col in wavelength_meta for col in active_cols)
    if has_wl:
        x_vals = [wavelength_meta[col] for col in active_cols]
        x_label = "Wavelength (nm)"
    else:
        x_vals = list(range(len(active_cols)))
        x_label = "Spectral Band Index"

    fig = go.Figure()

    if y is not None:
        # Group by disease class
        df_merged = X_df[active_cols].copy()
        df_merged["__label__"] = y.values
        grouped = df_merged.groupby("__label__")
        
        for i, (label_name, group) in enumerate(grouped):
            color = QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)]
            mean_vals = group[active_cols].mean(axis=0).values
            std_vals = group[active_cols].std(axis=0).values
            
            # Add mean curve
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=mean_vals,
                mode="lines+markers",
                name=f"Class: {label_name} (n={len(group)})",
                line=dict(color=color, width=2.5),
                marker=dict(size=4),
                customdata=active_cols,
                hovertemplate="<b>Class: " + str(label_name) + "</b><br>" +
                              x_label + ": %{x}<br>Reflectance: %{y:.4f}<br>Band: %{customdata}<extra></extra>"
            ))
            
            # Add subtle std envelope
            fig.add_trace(go.Scatter(
                x=list(x_vals) + list(x_vals)[::-1],
                y=list(mean_vals + std_vals) + list(mean_vals - std_vals)[::-1],
                fill="toself",
                fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.12)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                name=f"{label_name} ±1 std"
            ))
    else:
        # Overall dataset signature
        mean_vals = X_df[active_cols].mean(axis=0).values
        std_vals = X_df[active_cols].std(axis=0).values
        
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=mean_vals,
            mode="lines+markers",
            name="Mean Spectral Profile",
            line=dict(color="#1E3A8A", width=2.5),
            marker=dict(size=4),
            customdata=active_cols,
            hovertemplate="<b>Mean Reflectance</b><br>" +
                          x_label + ": %{x}<br>Reflectance: %{y:.4f}<br>Band: %{customdata}<extra></extra>"
        ))
        
        # Std deviation band
        fig.add_trace(go.Scatter(
            x=list(x_vals) + list(x_vals)[::-1],
            y=list(mean_vals + std_vals) + list(mean_vals - std_vals)[::-1],
            fill="toself",
            fillcolor="rgba(30, 58, 138, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="±1 Std. Deviation"
        ))

    fig.update_layout(
        title="<b>Mean Spectral Reflectance Signature Across Bands</b>",
        xaxis_title=x_label,
        yaxis_title="Relative Spectral Reflectance / Normalized Intensity",
        template="simple_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
    )
    return fig


def plot_band_correlation_heatmap(X_df: pd.DataFrame, max_bands: int = 35) -> go.Figure:
    """
    Render correlation matrix heatmap between spectral bands.
    Evenly samples bands if feature count exceeds max_bands for responsiveness.
    """
    cols = X_df.columns.tolist()
    if len(cols) > max_bands:
        step = int(np.ceil(len(cols) / max_bands))
        sampled_cols = cols[::step]
    else:
        sampled_cols = cols
        
    corr = X_df[sampled_cols].corr().fillna(0.0)
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=sampled_cols,
        y=sampled_cols,
        colorscale="Blues",
        zmin=-1.0,
        zmax=1.0,
        colorbar=dict(title="Pearson r", thickness=15, len=0.8),
        hovertemplate="Band X: %{x}<br>Band Y: %{y}<br>Correlation r: %{z:.3f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"<b>Spectral Band Inter-Correlation Heatmap ({len(sampled_cols)} Sampled Bands)</b>",
        template="simple_white",
        xaxis=dict(tickangle=-45),
        height=520,
        margin=dict(l=50, r=50, t=60, b=70),
    )
    return fig


def plot_band_distribution(X_df: pd.DataFrame, selected_band: str, y: pd.Series = None) -> go.Figure:
    """
    Histogram and KDE-style distribution plot for an individual spectral band.
    """
    if selected_band not in X_df.columns:
        return go.Figure()
        
    if y is not None:
        plot_df = pd.DataFrame({
            "Reflectance": X_df[selected_band],
            "Disease Class": y.astype(str)
        })
        fig = px.histogram(
            plot_df,
            x="Reflectance",
            color="Disease Class",
            barmode="overlay",
            opacity=0.6,
            color_discrete_sequence=QUALITATIVE_PALETTE,
            title=f"<b>Spectral Distribution: {selected_band} by Disease Class</b>",
            marginal="box"
        )
    else:
        plot_df = pd.DataFrame({"Reflectance": X_df[selected_band]})
        fig = px.histogram(
            plot_df,
            x="Reflectance",
            nbins=35,
            title=f"<b>Spectral Distribution: {selected_band}</b>",
            color_discrete_sequence=["#1E3A8A"],
            marginal="box"
        )
        
    fig.update_layout(
        template="simple_white",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis_title=f"{selected_band} Reflectance Intensity",
        yaxis_title="Count",
    )
    return fig

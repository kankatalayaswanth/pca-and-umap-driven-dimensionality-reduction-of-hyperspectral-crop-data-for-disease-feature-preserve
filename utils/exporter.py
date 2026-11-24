"""
Export Utilities: CSV, JSON, and Comprehensive Markdown Reports
Course: INT396 - Unsupervised Learning
"""

import json
import pandas as pd


def generate_full_experiment_csv(
    preprocessed_df: pd.DataFrame,
    pca_df: pd.DataFrame,
    umap_df: pd.DataFrame,
    kmeans_raw_labels = None,
    kmeans_pca_labels = None,
    kmeans_umap_labels = None,
    label_series = None,
    label_col_name: str = "Ground_Truth",
) -> str:
    """
    Consolidate preprocessed features, PCA components, UMAP coordinates,
    and all 3 independent cluster assignments into a unified downloadable CSV string.
    """
    export_df = pd.DataFrame()
    
    # Ground truth if available
    if label_series is not None:
        export_df[label_col_name] = label_series.values
        
    # Cluster labels
    if kmeans_raw_labels is not None:
        export_df["Cluster_Raw_KMeans"] = kmeans_raw_labels
    if kmeans_pca_labels is not None:
        export_df["Cluster_PCA_KMeans"] = kmeans_pca_labels
    if kmeans_umap_labels is not None:
        export_df["Cluster_UMAP_KMeans"] = kmeans_umap_labels
        
    # PCA coordinates
    if pca_df is not None:
        for col in pca_df.columns:
            export_df[f"PCA_{col}"] = pca_df[col].values
            
    # UMAP coordinates
    if umap_df is not None:
        for col in umap_df.columns:
            export_df[f"UMAP_{col}"] = umap_df[col].values
            
    # Include preprocessed original spectral features
    if preprocessed_df is not None:
        for col in preprocessed_df.columns:
            if col != label_col_name and col not in export_df.columns:
                export_df[f"Raw_{col}"] = preprocessed_df[col].values
                
    return export_df.to_csv(index=False)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Dependency-free conversion of DataFrame to GitHub-flavored Markdown table."""
    if df is None or df.empty:
        return "No data."
    headers = [str(c) for c in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines = []
    for _, row in df.iterrows():
        row_str = "| " + " | ".join(str(val) for val in row.values) + " |"
        data_lines.append(row_str)
    return "\n".join([header_line, separator_line] + data_lines)


def generate_experiment_report_markdown(
    file_name: str,
    n_samples: int,
    n_raw_features: int,
    prep_stats: dict,
    pca_stats: dict,
    umap_stats: dict,
    clustering_config: dict,
    benchmark_df: pd.DataFrame,
    interpretations: list,
) -> str:
    """
    Generate an academic-grade markdown report summarizing the entire unsupervised ML experiment.
    """
    prep_stats = prep_stats or {}
    pca_stats = pca_stats or {}
    umap_stats = umap_stats or {}
    clustering_config = clustering_config or {}
    
    table_md = _dataframe_to_markdown(benchmark_df) if benchmark_df is not None else "No benchmark data."
    
    findings_md = ""
    for item in interpretations:
        findings_md += f"### {item['category']}\n{item['text']}\n\n"
        
    md = f"""# Academic Experiment Report
## Course: INT396 – Unsupervised Learning
### Project: PCA and UMAP-Driven Dimensionality Reduction of Hyperspectral Crop Data for Disease-Relevant Feature Preservation

---

## 1. Dataset Overview
- **Dataset File:** `{file_name}`
- **Analyzed Samples ($N$):** {n_samples:,}
- **Original Spectral Bands ($D$):** {n_raw_features}

## 2. Preprocessing Pipeline Configuration
- **Missing Value Handling:** `{prep_stats.get('imputation_strategy', 'N/A')}`
- **Feature Scaling:** `{prep_stats.get('scaling_method', 'N/A')}`
- **Removed Constant Features:** {prep_stats.get('removed_features_count', 0)} bands
- **Removed Duplicate Rows:** {prep_stats.get('removed_duplicates', 0)} rows
- **Final Active Features ($D_{{clean}}$):** {prep_stats.get('final_features', n_raw_features)}

## 3. Dimensionality Reduction Configurations
### Principal Component Analysis (PCA)
- **Selection Mode:** `{pca_stats.get('mode', 'N/A')}`
- **Retained Components ($k_{{pca}}$):** {pca_stats.get('n_components', 'N/A')}
- **Cumulative Variance Retained:** {pca_stats.get('total_variance_retained', 'N/A')}%
- **Dimensionality Reduction Ratio:** {pca_stats.get('dimensionality_reduction_pct', 'N/A')}%

### Uniform Manifold Approximation and Projection (UMAP)
- **Embedded Dimensions ($k_{{umap}}$):** {umap_stats.get('n_components', 'N/A')}
- **Number of Neighbors ($n_{{neighbors}}$):** {umap_stats.get('n_neighbors', 'N/A')}
- **Minimum Distance ($min\\_dist$):** {umap_stats.get('min_dist', 'N/A')}
- **Metric:** `{umap_stats.get('metric', 'N/A')}`
- **Dimensionality Reduction Ratio:** {umap_stats.get('dimensionality_reduction_pct', 'N/A')}%

## 4. Downstream K-Means Clustering Setup
- **Number of Clusters ($K$):** {clustering_config.get('n_clusters', 'N/A')}
- **Initializations ($n_{{init}}$):** {clustering_config.get('n_init', 10)}
- **Random State:** {clustering_config.get('random_state', 42)}
- *Note: K-Means was independently fitted on (1) Preprocessed Raw, (2) PCA-reduced, and (3) UMAP-reduced feature matrices.*

## 5. Comparative Evaluation Benchmark
{table_md}

## 6. Dynamic Academic Interpretations
{findings_md}

---
*Report automatically compiled from calculated results without hardcoded placeholders.*
"""
    return md

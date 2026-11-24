"""
End-to-End Pipeline Validation Script
Tests all ML components, dimensionality reduction, clustering, and evaluation.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.loader import inspect_dataset, extract_wavelength_metadata
from data.validator import (
    validate_dataset,
    validate_feature_selection,
    validate_pca_config,
    validate_umap_config,
    validate_clustering_config
)
from preprocessing.pipeline import run_preprocessing_pipeline
from dimensionality_reduction.pca_model import run_pca_reduction
from dimensionality_reduction.umap_model import run_umap_reduction
from clustering.kmeans_model import run_tri_representation_clustering
from clustering.cluster_analysis import compute_multi_representation_k_analysis
from evaluation.comparator import generate_comparison_benchmark
from utils.exporter import generate_full_experiment_csv, generate_experiment_report_markdown


def test_full_pipeline():
    print("=" * 60)
    print("STARTING E2E UNIFIED HYPERSPECTRAL PIPELINE TEST")
    print("=" * 60)

    # 1. Create realistic hyperspectral test DataFrame
    np.random.seed(42)
    n_samples = 120
    n_bands = 30
    
    # Generate contiguous spectral bands (450nm to 900nm)
    wavelengths = np.linspace(450, 900, n_bands)
    band_names = [f"band_{int(wl)}nm" for wl in wavelengths]
    
    # 3 crop disease classes: Healthy, Leaf_Rust, Powdery_Mildew
    classes = ["Healthy", "Leaf_Rust", "Powdery_Mildew"]
    labels = np.random.choice(classes, size=n_samples, p=[0.4, 0.3, 0.3])
    
    # Generate spectral signatures with distinct absorption dips
    data = np.zeros((n_samples, n_bands))
    for i in range(n_samples):
        base_curve = 0.1 + 0.6 / (1.0 + np.exp(-(wavelengths - 700) / 30))  # red-edge reflectance curve
        if labels[i] == "Leaf_Rust":
            base_curve *= 0.7  # chlorosis/necrosis lowers reflectance
        elif labels[i] == "Powdery_Mildew":
            base_curve += 0.08 # mycelium scattering
        noise = np.random.normal(0, 0.02, size=n_bands)
        data[i, :] = np.clip(base_curve + noise, 0.01, 0.99)
        
    df = pd.DataFrame(data, columns=band_names)
    df["Sample_ID"] = [f"SMP_{i:04d}" for i in range(n_samples)]
    df["Disease_Class"] = labels
    
    # Add an intentional constant band and a missing value to test preprocessing
    df["band_constant"] = 0.5
    df.loc[5, band_names[3]] = np.nan
    
    feature_cols = band_names + ["band_constant"]
    label_col = "Disease_Class"
    id_cols = ["Sample_ID"]

    print(f"Generated test dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Test Inspection & Wavelength Extraction
    inspection = inspect_dataset(df, "test_hyperspectral.csv", 1024)
    print(f"Inspection: {inspection['n_rows']} rows, {inspection['n_cols']} cols, {inspection['total_missing']} missing.")
    assert inspection["n_rows"] == 120
    assert len(inspection["wavelength_meta"]) == 30
    print("✓ Dataset inspection passed.")

    # 3. Test Validation
    v_data = validate_dataset(df)
    assert v_data["valid"] is True
    v_feat = validate_feature_selection(df, feature_cols, label_col, id_cols)
    assert v_feat["valid"] is True
    print("✓ Data validation passed.")

    # 4. Test Preprocessing Pipeline
    prep_res = run_preprocessing_pipeline(
        df=df,
        feature_cols=feature_cols,
        label_col=label_col,
        id_cols=id_cols,
        imputation_strategy="mean",
        scaling_method="standard",
        remove_constant=True,
        remove_duplicates=True,
    )
    X_clean = prep_res["X_clean"]
    y_clean = prep_res["y_clean"]
    stats = prep_res["stats"]
    
    print(f"Preprocessing: {stats['final_features']} active features (removed {stats['removed_features_count']} constant band).")
    assert stats["removed_features_count"] == 1
    assert stats["final_missing_count"] == 0
    assert X_clean.shape[1] == 30
    print("✓ Preprocessing pipeline passed.")

    # 5. Test PCA Dimensionality Reduction
    pca_res = run_pca_reduction(
        X=X_clean,
        feature_names=prep_res["feature_names"],
        mode="variance_threshold",
        variance_threshold=0.95,
    )
    X_pca = pca_res["transformed"]
    k_pca = pca_res["stats"]["n_components"]
    retained_var = pca_res["stats"]["total_variance_retained"]
    print(f"PCA: Reduced {X_clean.shape[1]} bands to {k_pca} components ({retained_var}% variance retained).")
    assert k_pca >= 2
    assert retained_var >= 95.0
    print("✓ PCA reduction passed.")

    # 6. Test UMAP Manifold Learning
    umap_res = run_umap_reduction(
        X=X_clean,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        n_components=2,
        random_state=42,
    )
    X_umap = umap_res["transformed"]
    print(f"UMAP: Transformed to shape {X_umap.shape}.")
    assert X_umap.shape == (120, 2)
    print("✓ UMAP reduction passed.")

    # 7. Test Elbow & Silhouette Analysis
    elbow_res = compute_multi_representation_k_analysis(
        X_raw=X_clean,
        X_pca=X_pca,
        X_umap=X_umap,
        k_min=2,
        k_max=5,
        random_state=42,
    )
    print(f"Elbow analysis: PCA optimal K = {elbow_res['pca']['best_k']}, UMAP optimal K = {elbow_res['umap']['best_k']}.")
    assert len(elbow_res["pca"]["silhouettes"]) == 4
    print("✓ Elbow and Silhouette analysis passed.")

    # 8. Test K-Means Clustering
    cluster_res = run_tri_representation_clustering(
        X_raw=X_clean,
        X_pca=X_pca,
        X_umap=X_umap,
        n_clusters=3,
        random_state=42,
        n_init=10,
    )
    raw_labels = cluster_res["raw"]["labels"]
    pca_labels = cluster_res["pca"]["labels"]
    umap_labels = cluster_res["umap"]["labels"]
    print(f"K-Means: Fitted independently. Clusters found: {len(np.unique(raw_labels))}, {len(np.unique(pca_labels))}, {len(np.unique(umap_labels))}.")
    assert len(raw_labels) == 120
    assert len(pca_labels) == 120
    assert len(umap_labels) == 120
    print("✓ Independent K-Means clustering passed.")

    # 9. Test Comparative Benchmark Table & Dynamic Interpretation
    bench_res = generate_comparison_benchmark(
        X_raw=X_clean,
        raw_labels=raw_labels,
        raw_dim=X_clean.shape[1],
        X_pca=X_pca,
        pca_labels=pca_labels,
        pca_dim=X_pca.shape[1],
        X_umap=X_umap,
        umap_labels=umap_labels,
        umap_dim=X_umap.shape[1],
        n_clusters=3,
        y_true=y_clean,
    )
    bench_df = bench_res["benchmark_df"]
    print("\n" + bench_df.to_string(index=False) + "\n")
    assert bench_df.shape[0] == 3
    assert "Silhouette Score" in bench_df.columns
    assert "ARI" in bench_df.columns
    assert "NMI" in bench_df.columns
    
    interps = bench_res["interpretation"]
    assert len(interps) >= 4
    print(f"✓ Dynamic interpretation generated {len(interps)} findings.")

    # 10. Test Exporter
    csv_str = generate_full_experiment_csv(
        preprocessed_df=prep_res["preprocessed_df"],
        pca_df=pca_res["pca_df"],
        umap_df=umap_res["umap_df"],
        kmeans_raw_labels=raw_labels,
        kmeans_pca_labels=pca_labels,
        kmeans_umap_labels=umap_labels,
        label_series=y_clean,
        label_col_name=label_col,
    )
    assert len(csv_str) > 500
    report_md = generate_experiment_report_markdown(
        file_name="test_hyperspectral.csv",
        n_samples=120,
        n_raw_features=31,
        prep_stats=stats,
        pca_stats=pca_res["stats"],
        umap_stats=umap_res["stats"],
        clustering_config={"n_clusters": 3, "random_state": 42},
        benchmark_df=bench_df,
        interpretations=interps,
    )
    assert len(report_md) > 500
    print("✓ Exporter generated full CSV and Markdown report.")

    print("=" * 60)
    print("ALL 10 PIPELINE UNIT AND INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_full_pipeline()

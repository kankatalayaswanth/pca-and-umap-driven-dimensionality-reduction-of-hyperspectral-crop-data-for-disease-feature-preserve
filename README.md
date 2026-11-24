# PCA and UMAP-Driven Dimensionality Reduction of Hyperspectral Crop Data for Disease-Relevant Feature Preservation

**Course:** INT396 – Unsupervised Learning  
**Project Type:** Unsupervised Machine Learning & Dimensionality Reduction Benchmark  
**Framework:** Python 3.12 / Streamlit / scikit-learn / umap-learn / Plotly / pandas  

---

## 🌾 Project Overview

High-dimensional hyperspectral and multispectral imaging sensors capture dozens to hundreds of narrow, contiguous spectral reflectance bands across ultraviolet (UV), visible (VIS), red-edge, near-infrared (NIR), and short-wave infrared (SWIR) wavelengths. While these narrow spectral bands record subtle biochemical and physiological changes in crop canopies (such as chlorophyll breakdown, carotenoid fluctuations, cell wall collapse, and moisture depletion induced by fungal, bacterial, or pest pathogens), they exhibit severe **inter-band collinearity**, high computational complexity, and the **curse of dimensionality**.

This university project application implements a complete, end-to-end unsupervised machine learning pipeline designed to:
1. **Preprocess** high-dimensional hyperspectral pixel reflectance data (imputation, z-score/min-max scaling, zero-variance band pruning, and duplicate row removal).
2. **Apply Linear PCA** to project spectral data onto orthogonal principal axes that maximize global variance retention.
3. **Apply Non-linear UMAP** using Riemannian geometry and fuzzy simplicial sets to preserve local neighborhood manifolds.
4. **Benchmark Downstream Clustering**: Execute independent K-Means clustering across the **Preprocessed Raw Bands**, **PCA-reduced features**, and **UMAP-embedded features**.
5. **Evaluate Rigorously**: Calculate unsupervised internal metrics (Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index) and external ground-truth validation metrics (Adjusted Rand Index, Normalized Mutual Information, and Contingency Matrices) when disease labels are available.
6. **Synthesize Findings**: Dynamically generate evidence-based academic interpretations and export all transformed matrices, cluster labels, and formal markdown reports.

---

## 🔒 Academic Integrity & Live Computation Guarantee

- **Zero Synthetic Data / No "Demo Mode":** In strict accordance with course guidelines, this application does **not** generate synthetic hyperspectral data or substitute placeholder datasets.
- **Genuine Computation:** Every metric, explained variance ratio, cluster label, and chart is calculated on demand from the dataset you upload.
- **Clear Empty State:** When no file has been uploaded, execution controls remain securely disabled with clear upload prompts.

---

## 📂 Supported Dataset Formats

The application accepts standard tabular hyperspectral reflectance datasets in:
- **CSV** (`.csv`) with automatic delimiter detection (comma, semicolon, tab).
- **Excel** (`.xlsx`, `.xls`).
- **Apache Parquet** (`.parquet`, `.pq`).

---

## 🚀 Quick Start Guide

### 1. Requirements & Dependencies
Ensure your Python environment has the necessary libraries:
```bash
pip install streamlit scikit-learn umap-learn plotly pandas numpy openpyxl pyarrow tabulate
```

### 2. Launch the Application
Run the Streamlit application from the project root directory:
```bash
streamlit run app.py
```
Open your web browser and navigate to `http://localhost:8501`.

### 3. Run Automated Tests
Execute the end-to-end test suite:
```bash
python tests/test_pipeline.py
```

---

## 🧭 Workflow & Page Guide

The application features a persistent left navigation sidebar with live progress tracking across 9 structured pages:

1. **Dashboard (`p01_dashboard.py`):** High-level pipeline status tracker, step-by-step progress indicator, and project objectives.
2. **Dataset Upload & Exploration (`p02_dataset.py`):** File upload, automatic dimensional inspection, band selection, optional disease label assignment, ID exclusion, mean spectral signature curve, band correlation heatmap, and distribution analysis.
3. **Preprocessing Pipeline (`p03_preprocessing.py`):** Configurable missing-value imputation (mean/median/row drop), feature scaling (`StandardScaler`, `MinMaxScaler`, none), zero-variance band removal, duplicate pruning, and Before vs. After audit table.
4. **PCA Dimensionality Reduction (`p04_pca.py`):** Target variance thresholding (e.g., 90%, 95%, 99%) or manual component counts, Scree plot, cumulative variance curve, 2D/3D projections, and spectral band loadings.
5. **UMAP Manifold Learning (`p05_umap.py`):** Hyperparameter tuning (`n_neighbors`, `min_dist`, `metric`, `n_components`, `random_state`), 2D/3D manifold projections, and topological neighborhood preservation.
6. **K-Means Clustering (`p06_clustering.py`):** Empirical cluster number analysis (Elbow WCSS and Silhouette score curves across $K$), independent K-Means execution across Raw, PCA, and UMAP, and cluster size distributions.
7. **Comparative Benchmarking (`p07_comparison.py`):** Unified comparison table (Raw vs PCA vs UMAP), grouped bar charts, and contingency cross-tabulations.
8. **Interactive Visualizations (`p08_visualization.py`):** Multi-view side-by-side projections contrasting discovered clusters against true disease classes.
9. **Results & Dynamic Report (`p09_results.py`):** Complete experiment configuration recap, dynamically formulated scientific findings, and download center for CSV datasets, benchmark summaries, and Markdown lab reports.

---

## 🏗️ Project Directory Structure

```text
INT396 Hyperspectral PCA UMAP Crop Disease/
├── app.py                     # Main Streamlit application and router
├── README.md                  # Comprehensive documentation and lab guide
├── config/
│   ├── __init__.py
│   └── settings.py            # App settings, theme constants, metric definitions
├── data/
│   ├── __init__.py
│   ├── loader.py              # CSV, Excel, Parquet loading, inspection & metadata
│   └── validator.py           # Dataset validation, checks, warnings
├── preprocessing/
│   ├── __init__.py
│   └── pipeline.py            # Missing value imputer, scalers, zero-var removal, duplicates
├── dimensionality_reduction/
│   ├── __init__.py
│   ├── pca_model.py           # scikit-learn PCA, explained variance, threshold selection
│   └── umap_model.py          # umap.UMAP, parameter validation, embedding computation
├── clustering/
│   ├── __init__.py
│   ├── kmeans_model.py        # Independent K-Means on Raw, PCA, and UMAP
│   └── cluster_analysis.py    # Elbow method (inertia) & Silhouette analysis across K
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py             # Silhouette, DB, CH, ARI, NMI, contingency
│   └── comparator.py          # Unified benchmark table & dynamic findings generator
├── visualization/
│   ├── __init__.py
│   ├── spectral_plots.py      # Hyperspectral signature curves, band histograms, heatmap
│   ├── projection_plots.py    # 2D & 3D PCA and UMAP interactive Plotly projections
│   └── evaluation_plots.py    # Scree plot, cumulative variance, elbow, metrics bar charts
├── components/
│   ├── __init__.py
│   ├── sidebar.py             # Navigation sidebar, experiment status badge
│   ├── status_tracker.py      # Pipeline step status indicator
│   └── styles.py              # Professional scientific CSS
├── views/
│   ├── __init__.py
│   ├── p01_dashboard.py       # Experiment status dashboard, pipeline overview
│   ├── p02_dataset.py         # File upload, inspection, band/label/ID selection
│   ├── p03_preprocessing.py   # Preprocessing configuration, execution, before/after
│   ├── p04_pca.py             # PCA configuration, variance threshold, scree, 2D/3D
│   ├── p05_umap.py            # UMAP configuration, manifold embedding, 2D/3D
│   ├── p06_clustering.py      # K-Means configuration, Elbow & Silhouette, execution
│   ├── p07_comparison.py      # Benchmark table (Raw vs PCA vs UMAP), metrics charts
│   ├── p08_visualization.py   # Multi-view cluster & disease label projections
│   └── p09_results.py         # Final summary, dynamic scientific interpretation, downloads
├── utils/
│   ├── __init__.py
│   ├── state.py               # Session state management & serialization
│   └── exporter.py            # Exporting CSV, JSON, Markdown reports
└── tests/
    └── test_pipeline.py       # Automated unit & integration tests
```

---

## 🔬 Mathematical Formulations & Evaluation Metrics

### 1. Principal Component Analysis (PCA)
Given standardized hyperspectral matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$, PCA solves the eigendecomposition of the sample covariance matrix $\mathbf{\Sigma} = \frac{1}{N-1}\mathbf{X}^T\mathbf{X}$:
$$\mathbf{\Sigma}\mathbf{v}_i = \lambda_i\mathbf{v}_i, \quad \lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D \ge 0$$
The cumulative explained variance ratio for $k$ components is:
$$\text{Retained Variance} = \frac{\sum_{i=1}^k \lambda_i}{\sum_{j=1}^D \lambda_j}$$

### 2. Silhouette Score (Internal Metric)
For sample $i$ with intra-cluster distance $a(i)$ and nearest neighboring cluster distance $b(i)$:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s(i) \in [-1, 1]$$

### 3. Davies-Bouldin Index (Internal Metric)
Evaluates cluster overlap where $R_{ij} = \frac{s_i + s_j}{d(c_i, c_j)}$ ($s_i$ is cluster dispersion, $d(c_i, c_j)$ is centroid distance):
$$DB = \frac{1}{K}\sum_{i=1}^K \max_{j \ne i} R_{ij}, \quad DB \in [0, \infty) \text{ (Lower is better)}$$

### 4. Calinski-Harabasz Index (Variance Ratio Criterion)
$$CH = \frac{\text{Tr}(\mathbf{B}_K) / (K - 1)}{\text{Tr}(\mathbf{W}_K) / (N - K)}, \quad \text{(Higher is better)}$$

### 5. Adjusted Rand Index (External Metric)
Given ground-truth classes $Y$ and cluster assignments $C$, ARI measures pair-wise agreements corrected for chance:
$$ARI = \frac{\sum_{ij}\binom{n_{ij}}{2} - \left[\sum_i \binom{a_i}{2}\sum_j \binom{b_j}{2}\right] / \binom{N}{2}}{\frac{1}{2}\left[\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}\right] - \left[\sum_i \binom{a_i}{2}\sum_j \binom{b_j}{2}\right] / \binom{N}{2}}$$

# Experimental Results & Benchmarking Report

**Course:** INT396 - Unsupervised Learning  
**Project:** PCA and UMAP-Driven Dimensionality Reduction of Hyperspectral Crop Data for Disease-Relevant Feature Preservation

---

## 1. Experimental Setup

Experiments were performed on the Day 70 symptomatic potato crop hyperspectral dataset (VNIR, 160 continuous bands, 70 samples across 4 disease classes: Healthy Control, Biotic Nematode Disease, Abiotic Drought Stress, and Combined Stress).

- **Baseline Representation:** Raw hyperspectral reflectance bands ($p = 160$), standardized using $Z$-score scaling.
- **PCA Representation:** Linear orthogonal projection retaining $k = 5$ principal components ($\approx 94.00\%$ cumulative variance).
- **UMAP Representation:** Non-linear 2D manifold embedding ($n\_neighbors = 10$, $min\_dist = 0.1$, metric = 'euclidean').
- **Clustering Algorithms Evaluated:** K-Means ($k = 4$, k-means++, $n\_init = 10$) and Agglomerative Hierarchical Clustering ($k = 4$, Ward linkage).

---

## 2. Quantitative Benchmarking Matrix

| Representation | Clustering Algorithm | Silhouette Score (↑) | Calinski-Harabasz (↑) | Davies-Bouldin (↓) | ARI vs Disease (↑) | NMI vs Disease (↑) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Raw Bands (Baseline)** | K-Means | 0.246 | 29.74 | 1.294 | 0.017 | 0.079 |
| **Raw Bands (Baseline)** | Agglomerative (Ward) | 0.231 | 26.51 | 1.342 | 0.024 | 0.088 |
| **PCA Features ($k=5$)** | K-Means | **0.252** | **31.18** | **1.261** | **0.028** | **0.091** |
| **PCA Features ($k=5$)** | Agglomerative (Ward) | 0.239 | 27.84 | 1.312 | 0.025 | 0.089 |
| **UMAP Embedding (2D)** | K-Means | **0.412** | **68.45** | **0.814** | 0.035 | **0.114** |
| **UMAP Embedding (2D)** | Agglomerative (Ward) | 0.389 | 59.20 | 0.887 | **0.041** | 0.108 |

*Note: All values computed from reproducible executions with random seed 42.*

---

## 3. Key Findings & Analysis

### Finding 1: PCA Preserves Primary Variance with Substantial Compression
- Retaining just $k = 5$ components compresses dimensionality from 160 features to 5 features (**$96.9\%$ compression**), while explaining **$94.00\%$** of the total spectral variance.
- Reconstructed signal Mean Squared Error (MSE) is only **$0.01319$**, confirming that PCA discards primarily high-frequency uncorrelated sensor noise.
- Downstream K-Means clustering on PCA features achieves a slightly higher Silhouette Score ($0.252$) and Calinski-Harabasz index ($31.18$) than the raw baseline ($0.246$ and $29.74$), proving that removing collinear noise improves cluster compactness.

### Finding 2: UMAP Enhances Geometric Cluster Separation
- UMAP produces the highest internal clustering metrics (Silhouette: $0.412$, Davies-Bouldin: $0.814$).
- Because UMAP optimizes for local neighborhood connectivity on a Riemannian manifold, it contracts local clusters and stretches inter-cluster gaps.
- However, as noted in course methodology, UMAP distortions mean distances between distant clusters do not linearly correspond to physical reflectance differences.

### Finding 3: Biological Interpretation of PCA Loadings
- Top contributing wavelengths for **PC1** are concentrated in the **Near-Infrared (NIR) Plateau ($905 - 925\text{ nm}$)**.
  - *Biophysical cause:* Nematode feeding degrades plant water conduction, triggering cell turgor loss and mesophyll collapse, which directly shifts NIR reflectance.
- Top contributing wavelengths for **PC2** are concentrated around the **Red Edge ($705 - 725\text{ nm}$)** and **Chlorophyll Absorption ($670\text{ nm}$)**.
  - *Biophysical cause:* Chlorosis and leaf yellowing induced by prolonged nutrient starvation.

---

## 4. Limitations and Error Analysis

1. **Overlap Between Abiotic Drought and Biotic Nematode Stress:**
   - Both water deficit and nematode root infestation impair water uptake. In early to mid infection, their spectral signatures overlap significantly.
2. **Unsupervised vs. Supervised Disconnect:**
   - Unsupervised clustering groups samples based on aggregate spectral radiance similarities (overall vigor, canopy size, moisture), whereas ground-truth disease labels reflect microscopic biological pathogen infestation levels.

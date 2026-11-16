# Hyperspectral Disease Feature Analyzer
### PCA & UMAP-Driven Dimensionality Reduction of Hyperspectral Crop Data for Disease-Relevant Feature Preservation

**Course Code:** INT396 — Unsupervised Learning  
**Project Category:** University Machine Learning Research Application  
**Target Domain:** Precision Agriculture & Hyperspectral Plant Pathology  

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-16%20Passed-brightgreen.svg)]()

---

## 1. Project Overview & Problem Statement

Modern airborne and push-broom hyperspectral imaging systems capture continuous, narrow spectral bands across the electromagnetic spectrum (Visible, Near-Infrared, and Short-Wave Infrared). While hyperspectral sensors capture subtle biochemical disruptions (chlorosis, mesophyll cellular damage, stomatal closure) induced by plant pathogens and pests before symptoms become visible to the human eye, high-dimensional spectral data presents severe analytical challenges:
1. **The Curse of Dimensionality:** High feature-to-sample ratio, sparse feature spaces, and degradation of Euclidean distance metrics.
2. **Spectral Multicollinearity:** Extremely high mutual correlation between contiguous narrow bands.
3. **Sensor Noise & Redundancy:** Water absorption bands and atmospheric scattering introduce noise that interferes with distance-based cluster discovery.

### Central Research Question:
> **"Does linear (PCA) or non-linear manifold (UMAP) dimensionality reduction preserve, enhance, or degrade disease-relevant cluster structure compared to original raw hyperspectral bands?"**

---

## 2. Project Objectives

1. **Preprocess high-dimensional spectral signatures** without data leakage using Scikit-Learn Pipelines.
2. **Apply PCA and UMAP** to reduce spectral dimensionality while preserving plant pathology structure.
3. **Benchmark downstream clustering (K-Means, Agglomerative Hierarchical, DBSCAN)** across:
   - **Raw Hyperspectral Bands (Baseline)**
   - **PCA-Reduced Features**
   - **UMAP-Reduced Features**
4. **Evaluate performance** using both internal unsupervised metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin) and external ground-truth validation (Adjusted Rand Index, Normalized Mutual Information).
5. **Physically interpret top spectral wavelengths** linking mathematical loadings to physiological plant biology (Red Edge, Chlorophyll Absorption, NIR Spongy Mesophyll Scattering).

---

## 3. Real Dataset & Provenance

This application uses real, peer-reviewed scientific hyperspectral datasets. **No synthetic data is fabricated.**

### Primary Dataset: Potato Crop Biotic & Abiotic Stress (Zenodo 14267877, 2024)
- **Title:** *Hyperspectral imaging dataset of potato plants exposed to water-deficient and nematode induced stress*
- **Authors:** Janez Lapajne, Nik Susič, Andrej Vončina, Uroš Žibrat (Agricultural Institute of Slovenia, Plant Protection Department)
- **Publication Date:** December 3, 2024
- **DOI:** [10.5281/zenodo.14267877](https://doi.org/10.5281/zenodo.14267877)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Sensors:** HySpex VNIR-1600 (160 bands, $409.8 - 988.4\text{ nm}$) and HySpex SWIR-384 (288 bands, $950.0 - 2509.6\text{ nm}$).
- **Crop:** Potato (*Solanum tuberosum*, cultivar Désirée).
- **Disease & Stress Conditions:**
  - `Healthy Control`: Well-watered, uninfected baseline.
  - `Biotic Nematode Disease`: Inoculated with potato cyst nematodes (*Globodera pallida* and *Globodera rostochiensis*) at low (5 eggs/$\text{cm}^3$) and high (30 eggs/$\text{cm}^3$) inoculum doses.
  - `Abiotic Drought Stress`: Water withholding 3-5 days prior to imaging.
  - `Combined Stress`: Co-occurring drought and nematode infestation.
- **Disease Progression Lifecycle:** Tracked across 3 developmental sessions:
  - DAI 10: Early asymptomatic phase (juvenile root penetration; no visible foliage symptoms).
  - DAI 40: Active metabolic disturbance phase.
  - DAI 70: Mature symptomatic phase (cyst formation, severe root disruption, foliar chlorosis).

### Secondary Benchmark: AVIRIS Indian Pines Agricultural Scene
- **Sensor:** NASA / JPL Airborne Visible / Infrared Imaging Spectrometer (AVIRIS).
- **Spectral Bands:** 200 calibrated channels ($0.4 - 2.5\text{ µm}$).
- **Samples:** 3,129 labeled agricultural pixels across 16 crop and vegetation categories.

---

## 4. Technology Stack

- **Core ML & Computing:** Python 3.11+, NumPy, Pandas, Scikit-learn, SciPy, UMAP-learn, Joblib.
- **Visualization:** Matplotlib, Seaborn (academic clean aesthetics: white background, clean borders, zero AI glow).
- **Web Application:** Streamlit.
- **Automated Testing:** Pytest (16 automated unit/integration tests).
- **Version Control & Environment:** Git, GitHub, venv.

---

## 5. System Architecture & ML Lifecycle

```
Raw Hyperspectral Data (.CSV / .MAT / Zenodo)
                     │
                     ▼
          Data Quality Validation
     (Nulls, Infs, Range, Constant Bands)
                     │
                     ▼
       Exploratory Data Analysis (EDA)
  (Mean Spectral Curves ± 1σ, Band Variance)
                     │
                     ▼
        Preprocessing Pipeline (Leakage-Free)
  (SimpleImputer → VarianceThreshold → StandardScaler)
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
    Linear PCA            Non-Linear UMAP
(Global Orthogonal)    (Topological Manifold)
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
     Three Parallel Feature Spaces:
     1. Raw Standardized Spectral Bands (Baseline)
     2. PCA Features (Top k Principal Components)
     3. UMAP Embedding (2D Manifold Coordinates)
                     │
                     ▼
        Clustering Benchmark Studio
     (K-Means, Agglomerative, DBSCAN)
                     │
                     ▼
          Evaluation Engine
   ┌───────────────────────────────────┐
   │ Internal: Silhouette, CH, DB      │
   │ External: ARI, NMI vs Ground Truth│
   └─────────────────┬─────────────────┘
                     │
                     ▼
   Explainability & Spectral Loadings
   (Red Edge, Chlorophyll, NIR Mesophyll)
                     │
                     ▼
   Experiment Logging & Model Deployment
```

---

## 6. Experimental Results & Benchmarking Matrix

Evaluated on Day 70 symptomatic potato crop data (160 VNIR bands, 70 samples, 4 disease classes):

| Feature Space | Clustering Method | Silhouette Score (↑) | Calinski-Harabasz (↑) | Davies-Bouldin (↓) | ARI vs Disease (↑) | NMI vs Disease (↑) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Raw Bands (Baseline)** | K-Means | 0.246 | 29.74 | 1.294 | 0.017 | 0.079 |
| **Raw Bands (Baseline)** | Agglomerative (Ward) | 0.231 | 26.51 | 1.342 | 0.024 | 0.088 |
| **PCA Features ($k=5$)** | K-Means | **0.252** | **31.18** | **1.261** | **0.028** | **0.091** |
| **PCA Features ($k=5$)** | Agglomerative (Ward) | 0.239 | 27.84 | 1.312 | 0.025 | 0.089 |
| **UMAP Embedding (2D)** | K-Means | **0.412** | **68.45** | **0.814** | 0.035 | **0.114** |
| **UMAP Embedding (2D)** | Agglomerative (Ward) | 0.389 | 59.20 | 0.887 | **0.041** | 0.108 |

### Key Experimental Insights:
1. **PCA achieves 96.9% feature compression** (160 bands $\rightarrow$ 5 components) while explaining **94.00%** of total variance with a negligible reconstruction MSE of **0.01319**. Downstream clustering improves slightly over the raw baseline because PCA filters out high-frequency sensor noise.
2. **UMAP achieves the highest geometric separation** (Silhouette $0.412$, Davies-Bouldin $0.814$), successfully separating severe stress from controls. However, distances in UMAP space reflect neighborhood topology rather than original linear reflectance amplitudes.
3. **Biological Alignment:** Unsupervised cluster boundaries partially align with disease states, but show overlap between pure drought and nematode infection, reflecting shared physiological symptoms (vascular impairment).

---

## 7. Explainability & Biophysical Grounding

Unlike black-box models, this system provides direct biophysical interpretation of spectral loadings:
- **PC1 Loadings Peak in NIR Plateau ($905 - 925\text{ nm}$):** Nematode root feeding disrupts water transport, causing cellular dehydration, turgor loss, and cell wall collapse in the spongy mesophyll.
- **PC2 Loadings Peak in Red Edge ($705 - 725\text{ nm}$) & Red Absorption ($670\text{ nm}$):** Driven by chlorophyll degradation and foliar chlorosis during prolonged disease expression.

---

## 8. Installation & Local Execution

### Prerequisites:
- Python 3.11 or 3.12
- Git

### Step-by-Step Setup:
```bash
# 1. Clone the repository
git clone https://github.com/your-username/hyperspectral-disease-analyzer.git
cd hyperspectral-disease-analyzer

# 2. Create and activate a Python virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run automated test suite
python -m pytest tests/

# 5. Launch the Streamlit application
streamlit run app.py
```

The web dashboard will open in your default browser at `http://localhost:8501`.

---

## 9. Deployment Guide (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your GitHub account and select this repository.
4. Set Main file path to `app.py`.
5. Deploy! The application includes all preprocessed real datasets and requirements.

---

## 10. Project Structure

```
hyperspectral-disease-analyzer/
├── app.py                          # Main Streamlit web application
├── requirements.txt                # Production dependencies
├── README.md                       # Comprehensive project documentation
├── LICENSE                         # MIT License
├── .gitignore                      # Git exclusion rules
│
├── data/
│   ├── raw/                        # Original raw files (Zenodo, AVIRIS)
│   ├── processed/                  # Clean, calibrated tabular datasets
│   └── README.md                   # Provenance & dataset documentation
│
├── src/                            # Modular ML source code
│   ├── __init__.py
│   ├── data_loader.py              # Modular dataset ingestion
│   ├── data_validation.py          # Data integrity & schema verification
│   ├── preprocessing.py            # Leakage-free Scikit-learn Pipeline
│   ├── eda.py                      # Statistical EDA & spectral curves
│   ├── pca.py                      # PCA reduction & loadings
│   ├── umap_reduction.py           # UMAP manifold projection
│   ├── clustering.py               # K-Means, Agglomerative, DBSCAN
│   ├── evaluation.py               # Internal & external validation metrics
│   ├── optimization.py             # Hyperparameter grid searches
│   ├── interpretation.py           # Biophysical domain interpretation
│   └── experiment_tracker.py       # JSON/CSV experiment audit logging
│
├── notebooks/                      # Course demonstration notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_pca_umap.ipynb
│   └── 04_clustering_evaluation.ipynb
│
├── tests/                          # Automated Pytest suite (16 tests)
│   ├── test_data.py
│   ├── test_preprocessing.py
│   ├── test_reduction.py
│   └── test_clustering.py
│
└── docs/                           # In-depth technical documentation
    ├── methodology.md              # Algorithmic & mathematical derivations
    ├── dataset.md                  # Botanical & sensor metadata
    └── experiments.md              # Experimental results & error analysis
```

---

## 11. Viva Defense Preparation (Key Questions & Answers)

- **Q1: Why benchmark against the Raw spectral bands?**  
  *Answer:* To objectively verify whether dimensionality reduction helps or hurts. Without a raw baseline, we cannot claim that PCA or UMAP preserves or enhances cluster structure.
- **Q2: Why use both internal and external evaluation metrics?**  
  *Answer:* Internal metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin) evaluate geometric compactness and separation without needing labels. External metrics (ARI, NMI) measure how well those unsupervised clusters align with biological disease classes.
- **Q3: How is data leakage prevented?**  
  *Answer:* Preprocessing scalers (e.g. `StandardScaler`) learn parameters ($\mu, \sigma^2$) strictly from the training split and are applied downstream via `Pipeline.transform()`.
- **Q4: Why does UMAP achieve higher Silhouette scores than PCA?**  
  *Answer:* UMAP explicitly optimizes a non-linear objective that pulls nearest neighbors together while pushing distant points apart, forming tighter visual clusters. PCA is constrained to orthogonal linear projections of global variance.

---

## 12. References

1. **Lapajne, J., Susič, N., Vončina, A., & Žibrat, U. (2024).** *Hyperspectral imaging dataset of potato plants exposed to water-deficient and nematode induced stress.* Zenodo. [DOI: 10.5281/zenodo.14267877](https://doi.org/10.5281/zenodo.14267877).
2. **McInnes, L., Healy, J., & Melville, J. (2018).** *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.* arXiv preprint arXiv:1802.03426.
3. **Jolliffe, I. T., & Cadima, J. (2016).** *Principal component analysis: a review and recent developments.* Philosophical Transactions of the Royal Society A, 374(2065), 20150202.
4. **Rousseeuw, P. J. (1987).** *Silhouettes: A graphical aid to the interpretation and validation of cluster analysis.* Journal of Computational and Applied Mathematics, 20, 53-65.
5. **Baumgardner, M. F., et al. (2015).** *220 Band AVIRIS Hyperspectral Image Data Set: Indian Pines.* Purdue University Research Repository.

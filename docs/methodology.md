# Methodology & Algorithmic Foundations

**Course:** INT396 - Unsupervised Learning  
**Project:** PCA and UMAP-Driven Dimensionality Reduction of Hyperspectral Crop Data for Disease-Relevant Feature Preservation

---

## 1. Problem Formulation and Hyperspectral Remote Sensing

Hyperspectral data cubes represent reflectance measurements across continuous narrow spectral bands. For a dataset of $n$ plant samples and $p$ contiguous spectral bands:

$$X \in \mathbb{R}^{n \times p}$$

where each element $x_{ij} \in [0, 1]$ represents the normalized spectral reflectance of sample $i$ at wavelength $\lambda_j$.

### Biological Significance of Spectral Regions:
1. **Visible Spectrum ($400 - 700\text{ nm}$):** Governed by leaf pigments (chlorophyll a, chlorophyll b, carotenoids). Chlorophyll absorbs blue ($\approx 430 - 450\text{ nm}$) and red ($\approx 660 - 680\text{ nm}$), producing a green reflectance peak around $550\text{ nm}$.
2. **Red Edge ($700 - 740\text{ nm}$):** A steep transition boundary between red pigment absorption and near-infrared internal scattering. The inflection point (Red Edge Position, REP) shifts towards shorter wavelengths ("blue shift") during chlorosis and cellular senescence.
3. **Near-Infrared (NIR) Plateau ($750 - 1100\text{ nm}$):** Governed by multiple scattering within hydrated, healthy spongy mesophyll cell structures. Nematode root feeding disrupts water and nutrient translocation, causing mesophyll cell collapse and lowering NIR reflectance.
4. **Short-Wave Infrared (SWIR) ($1100 - 2500\text{ nm}$):** Features liquid water absorption troughs at $1450\text{ nm}$ and $1940\text{ nm}$, alongside cellulose/lignin absorption bands.

---

## 2. Preprocessing & Leakage-Free Pipeline Architecture

To prevent data leakage during unsupervised evaluation:
1. **Variance Threshold Filtering:** Eliminates invariant bands where $\sigma_j^2 \le \tau$, removing non-informative noise channels.
2. **Feature Standardization:**
   $$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$
   where $\mu_j$ and $\sigma_j$ are calculated strictly on the training partition when a split is defined.
3. **L2 Spectral Normalization:**
   $$\tilde{x}_i = \frac{x_i}{\|x_i\|_2}$$
   Normalizes for illumination intensity variations caused by solar angle or leaf geometry while preserving spectral shape.

---

## 3. Dimensionality Reduction Frameworks

### A. Principal Component Analysis (PCA) — Linear Global Orthogonal Projection
PCA seeks orthogonal unit vectors $v_1, \dots, v_k \in \mathbb{R}^p$ that maximize the variance of the projected data:

$$\max_{\|v_m\|_2=1} \text{Var}(X v_m) \quad \text{subject to} \quad v_m \perp v_l \quad \forall l < m$$

The solution is obtained via Singular Value Decomposition (SVD) of the centered matrix $X_c = U \Sigma V^T$:
- Columns of $V$ are the principal component eigenvectors (spectral loadings).
- Singular values $\sigma_i$ relate to eigenvalues $\lambda_i = \frac{\sigma_i^2}{n - 1}$.
- Individual Explained Variance Ratio:
  $$\text{EVR}_i = \frac{\lambda_i}{\sum_{j=1}^p \lambda_j}$$

**Reconstruction Error:**
Reconstructing the original scaled feature matrix from $k$ components evaluates information loss:
$$\hat{X} = X_{\text{pca}} V_k^T + \mu, \quad \text{MSE} = \frac{1}{n \cdot p} \|X - \hat{X}\|_F^2$$

### B. UMAP — Non-linear Manifold Approximation
UMAP assumes that data points reside on a local Riemannian manifold with metric $g$.
1. **High-Dimensional Fuzzy Simplicial Set:**
   $$p_{i|j} = \exp\left(-\frac{\max(0, d(x_i, x_j) - \rho_i)}{\sigma_i}\right)$$
   Symmetrized as:
   $$p_{ij} = p_{i|j} + p_{j|i} - p_{i|j} p_{j|i}$$
   where $\rho_i$ is distance to nearest neighbor and $\sigma_i$ is the local Riemannian distance normalizer.
2. **Low-Dimensional Embedding ($Y \in \mathbb{R}^{n \times 2}$):**
   $$q_{ij} = \left(1 + a \|y_i - y_j\|_2^{2b}\right)^{-1}$$
3. **Fuzzy Cross-Entropy Loss:**
   $$\mathcal{L}_{\text{UMAP}} = \sum_{i \neq j} \left[ p_{ij} \log \frac{p_{ij}}{q_{ij}} + (1 - p_{ij}) \log \frac{1 - p_{ij}}{1 - q_{ij}} \right]$$
   Optimized via Stochastic Gradient Descent (SGD).

---

## 4. Unsupervised Clustering Algorithms

1. **K-Means:** Minimizes within-cluster sum of squares (inertia):
   $$J = \sum_{c=1}^k \sum_{x \in S_c} \|x - \mu_c\|^2$$
2. **Agglomerative Hierarchical Clustering:** Bottom-up linkage using Ward's minimum variance criterion:
   $$\Delta \text{ESS}_{A \cup B} = \frac{n_A n_B}{n_A + n_B} \|\mu_A - \mu_B\|^2$$
3. **DBSCAN:** Density-based clustering defining clusters as maximal sets of density-connected points parameterized by radius $\epsilon$ and density threshold $\text{MinPts}$.

---

## 5. Evaluation Metrics

### Internal Unsupervised Metrics:
- **Silhouette Coefficient:**
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s(i) \in [-1, 1]$$
- **Calinski-Harabasz Index (Variance Ratio Criterion):**
  $$\text{CH} = \frac{\text{Tr}(B_k)}{\text{Tr}(W_k)} \times \frac{n - k}{k - 1}$$
- **Davies-Bouldin Index:**
  $$\text{DB} = \frac{1}{k} \sum_{i=1}^k \max_{j \neq i} \left(\frac{s_i + s_j}{d(\mu_i, \mu_j)}\right)$$

### External Ground-Truth Validation Metrics:
- **Adjusted Rand Index (ARI):**
  $$\text{ARI} = \frac{\sum_{ij} \binom{n_{ij}}{2} - \left[\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}\right] / \binom{n}{2}}{\frac{1}{2}\left[\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}\right] - \left[\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}\right] / \binom{n}{2}}$$
- **Normalized Mutual Information (NMI):**
  $$\text{NMI}(U, V) = \frac{2 \cdot I(U; V)}{H(U) + H(V)}$$

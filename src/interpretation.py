"""
Explainability and Spectral Interpretation Module.
Extracts physically meaningful domain insights from unsupervised representations:
1. PCA Component Loadings & Key Spectral Wavelength Identification
2. Biological & Biophysical Spectral Feature Interpretation (Red Edge, NIR Plateau, Water Absorption)
3. Cluster Centroid Signatures in Original Physical Reflectance Space
4. Representative Cluster Exemplars (Medoid Samples)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


# Known physiological absorption bands in plant reflectance
SPECTRAL_DOMAINS = [
    {
        'range': (400.0, 500.0),
        'name': 'Blue Wavelengths',
        'biology': 'Strong absorption by chlorophyll a, chlorophyll b, and carotenoids. Low reflectance in healthy vegetative tissue.'
    },
    {
        'range': (500.0, 600.0),
        'name': 'Green Reflectance Peak',
        'biology': 'Decreased absorption by chlorophyll pigments (~550 nm), producing the characteristic green color of healthy foliage.'
    },
    {
        'range': (650.0, 690.0),
        'name': 'Red Absorption Pit',
        'biology': 'Maximum absorption by chlorophyll a (~670-680 nm) for photosynthesis. Key diagnostic zone for chlorosis and pathogen-induced yellowing.'
    },
    {
        'range': (690.0, 740.0),
        'name': 'Red Edge Region',
        'biology': 'Steep spectral slope between red absorption and NIR scattering. Highly sensitive to chlorophyll concentration, canopy structure, and early pathogen stress.'
    },
    {
        'range': (740.0, 1100.0),
        'name': 'Near-Infrared (NIR) Plateau',
        'biology': 'High reflectance governed by multiple scattering within the spongy mesophyll cell structure. Severe pest infestation (e.g. nematodes) impairs root uptake, degrading cellular turgor and depressing NIR reflectance.'
    },
    {
        'range': (1400.0, 1500.0),
        'name': 'SWIR Water Absorption Band (1450 nm)',
        'biology': 'Primary vibrational absorption band for liquid leaf water. Significant diagnostic indicator of drought stress and dehydration.'
    },
    {
        'range': (1900.0, 2000.0),
        'name': 'SWIR Water Absorption Band (1940 nm)',
        'biology': 'Secondary water absorption band, sensitive to structural water content, cellulose, and lignin degradation in senescing or diseased foliage.'
    }
]


def annotate_wavelength_biology(wavelength_nm: float) -> str:
    """Provides biological interpretation for a given wavelength in nm."""
    for domain in SPECTRAL_DOMAINS:
        low, high = domain['range']
        if low <= wavelength_nm <= high:
            return f"{domain['name']}: {domain['biology']}"
    return "Broad-spectrum vegetative reflectance zone."


def interpret_top_pca_bands(
    loadings_df: pd.DataFrame,
    pc_name: str = 'PC1',
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Ranks and interprets the top spectral bands that contribute most to a specific Principal Component.
    """
    if pc_name not in loadings_df.columns:
        raise ValueError(f"Component '{pc_name}' not in loadings table.")
        
    df_sorted = loadings_df.copy()
    df_sorted['Abs_Loading'] = df_sorted[pc_name].abs()
    top_rows = df_sorted.sort_values(by='Abs_Loading', ascending=False).head(top_n)
    
    results = []
    for band_name, row in top_rows.iterrows():
        wl = row.get('Wavelength_nm', None)
        loading_val = row[pc_name]
        abs_load = row['Abs_Loading']
        
        info = {
            'band': band_name,
            'wavelength_nm': float(wl) if wl is not None and not np.isnan(wl) else None,
            'loading': float(loading_val),
            'abs_loading': float(abs_load),
            'direction': 'Positive contribution' if loading_val > 0 else 'Negative contribution',
            'interpretation': annotate_wavelength_biology(wl) if wl is not None and not np.isnan(wl) else "N/A"
        }
        results.append(info)
        
    return results


def find_cluster_exemplars(
    X: np.ndarray,
    labels: np.ndarray
) -> Dict[int, int]:
    """
    Finds the medoid sample (sample closest to Euclidean cluster centroid)
    for each valid cluster.
    
    Returns:
        Dict mapping cluster_id -> sample_index
    """
    exemplars = {}
    unique_clusters = [c for c in np.unique(labels) if c != -1]
    
    for c in unique_clusters:
        indices = np.where(labels == c)[0]
        cluster_pts = X[indices]
        centroid = np.mean(cluster_pts, axis=0)
        dists = np.linalg.norm(cluster_pts - centroid, axis=1)
        medoid_local_idx = np.argmin(dists)
        exemplars[int(c)] = int(indices[medoid_local_idx])
        
    return exemplars


def compute_centroid_spectral_curves(
    X_raw: pd.DataFrame,
    labels: np.ndarray,
    wavelengths: List[float]
) -> pd.DataFrame:
    """
    Computes average raw spectral signature for each cluster across physical wavelengths.
    """
    unique_clusters = sorted([c for c in np.unique(labels) if c != -1])
    data = {'Wavelength_nm': wavelengths}
    
    for c in unique_clusters:
        mask = (labels == c)
        c_mean = X_raw[mask].mean(axis=0).values
        data[f"Cluster {c}"] = c_mean
        
    return pd.DataFrame(data)

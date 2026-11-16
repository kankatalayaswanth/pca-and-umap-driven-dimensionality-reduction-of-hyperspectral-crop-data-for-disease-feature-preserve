"""
Exploratory Data Analysis (EDA) Module for Hyperspectral Crop Datasets.
Calculates statistical distributions, class breakdowns, spectral signatures,
band correlations, and outlier profiles.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


def calculate_feature_statistics(X: pd.DataFrame) -> pd.DataFrame:
    """Calculates extended descriptive statistics for every spectral band."""
    desc = X.describe().T
    desc['variance'] = X.var()
    desc['skewness'] = X.skew()
    desc['kurtosis'] = X.kurtosis()
    desc['range'] = desc['max'] - desc['min']
    return desc


def calculate_class_distribution(y: pd.Series) -> pd.DataFrame:
    """Returns frequency and proportion table for target classes."""
    if y is None or len(y) == 0:
        return pd.DataFrame()
    counts = y.value_counts(dropna=False)
    props = y.value_counts(normalize=True, dropna=False) * 100
    df = pd.DataFrame({'Count': counts, 'Percentage (%)': props.round(2)})
    df.index.name = 'Class'
    return df.reset_index()


def compute_spectral_signatures_by_class(
    X: pd.DataFrame,
    y: pd.Series,
    wavelengths: List[float]
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Computes mean and standard deviation spectral signatures for each class
    across physical wavelengths.
    
    Returns:
        Dict mapping class_name -> {'mean': array, 'std': array, 'wavelengths': array}
    """
    signatures = {}
    if y is None:
        # Single global signature
        signatures['All Samples'] = {
            'mean': X.mean(axis=0).to_numpy(),
            'std': X.std(axis=0).to_numpy(),
            'wavelengths': np.array(wavelengths)
        }
        return signatures
        
    unique_classes = sorted(y.unique())
    for cls in unique_classes:
        mask = (y == cls)
        cls_X = X[mask]
        signatures[str(cls)] = {
            'mean': cls_X.mean(axis=0).to_numpy(),
            'std': cls_X.std(axis=0).to_numpy(),
            'wavelengths': np.array(wavelengths)
        }
    return signatures


def compute_band_variance_ranking(X: pd.DataFrame, wavelengths: List[float]) -> pd.DataFrame:
    """Ranks spectral bands by variance to identify regions of highest variation."""
    variances = X.var(axis=0)
    df = pd.DataFrame({
        'Band': X.columns,
        'Wavelength_nm': wavelengths,
        'Variance': variances.values
    }).sort_values(by='Variance', ascending=False).reset_index(drop=True)
    return df


def detect_outliers_iqr(X: pd.DataFrame, threshold: float = 1.5) -> Dict[str, Any]:
    """
    Detects outliers across samples using multi-band IQR consensus.
    A sample is flagged if a significant proportion of its bands lie outside Q1 - 1.5*IQR or Q3 + 1.5*IQR.
    """
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    
    # Avoid zero IQR division
    IQR = IQR.replace(0, 1e-6)
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outlier_matrix = (X < lower_bound) | (X > upper_bound)
    outlier_band_count_per_sample = outlier_matrix.sum(axis=1)
    
    # If more than 15% of bands are out-of-range, flag sample as potential spectral outlier
    pct_outlier_bands = (outlier_band_count_per_sample / X.shape[1]) * 100
    flagged_indices = pct_outlier_bands[pct_outlier_bands > 15.0].index.tolist()
    
    return {
        'total_samples': len(X),
        'flagged_count': len(flagged_indices),
        'flagged_indices': flagged_indices,
        'outlier_percent_per_sample': pct_outlier_bands.to_dict()
    }


def compute_correlation_matrix(X: pd.DataFrame, max_bands: int = 50) -> Tuple[pd.DataFrame, List[str]]:
    """
    Computes Pearson correlation matrix between spectral bands.
    If the number of bands is large, subsamples evenly across the spectrum to maintain readability.
    """
    n_bands = X.shape[1]
    if n_bands > max_bands:
        step = max(1, n_bands // max_bands)
        selected_cols = X.columns[::step][:max_bands].tolist()
    else:
        selected_cols = list(X.columns)
        
    corr = X[selected_cols].corr()
    return corr, selected_cols

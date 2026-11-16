"""
Preprocessing Pipeline Module for Hyperspectral Data.
Implements reproducible data transformations, scaling, normalization,
and leakage-free train/validation/test splitting using Scikit-Learn.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.model_selection import train_test_split


SCALER_REGISTRY = {
    'standard': ('StandardScaler (Z-score zero-mean, unit-variance)', StandardScaler),
    'minmax': ('MinMaxScaler (Scale features to [0, 1])', MinMaxScaler),
    'robust': ('RobustScaler (Robust to outliers using IQR)', RobustScaler),
    'l2_normalizer': ('L2 Normalizer (Unit-norm per spectral profile)', Normalizer),
    'none': ('Raw Reflectance (No Scaling)', None)
}


def build_preprocessing_pipeline(
    scaler_type: str = 'standard',
    variance_threshold: float = 0.0,
    impute_strategy: str = 'median'
) -> Pipeline:
    """
    Constructs a Scikit-Learn Pipeline for hyperspectral preprocessing.
    
    Steps:
    1. Imputer: handles missing/corrupt values without data leakage
    2. VarianceThreshold: removes constant/uninformative spectral bands
    3. Scaler: scales spectral features appropriately
    """
    steps = [
        ('imputer', SimpleImputer(strategy=impute_strategy))
    ]
    
    if variance_threshold > 0.0:
        steps.append(('var_filter', VarianceThreshold(threshold=variance_threshold)))
        
    if scaler_type != 'none' and scaler_type in SCALER_REGISTRY:
        scaler_cls = SCALER_REGISTRY[scaler_type][1]
        if scaler_cls is not None:
            steps.append(('scaler', scaler_cls()))
            
    return Pipeline(steps=steps)


def create_train_test_split(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    test_size: float = 0.25,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Splits data into train and test sets to guarantee no data leakage during pipeline fitting.
    Applies stratification if target classes are provided and balanced.
    """
    stratify = None
    if y is not None:
        # Only stratify if all classes have at least 2 samples
        val_counts = y.value_counts()
        if (val_counts >= 2).all():
            stratify = y
            
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
    else:
        X_train, X_test = train_test_split(
            X, test_size=test_size, random_state=random_state
        )
        y_train, y_test = None, None
        
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_indices': X_train.index.tolist(),
        'test_indices': X_test.index.tolist()
    }


def fit_transform_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
    band_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """
    Fits the preprocessing pipeline strictly on X_train, then transforms X_train and X_test.
    Prevents data leakage.
    
    Returns:
        X_train_proc (np.ndarray): Transformed training features
        X_test_proc (Optional[np.ndarray]): Transformed test features (or None)
        active_band_names (List[str]): Band names retained after variance filtering
    """
    X_train_proc = pipeline.fit_transform(X_train)
    X_test_proc = pipeline.transform(X_test) if X_test is not None else None
    
    # Track retained band names
    active_bands = list(band_names) if band_names is not None else [f"band_{i}" for i in range(X_train.shape[1])]
    if 'var_filter' in pipeline.named_steps:
        mask = pipeline.named_steps['var_filter'].get_support()
        active_bands = [b for b, m in zip(active_bands, mask) if m]
        
    return X_train_proc, X_test_proc, active_bands


def compare_before_after(
    X_raw: pd.DataFrame,
    X_proc: np.ndarray,
    band_names: List[str]
) -> pd.DataFrame:
    """Returns comparative summary dataframe before and after preprocessing."""
    raw_means = X_raw.mean().values[:len(band_names)]
    raw_stds = X_raw.std().values[:len(band_names)]
    raw_mins = X_raw.min().values[:len(band_names)]
    raw_maxs = X_raw.max().values[:len(band_names)]
    
    proc_means = np.mean(X_proc, axis=0)
    proc_stds = np.std(X_proc, axis=0)
    proc_mins = np.min(X_proc, axis=0)
    proc_maxs = np.max(X_proc, axis=0)
    
    summary = pd.DataFrame({
        'Band': band_names,
        'Raw_Mean': raw_means,
        'Proc_Mean': proc_means,
        'Raw_Std': raw_stds,
        'Proc_Std': proc_stds,
        'Raw_Min': raw_mins,
        'Proc_Min': proc_mins,
        'Raw_Max': raw_maxs,
        'Proc_Max': proc_maxs
    })
    return summary


def save_pipeline(pipeline: Pipeline, path: str):
    """Serializes fitted pipeline to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: str) -> Pipeline:
    """Deserializes saved pipeline from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pipeline file not found: {path}")
    return joblib.load(path)

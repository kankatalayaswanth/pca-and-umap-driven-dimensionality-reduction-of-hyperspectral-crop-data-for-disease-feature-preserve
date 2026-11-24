"""
Data and Pipeline Configuration Validator
Detects anomalies, missing values, infinite entries, and invalid parameters.
"""

import pandas as pd
import numpy as np


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Validate raw uploaded dataset for fundamental integrity.
    Returns dict with 'valid' (bool), 'errors' (list), and 'warnings' (list).
    """
    errors = []
    warnings = []
    
    if df is None or df.empty:
        errors.append("The uploaded dataset is empty.")
        return {"valid": False, "errors": errors, "warnings": warnings}
        
    n_rows, n_cols = df.shape
    
    if n_rows < 3:
        errors.append(f"Insufficient samples: dataset contains only {n_rows} rows. A minimum of 3 samples is required.")
        
    if n_cols < 2:
        errors.append(f"Insufficient columns: dataset contains only {n_cols} columns. A minimum of 2 columns is required.")
        
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) < 2:
        errors.append("Dataset does not contain at least 2 numerical columns suitable for spectral feature extraction.")
        
    # Check for duplicate rows
    n_duplicates = int(df.duplicated().sum())
    if n_duplicates > 0:
        warnings.append(f"Found {n_duplicates} duplicate row(s) ({n_duplicates/n_rows*100:.1f}% of dataset). Consider enabling duplicate removal in preprocessing.")
        
    # Check for missing values
    missing_count = int(df.isnull().sum().sum())
    if missing_count > 0:
        warnings.append(f"Dataset contains {missing_count} missing value(s). Preprocessing imputation or row removal will be required.")
        
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def validate_feature_selection(df: pd.DataFrame, feature_cols: list, label_col: str = None, id_cols: list = None) -> dict:
    """
    Validate chosen feature bands, label column, and ID columns.
    """
    errors = []
    warnings = []
    id_cols = id_cols or []
    
    if not feature_cols or len(feature_cols) < 2:
        errors.append("Please select at least 2 numerical spectral band columns to perform dimensionality reduction.")
        return {"valid": False, "errors": errors, "warnings": warnings}
        
    # Ensure all selected feature columns are present and numeric
    non_numeric_features = [col for col in feature_cols if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric_features:
        errors.append(f"The following selected feature columns contain non-numeric data: {', '.join(non_numeric_features[:5])}. Only numerical spectral bands can be processed.")
        
    # Check for overlap between feature columns and label column
    if label_col and label_col in feature_cols:
        errors.append(f"Column '{label_col}' is selected as both a Feature and the Ground-Truth Label. Labels must be excluded from unsupervised feature representations.")
        
    # Check for overlap between feature columns and ID columns
    overlap_ids = set(feature_cols).intersection(set(id_cols))
    if overlap_ids:
        errors.append(f"Columns {list(overlap_ids)[:3]} are marked as both ID columns and Spectral Features. Please exclude ID columns from features.")
        
    # Check for zero-variance / constant features
    if len(errors) == 0:
        numeric_subset = df[feature_cols].select_dtypes(include=[np.number])
        variances = numeric_subset.var(numeric_only=True)
        zero_var_cols = variances[variances == 0].index.tolist()
        if zero_var_cols:
            warnings.append(f"Found {len(zero_var_cols)} constant feature(s) with zero variance (e.g. {', '.join(zero_var_cols[:3])}). Preprocessing can remove them automatically.")
            
        # Check for infinite values
        inf_count = np.isinf(numeric_subset.to_numpy(dtype=float, na_value=np.nan)).sum()
        if inf_count > 0:
            warnings.append(f"Found {inf_count} infinite value(s) in selected features. These will need to be replaced during preprocessing.")
            
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def validate_pca_config(n_samples: int, n_features: int, mode: str, variance_threshold: float, n_components: int) -> dict:
    """Validate PCA hyperparameter settings against dataset dimensions."""
    errors = []
    warnings = []
    max_possible = min(n_samples - 1, n_features)
    
    if max_possible < 1:
        errors.append("Dataset dimensions are insufficient for PCA (need at least 2 samples and 1 feature).")
        return {"valid": False, "errors": errors, "warnings": warnings}
        
    if mode == "variance_threshold":
        if not (0.01 <= variance_threshold <= 1.0):
            errors.append("Variance retention threshold must be between 0.01 (1%) and 1.0 (100%).")
    elif mode == "manual":
        if n_components < 1:
            errors.append("Number of PCA components must be at least 1.")
        elif n_components > max_possible:
            errors.append(f"Number of PCA components ({n_components}) exceeds the maximum possible rank min(N-1, D) = {max_possible}.")
            
    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def validate_umap_config(n_samples: int, n_neighbors: int, min_dist: float, n_components: int) -> dict:
    """Validate UMAP parameters against sample count."""
    errors = []
    warnings = []
    
    if n_samples < 4:
        errors.append(f"UMAP requires at least 4 samples. Current dataset has {n_samples}.")
        return {"valid": False, "errors": errors, "warnings": warnings}
        
    if n_neighbors < 2:
        errors.append("UMAP 'n_neighbors' must be at least 2.")
    elif n_neighbors >= n_samples:
        errors.append(f"UMAP 'n_neighbors' ({n_neighbors}) cannot be greater than or equal to sample count ({n_samples}).")
        
    if not (0.0 <= min_dist <= 0.99):
        errors.append("UMAP 'min_dist' must be between 0.0 and 0.99.")
        
    if n_components not in [2, 3]:
        warnings.append(f"UMAP n_components is set to {n_components}. 2 or 3 components are recommended for visualization.")
        
    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def validate_clustering_config(n_samples: int, n_clusters: int) -> dict:
    """Validate K-Means configuration."""
    errors = []
    warnings = []
    
    if n_clusters < 2:
        errors.append("Number of clusters (K) must be at least 2.")
    elif n_clusters >= n_samples:
        errors.append(f"Number of clusters K ({n_clusters}) must be strictly less than sample count ({n_samples}).")
        
    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

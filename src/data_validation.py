"""
Data Validation Module for Hyperspectral Feature Matrices.
Performs rigorous structural, numerical, and spectral integrity checks.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


def validate_hyperspectral_data(
    X: pd.DataFrame,
    y: pd.Series = None,
    min_samples: int = 10,
    min_features: int = 2
) -> Dict[str, Any]:
    """
    Validates hyperspectral spectral band matrix X and optional target series y.
    
    Returns a dictionary containing:
        - is_valid (bool): Overall pass/fail status
        - checks (Dict[str, bool]): Individual check status
        - issues (List[str]): List of error/warning messages
        - statistics (Dict[str, Any]): Structural data statistics
    """
    issues = []
    checks = {}
    
    # 1. Type and dimension checks
    if not isinstance(X, pd.DataFrame):
        return {
            'is_valid': False,
            'checks': {'is_dataframe': False},
            'issues': ['Input feature matrix X must be a pandas DataFrame.'],
            'statistics': {}
        }
        
    n_samples, n_features = X.shape
    checks['min_samples'] = bool(n_samples >= min_samples)
    if not checks['min_samples']:
        issues.append(f"Insufficient samples: found {n_samples}, minimum required is {min_samples}.")
        
    checks['min_features'] = bool(n_features >= min_features)
    if not checks['min_features']:
        issues.append(f"Insufficient spectral bands: found {n_features}, minimum required is {min_features}.")
        
    # 2. Non-numeric columns check
    non_numeric_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
    checks['all_numeric'] = len(non_numeric_cols) == 0
    if not checks['all_numeric']:
        issues.append(f"Non-numeric columns detected in spectral bands: {non_numeric_cols[:5]}...")
        
    # 3. Missing and Infinite values
    missing_count = int(X.isna().sum().sum())
    checks['no_missing'] = (missing_count == 0)
    if not checks['no_missing']:
        issues.append(f"Detected {missing_count} missing (NaN) values across spectral bands.")
        
    # Infinite values check
    numeric_x = X.select_dtypes(include=[np.number])
    inf_count = int(np.isinf(numeric_x.to_numpy()).sum()) if checks['all_numeric'] else 0
    checks['no_infinite'] = (inf_count == 0)
    if not checks['no_infinite']:
        issues.append(f"Detected {inf_count} infinite (Inf) values in spectral data.")
        
    # 4. Zero-variance / constant bands check
    var_series = numeric_x.var()
    zero_var_cols = var_series[var_series == 0].index.tolist()
    checks['no_constant_bands'] = (len(zero_var_cols) == 0)
    if not checks['no_constant_bands']:
        issues.append(f"Found {len(zero_var_cols)} constant (zero-variance) spectral bands.")
        
    # 5. Duplicate records
    duplicate_rows = int(X.duplicated().sum())
    checks['no_duplicates'] = (duplicate_rows == 0)
    if not checks['no_duplicates']:
        issues.append(f"Found {duplicate_rows} exact duplicate spectral rows.")
        
    # 6. Physical spectral range check (Reflectance typically between 0.0 and 1.0 or DN values > 0)
    min_val = float(numeric_x.min().min()) if not numeric_x.empty else 0.0
    max_val = float(numeric_x.max().max()) if not numeric_x.empty else 0.0
    
    checks['non_negative'] = (min_val >= -1e-5)
    if not checks['non_negative']:
        issues.append(f"Negative reflectance values detected (min value: {min_val:.4f}). Physical reflectance should be >= 0.")
        
    # 7. Target label checks (if present)
    target_stats = {}
    if y is not None:
        checks['target_aligned'] = (len(y) == n_samples)
        if not checks['target_aligned']:
            issues.append(f"Target length mismatch: X has {n_samples} rows but y has {len(y)} items.")
        target_missing = int(y.isna().sum())
        checks['target_no_missing'] = (target_missing == 0)
        if not checks['target_no_missing']:
            issues.append(f"Target labels contain {target_missing} missing values.")
        target_stats['n_classes'] = int(y.nunique(dropna=True))
        target_stats['class_distribution'] = y.value_counts(dropna=False).to_dict()
    else:
        checks['target_aligned'] = True
        checks['target_no_missing'] = True
        
    # Overall valid if all critical structural checks pass
    critical_checks = ['min_samples', 'min_features', 'all_numeric', 'no_infinite']
    is_valid = all(checks[k] for k in critical_checks if k in checks)
    
    return {
        'is_valid': is_valid,
        'checks': checks,
        'issues': issues,
        'statistics': {
            'n_samples': n_samples,
            'n_features': n_features,
            'missing_values': missing_count,
            'infinite_values': inf_count,
            'duplicate_rows': duplicate_rows,
            'min_value': min_val,
            'max_value': max_val,
            'constant_bands_count': len(zero_var_cols),
            'target': target_stats
        }
    }

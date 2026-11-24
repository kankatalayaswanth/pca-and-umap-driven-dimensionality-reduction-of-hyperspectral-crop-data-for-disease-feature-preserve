"""
Preprocessing Pipeline for Hyperspectral Feature Matrix
Handles missing values, scaling, constant features, and duplicates.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer


def run_preprocessing_pipeline(
    df: pd.DataFrame,
    feature_cols: list,
    label_col: str = None,
    id_cols: list = None,
    imputation_strategy: str = "mean",     # 'mean', 'median', 'drop_rows'
    scaling_method: str = "standard",       # 'standard', 'minmax', 'none'
    remove_constant: bool = True,
    remove_duplicates: bool = True,
) -> dict:
    """
    Execute preprocessing pipeline consistently on spectral feature columns.
    Preserves aligned label and ID columns.
    
    Returns a dict containing:
    - 'X_clean': np.ndarray of cleaned & scaled features
    - 'X_clean_df': pd.DataFrame with feature column names
    - 'preprocessed_df': pd.DataFrame with features + aligned label/ID columns
    - 'y_clean': pd.Series of labels (if label_col provided)
    - 'stats': Before vs After comparison statistics
    - 'feature_names': list of active features remaining after filtering
    """
    id_cols = id_cols or []
    
    # 1. Record Initial State
    initial_samples = len(df)
    initial_features = len(feature_cols)
    initial_missing = int(df[feature_cols].isnull().sum().sum())
    initial_duplicates = int(df.duplicated(subset=feature_cols).sum())
    
    # Work on a copy
    work_df = df.copy()
    
    # Replace +/- inf with NaN in feature columns
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(work_df[col]):
            work_df[col] = work_df[col].replace([np.inf, -np.inf], np.nan)
            
    # Update missing count after inf replacement
    total_missing_after_inf = int(work_df[feature_cols].isnull().sum().sum())
    inf_count_found = total_missing_after_inf - initial_missing
    
    # 2. Duplicate Row Removal (if requested)
    removed_duplicates_count = 0
    if remove_duplicates:
        before_dup = len(work_df)
        work_df = work_df.drop_duplicates(subset=feature_cols)
        removed_duplicates_count = before_dup - len(work_df)
        
    # 3. Missing Value Handling
    removed_rows_by_na = 0
    if imputation_strategy == "drop_rows":
        before_drop = len(work_df)
        work_df = work_df.dropna(subset=feature_cols)
        removed_rows_by_na = before_drop - len(work_df)
    elif imputation_strategy in ["mean", "median"]:
        imputer = SimpleImputer(strategy=imputation_strategy)
        work_df[feature_cols] = imputer.fit_transform(work_df[feature_cols])
        
    # Check if we have samples left
    if len(work_df) < 3:
        raise ValueError(f"Preprocessing resulted in only {len(work_df)} samples. Please adjust missing-value strategy.")
        
    # 4. Constant / Zero-Variance Feature Removal
    removed_constant_cols = []
    active_features = list(feature_cols)
    if remove_constant:
        var_series = work_df[active_features].var(numeric_only=True)
        zero_var_cols = var_series[var_series == 0].index.tolist()
        # Also check for single unique value (e.g. all 1s or all 0s)
        for col in active_features:
            if col not in zero_var_cols and work_df[col].nunique(dropna=True) <= 1:
                zero_var_cols.append(col)
                
        zero_var_cols = list(set(zero_var_cols))
        if zero_var_cols:
            removed_constant_cols = zero_var_cols
            active_features = [c for c in active_features if c not in zero_var_cols]
            
    if len(active_features) < 2:
        raise ValueError("After removing constant features, fewer than 2 spectral bands remain. Cannot continue.")
        
    # Extract feature matrix
    X_raw = work_df[active_features].to_numpy(dtype=float)
    
    # 5. Feature Scaling
    scaler_name = "None"
    if scaling_method == "standard":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        scaler_name = "Standard Normalization (Z-Score: mean=0, std=1)"
    elif scaling_method == "minmax":
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler.fit_transform(X_raw)
        scaler_name = "Min-Max Normalization (0.0 to 1.0 range)"
    else:
        X_scaled = X_raw.copy()
        scaler_name = "Raw Intensities (No Scaling Applied)"
        
    # Prepare Output DataFrames
    X_clean_df = pd.DataFrame(X_scaled, columns=active_features, index=work_df.index)
    
    # Combine with label and ID columns if available
    preprocessed_df = X_clean_df.copy()
    y_clean = None
    if label_col and label_col in work_df.columns:
        y_clean = work_df[label_col].copy()
        preprocessed_df[label_col] = y_clean
        
    for idc in id_cols:
        if idc in work_df.columns and idc not in preprocessed_df.columns:
            preprocessed_df[idc] = work_df[idc]
            
    # Reset index for consistent integer alignment across downstream representations
    preprocessed_df = preprocessed_df.reset_index(drop=True)
    X_clean_df = X_clean_df.reset_index(drop=True)
    if y_clean is not None:
        y_clean = y_clean.reset_index(drop=True)
        
    # Summary Statistics (Before vs After)
    stats = {
        "initial_samples": initial_samples,
        "final_samples": len(work_df),
        "removed_rows": initial_samples - len(work_df),
        "removed_duplicates": removed_duplicates_count,
        "removed_na_rows": removed_rows_by_na,
        "initial_features": initial_features,
        "final_features": len(active_features),
        "removed_features_count": len(removed_constant_cols),
        "removed_features_list": removed_constant_cols,
        "initial_missing_count": initial_missing,
        "final_missing_count": int(np.isnan(X_scaled).sum()),
        "inf_count_replaced": inf_count_found,
        "imputation_strategy": imputation_strategy,
        "scaling_method": scaler_name,
        "sample_retention_pct": round((len(work_df) / initial_samples) * 100, 2),
        "feature_retention_pct": round((len(active_features) / initial_features) * 100, 2),
    }
    
    return {
        "X_clean": X_scaled,
        "X_clean_df": X_clean_df,
        "preprocessed_df": preprocessed_df,
        "y_clean": y_clean,
        "stats": stats,
        "feature_names": active_features,
    }

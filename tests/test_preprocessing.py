"""
Unit tests for hyperspectral preprocessing pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from preprocessing.pipeline import run_preprocessing_pipeline


def test_preprocessing_pipeline():
    df = pd.DataFrame({
        "b1": [1.0, 2.0, np.nan, 4.0, 5.0],
        "b2": [10.0, 20.0, 30.0, 40.0, 50.0],
        "const_band": [1.0, 1.0, 1.0, 1.0, 1.0],  # zero variance
        "label": ["healthy", "rust", "rust", "healthy", "rust"]
    })
    
    res = run_preprocessing_pipeline(
        df=df,
        feature_cols=["b1", "b2", "const_band"],
        label_col="label",
        imputation_strategy="mean",
        scaling_method="standard",
        remove_constant=True,
        remove_duplicates=True,
    )
    
    assert res["stats"]["removed_features_count"] == 1
    assert "const_band" in res["stats"]["removed_features_list"]
    assert res["X_clean"].shape[1] == 2
    assert res["stats"]["final_missing_count"] == 0
    # StandardScaler output has approximately 0 mean and 1 variance
    assert np.allclose(res["X_clean"].mean(axis=0), [0.0, 0.0], atol=1e-7)
    assert np.allclose(res["X_clean"].std(axis=0), [1.0, 1.0], atol=1e-7)

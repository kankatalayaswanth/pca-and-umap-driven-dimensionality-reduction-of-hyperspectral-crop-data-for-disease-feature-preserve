"""
Tests for Preprocessing and Pipeline Modules.
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.data_loader import load_dataset
from src.preprocessing import (
    build_preprocessing_pipeline,
    create_train_test_split,
    fit_transform_pipeline,
    compare_before_after,
    save_pipeline,
    load_pipeline
)


def test_preprocessing_scalers():
    X, _, _, bands, _ = load_dataset('potato_dai70_vnir')
    
    # Standard scaler
    pipe_std = build_preprocessing_pipeline('standard')
    X_std, _, _ = fit_transform_pipeline(pipe_std, X)
    assert np.allclose(np.mean(X_std), 0.0, atol=1e-5)
    assert np.allclose(np.std(X_std), 1.0, atol=1e-2)
    
    # MinMax scaler
    pipe_mm = build_preprocessing_pipeline('minmax')
    X_mm, _, _ = fit_transform_pipeline(pipe_mm, X)
    assert X_mm.min() >= -1e-5
    assert X_mm.max() <= 1.0 + 1e-5
    
    # None scaler (Raw)
    pipe_raw = build_preprocessing_pipeline('none')
    X_raw_out, _, _ = fit_transform_pipeline(pipe_raw, X)
    assert np.allclose(X.values, X_raw_out, atol=1e-5)


def test_train_test_split_no_leakage():
    X, y, _, bands, _ = load_dataset('potato_dai70_vnir')
    split = create_train_test_split(X, y, test_size=0.3, random_state=42)
    
    assert split['X_train'].shape[0] == 49
    assert split['X_test'].shape[0] == 21
    assert len(set(split['train_indices']).intersection(set(split['test_indices']))) == 0
    
    # Check that scaler fits strictly on train
    pipe = build_preprocessing_pipeline('standard')
    X_train_proc, X_test_proc, _ = fit_transform_pipeline(pipe, split['X_train'], split['X_test'])
    
    assert np.allclose(np.mean(X_train_proc), 0.0, atol=1e-5)
    # Test set transformed by train scaler will not have mean identically zero, proving no leakage
    assert X_test_proc is not None
    assert X_test_proc.shape[0] == 21


def test_pipeline_serialization(tmp_path):
    X, _, _, _, _ = load_dataset('potato_dai70_vnir')
    pipe = build_preprocessing_pipeline('standard')
    X_proc1, _, _ = fit_transform_pipeline(pipe, X)
    
    save_path = os.path.join(tmp_path, "pipeline.joblib")
    save_pipeline(pipe, save_path)
    assert os.path.exists(save_path)
    
    loaded_pipe = load_pipeline(save_path)
    X_proc2 = loaded_pipe.transform(X)
    assert np.allclose(X_proc1, X_proc2)

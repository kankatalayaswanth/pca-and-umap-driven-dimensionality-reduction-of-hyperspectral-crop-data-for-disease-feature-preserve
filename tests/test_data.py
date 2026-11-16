"""
Tests for Data Loader and Data Validation Modules.
"""

import os
import io
import pytest
import pandas as pd
import numpy as np
from src.data_loader import load_dataset, get_available_datasets, load_custom_file
from src.data_validation import validate_hyperspectral_data


def test_get_available_datasets():
    datasets = get_available_datasets()
    assert isinstance(datasets, dict)
    assert 'potato_dai70_vnir' in datasets
    assert 'indian_pines_crops' in datasets


def test_load_potato_dai70_vnir():
    X, y, meta, bands, wls = load_dataset('potato_dai70_vnir')
    assert isinstance(X, pd.DataFrame)
    assert X.shape[0] == 70
    assert X.shape[1] == 160
    assert y is not None
    assert len(y) == 70
    assert len(bands) == 160
    assert len(wls) == 160
    assert wls[0] < wls[-1]  # Monotonically increasing wavelengths


def test_load_indian_pines():
    X, y, meta, bands, wls = load_dataset('indian_pines_crops')
    assert isinstance(X, pd.DataFrame)
    assert X.shape[0] > 1000
    assert X.shape[1] == 200
    assert y is not None


def test_data_validation_valid():
    X, y, _, _, _ = load_dataset('potato_dai70_vnir')
    res = validate_hyperspectral_data(X, y)
    assert res['is_valid'] is True
    assert len(res['issues']) == 0
    assert res['statistics']['n_samples'] == 70
    assert res['statistics']['n_features'] == 160


def test_data_validation_invalid_shapes():
    # Less than minimum samples
    X_small = pd.DataFrame(np.random.randn(3, 5))
    res = validate_hyperspectral_data(X_small, min_samples=10)
    assert res['is_valid'] is False
    assert any("Insufficient samples" in s for s in res['issues'])


def test_load_custom_file_simulated():
    csv_data = "sample_id,class_name,band_450nm,band_550nm,band_650nm\ns1,healthy,0.1,0.2,0.3\ns2,diseased,0.2,0.3,0.4\n"
    file_obj = io.StringIO(csv_data)
    file_obj.name = "custom_test.csv"
    
    # Load with pandas StringIO wrapper
    df = pd.read_csv(io.StringIO(csv_data))
    tmp_path = "tests/temp_test.csv"
    df.to_csv(tmp_path, index=False)
    
    with open(tmp_path, 'rb') as f:
        X, y, meta, bands, wls = load_custom_file(f, target_col='class_name')
        
    os.remove(tmp_path)
    
    assert X.shape == (2, 3)
    assert list(bands) == ['band_450nm', 'band_550nm', 'band_650nm']
    assert wls == [450.0, 550.0, 650.0]
    assert y.tolist() == ['healthy', 'diseased']

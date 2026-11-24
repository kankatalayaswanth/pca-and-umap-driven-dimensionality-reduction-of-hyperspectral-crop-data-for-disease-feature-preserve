"""
Unit tests for data loading and validation modules.
"""

import pytest
import numpy as np
import pandas as pd
from data.loader import inspect_dataset, extract_wavelength_metadata
from data.validator import validate_dataset, validate_feature_selection


def test_inspect_dataset():
    df = pd.DataFrame({
        "band_450nm": [0.1, 0.2, 0.3],
        "band_550nm": [0.4, 0.5, 0.6],
        "label": ["A", "B", "A"]
    })
    meta = inspect_dataset(df, "test.csv")
    assert meta["n_rows"] == 3
    assert meta["n_cols"] == 3
    assert meta["n_numerical"] == 2
    assert meta["n_non_numerical"] == 1
    assert "band_450nm" in meta["wavelength_meta"]
    assert meta["wavelength_meta"]["band_450nm"] == 450.0


def test_validate_dataset():
    # Empty
    assert validate_dataset(pd.DataFrame())["valid"] is False
    # Insufficient rows
    assert validate_dataset(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))["valid"] is False
    # Valid
    df = pd.DataFrame({"b1": [1, 2, 3], "b2": [4, 5, 6]})
    assert validate_dataset(df)["valid"] is True


def test_validate_feature_selection():
    df = pd.DataFrame({
        "b1": [1.0, 2.0, 3.0],
        "b2": [4.0, 5.0, 6.0],
        "class": ["H", "D", "H"],
        "id": ["s1", "s2", "s3"]
    })
    # Valid selection
    res = validate_feature_selection(df, ["b1", "b2"], label_col="class", id_cols=["id"])
    assert res["valid"] is True
    
    # Overlap between label and feature
    res_overlap = validate_feature_selection(df, ["b1", "class"], label_col="class")
    assert res_overlap["valid"] is False


def test_curated_datasets():
    from data.loader import get_curated_datasets, load_curated_dataset
    datasets = get_curated_datasets()
    assert len(datasets) >= 5
    assert "potato_vnir" in datasets
    assert "wheat_rust" in datasets
    assert "citrus_hlb" in datasets

    # Test loading one curated dataset
    df = load_curated_dataset("potato_vnir")
    assert not df.empty
    assert len(df) == 70
    assert "Disease_Class" in df.columns

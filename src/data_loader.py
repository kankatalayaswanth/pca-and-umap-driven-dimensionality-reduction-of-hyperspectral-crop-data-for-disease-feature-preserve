"""
Data Loader Module for Hyperspectral Crop Datasets.
Supports:
1. Potato Crop Disease & Stress Hyperspectral Dataset (Zenodo 14267877)
2. Indian Pines AVIRIS Benchmark Hyperspectral Dataset
3. Custom user-uploaded CSV / Parquet spectral tables
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')

AVAILABLE_DATASETS = {
    'potato_dai70_vnir': {
        'name': 'Potato Crop Disease - DAI 70 (VNIR 160 Bands)',
        'file': 'potato_disease_dai70_vnir.csv',
        'target_col': 'Disease_Class',
        'description': 'Leaf-level hyperspectral reflectance (409-988 nm) of potato plants at Day 70 (symptomatic cyst maturation stage) exposed to biotic nematode pests and abiotic drought stress.',
        'citation': 'Lapajne et al. (2024), Zenodo DOI: 10.5281/zenodo.14267877, Agricultural Institute of Slovenia.',
        'spectral_type': 'VNIR (160 bands, 409.8 - 988.4 nm)',
        'meta_cols': ['Pot', 'Stage', 'Disease_Class', 'Setup_clean', 'Watering', 'Nematode', 'Inoculum', 'Pf_cyst']
    },
    'potato_dai70_allbands': {
        'name': 'Potato Crop Disease - DAI 70 (VNIR + SWIR 448 Bands)',
        'file': 'potato_disease_dai70_allbands.csv',
        'target_col': 'Disease_Class',
        'description': 'Full-range hyperspectral reflectance (409-2509 nm, 448 bands) of potato plants at Day 70, capturing both chlorophyll/pigment dynamics (VNIR) and moisture/lignin structure (SWIR).',
        'citation': 'Lapajne et al. (2024), Zenodo DOI: 10.5281/zenodo.14267877, Agricultural Institute of Slovenia.',
        'spectral_type': 'VNIR + SWIR (448 bands, 409.8 - 2509.6 nm)',
        'meta_cols': ['Pot', 'Stage', 'Disease_Class', 'Setup_clean', 'Watering', 'Nematode', 'Inoculum', 'Pf_cyst']
    },
    'potato_multistage_vnir': {
        'name': 'Potato Crop Disease - Multi-Stage Progression (VNIR 160 Bands)',
        'file': 'potato_disease_multistage_vnir.csv',
        'target_col': 'Disease_Class',
        'description': 'Multi-temporal hyperspectral signatures tracked across 3 lifecycle stages: DAI 10 (early asymptomatic), DAI 40 (metabolically active), and DAI 70 (symptomatic maturation). Total 213 samples.',
        'citation': 'Lapajne et al. (2024), Zenodo DOI: 10.5281/zenodo.14267877, Agricultural Institute of Slovenia.',
        'spectral_type': 'VNIR (160 bands, 409.8 - 988.4 nm)',
        'meta_cols': ['Pot', 'Stage', 'Disease_Class', 'Setup_clean', 'Watering', 'Nematode', 'Inoculum', 'Pf_cyst']
    },
    'indian_pines_crops': {
        'name': 'Indian Pines AVIRIS Benchmark (200 Bands, 16 Classes)',
        'file': 'indian_pines_crops.csv',
        'target_col': 'Crop_Class',
        'description': 'Benchmark airborne hyperspectral scene over northwestern Indiana agricultural fields, measuring 200 continuous bands across 16 crop/vegetation classes.',
        'citation': 'Purposely curated AVIRIS benchmark dataset by Purdue University / NASA JPL.',
        'spectral_type': 'AVIRIS (200 bands, 0.4 - 2.5 µm)',
        'meta_cols': ['Class_ID', 'Crop_Class']
    }
}


def get_available_datasets() -> Dict[str, Dict[str, Any]]:
    """Returns metadata dictionaries of all pre-configured datasets."""
    return AVAILABLE_DATASETS


def load_dataset(dataset_key: str) -> Tuple[pd.DataFrame, Optional[pd.Series], pd.DataFrame, List[str], List[float]]:
    """
    Loads one of the pre-configured hyperspectral datasets.
    
    Returns:
        X (pd.DataFrame): Spectral band features only (numeric)
        y (Optional[pd.Series]): Categorical target labels (if present)
        metadata_df (pd.DataFrame): Non-spectral metadata columns
        band_names (List[str]): List of band column names
        wavelengths (List[float]): Physical wavelength values in nm if available, else index numbers
    """
    if dataset_key not in AVAILABLE_DATASETS:
        raise ValueError(f"Unknown dataset key '{dataset_key}'. Choose from: {list(AVAILABLE_DATASETS.keys())}")
        
    info = AVAILABLE_DATASETS[dataset_key]
    file_path = os.path.join(PROCESSED_DATA_DIR, info['file'])
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}. Please verify data generation.")
        
    df = pd.read_csv(file_path)
    meta_cols = [c for c in info['meta_cols'] if c in df.columns]
    metadata_df = df[meta_cols].copy()
    
    target_col = info['target_col']
    y = df[target_col].copy() if target_col in df.columns else None
    
    band_cols = [c for c in df.columns if c not in meta_cols]
    X = df[band_cols].copy()
    
    # Extract numerical wavelengths if encoded in column name (e.g., band_vnir_409.8nm)
    wavelengths = []
    for col in band_cols:
        parts = col.replace('nm', '').split('_')
        try:
            wl = float(parts[-1])
            wavelengths.append(wl)
        except ValueError:
            wavelengths.append(float(len(wavelengths) + 1))
            
    return X, y, metadata_df, band_cols, wavelengths


def load_custom_file(uploaded_file, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.Series], pd.DataFrame, List[str], List[float]]:
    """
    Loads a user-uploaded CSV or Parquet file.
    Automatically segregates numeric spectral bands from categorical metadata.
    """
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.parquet'):
        df = pd.read_parquet(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Please upload a .csv or .parquet file.")
        
    y = None
    meta_cols = []
    if target_col and target_col in df.columns:
        y = df[target_col].copy()
        meta_cols.append(target_col)
        
    # Categorize non-numeric columns as metadata
    for col in df.columns:
        if col != target_col and not pd.api.types.is_numeric_dtype(df[col]):
            meta_cols.append(col)
            
    band_cols = [c for c in df.columns if c not in meta_cols]
    X = df[band_cols].copy()
    metadata_df = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)
    
    wavelengths = []
    for col in band_cols:
        parts = col.replace('nm', '').split('_')
        try:
            wl = float(parts[-1])
            wavelengths.append(wl)
        except ValueError:
            wavelengths.append(float(len(wavelengths) + 1))
            
    return X, y, metadata_df, band_cols, wavelengths

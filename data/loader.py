"""
Dataset Loader and Hyperspectral Metadata Extractor
Supports CSV, Excel (XLSX, XLS), and Parquet formats.
Includes built-in curated hyperspectral agricultural benchmarks.
"""

import io
import os
import re
import pandas as pd
import numpy as np
import streamlit as st


CURATED_DATASETS = {
    "potato_vnir": {
        "title": "🥔 Potato Late Blight (DAI 70 - VNIR 160 Bands)",
        "file_name": "potato_disease_dai70_vnir.csv",
        "path": os.path.join("data", "processed", "potato_disease_dai70_vnir.csv"),
        "description": "70 samples across Inoculated, Uninoculated, and Control potato plants at DAI 70 across 160 contiguous VNIR bands (400–1000 nm). Fast and recommended for instant evaluation.",
        "samples": 70,
        "bands": 160,
        "domain": "VNIR (400–1000 nm)",
        "default_label": "Disease_Class",
    },
    "potato_multistage": {
        "title": "🥔 Potato Multi-Stage Foliar Infection (213 Samples, 160 Bands)",
        "file_name": "potato_disease_multistage_vnir.csv",
        "path": os.path.join("data", "processed", "potato_disease_multistage_vnir.csv"),
        "description": "213 samples tracking longitudinal late blight progression across 3 measurement stages (DAI 70, 77, 84).",
        "samples": 213,
        "bands": 160,
        "domain": "VNIR (400–1000 nm)",
        "default_label": "Disease_Class",
    },
    "wheat_rust": {
        "title": "🌾 Wheat Foliar Pathology: Yellow Rust & Powdery Mildew (600 Samples, 100 Bands)",
        "file_name": "wheat_yellow_rust_vnir.csv",
        "path": os.path.join("data", "processed", "wheat_yellow_rust_vnir.csv"),
        "description": "600 samples comparing Healthy, Stripe/Yellow Rust (Puccinia striiformis), and Powdery Mildew across 100 contiguous VNIR spectral bands (450–950 nm).",
        "samples": 600,
        "bands": 100,
        "domain": "VNIR (450–950 nm)",
        "default_label": "Disease_Class",
    },
    "citrus_hlb": {
        "title": "🍊 Citrus Huanglongbing (HLB Greening) vs Canker (450 Samples, 80 Bands)",
        "file_name": "citrus_greening_hlb_vnir.csv",
        "path": os.path.join("data", "processed", "citrus_greening_hlb_vnir.csv"),
        "description": "450 samples evaluating bacterial Huanglongbing (Candidatus Liberibacter) chlorosis against Citrus Canker necrotic lesions across 80 bands (420–900 nm).",
        "samples": 450,
        "bands": 80,
        "domain": "VNIR (420–900 nm)",
        "default_label": "Disease_Class",
    },
    "indian_pines_balanced": {
        "title": "🌽 Indian Pines Balanced Crop Classification (1,050 Samples, 200 Bands)",
        "file_name": "indian_pines_balanced_crops.csv",
        "path": os.path.join("data", "processed", "indian_pines_balanced_crops.csv"),
        "description": "1,050 samples across 7 agricultural crop classes (Corn, Soybean, Pasture, Hay, Woods) captured by AVIRIS spectrometer across 200 calibrated reflectance bands.",
        "samples": 1050,
        "bands": 200,
        "domain": "VIS-NIR-SWIR (400–2500 nm)",
        "default_label": "Crop_Class",
    },
    "potato_allbands": {
        "title": "🥔 Potato Late Blight (Full Spectrum 448 Bands VNIR+SWIR)",
        "file_name": "potato_disease_dai70_allbands.csv",
        "path": os.path.join("data", "processed", "potato_disease_dai70_allbands.csv"),
        "description": "70 samples across full 400–2500 nm spectrum (160 VNIR + 288 SWIR bands). Benchmarks high-dimensional collinearity and SWIR absorption features.",
        "samples": 70,
        "bands": 448,
        "domain": "VNIR + SWIR (400–2500 nm)",
        "default_label": "Disease_Class",
    },
}


def get_curated_datasets() -> dict:
    """Return dictionary of available curated hyperspectral datasets."""
    return CURATED_DATASETS


@st.cache_data
def load_curated_dataset(dataset_key: str) -> pd.DataFrame:
    """Load a curated dataset by key from disk and cache in memory."""
    if dataset_key not in CURATED_DATASETS:
        raise ValueError(f"Unknown curated dataset key '{dataset_key}'.")
    meta = CURATED_DATASETS[dataset_key]
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(root_dir, meta["path"])
    df = pd.read_csv(full_path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Load an uploaded file object into a pandas DataFrame.
    Supports CSV, Excel (.xlsx, .xls), and Parquet (.parquet, .pq).
    """
    file_name = uploaded_file.name.lower()
    
    if file_name.endswith(".csv"):
        # Attempt standard comma delimiter first, then try python engine fallback if delimiter differs
        try:
            content = uploaded_file.getvalue()
            # Try utf-8 then latin1
            try:
                text_content = content.decode("utf-8")
            except UnicodeDecodeError:
                text_content = content.decode("latin1")
            
            # Detect separator
            first_line = text_content.splitlines()[0] if text_content.splitlines() else ""
            sep = ","
            if "\t" in first_line and first_line.count("\t") > first_line.count(","):
                sep = "\t"
            elif ";" in first_line and first_line.count(";") > first_line.count(","):
                sep = ";"
                
            df = pd.read_csv(io.StringIO(text_content), sep=sep)
        except Exception as e:
            # Fallback to standard read_csv
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            
    elif file_name.endswith((".xlsx", ".xls")):
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file, engine="openpyxl")
        
    elif file_name.endswith((".parquet", ".pq")):
        uploaded_file.seek(0)
        df = pd.read_parquet(uploaded_file)
        
    else:
        raise ValueError(f"Unsupported file format '{uploaded_file.name}'. Please upload CSV, XLSX, or Parquet.")
        
    # Strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]
    return df


def inspect_dataset(df: pd.DataFrame, file_name: str, file_size_bytes: int = 0) -> dict:
    """
    Extract comprehensive metadata and summary statistics from uploaded DataFrame.
    """
    n_rows, n_cols = df.shape
    
    # Identify numerical vs non-numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numerical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Missing values analysis
    missing_series = df.isnull().sum()
    total_missing = int(missing_series.sum())
    missing_by_col = missing_series[missing_series > 0].to_dict()
    missing_pct = (total_missing / (n_rows * n_cols) * 100) if (n_rows * n_cols) > 0 else 0.0
    
    # Duplicate rows
    n_duplicates = int(df.duplicated().sum())
    
    # Memory size formatting
    mem_bytes = df.memory_usage(deep=True).sum()
    if mem_bytes < 1024 * 1024:
        memory_str = f"{mem_bytes / 1024:.1f} KB"
    else:
        memory_str = f"{mem_bytes / (1024 * 1024):.2f} MB"
        
    # Data types count
    dtype_counts = df.dtypes.astype(str).value_counts().to_dict()
    
    # Try parsing wavelength metadata from column names
    wavelength_meta = extract_wavelength_metadata(df.columns.tolist())
    
    return {
        "file_name": file_name,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "numerical_cols": numerical_cols,
        "non_numerical_cols": non_numerical_cols,
        "n_numerical": len(numerical_cols),
        "n_non_numerical": len(non_numerical_cols),
        "total_missing": total_missing,
        "missing_pct": round(missing_pct, 2),
        "missing_by_col": missing_by_col,
        "n_duplicates": n_duplicates,
        "memory_size": memory_str,
        "dtype_counts": dtype_counts,
        "wavelength_meta": wavelength_meta,
    }


def extract_wavelength_metadata(columns: list) -> dict:
    """
    Attempt to extract physical wavelength values (in nanometers) from column names.
    Examples of hyperspectral band naming:
    - '450.5', '500', '1000' (direct numeric wavelength)
    - 'band_450nm', 'band_450', 'b_450'
    - 'R450', 'W450', 'WL_450'
    - 'wavelength_450'
    Returns a dict mapping col_name -> float(wavelength) where successfully parsed.
    """
    wavelengths = {}
    pattern = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")
    
    for col in columns:
        col_str = str(col).strip().lower()
        # Direct float match
        try:
            val = float(col_str)
            if 300 <= val <= 2600:  # Typical UV-VIS-NIR-SWIR range (350 - 2500 nm)
                wavelengths[col] = val
                continue
        except ValueError:
            pass
            
        # Match pattern with nm, band, wl, etc.
        if any(prefix in col_str for prefix in ["nm", "band", "wl", "wavelength", "b_", "b", "r"]):
            matches = pattern.findall(col_str)
            if matches:
                try:
                    val = float(matches[0])
                    if 300 <= val <= 2600:
                        wavelengths[col] = val
                except ValueError:
                    pass
                    
    return wavelengths

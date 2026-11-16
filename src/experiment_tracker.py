"""
Experiment Tracker Module for Hyperspectral Unsupervised Learning.
Persists and manages experimental records (JSON/CSV) for reproducible ML lifecycle tracking.
"""

import os
import json
import uuid
from datetime import datetime
import pandas as pd
from typing import Dict, Any, List, Optional


DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
DEFAULT_EXPERIMENTS_FILE = os.path.join(DEFAULT_RESULTS_DIR, 'experiments.json')


def log_experiment(
    dataset_name: str,
    representation: str,
    algorithm: str,
    preprocessing_config: Dict[str, Any],
    hyperparameters: Dict[str, Any],
    metrics: Dict[str, Any],
    representation_params: Optional[Dict[str, Any]] = None,
    random_seed: int = 42,
    notes: str = "",
    file_path: str = DEFAULT_EXPERIMENTS_FILE
) -> Dict[str, Any]:
    """
    Logs a completed clustering experiment with full metadata and evaluation metrics.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    internal = metrics.get('internal_metrics', {})
    external = metrics.get('external_metrics', {})
    
    experiment_record = {
        'experiment_id': str(uuid.uuid4())[:8],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'dataset': dataset_name,
        'representation': representation,
        'algorithm': algorithm,
        'preprocessing': preprocessing_config,
        'representation_params': representation_params or {},
        'hyperparameters': hyperparameters,
        'n_clusters': metrics.get('n_clusters', 0),
        'noise_count': metrics.get('noise_count', 0),
        'silhouette': internal.get('silhouette_score', None),
        'calinski_harabasz': internal.get('calinski_harabasz_score', None),
        'davies_bouldin': internal.get('davies_bouldin_score', None),
        'ari': external.get('adjusted_rand_index', None),
        'nmi': external.get('normalized_mutual_info', None),
        'v_measure': external.get('v_measure', None),
        'random_seed': random_seed,
        'notes': notes
    }
    
    # Load existing records
    records = []
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                records = json.load(f)
        except Exception:
            records = []
            
    records.append(experiment_record)
    
    # Save back to JSON
    with open(file_path, 'w') as f:
        json.dump(records, f, indent=2)
        
    # Also save synchronized CSV
    csv_path = file_path.replace('.json', '.csv')
    df = pd.DataFrame(records)
    # Flatten nested dictionaries for clean CSV display
    df_flat = df.copy()
    df_flat['preprocessing'] = df_flat['preprocessing'].apply(json.dumps)
    df_flat['representation_params'] = df_flat['representation_params'].apply(json.dumps)
    df_flat['hyperparameters'] = df_flat['hyperparameters'].apply(json.dumps)
    df_flat.to_csv(csv_path, index=False)
    
    return experiment_record


def get_all_experiments(file_path: str = DEFAULT_EXPERIMENTS_FILE) -> pd.DataFrame:
    """
    Retrieves all past experimental records as a structured Pandas DataFrame.
    """
    if not os.path.exists(file_path):
        return pd.DataFrame()
        
    try:
        with open(file_path, 'r') as f:
            records = json.load(f)
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame()


def clear_experiments(file_path: str = DEFAULT_EXPERIMENTS_FILE):
    """Clears all stored experiments."""
    if os.path.exists(file_path):
        os.remove(file_path)
    csv_path = file_path.replace('.json', '.csv')
    if os.path.exists(csv_path):
        os.remove(csv_path)

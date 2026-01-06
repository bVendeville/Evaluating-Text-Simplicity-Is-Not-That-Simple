"""
Data loading utilities for metric evaluation.

Each dataset file contains pre-computed metric scores for evaluation:
- arts94.pkl, arts300.pkl, arts3k.pkl: ARTS datasets
- sda.pkl: Simplicity-DA
- st_sent.pkl, st_para.pkl, d_wiki.pkl: Human-annotated datasets

File structure:
- metric_scores: Dict[str, np.ndarray] mapping metric name to scores
  - 'human': ground truth human scores
  - 'SLE', 'LENS-SALSA', 'BATS-GB', 'BATS-RF', 'FKGL', etc.
- n_samples: number of samples
- scale: (optional) score scale tuple
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


def _load_pickle(path: Path) -> Any:
    """Load a pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_dataset(name: str) -> Dict:
    """
    Load a dataset with metric scores.

    Args:
        name: Dataset name (arts94, arts300, arts3k, sda,
              st_sent, st_para, d_wiki)

    Returns:
        Dict with:
        - metric_scores: Dict[str, np.ndarray] of metric scores
        - human_scores: np.ndarray of human scores (ground truth)
        - n_samples: int
        - scale: optional tuple
    """
    path = DATA_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = _load_pickle(path)

    return {
        'metric_scores': data['metric_scores'],
        'human_scores': data['metric_scores']['human'],
        'n_samples': data['n_samples'],
        'scale': data.get('scale'),
    }


def get_metric_names(data: Dict) -> List[str]:
    """Get list of metric names (excluding 'human')."""
    return [k for k in data['metric_scores'].keys() if k != 'human']


# =============================================================================
# CONVENIENCE LOADERS
# =============================================================================

def load_arts94() -> Dict:
    """Load ARTS94 dataset."""
    data = load_dataset('arts94')
    print(f"  ARTS94: {data['n_samples']} samples, {len(get_metric_names(data))} metrics")
    return data


def load_arts300() -> Dict:
    """Load ARTS300 dataset."""
    data = load_dataset('arts300')
    print(f"  ARTS300: {data['n_samples']} samples, {len(get_metric_names(data))} metrics")
    return data


def load_arts3k() -> Dict:
    """Load ARTS3k dataset."""
    data = load_dataset('arts3k')
    print(f"  ARTS3k: {data['n_samples']} samples, {len(get_metric_names(data))} metrics")
    return data


def load_sda() -> Dict:
    """Load Simplicity-DA dataset."""
    data = load_dataset('sda')
    print(f"  SDA: {data['n_samples']} samples, {len(get_metric_names(data))} metrics")
    return data


def load_st_sent() -> Dict:
    """Load ST-sent dataset."""
    data = load_dataset('st_sent')
    print(f"  ST-sent: {data['n_samples']} samples, {len(get_metric_names(data))} metrics, scale {data.get('scale')}")
    return data


def load_st_para() -> Dict:
    """Load ST-para dataset."""
    data = load_dataset('st_para')
    print(f"  ST-para: {data['n_samples']} samples, {len(get_metric_names(data))} metrics, scale {data.get('scale')}")
    return data


def load_d_wiki() -> Dict:
    """Load D-Wiki dataset."""
    data = load_dataset('d_wiki')
    print(f"  D-Wiki: {data['n_samples']} samples, {len(get_metric_names(data))} metrics, scale {data.get('scale')}")
    return data


# =============================================================================
# GROUPED LOADERS
# =============================================================================

def load_arts_datasets() -> Dict[str, Dict]:
    """Load all ARTS datasets."""
    return {
        'ARTS94': load_arts94(),
        'ARTS300': load_arts300(),
        'ARTS3k': load_arts3k(),
    }


def load_human_annotated_datasets() -> Dict[str, Dict]:
    """Load all human-annotated datasets."""
    return {
        'ST-sent': load_st_sent(),
        'ST-para': load_st_para(),
        'D-Wiki': load_d_wiki(),
    }


def load_all_datasets() -> Dict[str, Dict]:
    """Load all available datasets."""
    print("Loading all datasets...")

    datasets = {}

    print("\nARTS:")
    datasets.update(load_arts_datasets())

    print("\nSDA:")
    datasets['SDA'] = load_sda()

    print("\nHuman-annotated:")
    datasets.update(load_human_annotated_datasets())

    return datasets


# =============================================================================
# SUMMARY
# =============================================================================

def get_dataset_summary() -> pd.DataFrame:
    """Get summary of all datasets and their metrics."""
    datasets = load_all_datasets()

    summary_data = []
    for name, data in datasets.items():
        metrics = get_metric_names(data)
        human = data['human_scores']

        summary_data.append({
            'Dataset': name,
            'Samples': data['n_samples'],
            'Metrics': len(metrics),
            'Score Range': f"[{np.nanmin(human):.2f}, {np.nanmax(human):.2f}]",
            'Scale': str(data.get('scale', '-')),
        })

    return pd.DataFrame(summary_data)


def get_available_metrics() -> Dict[str, List[str]]:
    """Get available metrics for each dataset."""
    datasets = load_all_datasets()
    return {name: get_metric_names(data) for name, data in datasets.items()}


if __name__ == '__main__':
    print("=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    summary = get_dataset_summary()
    print(summary.to_string(index=False))

    print("\n" + "=" * 60)
    print("Available Metrics")
    print("=" * 60)
    metrics = get_available_metrics()
    for name, m_list in metrics.items():
        print(f"\n{name}: {', '.join(m_list)}")
